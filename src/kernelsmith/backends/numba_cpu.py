"""Numba parallel CPU backend.

The first backend that generates code rather than interpreting. The whole graph
becomes one ``@njit(parallel=True)`` function whose outer loop is ``prange``
over parameter sets, so a sweep uses every core.

Layout is ``(P, slots, T)`` - time innermost - because each core owns one
parameter set and walks time sequentially, so consecutive reads should be
consecutive addresses. The CUDA backend will choose the opposite for the same
reason applied to warps. Nothing outside `_reference` knows either layout.

Feature kernels are opaque and stay function calls; elementwise expressions are
inlined as ``for t`` loops, which is also where fusion will land later.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange, types

from kernelsmith.backends.base import Backend, CompiledProgram, check_call_arguments
from kernelsmith.backends.binding import Binding, bind
from kernelsmith.dsl import Call, DType, Expr, Graph, Op, Shape, Signature, ValueNode, VarRole
from kernelsmith.errors import GraphError, KernelsmithError
from kernelsmith.ir import Allocation, Liveness, PoolKind, allocate, cse
from kernelsmith.ir.fuse import FusedExpr, fuse

NUMBA_CPU_FN_REGISTER: Dict[object, "NumbaCPU_FPointer"] = {}

_DTYPE_TAG = {DType.BOOL: "b1", DType.INT32: "i32", DType.FLOAT32: "f32"}
_NUMPY_DTYPE = {DType.BOOL: np.bool_, DType.INT32: np.int32, DType.FLOAT32: np.float32}
_SHAPE_TAG = {Shape.VECTOR: "v", Shape.SCALAR: "s"}
_POOL_TAG = {PoolKind.TEMP: "tmp", PoolKind.OUTPUT: "out"}

# every DSL operator is valid python except these two
_UNARY_TEMPLATE = {"neg": "-({operand})", "~": "not ({operand})"}


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NumbaCPU_FPointer:
    func_name: str                       # the name the generated source calls
    func: object                         # the njit'd callable, bound at exec time
    input_signature: Tuple[Signature, ...]

    def __repr__(self):
        args = " , ".join(str(s) for s in self.input_signature)
        return f"{self.func_name}({args})"


def retrieve_func_metadata(func):
    """Parse an eagerly-compiled njit function's signature into our Signatures.

    Only usable when the kernel was given an explicit signature; a lazily
    compiled one has no ``signatures`` yet. Used for verification, never as the
    source of truth - the CallFactory owns the types.
    """
    type_map = {
        "bool": DType.BOOL,
        "int32": DType.INT32,
        "float32": DType.FLOAT32,
    }
    parsed_args = []

    numba_sig = func.signatures[0]
    for arg_type in numba_sig:
        if isinstance(arg_type, types.Array):
            base_type_str = str(arg_type.dtype)
            ndim = arg_type.ndim
        else:
            base_type_str = str(arg_type)
            ndim = 0

        if base_type_str not in type_map:
            raise TypeError(f"Unsupported Numba type: {base_type_str}")

        my_sig = Signature(type_map[base_type_str])
        if ndim == 1:
            my_sig = my_sig[:]
        elif ndim != 0:
            raise NotImplementedError("Dimensions > 1 are not supported by the Framework.")

        parsed_args.append(my_sig)
    return func.__name__, parsed_args


def register_numba_cpu(callFactory):
    """Register an njit kernel for ``callFactory``.

    The kernel takes (inputs..., scratch..., outputs...) and writes into the
    output arrays. When it was compiled eagerly its signature is checked
    against the factory; a lazy kernel is taken on trust and typed on first
    call by numba.
    """
    expected = tuple(
        callFactory.input_signature
        + callFactory.buffer_signature
        + callFactory.output_signature
    )

    def decorator(func):
        if getattr(func, "signatures", None):
            _, declared = retrieve_func_metadata(func)
            if tuple(declared) != expected:
                raise KernelsmithError(
                    f"numba kernel for '{callFactory.func_name}' declares"
                    f" ({' , '.join(map(str, declared))}) but the feature signature is"
                    f" ({' , '.join(map(str, expected))})"
                )

        NUMBA_CPU_FN_REGISTER[callFactory] = NumbaCPU_FPointer(
            func_name=f"{callFactory.func_name}_numba_cpu",
            func=func,
            input_signature=expected,
        )
        return func

    return decorator


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

def _pool_array_name(key) -> str:
    kind, dtype, shape = key
    return f"{_POOL_TAG[kind]}_{_DTYPE_TAG[dtype]}_{_SHAPE_TAG[shape]}"


def _input_array_name(dtype: DType) -> str:
    return f"inp_{_DTYPE_TAG[dtype]}"


def _param_array_name(dtype: DType) -> str:
    return f"par_{_DTYPE_TAG[dtype]}"


def _reference(
    value: ValueNode,
    allocation: Allocation,
    binding: Binding,
    time_index: Optional[str] = None,
) -> str:
    """Source text for reading or writing ``value``.

    The only place the memory layout is spelled out. ``time_index`` is the loop
    variable when we are inside a ``for t`` loop, and None outside it, where a
    series is referred to whole.
    """
    if value.role is VarRole.CONST:
        return repr(value.val)

    if value.role is VarRole.INPUT:
        column = binding.input_column(value.name, value.dtype)
        row = time_index if time_index is not None else ":"
        return f"{_input_array_name(value.dtype)}[{row}, {column}]"

    if value.role is VarRole.PARAM:
        column = binding.param_column(value.name, value.dtype)
        return f"{_param_array_name(value.dtype)}[p, {column}]"

    key, index = allocation.slots[value]
    name = _pool_array_name(key)
    if value.shape is Shape.SCALAR:
        return f"{name}[p, {index}]"
    if time_index is None:
        return f"{name}[p, {index}]"
    return f"{name}[p, {index}, {time_index}]"


def _slot_reference(slot, allocation: Allocation) -> str:
    """Scratch buffers have slots but no value node."""
    key, index = slot
    return f"{_pool_array_name(key)}[p, {index}]"


def kernel_parameters(graph: Graph, allocation: Allocation, binding: Binding) -> List[str]:
    """The generated kernel's parameters, in a fixed order the caller mirrors."""
    names = [_input_array_name(d) for d in sorted(binding.input_columns, key=lambda d: d.value)]
    names += [_param_array_name(d) for d in sorted(binding.param_columns, key=lambda d: d.value)]
    names += [
        _pool_array_name(key)
        for key in sorted(allocation.pool_size, key=lambda k: (k[0].value, k[1].value, k[2].value))
    ]
    # passed explicitly: a graph with no params or no inputs has no array to
    # read the counts from
    names += ["n_params", "n_bars"]
    return names


def code_gen(
    graph: Graph,
    ops: List[Op],
    allocation: Allocation,
    binding: Binding,
    replace: Dict[ValueNode, ValueNode],
    kernel_name: str = "kernel",
) -> str:
    """Emit the whole graph as one parallel njit function."""
    parameters = kernel_parameters(graph, allocation, binding)
    lines = [
        "@njit(parallel=True, cache=False)",
        f"def {kernel_name}({', '.join(parameters)}):",
    ]

    lines.append("    for p in prange(n_params):")

    def arg(value):
        return _reference(replace.get(value, value), allocation, binding)

    for op in ops:
        if isinstance(op, Call):
            pointer = NUMBA_CPU_FN_REGISTER[op.factory]
            call_args = [arg(a) for a in op.args]
            call_args += [_slot_reference(s, allocation) for s in allocation.scratch.get(op, ())]
            call_args += [_reference(o, allocation, binding) for o in op.outs]
            lines.append(f"        {pointer.func_name}({', '.join(call_args)})")
            continue

        if isinstance(op, FusedExpr):
            # one loop for the whole group; every value produced inside it is a
            # local, so only the root reaches a buffer
            lines.append("        for t in range(n_bars):")
            names: Dict[ValueNode, str] = {}
            for index, member in enumerate(op.members):
                operands = []
                for operand in member.args:
                    operand = replace.get(operand, operand)
                    operands.append(
                        names[operand] if operand in names
                        else _reference(operand, allocation, binding, "t")
                    )
                produced = member.outs[0]
                if member is op.root:
                    destination = _reference(produced, allocation, binding, "t")
                else:
                    destination = f"_v{index}"
                    names[produced] = destination
                lines.append(f"            {destination} = {_expression(member, operands)}")
            continue

        target = op.outs[0]
        if target.shape is Shape.SCALAR:
            operands = [_reference(replace.get(a, a), allocation, binding) for a in op.args]
            lines.append(f"        {_reference(target, allocation, binding)} = {_expression(op, operands)}")
        else:
            operands = [_reference(replace.get(a, a), allocation, binding, "t") for a in op.args]
            lines.append("        for t in range(n_bars):")
            destination = _reference(target, allocation, binding, "t")
            lines.append(f"            {destination} = {_expression(op, operands)}")

    return "\n".join(lines) + "\n"


def _expression(op: Expr, operands: List[str]) -> str:
    if len(operands) == 1:
        template = _UNARY_TEMPLATE.get(op.name)
        if template is None:
            raise GraphError(f"no numba emission for unary operator '{op.name}'")
        return template.format(operand=operands[0])
    return f"({operands[0]}) {op.name} ({operands[1]})"


# --------------------------------------------------------------------------
# compile / run
# --------------------------------------------------------------------------

class NumbaProgram(CompiledProgram):
    def __init__(self, graph, ops, allocation, binding, source, kernel, replace):
        self.graph = graph
        self.ops = ops
        self.allocation = allocation
        self.binding = binding
        self.source = source
        self.kernel = kernel
        self.replace = replace
        self._scratch: Dict[str, np.ndarray] = {}

    def _buffers(self, n_params: int, n_bars: int) -> Dict[str, np.ndarray]:
        """Scratch is reused between runs; outputs are always fresh.

        Temporaries dominate the footprint and their pages cost more to fault
        in than the arithmetic costs to run, so holding on to them is most of
        this backend's speed. Output buffers stay per-run because the arrays
        handed back are views into them - recycling those would silently
        rewrite results the caller is still holding.
        """
        buffers = {}
        for key, size in self.allocation.pool_size.items():
            kind, dtype, shape = key
            dims = (n_params, size, n_bars) if shape is Shape.VECTOR else (n_params, size)
            name = _pool_array_name(key)

            # np.empty is safe because a kernel must write every element of
            # every output on every path - see features/numba_cpu_impl.py
            if kind is PoolKind.OUTPUT:
                buffers[name] = np.empty(dims, dtype=_NUMPY_DTYPE[dtype])
                continue

            cached = self._scratch.get(name)
            if cached is None or cached.shape != dims:
                cached = np.empty(dims, dtype=_NUMPY_DTYPE[dtype])
                self._scratch[name] = cached
            buffers[name] = cached
        return buffers

    def run(self, inputs: dict, params: dict) -> dict:
        params, n_params, n_bars = check_call_arguments(self.graph, inputs, params)

        arrays: Dict[str, np.ndarray] = {}
        for dtype, columns in self.binding.input_columns.items():
            packed = np.zeros((n_bars, len(columns)), dtype=_NUMPY_DTYPE[dtype])
            for name, column in columns.items():
                packed[:, column] = np.asarray(inputs[name], dtype=_NUMPY_DTYPE[dtype])
            arrays[_input_array_name(dtype)] = packed

        for dtype, columns in self.binding.param_columns.items():
            packed = np.zeros((n_params, len(columns)), dtype=_NUMPY_DTYPE[dtype])
            for name, column in columns.items():
                packed[:, column] = np.asarray(params[name], dtype=_NUMPY_DTYPE[dtype])
            arrays[_param_array_name(dtype)] = packed

        arrays.update(self._buffers(n_params, n_bars))

        arrays["n_params"] = n_params
        arrays["n_bars"] = n_bars
        ordered = [arrays[name] for name in kernel_parameters(self.graph, self.allocation, self.binding)]
        self.kernel(*ordered)

        results = {}
        for name, node in self.graph.outputs.items():
            node = self.replace.get(node, node)
            if node.role is not VarRole.TEMP:
                # a bare input or param registered as an output never enters a buffer
                results[name] = _read_provided(node, arrays, self.binding, n_params)
                continue
            key, index = self.allocation.slots[node]
            buffer = arrays[_pool_array_name(key)]
            results[name] = buffer[:, index, :] if node.shape is Shape.VECTOR else buffer[:, index]
        return results


def _read_provided(node, arrays, binding, n_params):
    """An output that is itself an input or param never enters a buffer."""
    if node.role is VarRole.INPUT:
        column = binding.input_column(node.name, node.dtype)
        series = arrays[_input_array_name(node.dtype)][:, column]
        return np.broadcast_to(series, (n_params, series.shape[0])).copy()
    column = binding.param_column(node.name, node.dtype)
    return arrays[_param_array_name(node.dtype)][:, column]


class NumbaCPU_Backend(Backend):
    name = "numba"

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def compile(self, graph: Graph, verbose: bool = False) -> NumbaProgram:
        ops = graph.build()
        ops, replace = cse(ops)
        # fuse before liveness: values that became locals never appear in an
        # op's args or outs again, so they are never given a buffer
        ops, _levels = fuse(ops, graph.outputs.values(), replace)
        live = Liveness(ops, graph.outputs, replace)
        allocation = allocate(ops, live)
        binding = bind(graph)

        missing = [
            op.name for op in ops
            if isinstance(op, Call) and op.factory not in NUMBA_CPU_FN_REGISTER
        ]
        if missing:
            raise GraphError(
                "no numba CPU implementation for: " + ", ".join(sorted(set(missing)))
            )

        source = code_gen(graph, ops, allocation, binding, replace)
        if verbose:
            print(source)

        digest = hashlib.sha256(source.encode()).hexdigest()
        kernel = self._cache.get(digest)
        if kernel is None:
            namespace = {"njit": njit, "prange": prange, "np": np}
            for op in ops:
                if isinstance(op, Call):
                    pointer = NUMBA_CPU_FN_REGISTER[op.factory]
                    namespace[pointer.func_name] = pointer.func
            exec(compile(source, f"<kernelsmith:{digest[:12]}>", "exec"), namespace)
            kernel = namespace["kernel"]
            self._cache[digest] = kernel

        return NumbaProgram(graph, ops, allocation, binding, source, kernel, replace)
