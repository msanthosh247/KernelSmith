"""Shared machinery for backends that generate a kernel from the op list.

Every generating backend runs the same pipeline - build, CSE, fusion, liveness,
allocation - and emits the same shape of source: one kernel, one body per
parameter set, feature kernels called as opaque functions, elementwise work
inlined as ``for t`` loops. What differs between targets is small and lives in
two places:

``Layout`` subclass   how memory is laid out and addressed: array shapes, index
                      expressions, how a parameter set's index ``p`` is obtained,
                      and how results are read back. The only backend-specific
                      part of the *generated source*.
``Program`` hooks     how arrays are allocated, moved and launched on the target
                      (``_empty``, ``_upload``, ``_download``, ``_launch``).

Everything else - registration, emission, buffer reuse, result collection,
kernel caching - is written once, here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
from numba import types

from kernelsmith.backends.base import Backend, CompiledProgram
from kernelsmith.backends.binding import Binding, bind
from kernelsmith.dsl import (
    Call, DType, Expr, Graph, Op, OpCategory, Shape, Signature, ValueNode, VarRole,
)
from kernelsmith.errors import GraphError, KernelsmithError
from kernelsmith.ir import Allocation, Liveness, PoolKind, allocate, cse
from kernelsmith.ir.fuse import FusedExpr, fuse

_DTYPE_TAG = {DType.BOOL: "b1", DType.INT32: "i32", DType.FLOAT32: "f32"}
NUMPY_DTYPE = {DType.BOOL: np.bool_, DType.INT32: np.int32, DType.FLOAT32: np.float32}
_SHAPE_TAG = {Shape.VECTOR: "v", Shape.SCALAR: "s", Shape.TABLE: "t"}
_POOL_TAG = {PoolKind.TEMP: "tmp", PoolKind.OUTPUT: "out"}

# every DSL operator is valid python except these two
_UNARY_TEMPLATE = {"neg": "-({operand})", "~": "not ({operand})"}

# Numba's scalar typing is not the DSL's: a bare 2 is int64 to numba, a bare 2.0
# is float64, float32 / int32 is float64 and int32 + int32 is int64. Each of
# those widens everything downstream of it - on a GeForce GPU onto FP64, which
# runs at 1/64 rate. So the source spells every dtype out: constants carry
# theirs, operands are cast to the type the DSL says the op computes in, and
# integer results are narrowed back to int32.
_CAST = {DType.INT32: "np.int32", DType.FLOAT32: "np.float32"}


def typed_constant(value: ValueNode, dtype: Optional[DType] = None) -> str:
    """A constant, spelled so numba gives it ``dtype`` - by default the one the
    DSL inferred for it."""
    dtype = dtype or value.dtype
    if dtype is DType.BOOL:
        return repr(bool(value.val))
    if dtype is DType.INT32:
        return f"np.int32({int(value.val)!r})"
    number = float(value.val)
    if math.isnan(number):
        return "np.float32(np.nan)"
    if math.isinf(number):
        return "np.float32(np.inf)" if number > 0 else "np.float32(-np.inf)"
    return f"np.float32({number!r})"


def _compute_dtype(op: Expr) -> Optional[DType]:
    """The dtype a binary op's operands must share, or None if nothing to cast."""
    category = OpCategory.classify(op.name)
    if category is OpCategory.LOGIC:
        return None
    if op.name == "/":
        return DType.FLOAT32                        # true division, numpy-style
    left, right = (arg.dtype for arg in op.args)
    return left.join(right)

_INDENT = "    "


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionPointer:
    """A feature's compiled kernel, and the full argument list it expects."""

    func_name: str                        # the name the generated source calls
    func: object                          # the compiled callable, bound at exec time
    arg_signature: Tuple[Signature, ...]  # inputs + scratch + outputs, in call order

    def __repr__(self) -> str:
        args = " , ".join(str(s) for s in self.arg_signature)
        return f"{self.func_name}({args})"


def retrieve_func_metadata(func):
    """Parse an eagerly-compiled kernel's signature into our Signatures.

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


class KernelRegistry:
    """One backend's feature kernels, keyed by CallFactory.

    ``suffix`` becomes part of the name the generated source calls
    (``sma`` -> ``sma_numba_cpu``), so two backends never collide in one
    namespace; ``label`` is how error messages name the backend.
    """

    def __init__(self, suffix: str, label: str):
        self.suffix = suffix
        self.label = label
        self.entries: Dict[object, FunctionPointer] = {}

    def __contains__(self, factory) -> bool:
        return factory in self.entries

    def __getitem__(self, factory) -> FunctionPointer:
        return self.entries[factory]

    def register(self, callFactory):
        """Register a kernel for ``callFactory``.

        The kernel takes (inputs..., scratch..., outputs...) and writes into the
        output arrays. When it was compiled eagerly its signature is checked
        against the factory; a lazy kernel is taken on trust and typed on first
        call.
        """
        expected = tuple(
            callFactory.input_signature
            + callFactory.buffer_signature
            + callFactory.output_signature
        )
        emitted_name = f"{callFactory.func_name}_{self.suffix}"

        def decorator(func):
            if getattr(func, "signatures", None):
                _, declared = retrieve_func_metadata(func)
                if tuple(declared) != expected:
                    raise KernelsmithError(
                        f"{self.label} kernel for '{callFactory.func_name}' declares"
                        f" ({' , '.join(map(str, declared))}) but the feature signature is"
                        f" ({' , '.join(map(str, expected))})"
                    )

            # the exec namespace binds kernels by name, and the source cache is
            # keyed on source text alone - two factories sharing a name would let
            # one graph's cached kernel be reused with the other's function bound in
            clash = next(
                (
                    factory
                    for factory, pointer in self.entries.items()
                    if pointer.func_name == emitted_name and factory is not callFactory
                ),
                None,
            )
            if clash is not None:
                raise KernelsmithError(
                    f"two features both register the {self.label} kernel '{emitted_name}';"
                    " feature names must be unique within a backend"
                )

            self.entries[callFactory] = FunctionPointer(
                func_name=emitted_name,
                func=func,
                arg_signature=expected,
            )
            return func

        return decorator


# --------------------------------------------------------------------------
# addressing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Every array name and index expression the generated source uses.

    The base class owns what every target shares: array names, the CSE
    replacement map (applied in ``resolve`` and nowhere else, so no caller has
    to remember it), and the kernel's call convention. A subclass supplies the
    memory layout - index expressions, array shapes, how ``p`` is obtained, and
    how results are read back. Swapping the subclass is what a new target does.
    """

    allocation: Allocation
    binding: Binding
    replace: Dict[ValueNode, ValueNode]

    # the decorator line placed above the generated kernel
    decorator = ""
    # the axis of every pool and param array that indexes the parameter set
    param_axis = 0

    # -- names ------------------------------------------------------------

    def pool_array(self, key) -> str:
        kind, dtype, shape = key
        return f"{_POOL_TAG[kind]}_{_DTYPE_TAG[dtype]}_{_SHAPE_TAG[shape]}"

    def constant(self, value: ValueNode) -> str:
        return typed_constant(value)

    def input_array(self, dtype: DType) -> str:
        return f"inp_{_DTYPE_TAG[dtype]}"

    def param_array(self, dtype: DType) -> str:
        return f"par_{_DTYPE_TAG[dtype]}"

    def table_array(self, name: str) -> str:
        return f"tab_{name}"

    # -- references -------------------------------------------------------

    def resolve(self, value: ValueNode) -> ValueNode:
        """Follow the CSE replacement map to the value that survived."""
        return self.replace.get(value, value)

    def scratch_refs(self, op: Op) -> List[str]:
        return [self.scratch_ref(s) for s in self.allocation.scratch.get(op, ())]

    def call_output_ref(self, value: ValueNode) -> str:
        """How a feature kernel is handed an output to write into.

        A series is passed as its view, as everywhere. A scalar cannot be passed
        as its element - that is a value, and the kernel's write would be lost
        - so it goes as a one-element view the kernel writes ``out[0]`` of.
        """
        if value.shape is Shape.SCALAR:
            return self.scalar_view(value)
        return self.ref(value)

    # -- call convention --------------------------------------------------

    def kernel_parameters(self) -> Tuple[str, ...]:
        """The generated kernel's parameters, in a fixed order the caller mirrors.

        Computed once at compile time and carried on the program, so the
        emitted signature and the argument list built by ``run`` cannot drift.
        """
        names = [
            self.input_array(d)
            for d in sorted(self.binding.input_columns, key=lambda d: d.value)
        ]
        names += [self.table_array(name) for name in self.binding.tables]
        names += [
            self.param_array(d)
            for d in sorted(self.binding.param_columns, key=lambda d: d.value)
        ]
        names += [
            self.pool_array(key)
            for key in sorted(
                self.allocation.pool_size,
                key=lambda k: (k[0].value, k[1].value, k[2].value),
            )
        ]
        # passed explicitly: a graph with no params or no inputs has no array to
        # read the counts from
        names += ["n_params", "n_bars"]
        return tuple(names)

    # -- per target -------------------------------------------------------

    def ref(self, value: ValueNode, t: Optional[str] = None) -> str:
        """Source text for reading or writing ``value``.

        ``t`` is the loop variable when we are inside a ``for t`` loop, and
        None outside it, where a series is referred to whole.
        """
        raise NotImplementedError

    def scratch_ref(self, slot) -> str:
        """Scratch buffers have slots but no value node."""
        raise NotImplementedError

    def scalar_view(self, value: ValueNode) -> str:
        """A scalar's slot as a one-element array (see ``call_output_ref``)."""
        raise NotImplementedError

    @contextmanager
    def parameter_scope(self, writer: "SourceWriter") -> Generator[None]:
        """Open the region of the kernel where ``p`` names one parameter set."""
        raise NotImplementedError
        yield  # pragma: no cover

    def pool_shape(self, key, size: int, n_params: int, n_bars: int) -> Tuple[int, ...]:
        raise NotImplementedError

    def input_shape(self, n_bars: int, width: int) -> Tuple[int, ...]:
        raise NotImplementedError

    def input_slot(self, column: int) -> Tuple[Any, ...]:
        """Index selecting one input's whole series in the packed input array."""
        raise NotImplementedError

    def param_shape(self, n_params: int, width: int) -> Tuple[int, ...]:
        raise NotImplementedError

    def param_slot(self, column: int) -> Tuple[Any, ...]:
        """Index selecting one param's values, one per parameter set."""
        raise NotImplementedError

    def output_view(self, host: np.ndarray, index: int, shape: Shape) -> np.ndarray:
        """One output slot as ``(P, T)`` for a vector or ``(P,)`` for a scalar."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

@dataclass
class SourceWriter:
    """Builds the kernel source: indentation primitives plus one emitter per op.

    Indentation is a property of the nesting, not something each ``append``
    counts out in spaces, so a loop body cannot be written at the wrong depth.
    """

    layout: Layout
    registry: KernelRegistry
    _lines: List[str] = field(default_factory=list)
    _depth: int = 0

    # -- primitives -------------------------------------------------------

    def line(self, text: str) -> None:
        self._lines.append(_INDENT * self._depth + text)

    @contextmanager
    def block(self, header: str) -> Generator[None]:
        """Write ``header`` and indent everything written inside the with-body."""
        self.line(header)
        self._depth += 1
        mark = len(self._lines)
        try:
            yield
        finally:
            if len(self._lines) == mark:
                # a graph whose outputs are all bare inputs or params produces
                # no ops; an empty body would not be valid python
                self.line("pass")
            self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"

    # -- emitters ---------------------------------------------------------

    def emit_kernel(self, ops: List[Op], kernel_name: str = "kernel") -> None:
        parameters = ", ".join(self.layout.kernel_parameters())
        self.line(self.layout.decorator)
        with self.block(f"def {kernel_name}({parameters}):"):
            with self.layout.parameter_scope(self):
                for op in ops:
                    self.emit_op(op)

    def emit_op(self, op: Op) -> None:
        if isinstance(op, Call):
            self.emit_call(op)
        elif isinstance(op, FusedExpr):
            self.emit_fused(op)
        elif op.outs[0].shape is Shape.SCALAR:
            self.emit_scalar(op)
        else:
            self.emit_elementwise(op)

    def emit_call(self, op: Call) -> None:
        """Feature kernels are opaque: one call, writing into its output slots."""
        pointer = self.registry[op.factory]
        args = [self.layout.ref(a) for a in op.args]
        args += self.layout.scratch_refs(op)
        args += [self.layout.call_output_ref(o) for o in op.outs]
        self.line(f"{pointer.func_name}({', '.join(args)})")

    def emit_scalar(self, op: Expr) -> None:
        operands = [self.layout.ref(a) for a in op.args]
        self.line(f"{self.layout.ref(op.outs[0])} = {self.expression(op, operands)}")

    def emit_elementwise(self, op: Expr) -> None:
        with self.block("for t in range(n_bars):"):
            operands = [self.layout.ref(a, "t") for a in op.args]
            destination = self.layout.ref(op.outs[0], "t")
            self.line(f"{destination} = {self.expression(op, operands)}")

    def emit_fused(self, op: FusedExpr) -> None:
        """One loop for the whole group.

        Every value produced inside it is a local, so only the root reaches a
        buffer. Locals are keyed by the resolved node, the same form ``ref``
        works in, so a member's operand matches the member that produced it.
        """
        with self.block("for t in range(n_bars):"):
            locals_: Dict[ValueNode, str] = {}
            for index, member in enumerate(op.members):
                operands = []
                for arg in member.args:
                    arg = self.layout.resolve(arg)
                    if arg in locals_:
                        operands.append(locals_[arg])
                    else:
                        operands.append(self.layout.ref(arg, "t"))

                produced = self.layout.resolve(member.outs[0])
                if member is op.root:
                    destination = self.layout.ref(produced, "t")
                else:
                    destination = f"_v{index}"
                    locals_[produced] = destination

                self.line(f"{destination} = {self.expression(member, operands)}")

    @staticmethod
    def expression(op: Expr, operands: List[str]) -> str:
        """``op`` over already-rendered operands, with every dtype spelled out
        (see ``_CAST``)."""
        if len(operands) == 1:
            template = _UNARY_TEMPLATE.get(op.name)
            if template is None:
                raise GraphError(f"no emission for unary operator '{op.name}'")
            text = template.format(operand=operands[0])
        else:
            target = _compute_dtype(op)
            if target is not None:
                operands = [
                    text if arg.dtype is target
                    # a constant is simply written in the target type: 2.0, not float(2)
                    else typed_constant(arg, target) if arg.role is VarRole.CONST
                    else f"{_CAST[target]}({text})"
                    for arg, text in zip(op.args, operands)
                ]
            text = f"({operands[0]}) {op.name} ({operands[1]})"

        # numba widens int32 arithmetic to int64; the DSL says int32
        if op.outs[0].dtype is DType.INT32:
            return f"np.int32({text})"
        return text


def code_gen(ops: List[Op], layout: Layout, registry: KernelRegistry,
             kernel_name: str = "kernel") -> str:
    """Emit the whole graph as one kernel."""
    writer = SourceWriter(layout, registry)
    writer.emit_kernel(ops, kernel_name)
    return writer.render()


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

class GeneratedProgram(CompiledProgram):
    """A compiled kernel plus the buffers it runs against.

    Shapes come from the layout and array handling from four hooks, so a target
    only says *how* to allocate, move and launch - never what to allocate.

    Not reentrant: scratch is reused between runs, so one program must not be
    run from two threads at once.
    """

    def __init__(self, graph, ops, layout: Layout, source, kernel, param_names):
        self.graph = graph
        self.ops = ops
        self.layout = layout
        self.source = source
        self.kernel = kernel
        self.param_names = param_names
        self._scratch: Dict[str, Any] = {}

    # kept so callers that reached into the old attributes still work
    @property
    def allocation(self) -> Allocation:
        return self.layout.allocation

    @property
    def binding(self) -> Binding:
        return self.layout.binding

    @property
    def replace(self) -> Dict[ValueNode, ValueNode]:
        return self.layout.replace

    def get_generated_code(self) -> str:
        return self.source

    # -- target hooks -----------------------------------------------------

    def _empty(self, dims: Tuple[int, ...], dtype):
        raise NotImplementedError

    def _upload(self, host: np.ndarray):
        raise NotImplementedError

    def _download(self, array) -> np.ndarray:
        raise NotImplementedError

    def _download_in_order(self, array, restore: np.ndarray) -> np.ndarray:
        """Download a pool with the parameter axis put back in the caller's order.

        On the host this is one ``take``: cheap when the parameter axis is the
        outermost, so each parameter set is a contiguous block to copy. A target
        whose parameter axis is innermost should override this and gather on
        the device instead.
        """
        return np.take(self._download(array), restore, axis=self.layout.param_axis)

    def _launch(self, args: List[Any], n_params: int) -> None:
        raise NotImplementedError

    # -- shared -----------------------------------------------------------

    def _buffers(self, n_params: int, n_bars: int) -> Dict[str, Any]:
        """Scratch is reused between runs; outputs are always fresh.

        Temporaries dominate the footprint and their pages cost more to fault
        in than the arithmetic costs to run, so holding on to them is most of
        the speed. Output buffers stay per-run because the arrays handed back
        may be views into them - recycling those would silently rewrite results
        the caller is still holding.
        """
        buffers = {}
        for key, size in self.layout.allocation.pool_size.items():
            kind, dtype, _ = key
            dims = self.layout.pool_shape(key, size, n_params, n_bars)
            name = self.layout.pool_array(key)

            # empty rather than zeroed is safe because a kernel must write every
            # element of every output on every path - see features/kernels.py
            if kind is PoolKind.OUTPUT:
                buffers[name] = self._empty(dims, NUMPY_DTYPE[dtype])
                continue

            cached = self._scratch.get(name)
            if cached is None or tuple(cached.shape) != dims:
                cached = self._empty(dims, NUMPY_DTYPE[dtype])
                self._scratch[name] = cached
            buffers[name] = cached
        return buffers

    def _pack_inputs(self, inputs: dict, n_bars: int) -> Dict[str, np.ndarray]:
        arrays = {}
        for dtype, columns in self.layout.binding.input_columns.items():
            packed = np.zeros(self.layout.input_shape(n_bars, len(columns)), dtype=NUMPY_DTYPE[dtype])
            for name, column in columns.items():
                packed[self.layout.input_slot(column)] = np.asarray(inputs[name], dtype=NUMPY_DTYPE[dtype])
            arrays[self.layout.input_array(dtype)] = packed
        return arrays

    def _pack_params(self, params: dict, n_params: int) -> Dict[str, np.ndarray]:
        arrays = {}
        for dtype, columns in self.layout.binding.param_columns.items():
            packed = np.zeros(self.layout.param_shape(n_params, len(columns)), dtype=NUMPY_DTYPE[dtype])
            for name, column in columns.items():
                packed[self.layout.param_slot(column)] = np.asarray(params[name], dtype=NUMPY_DTYPE[dtype])
            arrays[self.layout.param_array(dtype)] = packed
        return arrays

    def _collect_results(self, arrays: Dict[str, Any], host: Dict[str, np.ndarray],
                         n_params: int, restore: Optional[np.ndarray] = None) -> dict:
        """Download each output pool once and slice every output out of it.

        ``restore`` undoes a ``sort_by`` permutation. It is applied to the whole
        pool, before slicing: on the GPU the outputs come back as transposed
        views, and permuting those would be the same strided gather
        ``output_view`` exists to avoid.
        """
        results = {}
        downloaded: Dict[str, np.ndarray] = {}
        for name, node in self.graph.outputs.items():
            node = self.layout.resolve(node)
            if node.role is not VarRole.TEMP:
                # a bare input or param registered as an output never enters a buffer
                results[name] = self._read_provided(node, host, n_params, restore)
                continue
            key, index = self.layout.allocation.slots[node]
            pool = self.layout.pool_array(key)
            if pool not in downloaded:
                if restore is None:
                    downloaded[pool] = self._download(arrays[pool])
                else:
                    downloaded[pool] = self._download_in_order(arrays[pool], restore)
            results[name] = self.layout.output_view(downloaded[pool], index, node.shape)
        return results

    def _read_provided(self, node, host: Dict[str, np.ndarray], n_params: int,
                       restore: Optional[np.ndarray] = None):
        """An output that is itself an input or param, read from the host copy."""
        if node.role is VarRole.INPUT:
            column = self.layout.binding.input_column(node.name, node.dtype)
            series = host[self.layout.input_array(node.dtype)][self.layout.input_slot(column)]
            return np.broadcast_to(series, (n_params, series.shape[0])).copy()
        column = self.layout.binding.param_column(node.name, node.dtype)
        values = host[self.layout.param_array(node.dtype)][self.layout.param_slot(column)]
        return values if restore is None else values[restore]

    def _pack_tables(self, inputs: dict) -> Dict[str, np.ndarray]:
        return {
            self.layout.table_array(name): np.ascontiguousarray(inputs[name], dtype=NUMPY_DTYPE[dtype])
            for name, dtype in self.layout.binding.tables.items()
        }

    def _prepare(self, inputs: dict, params: dict, n_bars: Optional[int] = None):
        """Validate and lay out everything the kernel needs, ready to launch."""
        params, n_params, n_bars = self.check_call_arguments(self.graph, inputs, params, n_bars)

        host: Dict[str, np.ndarray] = {}
        host.update(self._pack_inputs(inputs, n_bars))
        host.update(self._pack_tables(inputs))
        host.update(self._pack_params(params, n_params))

        arrays: Dict[str, Any] = {name: self._upload(array) for name, array in host.items()}
        arrays.update(self._buffers(n_params, n_bars))
        # int32, like every other integer in the kernel: a Python int would be
        # typed int64 and make every loop counter compared against it int64 too
        arrays["n_params"] = np.int32(n_params)
        arrays["n_bars"] = np.int32(n_bars)
        return arrays, host, n_params

    def run(self, inputs: dict, params: dict, sort_by=None, n_bars: Optional[int] = None) -> dict:
        checked, _, _ = self.check_call_arguments(self.graph, inputs, params, n_bars)
        order = self.parameter_order(self.graph, checked, sort_by)
        if order is not None:
            params = {name: values[order] for name, values in checked.items()}

        arrays, host, n_params = self._prepare(inputs, params, n_bars)
        self._launch([arrays[name] for name in self.param_names], n_params)

        restore = None if order is None else np.argsort(order)
        return self._collect_results(arrays, host, n_params, restore)


# --------------------------------------------------------------------------
# compiling
# --------------------------------------------------------------------------

class GeneratedBackend(Backend):
    """Lowers a graph, emits one kernel, and compiles it once per source text.

    A target supplies ``registry``, ``layout_cls``, ``program_cls`` and the
    names its generated source needs in scope (``_namespace``).
    """

    registry: KernelRegistry
    layout_cls: type = Layout
    program_cls: type = GeneratedProgram

    # what the feature kernels compute in (see features/kernels.py, "Precision").
    # Series are float32 on every target; this is the accumulators and indices.
    float_type: type = np.float64
    int_type: type = np.int64

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def compile(self, graph: Graph, verbose: bool = False) -> GeneratedProgram:
        # registers every feature kernel, including the DSL's own (series[i]).
        # Here rather than at import time: the feature modules import the backends.
        import kernelsmith.features  # noqa: F401

        ops, layout = self._lower(graph)
        self._check_implementations(ops)

        source = code_gen(ops, layout, self.registry)
        if verbose:
            print(source)

        kernel = self._kernel_for(source, ops)
        return self._make_program(graph, ops, layout, source, kernel)

    def _make_program(self, graph, ops, layout, source, kernel) -> GeneratedProgram:
        return self.program_cls(graph, ops, layout, source, kernel, layout.kernel_parameters())

    def _lower(self, graph: Graph) -> Tuple[List[Op], Layout]:
        ops = graph.build()
        ops, replace = cse(ops)
        # fuse before liveness: values that became locals never appear in an
        # op's args or outs again, so they are never given a buffer
        ops, _levels = fuse(ops, graph.outputs.values(), replace)
        live = Liveness(ops, graph.outputs, replace)
        return ops, self.layout_cls(allocate(ops, live), bind(graph), replace)

    def _check_implementations(self, ops: List[Op]) -> None:
        missing = [
            op.name for op in ops
            if isinstance(op, Call) and op.factory not in self.registry
        ]
        if missing:
            raise GraphError(
                f"no {self.registry.label} implementation for: "
                + ", ".join(sorted(set(missing)))
            )

    def _namespace(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _kernel_for(self, source: str, ops: List[Op]):
        """Compile ``source``, reusing an earlier kernel when the text matches."""
        digest = hashlib.sha256(source.encode()).hexdigest()
        kernel = self._cache.get(digest)
        if kernel is not None:
            return kernel

        namespace = self._namespace()
        for op in ops:
            if isinstance(op, Call):
                pointer = self.registry[op.factory]
                namespace[pointer.func_name] = pointer.func

        exec(compile(source, f"<kernelsmith:{digest[:12]}>", "exec"), namespace)
        kernel = namespace["kernel"]
        self._cache[digest] = kernel
        return kernel
