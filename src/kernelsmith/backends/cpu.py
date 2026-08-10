"""CPU reference backend.

Executes a graph with numpy, one parameter set at a time. It is deliberately
simple: this is the oracle every faster backend is diffed against, and the
path that lets the whole project be developed and tested without a GPU.

Feature implementations are registered here, keyed by CallFactory:

    @cpu_impl(sma)
    def _sma(close, n):
        return (result,)          # always a tuple, one entry per output

Registration order does not matter - implementations are resolved at compile
time, and a missing one is a compile error naming the feature.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from kernelsmith.backends.base import Backend, CompiledProgram
from kernelsmith.dsl.graph import Call, CallFactory, Expr, Graph, Op, ValueNode
from kernelsmith.dsl.types import VarRole
from kernelsmith.errors import GraphError
from kernelsmith.ir import cse

# elementwise operators: the Expr counterpart of the feature registry
NUMPY_OPS: Dict[str, Callable] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    "&": np.logical_and,
    "|": np.logical_or,
    "^": np.logical_xor,
    "~": np.logical_not,
    "neg": np.negative,
}

# feature implementations, keyed by the factory object (not its name, so two
# features sharing a name cannot collide)
CPU_FEATURES: Dict[CallFactory, Callable] = {}


def cpu_impl(factory: CallFactory):
    """Register a numpy implementation for ``factory``. The function must
    return a tuple with one entry per output in the factory's signature."""

    def decorator(fn: Callable) -> Callable:
        CPU_FEATURES[factory] = fn
        return fn

    return decorator


class CpuProgram(CompiledProgram):
    """A graph with every op resolved to a numpy callable."""

    def __init__(self, graph: Graph, ops: List[Op], impls: Dict[Op, Callable] , replace: Dict[ValueNode, ValueNode] = None):
        self.graph = graph
        self.ops = ops
        self.impls = impls
        if replace is None:
            replace = dict()
        self.replace = replace

    def _seed(self, inputs: dict, params: dict, index: int) -> dict:
        env = {}
        for name, node in self.graph.inputs.items():
            if name not in inputs:
                raise GraphError(f"missing input '{name}'")
            env[node] = np.asarray(inputs[name])
        for name, node in self.graph.params.items():
            env[node] = params[name][index]
        return env

    def _read(self, node: ValueNode, env: dict):
        node = self.replace.get(node , node)
        if node.role is VarRole.CONST:
            return node.val
        if node not in env:
            raise GraphError(f"value {node!r} was never produced - graph is inconsistent")
        return env[node]

    def run(self, inputs: dict, params: dict) -> dict:
        params = {name: np.atleast_1d(vals) for name, vals in params.items()}

        for name in self.graph.params:
            if name not in params:
                raise GraphError(f"missing param '{name}'")

        lengths = {len(v) for v in params.values()} or {1}
        if len(lengths) > 1:
            raise GraphError(f"all params must have the same length, got {sorted(lengths)}")
        n_params = lengths.pop()

        collected = {name: [] for name in self.graph.outputs}

        for i in range(n_params):
            env = self._seed(inputs, params, i)

            for op in self.ops:
                values = [self._read(arg, env) for arg in op.args]
                results = self.impls[op](*values)

                if not isinstance(results, tuple):
                    raise GraphError(
                        f"CPU implementation of '{op.name}' must return a tuple,"
                        f" got {type(results).__name__}"
                    )
                if len(results) != len(op.outs):
                    raise GraphError(
                        f"CPU implementation of '{op.name}' returned {len(results)} value(s),"
                        f" signature declares {len(op.outs)}"
                    )
                for node, value in zip(op.outs, results):
                    env[node] = value

            for name, node in self.graph.outputs.items():
                collected[name].append(self._read(node, env))

        return {name: np.stack(vals) for name, vals in collected.items()}


class CpuBackend(Backend):
    name = "cpu"

    def compile(self, graph: Graph) -> CpuProgram:
        ops = graph.build()
        ops , replace = cse(ops)

        impls: Dict[Op, Callable] = {}
        missing: List[str] = []

        for op in ops:
            if isinstance(op, Expr):
                fn = NUMPY_OPS.get(op.name)
                if fn is None:
                    missing.append(f"operator '{op.name}'")
                else:
                    # wrap so every op has the same "returns a tuple" contract
                    impls[op] = lambda *values, _fn=fn: (_fn(*values),)
            elif isinstance(op, Call):
                fn = CPU_FEATURES.get(op.factory)
                if fn is None:
                    missing.append(f"feature '{op.name}'")
                else:
                    impls[op] = fn
            else:
                missing.append(f"unknown operation kind {type(op).__name__}")

        if missing:
            raise GraphError(
                "no CPU implementation for: " + ", ".join(sorted(set(missing)))
            )

        return CpuProgram(graph, ops, impls , replace)
