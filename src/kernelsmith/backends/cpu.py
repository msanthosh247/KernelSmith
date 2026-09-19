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

import inspect
from typing import Callable, Dict, List

import numpy as np

from kernelsmith.backends.base import Backend, CompiledProgram
from kernelsmith.dsl.graph import Call, CallFactory, Expr, Graph, Op, ValueNode
from kernelsmith.dsl.types import VarRole
from kernelsmith.errors import GraphError
from kernelsmith.ir import cse
from kernelsmith.ir.fuse import FusedExpr

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


class CPUProgram(CompiledProgram):
    """A graph with every op resolved to a numpy callable."""

    def __init__(self, graph: Graph, ops: List[Op], impls: Dict[Op, Callable] , replace: Dict[ValueNode, ValueNode] = None):
        self.graph = graph
        self.ops = ops
        self.impls = impls
        # features whose output length cannot be read off their arguments - a
        # simulation from a table - declare a keyword-only n_bars and get it
        self.sized = {
            op for op, fn in impls.items()
            if isinstance(op, Call) and "n_bars" in inspect.signature(fn).parameters
        }
        if replace is None:
            replace = dict()
        self.replace = replace

    def _seed(self, inputs: dict, params: dict, index: int) -> dict:
        env = {}
        for name, node in self.graph.inputs.items():
            if name not in inputs:
                raise GraphError(f"missing input '{name}'")
            env[node] = np.asarray(inputs[name])
        for name, node in self.graph.tables.items():
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

    def run(self, inputs: dict, params: dict, sort_by=None, n_bars=None) -> dict:
        params, n_params, n_bars = self.check_call_arguments(self.graph, inputs, params, n_bars)
        # parameter sets are evaluated one at a time here, so their order cannot
        # matter - but a bad sort_by is still the caller's error on every backend
        self.parameter_order(self.graph, params, sort_by)

        collected = {name: [] for name in self.graph.outputs}

        for i in range(n_params):
            env = self._seed(inputs, params, i)

            for op in self.ops:
                if isinstance(op , FusedExpr):
                    raise NotImplementedError("CPU code doesn't support fused expressions yet")
                values = [self._read(arg, env) for arg in op.args]
                if op in self.sized:
                    results = self.impls[op](*values, n_bars=n_bars)
                else:
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


class CPUBackend(Backend):
    name = "cpu"

    def compile(self, graph: Graph) -> CPUProgram:
        # registers the numpy implementations, including the DSL's own (series[i])
        import kernelsmith.features  # noqa: F401

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
            elif isinstance(op , FusedExpr):
                raise NotImplementedError("CPU Backend does not support expression inlining yet!")
            else:
                missing.append(f"unknown operation kind {type(op).__name__}")

        if missing:
            raise GraphError(
                "no CPU implementation for: " + ", ".join(sorted(set(missing)))
            )

        return CPUProgram(graph, ops, impls , replace)
