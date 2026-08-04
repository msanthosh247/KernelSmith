"""Backend interface.

A backend turns a Graph into a CompiledProgram. Backends are stateless and
reusable: compiling two graphs gives two programs, never one backend holding
one graph. Implementations of features live in per-backend registries, keyed
by CallFactory, so the DSL layer never learns about execution.
"""
from __future__ import annotations

from enum import Enum


class Backends(Enum):
    CPU = "cpu"
    NUMBA_CPU = "numba"
    NUMBA_CUDA = "cuda"


class CompiledProgram:
    """A graph compiled for one backend."""

    def run(self, inputs: dict, params: dict) -> dict:
        """Evaluate the graph.

        inputs: {input_name: array of length T}
        params: {param_name: sequence of P values, one per parameter set}
        returns {output_name: array}, stacked along the parameter axis.
        """
        raise NotImplementedError(f"{type(self).__name__}.run is not implemented")


class Backend:
    """Compiles graphs into CompiledPrograms."""

    name: str = "base"

    def compile(self, graph) -> CompiledProgram:
        raise NotImplementedError(f"{type(self).__name__}.compile is not implemented")
