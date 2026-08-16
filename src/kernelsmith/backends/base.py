"""Backend interface.

A backend turns a Graph into a CompiledProgram. Backends are stateless and
reusable: compiling two graphs gives two programs, never one backend holding
one graph. Implementations of features live in per-backend registries, keyed
by CallFactory, so the DSL layer never learns about execution.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

import numpy as np

from kernelsmith.errors import GraphError


def check_call_arguments(graph, inputs: dict, params: dict) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Validate a run's inputs and params against the graph.

    Shared by every backend so the messages never drift apart. Returns the
    params coerced to 1-d arrays, the number of parameter sets, and the series
    length.
    """
    params = {name: np.atleast_1d(values) for name, values in params.items()}

    for name in graph.inputs:
        if name not in inputs:
            raise GraphError(f"missing input '{name}'")
    for name in graph.params:
        if name not in params:
            raise GraphError(f"missing param '{name}'")

    lengths = {len(v) for v in params.values()} or {1}
    if len(lengths) > 1:
        raise GraphError(f"all params must have the same length, got {sorted(lengths)}")
    n_params = lengths.pop()

    series_lengths = {len(np.asarray(inputs[name])) for name in graph.inputs}
    if len(series_lengths) > 1:
        raise GraphError(f"all inputs must have the same length, got {sorted(series_lengths)}")
    n_bars = series_lengths.pop() if series_lengths else 0

    return params, n_params, n_bars


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
