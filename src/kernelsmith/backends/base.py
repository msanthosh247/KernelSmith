"""Backend interface.

A backend turns a Graph into a CompiledProgram. Backends are stateless and
reusable: compiling two graphs gives two programs, never one backend holding
one graph. Implementations of features live in per-backend registries, keyed
by CallFactory, so the DSL layer never learns about execution.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from kernelsmith.errors import GraphError




class Backends(Enum):
    CPU = "cpu"
    NUMBA_CPU = "numba"
    NUMBA_CUDA = "cuda"


class CompiledProgram:
    """A graph compiled for one backend."""

    def run(self, inputs: dict, params: dict, sort_by=None, n_bars=None) -> dict:
        """Evaluate the graph.

        inputs: {input_name: array of length T, table_name: array of any length}
        params: {param_name: sequence of P values, one per parameter set}
        sort_by: a scheduling hint - never changes results, only speed. None
            runs the parameter sets in the order given. A param name, or a
            sequence of names with the primary key first, runs them sorted by
            those params; True sorts by every param in declaration order.
        n_bars: the length of the time axis, for a graph that has no series
            input to take it from - a simulation whose only data is a table.
        returns {output_name: array}, stacked along the parameter axis, in the
            order the parameter sets were given whatever ``sort_by`` is.
        """
        raise NotImplementedError(f"{type(self).__name__}.run is not implemented")

    def get_generated_code(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.get_generated_code is not implemented")

    @staticmethod
    def check_call_arguments(graph, inputs: dict, params: dict,
                             n_bars: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], int, int]:
        """Validate a run's inputs, tables and params against the graph.

        Shared by every backend so the messages never drift apart. Returns the
        params coerced to 1-d arrays, the number of parameter sets, and the series
        length - from the series inputs, or ``n_bars`` when there are none.
        """
        params = {name: np.atleast_1d(values) for name, values in params.items()}

        for name in graph.inputs:
            if name not in inputs:
                raise GraphError(f"missing input '{name}'")
        for name in getattr(graph, "tables", {}):
            if name not in inputs:
                raise GraphError(f"missing table '{name}'")
        for name in graph.params:
            if name not in params:
                raise GraphError(f"missing param '{name}'")

        lengths = {len(v) for v in params.values()} or {1}
        if len(lengths) > 1:
            raise GraphError(f"all params must have the same length, got {sorted(lengths)}")
        n_params = lengths.pop()

        # tables are off the time axis: their lengths are their own business
        series_lengths = {len(np.asarray(inputs[name])) for name in graph.inputs}
        if len(series_lengths) > 1:
            raise GraphError(f"all inputs must have the same length, got {sorted(series_lengths)}")
        if series_lengths:
            length = series_lengths.pop()
            if n_bars is not None and n_bars != length:
                raise GraphError(
                    f"n_bars={n_bars} contradicts the series inputs, which are {length} long;"
                    " n_bars is only for graphs with no series input"
                )
            n_bars = length
        elif n_bars is None:
            if getattr(graph, "tables", None):
                raise GraphError(
                    "this graph has no series input to set the time axis: pass n_bars,"
                    " e.g. run(inputs, params, n_bars=63) for a 63-step simulation"
                )
            n_bars = 0
        elif n_bars < 0:
            raise GraphError(f"n_bars must be non-negative, got {n_bars}")

        return params, n_params, n_bars

    @staticmethod
    def parameter_order(graph, params: Dict[str, np.ndarray], sort_by) -> Optional[np.ndarray]:
        """The permutation ``sort_by`` asks for, or None to keep the given order.

        Why it can matter: on the GPU a warp is 32 consecutive parameter sets
        running in lockstep. A window kernel reads ``period`` bars back, so 32
        different periods read 32 different rows - 32 memory transactions where
        equal periods would need one. Sorting puts similar periods in the same
        warp. Whether it pays depends on which params set a lookback, which is
        why it is opt-in and names the params rather than guessing.

        ``params`` must already have been through ``check_call_arguments``.
        """
        if sort_by is None or sort_by is False:
            return None
        if sort_by is True:
            names = list(graph.params)
        elif isinstance(sort_by, str):
            names = [sort_by]
        else:
            names = list(sort_by)
        for name in names:
            if name not in graph.params:
                raise GraphError(f"cannot sort by '{name}': the graph has no such param")
        if not names:
            return None
        # lexsort takes the primary key last; it is stable, so ties keep their order
        return np.lexsort([params[name] for name in reversed(names)])


class Backend:
    """Compiles graphs into CompiledPrograms."""

    name: str = "base"

    def compile(self, graph) -> CompiledProgram:
        raise NotImplementedError(f"{type(self).__name__}.compile is not implemented")
