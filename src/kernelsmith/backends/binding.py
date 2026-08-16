"""Column assignment for a graph's named inputs and params.

Values reach a compiled program packed into arrays, one per dtype: every named
input gets a column of ``inp_<dtype>`` and every param a column of
``par_<dtype>``. This decides which column, in graph registration order, so the
result is a pure function of the graph and identical between runs.

Backend-agnostic on purpose - it says which column, never which axis. Layout is
the backend's business.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from kernelsmith.dsl import DType, Graph


@dataclass(frozen=True)
class Binding:
    input_columns: Dict[DType, Dict[str, int]]
    param_columns: Dict[DType, Dict[str, int]]

    def input_column(self, name: str, dtype: DType) -> int:
        return self.input_columns[dtype][name]

    def param_column(self, name: str, dtype: DType) -> int:
        return self.param_columns[dtype][name]

    def input_width(self, dtype: DType) -> int:
        return len(self.input_columns.get(dtype, ()))

    def param_width(self, dtype: DType) -> int:
        return len(self.param_columns.get(dtype, ()))


def _columns(nodes: dict) -> Dict[DType, Dict[str, int]]:
    grouped: Dict[DType, Dict[str, int]] = {}
    for name, node in nodes.items():          # registration order
        columns = grouped.setdefault(node.dtype, {})
        columns[name] = len(columns)
    return grouped


def bind(graph: Graph) -> Binding:
    return Binding(
        input_columns=_columns(graph.inputs),
        param_columns=_columns(graph.params),
    )
