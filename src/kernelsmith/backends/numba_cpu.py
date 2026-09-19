"""Numba parallel CPU backend.

The first backend that generates code rather than interpreting. The whole graph
becomes one ``@njit(parallel=True)`` function whose outer loop is ``prange``
over parameter sets, so a sweep uses every core.

Layout is ``(P, slots, T)`` - time innermost - because each core owns one
parameter set and walks time sequentially, so consecutive reads should be
consecutive addresses. The CUDA backend chooses the opposite for the same
reason applied to warps.

Everything shared with other generating backends - registration, emission,
buffer reuse, kernel caching - lives in ``generated.py``. What is here is only
what makes this target this target: ``CpuLayout`` and the array hooks.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from kernelsmith.backends.generated import (
    FunctionPointer,
    GeneratedBackend,
    GeneratedProgram,
    KernelRegistry,
    Layout,
    SourceWriter,
    code_gen,
    retrieve_func_metadata,
)
from kernelsmith.dsl import Shape, ValueNode, VarRole

NUMBA_CPU = KernelRegistry(suffix="numba_cpu", label="numba CPU")

# names kept from before the shared module existed
NUMBA_CPU_FN_REGISTER = NUMBA_CPU.entries
register_numba_cpu = NUMBA_CPU.register
NumbaCPU_FPointer = FunctionPointer

__all__ = [
    "CpuLayout", "NUMBA_CPU", "NUMBA_CPU_FN_REGISTER", "NumbaCPU_Backend",
    "NumbaCPU_FPointer", "NumbaProgram", "SourceWriter", "code_gen",
    "register_numba_cpu", "retrieve_func_metadata",
]


class CpuLayout(Layout):
    """``(P, slots, T)``: each core owns a parameter set and walks its series."""

    decorator = "@njit(parallel=True, cache=False)"

    @contextmanager
    def parameter_scope(self, writer: SourceWriter) -> Generator[None]:
        with writer.block("for p in prange(n_params):"):
            yield

    def ref(self, value: ValueNode, t: Optional[str] = None) -> str:
        value = self.resolve(value)

        if value.role is VarRole.CONST:
            return self.constant(value)

        if value.shape is Shape.TABLE:
            return self.table_array(value.name)

        if value.role is VarRole.INPUT:
            column = self.binding.input_column(value.name, value.dtype)
            row = t if t is not None else ":"
            return f"{self.input_array(value.dtype)}[{row}, {column}]"

        if value.role is VarRole.PARAM:
            column = self.binding.param_column(value.name, value.dtype)
            return f"{self.param_array(value.dtype)}[p, {column}]"

        key, index = self.allocation.slots[value]
        name = self.pool_array(key)
        if value.shape is Shape.SCALAR or t is None:
            return f"{name}[p, {index}]"
        return f"{name}[p, {index}, {t}]"

    def scalar_view(self, value: ValueNode) -> str:
        key, index = self.allocation.slots[self.resolve(value)]
        return f"{self.pool_array(key)}[p, {index}:{index + 1}]"

    def scratch_ref(self, slot) -> str:
        key, index = slot
        return f"{self.pool_array(key)}[p, {index}]"

    def pool_shape(self, key, size: int, n_params: int, n_bars: int) -> Tuple[int, ...]:
        _, _, shape = key
        return (n_params, size, n_bars) if shape is Shape.VECTOR else (n_params, size)

    def input_shape(self, n_bars: int, width: int) -> Tuple[int, ...]:
        return (n_bars, width)

    def input_slot(self, column: int) -> Tuple[Any, ...]:
        return (slice(None), column)

    def param_shape(self, n_params: int, width: int) -> Tuple[int, ...]:
        return (n_params, width)

    def param_slot(self, column: int) -> Tuple[Any, ...]:
        return (slice(None), column)

    def output_view(self, host: np.ndarray, index: int, shape: Shape) -> np.ndarray:
        return host[:, index, :] if shape is Shape.VECTOR else host[:, index]


class NumbaProgram(GeneratedProgram):
    """Host arrays throughout: nothing to move, the kernel is a plain call."""

    def _empty(self, dims, dtype):
        return np.empty(dims, dtype=dtype)

    def _upload(self, host: np.ndarray):
        return host

    def _download(self, array) -> np.ndarray:
        return array

    def _launch(self, args: List[Any], n_params: int) -> None:
        self.kernel(*args)


class NumbaCPU_Backend(GeneratedBackend):
    name = "numba"
    registry = NUMBA_CPU
    layout_cls = CpuLayout
    program_cls = NumbaProgram

    def _namespace(self) -> Dict[str, Any]:
        return {"njit": njit, "prange": prange, "np": np}
