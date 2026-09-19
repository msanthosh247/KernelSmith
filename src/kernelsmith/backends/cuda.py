"""CUDA backend.

The same pipeline and the same emitted shape as the Numba CPU backend - one
kernel, feature kernels called as device functions, elementwise work inlined as
``for t`` loops - with one thread per parameter set instead of one core.

Layout is ``(T, slots, P)`` - the parameter index innermost. The 32 threads of a
warp are 32 consecutive parameter sets, and at any moment they read the same
``(t, slot)`` of their own series: with ``P`` innermost those are 32 adjacent
addresses, which the hardware serves as one coalesced transaction. Time
innermost, as on the CPU, would scatter the same 32 reads across 32 cache lines
- the defect this layout exists to avoid. Inputs are shared by every thread, so
they keep ``(T, F)``: all 32 threads read one address, which is a broadcast.

A parameter set's series is a strided column, ``tmp[:, slot, p]``; feature
kernels must not assume contiguity (see ``features/kernels.py``).

Occupancy is decided by ``P``, not the block size: with one thread per
parameter set, filling a GPU takes tens of thousands of parameter sets, and a
small sweep leaves most of the device idle whatever the launch configuration.
"""
from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import cuda

from kernelsmith.backends.generated import (
    GeneratedBackend,
    GeneratedProgram,
    KernelRegistry,
    Layout,
    SourceWriter,
)
from kernelsmith.dsl import Shape, ValueNode, VarRole
from kernelsmith.errors import KernelsmithError

CUDA = KernelRegistry(suffix="cuda", label="CUDA")
register_cuda = CUDA.register

DEFAULT_BLOCK_SIZE = 32

__all__ = [
    "CUDA", "CudaBackend", "CudaLayout", "CudaProgram", "DEFAULT_BLOCK_SIZE",
    "register_cuda",
]


class CudaLayout(Layout):
    """``(T, slots, P)``: a warp's 32 threads read 32 adjacent addresses."""

    decorator = "@cuda.jit"
    param_axis = -1

    @contextmanager
    def parameter_scope(self, writer: SourceWriter) -> Generator[None]:
        writer.line("p = cuda.grid(1)")
        # the last block is usually partial; threads past the sweep do nothing
        with writer.block("if p < n_params:"):
            yield

    def ref(self, value: ValueNode, t: Optional[str] = None) -> str:
        value = self.resolve(value)

        if value.role is VarRole.CONST:
            return self.constant(value)

        if value.shape is Shape.TABLE:
            # shared by every thread: all 32 lanes of a warp read one address
            return self.table_array(value.name)

        if value.role is VarRole.INPUT:
            column = self.binding.input_column(value.name, value.dtype)
            row = t if t is not None else ":"
            return f"{self.input_array(value.dtype)}[{row}, {column}]"

        if value.role is VarRole.PARAM:
            column = self.binding.param_column(value.name, value.dtype)
            return f"{self.param_array(value.dtype)}[{column}, p]"

        key, index = self.allocation.slots[value]
        name = self.pool_array(key)
        if value.shape is Shape.SCALAR:
            return f"{name}[{index}, p]"
        row = t if t is not None else ":"
        return f"{name}[{row}, {index}, p]"

    def scalar_view(self, value: ValueNode) -> str:
        key, index = self.allocation.slots[self.resolve(value)]
        return f"{self.pool_array(key)}[{index}:{index + 1}, p]"

    def scratch_ref(self, slot) -> str:
        key, index = slot
        _, _, shape = key
        name = self.pool_array(key)
        return f"{name}[:, {index}, p]" if shape is Shape.VECTOR else f"{name}[{index}, p]"

    def pool_shape(self, key, size: int, n_params: int, n_bars: int) -> Tuple[int, ...]:
        _, _, shape = key
        return (n_bars, size, n_params) if shape is Shape.VECTOR else (size, n_params)

    def input_shape(self, n_bars: int, width: int) -> Tuple[int, ...]:
        return (n_bars, width)

    def input_slot(self, column: int) -> Tuple[Any, ...]:
        return (slice(None), column)

    def param_shape(self, n_params: int, width: int) -> Tuple[int, ...]:
        return (width, n_params)

    def param_slot(self, column: int) -> Tuple[Any, ...]:
        return (column, slice(None))

    def output_view(self, host: np.ndarray, index: int, shape: Shape) -> np.ndarray:
        if shape is Shape.VECTOR:
            # device order is (T, P); callers get (P, T) like every other backend.
            # A transposed *view* - making it contiguous is a strided gather over
            # the whole array, which measured ~45x the cost of the device-to-host
            # copy itself. Callers who need C order can pay for it explicitly.
            return host[:, index, :].T
        return host[index, :]


# threads per block for the gather below: a warp along the parameter axis, so
# writes coalesce, and a few rows deep
_GATHER_BLOCK = (32, 8)


@cuda.jit
def _gather_columns(source, order, target):
    """target[row, q] = source[row, order[q]].

    Undoes a ``sort_by`` permutation on the device. The reads scatter only
    within one row - P elements, small enough to sit in L2 - and the writes
    are coalesced. The same permutation on the host is a gather over the whole
    array and cost ~4x the kernel it was meant to speed up.
    """
    column, row = cuda.grid(2)
    if row < source.shape[0] and column < source.shape[1]:
        target[row, column] = source[row, order[column]]


class CudaProgram(GeneratedProgram):
    """Device arrays: inputs are copied up, outputs copied back, scratch stays.

    Scratch lives on the device between runs for the same reason it is kept on
    the CPU, and more so - allocating device memory is not free either.
    """

    def __init__(self, graph, ops, layout, source, kernel, param_names,
                 block_size: int = DEFAULT_BLOCK_SIZE):
        super().__init__(graph, ops, layout, source, kernel, param_names)
        self.block_size = block_size

    def launch_config(self, n_params: int) -> Tuple[int, int]:
        """(blocks, threads per block) - one thread per parameter set."""
        return max(1, math.ceil(n_params / self.block_size)), self.block_size

    def _empty(self, dims, dtype):
        return cuda.device_array(dims, dtype=dtype)

    def _upload(self, host: np.ndarray):
        return cuda.to_device(host)

    def _download(self, array) -> np.ndarray:
        return array.copy_to_host()

    def _download_in_order(self, array, restore: np.ndarray) -> np.ndarray:
        # every pool has the parameter axis last: fold the rest into rows
        *_, n_params = array.shape            # device shapes refuse negative indexing
        rows = array.size // n_params
        source = array.reshape(rows, n_params)
        target = cuda.device_array((rows, n_params), dtype=array.dtype)
        blocks = (math.ceil(n_params / _GATHER_BLOCK[0]), math.ceil(rows / _GATHER_BLOCK[1]))
        _gather_columns[blocks, _GATHER_BLOCK](source, cuda.to_device(restore), target)
        return target.copy_to_host().reshape(array.shape)

    def _launch(self, args: List[Any], n_params: int) -> None:
        blocks, threads = self.launch_config(n_params)
        self.kernel[blocks, threads](*args)
        cuda.synchronize()


class CudaBackend(GeneratedBackend):
    name = "cuda"
    registry = CUDA
    layout_cls = CudaLayout
    program_cls = CudaProgram
    # FP64 on a GeForce part runs at 1/64 rate; int64 arithmetic is emulated
    float_type = np.float32
    int_type = np.int32

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        super().__init__()
        if not 1 <= block_size <= 1024:
            raise KernelsmithError(f"block size must be between 1 and 1024, got {block_size}")
        self.block_size = block_size

    def _make_program(self, graph, ops, layout, source, kernel) -> CudaProgram:
        return CudaProgram(
            graph, ops, layout, source, kernel, layout.kernel_parameters(),
            block_size=self.block_size,
        )

    def _namespace(self) -> Dict[str, Any]:
        return {"cuda": cuda, "np": np}
