"""Feature kernels for the CUDA backend.

The same bodies the Numba CPU backend compiles with ``njit`` (see
``kernels.py``), specialised to float32 / int32 and compiled here as device
functions: FP64 on a GeForce part runs at 1/64 rate, and int64 arithmetic is
emulated. Compilation is lazy, so importing this module needs neither a GPU
nor a driver.
"""
from __future__ import annotations

from numba import cuda

from kernelsmith.backends.cuda import CudaBackend, register_cuda
from kernelsmith.features import kernels
from kernelsmith.features.specs import FEATURES

SINGLE = kernels.specialise(CudaBackend.float_type, CudaBackend.int_type)

# feature -> its device function, e.g. sma -> the float32 sma_kernel
CUDA_KERNELS = {
    factory: register_cuda(factory)(
        cuda.jit(device=True)(getattr(SINGLE, f"{factory.func_name}_kernel"))
    )
    for factory in FEATURES
}
