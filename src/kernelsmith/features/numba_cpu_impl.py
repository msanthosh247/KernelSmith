"""Feature kernels for the Numba CPU backend.

The bodies live in ``kernels.py`` and are shared with the CUDA backend; this
module compiles each with ``njit`` and registers it. See ``kernels.py`` for the
contract every kernel must honour.
"""
from __future__ import annotations

from numba import njit

from kernelsmith.backends.numba_cpu import register_numba_cpu
from kernelsmith.features import kernels
from kernelsmith.features.specs import FEATURES, sma

# feature -> its compiled kernel, e.g. sma -> kernels.sma_kernel under njit
NUMBA_CPU_KERNELS = {
    factory: register_numba_cpu(factory)(
        njit(cache=True)(getattr(kernels, f"{factory.func_name}_kernel"))
    )
    for factory in FEATURES
}

# called directly by the hand-written baseline in benchmarks/numba_cpu.py
sma_numba_cpu = NUMBA_CPU_KERNELS[sma]
