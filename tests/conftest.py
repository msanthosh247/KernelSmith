import importlib.util

import pytest


def _generating_backends():
    """Every backend that emits and compiles code and can run here: the Numba
    CPU backend whenever numba is installed, CUDA on a GPU or under the
    simulator (NUMBA_ENABLE_CUDASIM=1)."""
    backends = []
    if importlib.util.find_spec("numba") is None:
        return backends
    from kernelsmith.backends.numba_cpu import NumbaCPU_Backend
    backends.append(pytest.param(NumbaCPU_Backend, id="numba"))

    from kernelsmith.availability import cuda_usable
    if cuda_usable():
        from kernelsmith.backends.cuda import CudaBackend
        backends.append(pytest.param(CudaBackend, id="cuda"))
    return backends


def tolerance(backend) -> dict:
    """assert_allclose tolerances for a backend's results against the float64
    numpy oracle. Double-precision kernels agree to rounding; single-precision
    ones carry float32 rounding, which ratios of small spreads (z-score, CCI,
    RSI over a flat stretch) amplify - a bug is orders of magnitude past this."""
    import numpy as np
    if getattr(backend, "float_type", np.float64) is np.float32:
        return {"rtol": 1e-3, "atol": 1e-3}
    return {"rtol": 1e-5, "atol": 1e-4}


@pytest.fixture(params=_generating_backends())
def generating_backend(request):
    """A generating backend class; the test runs once for each available one."""
    return request.param
