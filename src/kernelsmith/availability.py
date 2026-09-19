"""Which optional backends can run here.

numba's CUDA target ships separately, as numba-cuda, and being installed is not
the same as being importable: recent versions load the CUDA runtime library
(libcudart) while ``numba.cuda`` is being imported, and on a machine without it
the import raises - and not an ImportError. So asking whether the package is
installed is the wrong question; the only reliable test is the import itself.

The simulator (``NUMBA_ENABLE_CUDASIM=1``) needs no runtime and imports anywhere.
"""
from __future__ import annotations

import functools
import importlib.util
from typing import Optional

# why the CUDA target is unavailable, for anyone wondering; None when it imports
cuda_unavailable_reason: Optional[str] = None


@functools.lru_cache(maxsize=None)
def cuda_importable() -> bool:
    """numba.cuda imports: kernels can be compiled for it (or simulated)."""
    global cuda_unavailable_reason
    if importlib.util.find_spec("numba_cuda") is None:
        cuda_unavailable_reason = "numba-cuda is not installed"
        return False
    try:
        from numba import cuda  # noqa: F401
    except Exception as error:      # e.g. DynamicLibNotFoundError: no CUDA runtime
        cuda_unavailable_reason = f"numba.cuda does not import: {type(error).__name__}: {error}"
        return False
    return True


@functools.lru_cache(maxsize=None)
def cuda_usable() -> bool:
    """A device - or the simulator - is there to run on."""
    if not cuda_importable():
        return False
    import warnings
    from numba import cuda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(cuda.is_available())


__all__ = ["cuda_importable", "cuda_unavailable_reason", "cuda_usable"]
