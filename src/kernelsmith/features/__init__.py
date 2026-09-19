"""A small starter library of features.

A feature is one signature (``specs.py``) plus one implementation per backend,
each in its own module. Indicators (``indicators.py``) are compositions of
features and arithmetic, with no implementation of their own. Implementations register themselves on import, so this
package pulls them in - optional ones only when their dependency is installed.

The dependency is probed with ``find_spec`` rather than caught as an
ImportError, so a genuine failure inside a kernel module still propagates
instead of silently leaving the backend unregistered.
"""
from __future__ import annotations

import importlib.util

from kernelsmith.features import indicators, specs
from kernelsmith.features.indicators import *  # noqa: F401,F403
from kernelsmith.features.specs import *  # noqa: F401,F403
from kernelsmith.features import numpy_impl  # noqa: F401  - registers the CPU oracle

if importlib.util.find_spec("numba") is not None:
    from kernelsmith.features import numba_cpu_impl  # noqa: F401

    # numba's CUDA target ships separately, as numba-cuda
    if importlib.util.find_spec("numba_cuda") is not None:
        from kernelsmith.features import cuda_impl  # noqa: F401

__all__ = [name for name in specs.__all__ if name != "FEATURES"] + indicators.__all__
