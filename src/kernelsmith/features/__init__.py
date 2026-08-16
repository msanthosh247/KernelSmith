"""A small starter library of features.

A feature is one signature (``specs.py``) plus one implementation per backend,
each in its own module. Implementations register themselves on import, so this
package pulls them in - optional ones only when their dependency is installed.

The dependency is probed with ``find_spec`` rather than caught as an
ImportError, so a genuine failure inside a kernel module still propagates
instead of silently leaving the backend unregistered.
"""
from __future__ import annotations

import importlib.util

from kernelsmith.features.specs import ema, rolling_min_max, sma
from kernelsmith.features import numpy_impl  # noqa: F401  - registers the CPU oracle

if importlib.util.find_spec("numba") is not None:
    from kernelsmith.features import numba_cpu_impl  # noqa: F401

__all__ = ["ema", "rolling_min_max", "sma"]
