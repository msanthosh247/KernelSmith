"""njit kernels for the numba CPU backend.

The contract is (inputs..., scratch..., outputs...): a kernel writes into the
output arrays it is handed and returns nothing. Bodies are plain loops with no
whole-array numpy calls, so the same source compiles under
``@cuda.jit(device=True)`` when the CUDA backend arrives.

**Every element of every output must be written, on every path** - including
the degenerate ones, where the answer is NaN. Buffers are recycled between ops
and between runs and are never cleared, so an element a kernel skips keeps
whatever the previous tenant left there.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from kernelsmith.backends.numba_cpu import register_numba_cpu
from kernelsmith.features.specs import ema, rolling_min_max, sma

NAN = np.float32(np.nan)


@register_numba_cpu(sma)
@njit(cache=True)
def sma_numba_cpu(values, period, out):
    n = values.shape[0]
    if period <= 0 or period > n:
        for i in range(n):
            out[i] = NAN
        return
    for i in range(period - 1):
        out[i] = NAN
    total = 0.0
    for i in range(period):
        total += values[i]
    out[period - 1] = total / period
    for i in range(period, n):
        total += values[i] - values[i - period]
        out[i] = total / period


@register_numba_cpu(ema)
@njit(cache=True)
def ema_numba_cpu(values, period, out):
    n = values.shape[0]
    if period <= 0 or period > n:
        for i in range(n):
            out[i] = NAN
        return
    for i in range(period - 1):
        out[i] = NAN
    alpha = 2.0 / (period + 1.0)
    total = 0.0
    for i in range(period):
        total += values[i]
    prev = total / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = prev * (1.0 - alpha) + values[i] * alpha
        out[i] = prev


@register_numba_cpu(rolling_min_max)
@njit(cache=True)
def rolling_min_max_numba_cpu(values, period, lows, highs):
    n = values.shape[0]
    if period <= 0 or period > n:
        for i in range(n):
            lows[i] = NAN
            highs[i] = NAN
        return
    for i in range(period - 1):
        lows[i] = NAN
        highs[i] = NAN
    for i in range(period - 1, n):
        low = values[i]
        high = values[i]
        for j in range(i - period + 1, i + 1):
            if values[j] < low:
                low = values[j]
            if values[j] > high:
                high = values[j]
        lows[i] = low
        highs[i] = high
