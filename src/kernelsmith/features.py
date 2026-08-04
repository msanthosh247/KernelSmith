"""A small starter library of features.

Each feature is a signature (the CallFactory) plus one implementation per
backend. Only the CPU reference exists today; the CUDA implementations attach
to the same factories later without touching this file's specs.
"""
from __future__ import annotations

import numpy as np

from kernelsmith.backends.cpu import cpu_impl
from kernelsmith.dsl.graph import CallFactory
from kernelsmith.dsl.types import F4, I4, SCAL, VEC

sma = CallFactory(
    "sma",
    input_signature=[VEC(F4), SCAL(I4)],
    buffer_signature=[],
    output_signature=[VEC(F4)],
)

ema = CallFactory(
    "ema",
    input_signature=[VEC(F4), SCAL(I4)],
    buffer_signature=[],
    output_signature=[VEC(F4)],
)

rolling_min_max = CallFactory(
    "rolling_min_max",
    input_signature=[VEC(F4), SCAL(I4)],
    buffer_signature=[],
    output_signature=[VEC(F4), VEC(F4)],
)


@cpu_impl(sma)
def _sma(values, period):
    """Simple moving average; the first period-1 entries are NaN."""
    period = int(period)
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if period > 0 and period <= values.size:
        cumsum = np.cumsum(np.insert(values, 0, 0.0))
        out[period - 1:] = ((cumsum[period:] - cumsum[:-period]) / period).astype(np.float32)
    return (out,)


@cpu_impl(ema)
def _ema(values, period):
    """Exponential moving average seeded with the first ``period`` mean."""
    period = int(period)
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if period > 0 and period <= values.size:
        alpha = 2.0 / (period + 1.0)
        prev = values[:period].mean()
        out[period - 1] = prev
        for i in range(period, values.size):
            prev = prev * (1.0 - alpha) + values[i] * alpha
            out[i] = prev
    return (out,)


@cpu_impl(rolling_min_max)
def _rolling_min_max(values, period):
    """Rolling window minimum and maximum, NaN-padded at the front."""
    period = int(period)
    values = np.asarray(values, dtype=np.float64)
    lows = np.full(values.shape, np.nan, dtype=np.float32)
    highs = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(period - 1, values.size):
        window = values[i - period + 1: i + 1]
        lows[i] = window.min()
        highs[i] = window.max()
    return (lows, highs)
