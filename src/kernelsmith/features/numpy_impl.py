"""Reference implementations for the numpy CPU backend.

These are the oracle: every faster backend is diffed against them, so they are
written for obviousness rather than speed - each windowed feature recomputes
its window from the definition instead of sliding it, so an error in a
kernel's O(1) update cannot be mirrored here. Importing this module registers
them.

The conventions match ``kernels.py``: leading NaNs are skipped, a period below
1 gives all NaN, outputs are float32 and computed in float64.
"""
from __future__ import annotations

import numpy as np

from kernelsmith.backends.cpu import cpu_impl
from kernelsmith.features.specs import (
    adx, bootstrap_path, element, ema, mean_deviation, rma, rolling_arg_min_max,
    rolling_min_max, rolling_std, rolling_sum, rsi, shift, sma, stochastic_k,
    true_range, wma,
)


def _series(values):
    return np.asarray(values, dtype=np.float64)


def _start(*series):
    """Index at which every series has started (its first non-NaN element)."""
    start = 0
    for values in series:
        valid = np.flatnonzero(~np.isnan(values))
        start = max(start, valid[0] if valid.size else values.size)
    return start


def _nans(n):
    return np.full(n, np.nan, dtype=np.float32)


def _windowed(values, period, reduce):
    """``reduce`` over every full window of ``period`` valid elements.

    ``reduce`` gets all windows at once, one per row - ``sliding_window_view``
    is a view, not a copy - and returns one value per row. Still each window
    from its definition, nothing carried from one to the next; just not one
    Python call per bar.
    """
    period = int(period)
    out = _nans(values.size)
    start = _start(values)
    if period < 1 or start + period > values.size:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values[start:], period)
    out[start + period - 1:] = reduce(windows)
    return out


def _smoothed(values, period, alpha):
    """Mean of the first ``period`` valid values, then exponential smoothing."""
    period = int(period)
    out = _nans(values.size)
    start = _start(values)
    if period < 1 or start + period > values.size:
        return out
    previous = values[start: start + period].mean()
    out[start + period - 1] = previous
    for i in range(start + period, values.size):
        previous = previous * (1.0 - alpha) + values[i] * alpha
        out[i] = previous
    return out


# ---- averages and volatility ----------------------------------------------------

@cpu_impl(sma)
def _sma(values, period):
    return (_windowed(_series(values), period, lambda w: w.mean(axis=1)),)


@cpu_impl(rolling_sum)
def _rolling_sum(values, period):
    return (_windowed(_series(values), period, lambda w: w.sum(axis=1)),)


@cpu_impl(ema)
def _ema(values, period):
    return (_smoothed(_series(values), period, 2.0 / (int(period) + 1.0)),)


@cpu_impl(rma)
def _rma(values, period):
    period = int(period)
    return (_smoothed(_series(values), period, 1.0 / period if period > 0 else 0.0),)


@cpu_impl(wma)
def _wma(values, period):
    weights = np.arange(1, int(period) + 1, dtype=np.float64)
    return (_windowed(_series(values), period, lambda w: w @ weights / weights.sum()),)


@cpu_impl(rolling_std)
def _rolling_std(values, period):
    return (_windowed(_series(values), period, lambda w: w.std(axis=1)),)       # ddof=0


@cpu_impl(mean_deviation)
def _mean_deviation(values, period):
    return (_windowed(_series(values), period,
                      lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)),)


# ---- momentum and oscillators ---------------------------------------------------

@cpu_impl(rsi)
def _rsi(values, period):
    values, period = _series(values), int(period)
    out = _nans(values.size)
    start = _start(values)
    if period < 1 or start + period >= values.size:
        return (out,)
    changes = np.diff(values[start:])
    gains, losses = np.maximum(changes, 0.0), np.maximum(-changes, 0.0)
    gain, loss = gains[:period].mean(), losses[:period].mean()

    def value(gain, loss):
        if loss == 0.0:
            return 100.0 if gain > 0.0 else 50.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[start + period] = value(gain, loss)
    for k in range(period, changes.size):
        gain = (gain * (period - 1) + gains[k]) / period
        loss = (loss * (period - 1) + losses[k]) / period
        out[start + k + 1] = value(gain, loss)
    return (out,)


@cpu_impl(stochastic_k)
def _stochastic_k(high, low, close, period):
    high, low, close, period = _series(high), _series(low), _series(close), int(period)
    out = _nans(close.size)
    if period < 1:
        return (out,)
    for i in range(_start(high, low, close) + period - 1, close.size):
        highest = high[i - period + 1: i + 1].max()
        lowest = low[i - period + 1: i + 1].min()
        span = highest - lowest
        out[i] = 50.0 if span == 0.0 else 100.0 * (close[i] - lowest) / span
    return (out,)


# ---- range and trend ------------------------------------------------------------

def _true_ranges(high, low, close, start):
    """True range from ``start`` on; the first bar has no previous close."""
    ranges = high[start:] - low[start:]
    previous = close[start:-1]
    ranges[1:] = np.maximum.reduce([
        ranges[1:], np.abs(high[start + 1:] - previous), np.abs(low[start + 1:] - previous),
    ])
    return ranges


@cpu_impl(true_range)
def _true_range(high, low, close):
    high, low, close = _series(high), _series(low), _series(close)
    out = _nans(close.size)
    start = _start(high, low, close)
    if start < close.size:
        out[start:] = _true_ranges(high, low, close, start)
    return (out,)


@cpu_impl(adx)
def _adx(high, low, close, period):
    high, low, close, period = _series(high), _series(low), _series(close), int(period)
    n = close.size
    plus_di, minus_di, average = _nans(n), _nans(n), _nans(n)
    start = _start(high, low, close)
    if period < 1 or start + period >= n:
        return plus_di, minus_di, average

    up = np.diff(high[start:])
    down = -np.diff(low[start:])
    plus_moves = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_moves = np.where((down > up) & (down > 0.0), down, 0.0)
    ranges = _true_ranges(high, low, close, start)[1:]           # one per change

    def wilder_sums(moves):
        """Sum of the first ``period``, then S = S - S / period + x."""
        sums = np.empty(moves.size - period + 1)
        sums[0] = moves[:period].sum()
        for k in range(1, sums.size):
            sums[k] = sums[k - 1] - sums[k - 1] / period + moves[period - 1 + k]
        return sums

    range_sums = wilder_sums(ranges)
    safe = np.where(range_sums > 0.0, range_sums, 1.0)
    plus = np.where(range_sums > 0.0, 100.0 * wilder_sums(plus_moves) / safe, 0.0)
    minus = np.where(range_sums > 0.0, 100.0 * wilder_sums(minus_moves) / safe, 0.0)
    spread = plus + minus
    dx = np.where(spread > 0.0, 100.0 * np.abs(plus - minus) / np.where(spread > 0.0, spread, 1.0), 0.0)

    first = start + period                    # the bar of the period-th change
    plus_di[first:] = plus
    minus_di[first:] = minus
    if dx.size >= period:
        smoothed = dx[:period].mean()
        average[first + period - 1] = smoothed
        for k in range(period, dx.size):
            smoothed = (smoothed * (period - 1) + dx[k]) / period
            average[first + k] = smoothed
    return plus_di, minus_di, average


# ---- windows and signal primitives ----------------------------------------------

@cpu_impl(rolling_min_max)
def _rolling_min_max(values, period):
    values = _series(values)
    return (_windowed(values, period, lambda w: w.min(axis=1)),
            _windowed(values, period, lambda w: w.max(axis=1)))


@cpu_impl(rolling_arg_min_max)
def _rolling_arg_min_max(values, period):
    """Bars since the extreme; reversing the window makes argmin/argmax - which
    return the first occurrence - pick the most recent one."""
    values = _series(values)
    return (
        _windowed(values, period, lambda w: np.argmin(w[:, ::-1], axis=1)),
        _windowed(values, period, lambda w: np.argmax(w[:, ::-1], axis=1)),
    )


@cpu_impl(shift)
def _shift(values, period):
    values, period = _series(values), int(period)
    out = _nans(values.size)
    if 0 <= period < values.size:
        out[period:] = values[:values.size - period]
    return (out,)


# ---- indexing and simulation ----------------------------------------------------

@cpu_impl(element)
def _element(values, index):
    values, index = np.asarray(values), int(index)
    if index < 0:
        index += values.size
    return (np.float32(values[index]) if 0 <= index < values.size else np.float32(np.nan),)


_LOW32 = np.uint64(0xFFFFFFFF)


def _mix32(x):
    """lowbias32 over uint32 arrays - the same bits as kernels._mix32."""
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x = ((x.astype(np.uint64) * np.uint64(0x7FEB352D)) & _LOW32).astype(np.uint32)
    x ^= x >> np.uint32(15)
    x = ((x.astype(np.uint64) * np.uint64(0x846CA68B)) & _LOW32).astype(np.uint32)
    x ^= x >> np.uint32(16)
    return x


def hash3(seed, path, counter):
    """kernels._hash3, vectorised: a uint32 per counter."""
    counter = np.asarray(counter, dtype=np.int64).astype(np.uint32)
    inner = _mix32(np.uint32(path) ^ _mix32(counter))
    return _mix32(np.uint32(seed) ^ inner)


@cpu_impl(bootstrap_path)
def _bootstrap_path(close, block, path, seed, *, n_bars):
    """Circular moving-block bootstrap, written as the definition: the start of
    each block from the hash, the factor for step h is start + (h mod block),
    wrapped - no running counter, so a mistake in the kernel's bookkeeping
    cannot be mirrored here."""
    close, block = np.asarray(close, dtype=np.float64), int(block)
    factors = close.size - 1
    if block < 1 or factors < 1:
        return (_nans(n_bars),)
    growth = close[1:] / close[:-1]
    steps = np.arange(n_bars)
    starts = hash3(seed, path, steps // block) % np.uint32(factors)
    index = (starts.astype(np.int64) + steps % block) % factors
    return ((close[-1] * np.cumprod(growth[index])).astype(np.float32),)
