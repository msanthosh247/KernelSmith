"""Feature kernel bodies, written once and compiled by every generating backend.

These are plain Python functions - no decorator - so each backend compiles the
same source with its own compiler: ``njit`` for the Numba CPU backend,
``cuda.jit(device=True)`` for CUDA. Registration happens in the per-backend
modules (``numba_cpu_impl.py``, ``cuda_impl.py``). Helpers shared between
kernels are ``register_jitable``, which both compilers inline and which stays a
plain function under the CUDA simulator.

Rules that keep that possible:

- Plain loops only. No whole-array numpy calls, no allocation, no Python
  objects - device functions support none of them.
- The contract is (inputs..., scratch..., outputs...): write into the output
  arrays you are handed and return nothing.
- **Write every element of every output, on every path** - including the
  degenerate ones, where the answer is NaN. Buffers are recycled between ops
  and between runs and are never cleared, so an element a kernel skips keeps
  whatever the previous tenant left there.
- Do not assume the arrays are contiguous. On the CPU a series is a contiguous
  row; on the GPU it is a strided column, one element per time step.

Conventions every kernel follows:

- **One loop over time, warm-up predicated inside it.** On the GPU the 32
  threads of a warp are 32 parameter sets, usually with different periods. A
  kernel written as ``for i in range(period, n)`` starts each thread at a
  different ``i``, so at any moment they write 32 different rows - 32
  transactions instead of one. With a single ``for i in range(n)`` the threads
  stay in step and the writes coalesce; only reads that look back by
  ``period`` still scatter, and ``run(..., sort_by=...)`` narrows that.
- **Leading NaNs are skipped.** A series starts at its first non-NaN element,
  so a feature of a feature works: ``sma(rsi(x, 14), 5)`` is valid from the
  first bar both windows are full, not NaN forever. A NaN after the start
  propagates like any other arithmetic.
- A period below 1 gives an all-NaN output rather than an error; a period
  longer than the series simply never fills its window.

Precision
---------
Every scalar the kernels compute with is spelled ``FLOAT(...)`` or ``INT(...)``,
literals included. The module binds them to float64 / int64, which is what the
CPU backend compiles: FP64 is full rate on a CPU and keeps it bit-for-bit with
the numpy oracle. ``specialise(np.float32, np.int32)`` rebuilds every function
with them rebound, and that is what the CUDA backend compiles - on a GeForce
part FP64 runs at 1/64 of the FP32 rate and 64-bit integer arithmetic is
emulated with 32-bit instructions.

The spelling is not decoration. Numba types a bare ``1.0`` as float64 and a bare
``1`` as int64, and a float32 or int32 meeting either is widened - one literal
is enough to put the whole expression, and everything downstream of it, on the
slow path. ``tests/test_cuda.py`` reads the compiled PTX to hold this.

In float32 a running sum drifts - each add and subtract rounds, and over a few
thousand bars the rounding accumulates - so sliding sums are Kahan-compensated
(``_kahan``). In float64 the drift is negligible and the compensation would be
pure latency, so ``COMPENSATED`` switches it off at compile time.
"""
from __future__ import annotations

import math
import types

import numpy as np
from numba.extending import register_jitable

NAN = np.float32(np.nan)

# rebound by specialise(); see "Precision" above
FLOAT = np.float64
INT = np.int64
# Kahan compensation: needed in float32, pure latency in float64 - each
# compensated step is a chain of four dependent adds. A global is a compile-time
# constant to numba, so the unused branch of _kahan is removed entirely.
COMPENSATED = False


# ---- shared helpers ------------------------------------------------------------

@register_jitable
def first_valid(values):
    """Index of the first non-NaN element, or the length if there is none."""
    n = INT(values.shape[0])
    for i in range(n):
        if not math.isnan(values[i]):
            return i
    return n


@register_jitable
def first_valid_of(high, low, close):
    """Where all three series have started."""
    return max(first_valid(high), max(first_valid(low), first_valid(close)))


@register_jitable
def _kahan(total, carry, x):
    """Add ``x`` to ``total``, carrying the rounding error into the next add."""
    if not COMPENSATED:
        return total + x, carry
    y = x - carry
    moved = total + y
    carry = (moved - total) - y
    return moved, carry


@register_jitable
def _window_total(values, period, divisor, out):
    """Rolling sum over ``period`` valid elements, divided by ``divisor``."""
    n = INT(values.shape[0])
    start = first_valid(values)
    total = FLOAT(0.0)
    carry = FLOAT(0.0)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start:
            total, carry = _kahan(total, carry, FLOAT(values[i]))
            if i >= start + period:
                total, carry = _kahan(total, carry, -FLOAT(values[i - period]))
            if i >= start + period - INT(1):
                value = total / divisor
        out[i] = value


@register_jitable
def _exponential(values, period, alpha, out):
    """Exponential smoothing seeded with the mean of the first ``period`` values.
    The recurrence is contractive - old rounding decays by (1 - alpha) a bar -
    so it needs no compensation."""
    n = INT(values.shape[0])
    start = first_valid(values)
    total = FLOAT(0.0)
    previous = FLOAT(0.0)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start:
            if i < start + period:
                total += FLOAT(values[i])
                if i == start + period - INT(1):
                    previous = total / FLOAT(period)
                    value = previous
            else:
                previous = previous * (FLOAT(1.0) - alpha) + FLOAT(values[i]) * alpha
                value = previous
        out[i] = value


@register_jitable
def _rsi_value(gain, loss):
    if loss == FLOAT(0.0):
        # flat window: neutral
        return FLOAT(100.0) if gain > FLOAT(0.0) else FLOAT(50.0)
    return FLOAT(100.0) - FLOAT(100.0) / (FLOAT(1.0) + gain / loss)


# ---- averages and volatility ----------------------------------------------------

def sma_kernel(values, period, out):
    """Simple moving average over ``period`` bars."""
    _window_total(values, period, FLOAT(period), out)


def rolling_sum_kernel(values, period, out):
    """Sum over the last ``period`` bars."""
    _window_total(values, period, FLOAT(1.0), out)


def ema_kernel(values, period, out):
    """Exponential moving average, alpha = 2 / (period + 1), seeded with the first ``period`` mean."""
    alpha = FLOAT(2.0) / (FLOAT(period) + FLOAT(1.0)) if period > 0 else FLOAT(0.0)
    _exponential(values, period, alpha, out)


def rma_kernel(values, period, out):
    """Wilder's smoothing - an EMA with alpha = 1 / period (ATR, RSI, ADX)."""
    alpha = FLOAT(1.0) / FLOAT(period) if period > 0 else FLOAT(0.0)
    _exponential(values, period, alpha, out)


def wma_kernel(values, period, out):
    """Linearly weighted: the newest element weighs ``period``, the oldest 1.

    Sliding the window lowers every old weight by one - subtracting the plain
    window sum - and adds the new element at full weight, so each step is O(1).
    """
    n = INT(values.shape[0])
    start = first_valid(values)
    weight = FLOAT(period)
    norm = weight * (weight + FLOAT(1.0)) / FLOAT(2.0)
    total = FLOAT(0.0)
    total_carry = FLOAT(0.0)
    weighted = FLOAT(0.0)
    weighted_carry = FLOAT(0.0)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start:
            seen = i - start
            x = FLOAT(values[i])
            if seen < period:
                weighted, weighted_carry = _kahan(weighted, weighted_carry, FLOAT(seen + INT(1)) * x)
                total, total_carry = _kahan(total, total_carry, x)
            else:
                weighted, weighted_carry = _kahan(weighted, weighted_carry, weight * x - total)
                total, total_carry = _kahan(total, total_carry, x - FLOAT(values[i - period]))
            if seen >= period - INT(1):
                value = weighted / norm
        out[i] = value


def rolling_std_kernel(values, period, out):
    """Population standard deviation (ddof=0), the Bollinger convention.

    Welford's update, extended to a sliding window: replacing ``old`` with ``x``
    moves the sum of squared deviations by (x - old)(x - new_mean + old - mean).
    Unlike sum-of-squares minus squared sum, it does not cancel catastrophically
    when the spread is small next to the price.

    Precision, which matters in float32:
    - values are taken relative to an anchor, the series' first valid element.
      Variance does not care where zero is, and prices near the anchor subtract
      exactly (Sterbenz), so the mean is ~price-move sized rather than
      ~price sized, and so is its rounding;
    - the mean comes from a compensated window sum, not an incremental update;
    - the update is grouped as (x - new_mean) + (old - mean): two small
      differences, rather than a left-to-right sum through price magnitude;
    - the squares are compensated.

    Sliding still leaves a rounding residue from values that have left the
    window, so a flat window would read ~1e-6 instead of 0 - and a z-score
    over it a huge finite number instead of 0 / 0. Flatness is exact and cheap
    to track: count how many bars in a row repeat their predecessor.
    """
    n = INT(values.shape[0])
    start = first_valid(values)
    anchor = FLOAT(values[start]) if start < n else FLOAT(0.0)
    size = FLOAT(period)
    total = FLOAT(0.0)
    total_carry = FLOAT(0.0)
    mean = FLOAT(0.0)
    squares = FLOAT(0.0)
    squares_carry = FLOAT(0.0)
    repeats = INT(0)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start:
            seen = i - start
            x = FLOAT(values[i]) - anchor
            if seen > 0 and values[i] == values[i - INT(1)]:
                repeats += INT(1)
            else:
                repeats = INT(0)
            if seen < period:
                total, total_carry = _kahan(total, total_carry, x)
                new_mean = total / FLOAT(seen + INT(1))
                squares, squares_carry = _kahan(squares, squares_carry, (x - mean) * (x - new_mean))
            else:
                old = FLOAT(values[i - period]) - anchor
                total, total_carry = _kahan(total, total_carry, x - old)
                new_mean = total / size
                squares, squares_carry = _kahan(
                    squares, squares_carry, (x - old) * ((x - new_mean) + (old - mean)))
            mean = new_mean
            if seen >= period - INT(1):
                if repeats >= period - INT(1):
                    value = FLOAT(0.0)
                else:
                    value = math.sqrt(max(squares, FLOAT(0.0)) / size)
        out[i] = value


def mean_deviation_kernel(values, period, out):
    """Mean absolute deviation from the window mean (the CCI denominator).

    Not slideable: every deviation changes when the mean does, so each bar
    rereads its window. The sums are fresh each bar, so nothing drifts.
    """
    n = INT(values.shape[0])
    start = first_valid(values)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start + period - INT(1):
            total = FLOAT(0.0)
            for j in range(i - period + INT(1), i + INT(1)):
                total += FLOAT(values[j])
            mean = total / FLOAT(period)
            deviation = FLOAT(0.0)
            for j in range(i - period + INT(1), i + INT(1)):
                deviation += abs(FLOAT(values[j]) - mean)
            value = deviation / FLOAT(period)
        out[i] = value


# ---- momentum and oscillators ---------------------------------------------------

def rsi_kernel(values, period, out):
    """Wilder's RSI: average gain and loss seeded over the first ``period``
    changes, then smoothed with alpha = 1 / period. The first value is at the
    ``period``-th change, which needs period + 1 elements."""
    n = INT(values.shape[0])
    start = first_valid(values)
    size = FLOAT(period)
    gain = FLOAT(0.0)
    loss = FLOAT(0.0)
    for i in range(n):
        value = NAN
        if period > 0 and i > start:
            change = FLOAT(values[i]) - FLOAT(values[i - INT(1)])
            up = change if change > FLOAT(0.0) else FLOAT(0.0)
            down = -change if change < FLOAT(0.0) else FLOAT(0.0)
            if i <= start + period:
                gain += up
                loss += down
                if i == start + period:
                    gain /= size
                    loss /= size
                    value = _rsi_value(gain, loss)
            else:
                gain = (gain * (size - FLOAT(1.0)) + up) / size
                loss = (loss * (size - FLOAT(1.0)) + down) / size
                value = _rsi_value(gain, loss)
        out[i] = value


def stochastic_k_kernel(high, low, close, period, out):
    """Fast %K: where the close sits in the window's high-low range, 0..100.
    A flat window has no range; it reads 50."""
    n = INT(close.shape[0])
    start = first_valid_of(high, low, close)
    for i in range(n):
        value = NAN
        if period > 0 and i >= start + period - INT(1):
            highest = FLOAT(high[i])
            lowest = FLOAT(low[i])
            for j in range(i - period + INT(1), i):
                highest = max(highest, FLOAT(high[j]))
                lowest = min(lowest, FLOAT(low[j]))
            span = highest - lowest
            if span == FLOAT(0.0):
                value = FLOAT(50.0)
            else:
                value = FLOAT(100.0) * (FLOAT(close[i]) - lowest) / span
        out[i] = value


# ---- range and trend ------------------------------------------------------------

def true_range_kernel(high, low, close, out):
    """The bar's range, stretched to cover a gap from the previous close.
    The first bar has no previous close: its true range is high - low."""
    n = INT(close.shape[0])
    start = first_valid_of(high, low, close)
    for i in range(n):
        value = NAN
        if i >= start:
            h = FLOAT(high[i])
            lo = FLOAT(low[i])
            value = h - lo
            if i > start:
                previous = FLOAT(close[i - INT(1)])
                value = max(value, max(abs(h - previous), abs(lo - previous)))
        out[i] = value


def adx_kernel(high, low, close, period, plus_di, minus_di, adx):
    """Wilder's directional movement system.

    +DM / -DM are the parts of today's range outside yesterday's; they and the
    true range are summed over the first ``period`` changes and then carried
    with Wilder's running-sum update, S = S - S / period + x. DI is the ratio
    of the sums. DX = |+DI - -DI| / (+DI + -DI), and ADX is DX averaged over
    ``period`` values, then Wilder-smoothed. DI starts at the ``period``-th
    change, ADX ``period - 1`` bars later. The updates are contractive, like
    the EMA's, so they need no compensation.
    """
    n = INT(close.shape[0])
    start = first_valid_of(high, low, close)
    size = FLOAT(period)
    zero = FLOAT(0.0)
    hundred = FLOAT(100.0)
    range_sum = zero
    plus_sum = zero
    minus_sum = zero
    dx_sum = zero
    average = zero
    for i in range(n):
        plus_value = NAN
        minus_value = NAN
        adx_value = NAN
        if period > 0 and i > start:
            h = FLOAT(high[i])
            lo = FLOAT(low[i])
            previous = FLOAT(close[i - INT(1)])
            up = h - FLOAT(high[i - INT(1)])
            down = FLOAT(low[i - INT(1)]) - lo
            plus_move = up if up > down and up > zero else zero
            minus_move = down if down > up and down > zero else zero
            true_range = max(h - lo, max(abs(h - previous), abs(lo - previous)))

            changes = i - start
            if changes <= period:
                range_sum += true_range
                plus_sum += plus_move
                minus_sum += minus_move
            else:
                range_sum = range_sum - range_sum / size + true_range
                plus_sum = plus_sum - plus_sum / size + plus_move
                minus_sum = minus_sum - minus_sum / size + minus_move

            if changes >= period:
                plus = hundred * plus_sum / range_sum if range_sum > zero else zero
                minus = hundred * minus_sum / range_sum if range_sum > zero else zero
                spread = plus + minus
                dx = hundred * abs(plus - minus) / spread if spread > zero else zero
                plus_value = plus
                minus_value = minus
                if changes < INT(2) * period - INT(1):
                    dx_sum += dx
                elif changes == INT(2) * period - INT(1):
                    average = (dx_sum + dx) / size
                    adx_value = average
                else:
                    average = (average * (size - FLOAT(1.0)) + dx) / size
                    adx_value = average
        plus_di[i] = plus_value
        minus_di[i] = minus_value
        adx[i] = adx_value


# ---- windows and signal primitives ----------------------------------------------

def rolling_min_max_kernel(values, period, lows, highs):
    """Lowest and highest value over the last ``period`` bars (Donchian channel)."""
    n = INT(values.shape[0])
    start = first_valid(values)
    for i in range(n):
        lowest = NAN
        highest = NAN
        if period > 0 and i >= start + period - INT(1):
            lowest = values[i]
            highest = values[i]
            for j in range(i - period + INT(1), i):
                lowest = min(lowest, values[j])
                highest = max(highest, values[j])
        lows[i] = lowest
        highs[i] = highest


def rolling_arg_min_max_kernel(values, period, since_low, since_high):
    """Bars since the window's lowest and highest element (0 = this bar).
    Ties go to the most recent occurrence."""
    n = INT(values.shape[0])
    start = first_valid(values)
    for i in range(n):
        low_age = NAN
        high_age = NAN
        if period > 0 and i >= start + period - INT(1):
            lowest = i - period + INT(1)
            highest = lowest
            for j in range(lowest + INT(1), i + INT(1)):
                if values[j] <= values[lowest]:
                    lowest = j
                if values[j] >= values[highest]:
                    highest = j
            low_age = FLOAT(i - lowest)
            high_age = FLOAT(i - highest)
        since_low[i] = low_age
        since_high[i] = high_age


def shift_kernel(values, period, out):
    """``values`` delayed by ``period`` bars. Only non-negative shifts: a
    negative one would read the future, which a backtest must never do."""
    n = INT(values.shape[0])
    for i in range(n):
        value = NAN
        if period >= 0 and i >= period:
            value = values[i - period]
        out[i] = value


# ---- indexing and simulation -----------------------------------------------------

def element_kernel(values, index, out):
    """``series[index]`` into ``out[0]``; negative counts from the end, and an
    index past either end is NaN."""
    n = INT(values.shape[0])
    i = INT(index)
    if i < INT(0):
        i += n
    out[0] = values[i] if INT(0) <= i < n else NAN


# A counter-based hash stands in for a random number generator: no state, no
# generator object, just a function of (seed, path, counter). That is what
# makes a path the same bits under njit, cuda.jit and numpy - a stateful
# generator exists on none of them in the same form. The rounds are Chris
# Wellons' lowbias32. Multiplies go through uint64 and are masked back to 32
# bits: exact everywhere, silent under numpy (no scalar-overflow warning), and
# LLVM narrows a masked 64-bit product to a 32-bit multiply on the GPU.
_LOW32 = np.uint64(0xFFFFFFFF)


@register_jitable
def _mix32(x):
    x = x ^ (x >> np.uint32(16))
    x = np.uint32((np.uint64(x) * np.uint64(0x7FEB352D)) & _LOW32)
    x = x ^ (x >> np.uint32(15))
    x = np.uint32((np.uint64(x) * np.uint64(0x846CA68B)) & _LOW32)
    x = x ^ (x >> np.uint32(16))
    return x


@register_jitable
def _hash3(seed, path, counter):
    """A well-mixed uint32 from three integers."""
    return _mix32(np.uint32(seed) ^ _mix32(np.uint32(path) ^ _mix32(np.uint32(counter))))


def bootstrap_path_kernel(close, block, path, seed, out):
    """One simulated future path by circular moving-block bootstrap.

    The history ``close`` gives ``len(close) - 1`` one-bar growth factors,
    close[j + 1] / close[j]. Every ``block`` steps a new block starts at a
    random factor; within a block the factors are consecutive, wrapping from the
    end of the history to its start. Consecutive factors keep what a single
    draw would lose - volatility clustering, short-range autocorrelation -
    and ``block = 1`` is the plain (iid) bootstrap.

    ``out[h]`` is the price after h + 1 steps from the last close, so the path
    is as long as the graph's time axis (``run(..., n_bars=H)``). The start of
    block b is hash(seed, path, b): the same (seed, path) is the same path on
    every backend, and different paths are independent.

    The history must be free of NaNs; a block below 1, or a history with fewer
    than two prices, gives an all-NaN path.
    """
    n = INT(out.shape[0])
    factors = INT(close.shape[0]) - INT(1)
    level = FLOAT(close[factors]) if factors >= INT(0) else FLOAT(0.0)
    usable = block > 0 and factors > INT(0)
    left = INT(0)
    blocks = INT(0)     # counts blocks - h // block without a division, which
    j = INT(0)          # numba does in 64 bits and the GPU emulates
    for h in range(n):
        value = NAN
        if usable:
            if left == INT(0):
                draw = _hash3(seed, path, blocks)
                blocks += INT(1)
                j = INT(draw % np.uint32(factors))
                left = INT(block)
            level = level * (FLOAT(close[j + INT(1)]) / FLOAT(close[j]))
            value = level
            left -= INT(1)
            j += INT(1)
            if j == factors:
                j = INT(0)
        out[h] = value


# ---- precision -------------------------------------------------------------------

def specialise(float_type, int_type) -> types.SimpleNamespace:
    """Every kernel and helper above, rebuilt with FLOAT and INT rebound.

    Each function is re-created from its code object over a copy of this
    module's globals, so the copies call each other rather than the float64
    originals. Helpers are re-registered as jitable - registration is keyed by
    the function object, and these are new ones.
    """
    namespace = dict(globals())
    namespace.update(FLOAT=float_type, INT=int_type,
                     COMPENSATED=np.dtype(float_type).itemsize < 8)
    for name, fn in list(namespace.items()):
        if not isinstance(fn, types.FunctionType) or fn.__module__ != __name__ or fn is specialise:
            continue
        clone = types.FunctionType(fn.__code__, namespace, name, fn.__defaults__, fn.__closure__)
        clone.__doc__, clone.__qualname__ = fn.__doc__, fn.__qualname__
        namespace[name] = clone if name.endswith("_kernel") else register_jitable(clone)
    return types.SimpleNamespace(**namespace)
