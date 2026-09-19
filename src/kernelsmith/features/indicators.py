"""Indicators composed from feature kernels and arithmetic.

Nothing here has a kernel of its own. Each function builds graph nodes, so the
compiler sees through it: the arithmetic is fused into the surrounding
expression, and two indicators that share a sub-feature - Bollinger and a
z-score over the same window - compute it once after CSE.

Arguments are graph values or plain constants; periods are int params or ints,
multipliers float params or floats.

Division follows IEEE on every backend: a zero denominator gives inf, or NaN
for 0 / 0. That is the honest answer for a flat window; comparisons against NaN
are false, so a signal built on one simply does not fire.
"""
from __future__ import annotations

from kernelsmith.dsl import Shape, ValueNode
from kernelsmith.features.specs import (
    ema, mean_deviation, rma, rolling_min_max, rolling_std, shift, sma,
    stochastic_k, true_range,
)

__all__ = [
    "atr", "bollinger", "cci", "cross_over", "cross_under", "donchian",
    "keltner", "macd", "momentum", "roc", "stochastic", "williams_r", "zscore",
]


# ---- averages and volatility ----------------------------------------------------

def bollinger(values, period=20, width=2.0):
    """(lower, middle, upper): the SMA, plus and minus ``width`` population
    standard deviations over the same window."""
    middle = sma(values, period)
    band = rolling_std(values, period) * width
    return middle - band, middle, middle + band


def zscore(values, period=20):
    """How many standard deviations ``values`` sits from its moving average."""
    return (values - sma(values, period)) / rolling_std(values, period)


# ---- momentum and oscillators ---------------------------------------------------

def momentum(values, period=10):
    """Change over ``period`` bars."""
    return values - shift(values, period)


def roc(values, period=10):
    """Rate of change over ``period`` bars, in percent."""
    return (values / shift(values, period) - 1.0) * 100.0


def macd(values, fast=12, slow=26, signal=9):
    """(line, signal, histogram): fast EMA minus slow EMA, its own EMA, and the gap between them.

    The signal EMA starts once the line does - kernels skip leading NaNs - so
    it is valid from bar slow + signal - 2."""
    line = ema(values, fast) - ema(values, slow)
    trigger = ema(line, signal)
    return line, trigger, line - trigger


def stochastic(high, low, close, k_period=14, d_period=3):
    """(%K, %D): fast %K and its ``d_period`` simple average."""
    k = stochastic_k(high, low, close, k_period)
    return k, sma(k, d_period)


def williams_r(high, low, close, period=14):
    """Williams %R, -100..0: %K shifted down by 100."""
    return stochastic_k(high, low, close, period) - 100.0


def cci(high, low, close, period=20):
    """Commodity channel index over the typical price (high + low + close) / 3."""
    typical = (high + low + close) / 3.0
    return (typical - sma(typical, period)) / (mean_deviation(typical, period) * 0.015)


# ---- range and trend ------------------------------------------------------------

def atr(high, low, close, period=14):
    """Average true range: Wilder's smoothing of the true range."""
    return rma(true_range(high, low, close), period)


def keltner(high, low, close, period=20, atr_period=10, multiplier=2.0):
    """(lower, middle, upper): the close's EMA, plus and minus ``multiplier`` ATRs."""
    middle = ema(close, period)
    band = atr(high, low, close, atr_period) * multiplier
    return middle - band, middle, middle + band


def donchian(high, low, period=20):
    """(lower, middle, upper): the lowest low and highest high over ``period``."""
    lower, _ = rolling_min_max(low, period)
    _, upper = rolling_min_max(high, period)
    return lower, (lower + upper) / 2.0, upper


# ---- signal primitives ----------------------------------------------------------

def _previous(value):
    """The value one bar ago; a level - a constant or a scalar param - has no
    history, so it is its own previous value. ``rsi > 30`` crossings need this."""
    if isinstance(value, ValueNode) and value.shape is Shape.VECTOR:
        return shift(value, 1)
    return value


def cross_over(a, b):
    """True on the bar ``a`` moves from at-or-below ``b`` to above it.
    Either side may be a series or a level."""
    return (a > b) & (_previous(a) <= _previous(b))


def cross_under(a, b):
    """True on the bar ``a`` moves from at-or-above ``b`` to below it.
    Either side may be a series or a level."""
    return (a < b) & (_previous(a) >= _previous(b))
