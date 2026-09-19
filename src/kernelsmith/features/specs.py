"""Feature signatures.

A CallFactory is the single source of truth for a feature's types: every
backend's implementation is checked against these, and no implementation
declares its types twice.

Only features that need their own loop are here. Anything expressible as
arithmetic over these - Bollinger bands, MACD, ATR, crossovers - is composed in
``indicators.py`` and left to the compiler to fuse.
"""
from __future__ import annotations

from kernelsmith.dsl.graph import CallFactory, element
from kernelsmith.dsl.types import F4, I4

SERIES = F4[:]
TABLE = F4.table


def _feature(name, inputs, outputs=1):
    return CallFactory(
        name,
        input_signature=inputs,
        buffer_signature=[],
        output_signature=[SERIES] * outputs,
    )


# averages and volatility
sma = _feature("sma", [SERIES, I4])
ema = _feature("ema", [SERIES, I4])
rma = _feature("rma", [SERIES, I4])
wma = _feature("wma", [SERIES, I4])
rolling_sum = _feature("rolling_sum", [SERIES, I4])
rolling_std = _feature("rolling_std", [SERIES, I4])
mean_deviation = _feature("mean_deviation", [SERIES, I4])

# momentum and oscillators
rsi = _feature("rsi", [SERIES, I4])
stochastic_k = _feature("stochastic_k", [SERIES, SERIES, SERIES, I4])

# range and trend
true_range = _feature("true_range", [SERIES, SERIES, SERIES])
adx = _feature("adx", [SERIES, SERIES, SERIES, I4], outputs=3)

# windows and signal primitives
rolling_min_max = _feature("rolling_min_max", [SERIES, I4], outputs=2)
rolling_arg_min_max = _feature("rolling_arg_min_max", [SERIES, I4], outputs=2)
shift = _feature("shift", [SERIES, I4])

# simulation: the history is a table, the simulated path is the graph's series
bootstrap_path = _feature("bootstrap_path", [TABLE, I4, I4, I4])

FEATURES = (
    sma, ema, rma, wma, rolling_sum, rolling_std, mean_deviation,
    rsi, stochastic_k, true_range, adx,
    rolling_min_max, rolling_arg_min_max, shift,
    bootstrap_path,
    # backs series[i]; defined by the DSL itself, listed so every backend
    # registers a kernel for it like any other feature
    element,
)

# element is reached through series[i], not by name
__all__ = [factory.func_name for factory in FEATURES if factory is not element] + ["FEATURES"]
