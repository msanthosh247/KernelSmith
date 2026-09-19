"""The feature library: every kernel and indicator against the numpy oracle,
and the oracle against values worked out by hand.

Parity runs on every generating backend available - the Numba CPU backend
always, CUDA on a GPU or under the simulator:

    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_indicators.py
"""
import numpy as np
import pytest

pytest.importorskip("numba")

from kernelsmith import Graph  # noqa: E402
from kernelsmith.backends.cpu import CPUBackend  # noqa: E402
from kernelsmith.backends.numba_cpu import NumbaCPU_Backend  # noqa: E402
from kernelsmith import features as F  # noqa: E402
from conftest import tolerance  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Grid size"),
    # 0 / 0 over a flat window is NaN by design - see indicators.py
    pytest.mark.filterwarnings("ignore:invalid value encountered in .*divide:RuntimeWarning"),
]


N_BARS = 60
# degenerate periods included on purpose: below 1, exactly 1, longer than the series
PERIODS = np.array([0, 1, 2, 5, 14, 70], dtype=np.int32)


def ohlc(n=N_BARS, seed=3):
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    opn = close + rng.normal(0, 0.5, n)
    high = np.maximum(opn, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(opn, close) - np.abs(rng.normal(0, 0.4, n))
    high[20:26] = low[20:26] = close[20:26] = 101.0             # a flat stretch
    return {name: series.astype(np.float32)
            for name, series in (("high", high), ("low", low), ("close", close))}


def graph(build):
    """``build(high, low, close, n)`` -> value or tuple of values, one output each."""
    g = Graph()
    high, low, close = (g.register_input(name) for name in ("high", "low", "close"))
    result = build(high, low, close, g.int_param("n"))
    for i, value in enumerate(result if isinstance(result, tuple) else (result,)):
        g.register_output(f"out{i}", value)
    return g


def assert_matches_oracle(backend, build, params=None):
    params = params if params is not None else {"n": PERIODS}
    data = ohlc()
    expected = CPUBackend().compile(graph(build)).run(data, params)
    actual = backend().compile(graph(build)).run(data, params)
    for name, want in expected.items():
        got = np.asarray(actual[name])
        if want.dtype == np.bool_:
            np.testing.assert_array_equal(got, want, err_msg=name)
        else:
            np.testing.assert_allclose(got, want, equal_nan=True, err_msg=name, **tolerance(backend))
    return expected


FEATURES = {
    "sma": lambda h, l, c, n: F.sma(c, n),
    "ema": lambda h, l, c, n: F.ema(c, n),
    "rma": lambda h, l, c, n: F.rma(c, n),
    "wma": lambda h, l, c, n: F.wma(c, n),
    "rolling_sum": lambda h, l, c, n: F.rolling_sum(c, n),
    "rolling_std": lambda h, l, c, n: F.rolling_std(c, n),
    "mean_deviation": lambda h, l, c, n: F.mean_deviation(c, n),
    "rsi": lambda h, l, c, n: F.rsi(c, n),
    "stochastic_k": lambda h, l, c, n: F.stochastic_k(h, l, c, n),
    "true_range": lambda h, l, c, n: F.true_range(h, l, c),
    "adx": lambda h, l, c, n: F.adx(h, l, c, n),
    "rolling_min_max": lambda h, l, c, n: F.rolling_min_max(c, n),
    "rolling_arg_min_max": lambda h, l, c, n: F.rolling_arg_min_max(c, n),
    "shift": lambda h, l, c, n: F.shift(c, n),
    # a feature of a feature: the inner warm-up is a NaN prefix to skip
    "sma_of_rsi": lambda h, l, c, n: F.sma(F.rsi(c, n), 3),
    "rsi_of_ema": lambda h, l, c, n: F.rsi(F.ema(c, 4), n),
}

INDICATORS = {
    "bollinger": lambda h, l, c, n: F.bollinger(c, n, 2.0),
    "zscore": lambda h, l, c, n: F.zscore(c, n),
    "momentum": lambda h, l, c, n: F.momentum(c, n),
    "roc": lambda h, l, c, n: F.roc(c, n),
    "macd": lambda h, l, c, n: F.macd(c, 3, n, 4),
    "stochastic": lambda h, l, c, n: F.stochastic(h, l, c, n, 3),
    "williams_r": lambda h, l, c, n: F.williams_r(h, l, c, n),
    "cci": lambda h, l, c, n: F.cci(h, l, c, n),
    "atr": lambda h, l, c, n: F.atr(h, l, c, n),
    "keltner": lambda h, l, c, n: F.keltner(h, l, c, n, 5, 1.5),
    "donchian": lambda h, l, c, n: F.donchian(h, l, n),
    "cross_over": lambda h, l, c, n: F.cross_over(F.sma(c, 3), F.sma(c, n)),
    "cross_under": lambda h, l, c, n: F.cross_under(F.sma(c, 3), F.sma(c, n)),
    "cross_level": lambda h, l, c, n: F.cross_over(F.rsi(c, n), 50.0),
}


@pytest.mark.parametrize("build", FEATURES.values(), ids=FEATURES.keys())
def test_feature_matches_the_oracle(generating_backend, build):
    assert_matches_oracle(generating_backend, build)


@pytest.mark.parametrize("build", INDICATORS.values(), ids=INDICATORS.keys())
def test_indicator_matches_the_oracle(generating_backend, build):
    assert_matches_oracle(generating_backend, build)


def test_indicators_with_float_params(generating_backend):
    def graph_with_width():
        g = Graph()
        close = g.register_input("close")
        lower, _, upper = F.bollinger(close, g.int_param("n"), g.float_param("width"))
        g.register_output("lower", lower)
        g.register_output("upper", upper)
        return g

    data = {"close": ohlc()["close"]}
    params = {"n": [5, 10], "width": [1.0, 2.5]}
    expected = CPUBackend().compile(graph_with_width()).run(data, params)
    actual = generating_backend().compile(graph_with_width()).run(data, params)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], equal_nan=True,
                                   **tolerance(generating_backend))


# ---- the oracle against hand-worked values -------------------------------------

def oracle(build, series, n):
    """Run one feature on the numpy backend over ``series`` for high, low and close."""
    series = np.asarray(series, dtype=np.float32)
    data = {"high": series, "low": series, "close": series}
    result = CPUBackend().compile(graph(build)).run(data, {"n": [n]})
    return [result[name][0] for name in sorted(result)]


NAN = np.nan


def test_moving_averages_by_hand():
    x = [1, 2, 3, 4, 5]
    np.testing.assert_allclose(oracle(FEATURES["sma"], x, 3)[0], [NAN, NAN, 2, 3, 4])
    np.testing.assert_allclose(oracle(FEATURES["rolling_sum"], x, 3)[0], [NAN, NAN, 6, 9, 12])
    # weights 1, 2, 3 over the window: (1 + 4 + 9) / 6, (2 + 6 + 12) / 6, ...
    np.testing.assert_allclose(oracle(FEATURES["wma"], x, 3)[0], [NAN, NAN, 14 / 6, 20 / 6, 26 / 6])
    # seeded with mean(1, 2) = 1.5, then alpha = 2 / 3: on a ramp the lag settles at 0.5
    np.testing.assert_allclose(oracle(FEATURES["ema"], x, 2)[0],
                               [NAN, 1.5, 2.5, 3.5, 4.5], rtol=1e-6)


def test_std_and_mean_deviation_by_hand():
    x = [2, 4, 4, 4, 5, 5, 7, 9]                       # population std of all 8 is 2
    assert oracle(FEATURES["rolling_std"], x, 8)[0][-1] == pytest.approx(2.0)
    # mean 5, absolute deviations 3 1 1 1 0 0 2 4 -> 12 / 8
    assert oracle(FEATURES["mean_deviation"], x, 8)[0][-1] == pytest.approx(1.5)


def test_rsi_by_hand():
    rising = oracle(FEATURES["rsi"], np.arange(10.0), 3)[0]
    assert np.isnan(rising[:3]).all() and (rising[3:] == 100).all()
    flat = oracle(FEATURES["rsi"], np.full(10, 5.0), 3)[0]
    assert (flat[3:] == 50).all()
    # changes +2, -1, +1: gain 3/3 = 1, loss 1/3 -> 100 - 100 / (1 + 3) = 75
    assert oracle(FEATURES["rsi"], [10, 12, 11, 12], 3)[0][3] == pytest.approx(75.0)


def test_true_range_covers_the_gap():
    g = Graph()
    high, low, close = (g.register_input(name) for name in ("high", "low", "close"))
    g.register_output("tr", F.true_range(high, low, close))
    data = {"high": np.float32([10, 15, 11]), "low": np.float32([9, 14, 8]),
            "close": np.float32([9.5, 14.5, 10])}
    # bar 1 gaps up: 15 - 9.5; bar 2 gaps down past the close: 14.5 - 8
    np.testing.assert_allclose(CPUBackend().compile(g).run(data, {})["tr"][0], [1, 5.5, 6.5])


def test_adx_in_a_steady_uptrend():
    up = np.arange(30.0)
    plus, minus, adx = (series[-1] for series in oracle(FEATURES["adx"], up, 5))
    assert (plus, minus, adx) == (pytest.approx(100.0), 0.0, pytest.approx(100.0))


def test_stochastic_and_williams_by_hand():
    g = Graph()
    high, low, close = (g.register_input(name) for name in ("high", "low", "close"))
    g.register_output("k", F.stochastic_k(high, low, close, 2))
    g.register_output("r", F.williams_r(high, low, close, 2))
    data = {"high": np.float32([10, 12, 11]), "low": np.float32([8, 9, 9]),
            "close": np.float32([9, 11, 9])}
    result = CPUBackend().compile(g).run(data, {})
    # bar 1: range 8..12, close 11 -> 75; bar 2: range 9..12, close 9 -> 0
    np.testing.assert_allclose(result["k"][0], [NAN, 75, 0])
    np.testing.assert_allclose(result["r"][0], [NAN, -25, -100])


def test_windows_and_shift_by_hand():
    x = [3, 1, 1, 4, 4]
    low_age, high_age = oracle(FEATURES["rolling_arg_min_max"], x, 3)
    np.testing.assert_allclose(low_age, [NAN, NAN, 0, 1, 2])     # ties: most recent 1
    np.testing.assert_allclose(high_age, [NAN, NAN, 2, 0, 0])
    np.testing.assert_allclose(oracle(FEATURES["shift"], x, 2)[0], [NAN, NAN, 3, 1, 1])
    assert np.isnan(oracle(FEATURES["shift"], x, -1)[0]).all()   # never the future


def test_crossings_by_hand():
    g = Graph()
    a = g.register_input("a")
    g.register_output("over", F.cross_over(a, 2.0))
    g.register_output("under", F.cross_under(a, 2.0))
    result = CPUBackend().compile(g).run({"a": np.float32([1, 3, 3, 1, 2, 3])}, {})
    np.testing.assert_array_equal(result["over"][0], [False, True, False, False, False, True])
    np.testing.assert_array_equal(result["under"][0], [False, False, False, True, False, False])


def test_macd_signal_starts_after_the_line():
    g = Graph()
    close = g.register_input("close")
    line, trigger, histogram = F.macd(close, 3, 6, 4)
    for name, value in (("line", line), ("signal", trigger), ("hist", histogram)):
        g.register_output(name, value)
    result = CPUBackend().compile(g).run({"close": ohlc()["close"]}, {})
    first = {name: int(np.flatnonzero(~np.isnan(series[0]))[0]) for name, series in result.items()}
    assert first == {"line": 5, "signal": 8, "hist": 8}          # slow - 1, then + signal - 1


def test_shared_windows_are_computed_once():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    lower, middle, upper = F.bollinger(close, n)
    g.register_output("z", F.zscore(close, n))
    g.register_output("upper", upper)
    source = NumbaCPU_Backend().compile(g).source
    assert source.count("sma_numba_cpu(") == 1
    assert source.count("rolling_std_numba_cpu(") == 1
