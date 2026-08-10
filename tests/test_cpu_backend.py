import numpy as np
import pytest

from kernelsmith import CallFactory, F4, Graph, GraphError, I4, SCAL, VEC
from kernelsmith.backends.cpu import CpuBackend, cpu_impl
from kernelsmith.features import rolling_min_max, sma


def prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n)).astype(np.float32) + 100.0


def rolling_mean_reference(values, period):
    """Deliberately naive oracle: a different algorithm from the implementation."""
    out = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1: i + 1]) / period
    return out


def test_sma_matches_independent_reference():
    close = prices()
    g = Graph()
    g.output("fast", sma(g.input("close"), g.int_param("n")))

    result = CpuBackend().compile(g).run({"close": close}, {"n": [10]})["fast"][0]
    expected = rolling_mean_reference(close, 10)

    valid = ~np.isnan(expected)
    np.testing.assert_allclose(result[valid], expected[valid], rtol=1e-5)
    assert np.isnan(result[~valid]).all()


def test_crossover_strategy_end_to_end():
    close = prices()
    g = Graph()
    c = g.input("close")
    fast = sma(c, g.int_param("fast"))
    slow = sma(c, g.int_param("slow"))
    g.output("signal", (fast > slow) & (c > slow))

    signal = CpuBackend().compile(g).run({"close": close}, {"fast": [5], "slow": [20]})["signal"][0]

    f = rolling_mean_reference(close, 5)
    s = rolling_mean_reference(close, 20)
    with np.errstate(invalid="ignore"):
        expected = (f > s) & (close > s)
    np.testing.assert_array_equal(signal, expected)


def test_parameter_sweep_stacks_along_first_axis():
    close = prices()
    g = Graph()
    g.output("avg", sma(g.input("close"), g.int_param("n")))

    out = CpuBackend().compile(g).run({"close": close}, {"n": [5, 10, 20]})["avg"]
    assert out.shape == (3, len(close))
    # longer windows warm up later
    assert np.isnan(out[0]).sum() < np.isnan(out[2]).sum()


def test_elementwise_with_constant_and_broadcast():
    close, opn = prices(60, 1), prices(60, 2)
    g = Graph()
    g.output("med", (g.input("close") + g.input("open")) / 2)

    out = CpuBackend().compile(g).run({"close": close, "open": opn}, {})["med"]
    assert out.shape == (1, 60)
    np.testing.assert_allclose(out[0], (close + opn) / 2, rtol=1e-6)


def test_comparison_operators_execute():
    a, b = prices(40, 4), prices(40, 5)
    g = Graph()
    x, y = g.input("a"), g.input("b")
    g.output("ne", x != y)
    g.output("eq", x == y)
    g.output("ge", x >= y)

    out = CpuBackend().compile(g).run({"a": a, "b": b}, {})
    np.testing.assert_array_equal(out["ne"][0], a != b)
    np.testing.assert_array_equal(out["eq"][0], a == b)
    np.testing.assert_array_equal(out["ge"][0], a >= b)


def test_multi_output_feature():
    close = prices(50)
    g = Graph()
    lo, hi = rolling_min_max(g.input("close"), g.int_param("n"))
    g.output("low", lo)
    g.output("high", hi)

    out = CpuBackend().compile(g).run({"close": close}, {"n": [5]})
    assert (out["high"][0][4:] >= out["low"][0][4:]).all()
    np.testing.assert_allclose(out["high"][0][4], close[:5].max(), rtol=1e-6)


def test_output_can_be_a_bare_input():
    close = prices(20)
    g = Graph()
    c = g.input("close")
    g.output("passthrough", c)
    g.output("doubled", c * 2)

    out = CpuBackend().compile(g).run({"close": close}, {})
    np.testing.assert_allclose(out["passthrough"][0], close)


# ---- error paths -----------------------------------------------------------

def test_missing_implementation_fails_at_compile():
    mystery = CallFactory("mystery", [VEC(F4), SCAL(I4)], [], [VEC(F4)])
    g = Graph()
    g.output("x", mystery(g.input("close"), g.int_param("n")))

    with pytest.raises(GraphError, match="mystery"):
        CpuBackend().compile(g)


def test_implementation_must_return_tuple():
    bad = CallFactory("bad", [VEC(F4)], [], [VEC(F4)])

    @cpu_impl(bad)
    def _bad(x):
        return x  # not a tuple

    g = Graph()
    g.output("x", bad(g.input("close")))
    program = CpuBackend().compile(g)

    with pytest.raises(GraphError, match="must return a tuple"):
        program.run({"close": prices(10)}, {})


def test_wrong_output_count_is_caught():
    stingy = CallFactory("stingy", [VEC(F4)], [], [VEC(F4), VEC(F4)])

    @cpu_impl(stingy)
    def _stingy(x):
        return (x,)  # signature declares two

    g = Graph()
    a, _ = stingy(g.input("close"))
    g.output("a", a)

    with pytest.raises(GraphError, match="returned 1 value"):
        CpuBackend().compile(g).run({"close": prices(10)}, {})


def test_missing_input_and_param_are_named():
    g = Graph()
    g.output("avg", sma(g.input("close"), g.int_param("n")))
    program = CpuBackend().compile(g)

    with pytest.raises(GraphError, match="missing input 'close'"):
        program.run({}, {"n": [5]})
    with pytest.raises(GraphError, match="missing param 'n'"):
        program.run({"close": prices(10)}, {})


def test_params_must_agree_in_length():
    g = Graph()
    c = g.input("close")
    g.output("x", sma(c, g.int_param("a")) > sma(c, g.int_param("b")))
    program = CpuBackend().compile(g)

    with pytest.raises(GraphError, match="same length"):
        program.run({"close": prices(30)}, {"a": [5, 10], "b": [20]})
