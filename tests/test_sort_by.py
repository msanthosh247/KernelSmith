"""run(..., sort_by=...) is a scheduling hint: it may change speed, never results.

Parameter sets are shuffled on purpose - a permutation that is not undone, or
undone along the wrong axis, cannot pass by accident.
"""
import numpy as np
import pytest

from kernelsmith import Graph, GraphError
from kernelsmith.backends.cpu import CPUBackend
from kernelsmith.features import rolling_min_max, sma
from conftest import tolerance

pytestmark = pytest.mark.filterwarnings("ignore:.*Grid size")


def strategy():
    """Every kind of output: vector temp, scalar temp, bare param, bare input."""
    g = Graph()
    close = g.register_input("close")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    low, high = rolling_min_max(close, slow)
    g.register_output("signal", (sma(close, fast) > sma(close, slow)) & (close > low))
    g.register_output("width", high - low)
    g.register_output("span", slow - fast)
    g.register_output("slow", slow)
    g.register_output("close", close)
    return g


def sweep(n_params=45, n_bars=80, seed=1):
    rng = np.random.default_rng(seed)
    close = (np.cumsum(rng.normal(0, 1, n_bars)) + 100).astype(np.float32)
    params = {"fast": rng.integers(2, 10, n_params), "slow": rng.integers(10, 40, n_params)}
    return {"close": close}, params


@pytest.mark.parametrize("sort_by", ["slow", ["slow", "fast"], ("fast",), True])
def test_sorting_does_not_change_results(generating_backend, sort_by):
    inputs, params = sweep()
    program = generating_backend().compile(strategy())
    unsorted = program.run(inputs, params)
    ordered = program.run(inputs, params, sort_by=sort_by)

    expected = CPUBackend().compile(strategy()).run(inputs, params)
    for name in expected:
        assert np.asarray(ordered[name]).shape == expected[name].shape, name
        np.testing.assert_array_equal(np.asarray(ordered[name]), np.asarray(unsorted[name]), err_msg=name)
        np.testing.assert_allclose(np.asarray(ordered[name]), expected[name], equal_nan=True,
                                   err_msg=name, **tolerance(generating_backend))


def test_no_sort_keeps_the_given_order():
    graph = strategy()
    _, params = sweep()
    checked = {name: np.asarray(values) for name, values in params.items()}
    assert CPUBackend().compile(graph).parameter_order(graph, checked, None) is None
    assert CPUBackend().compile(graph).parameter_order(graph, checked, False) is None


def test_first_name_is_the_primary_key():
    graph = strategy()
    params = {"fast": np.array([3, 1, 2, 1]), "slow": np.array([10, 20, 10, 10])}
    order = CPUBackend().compile(graph).parameter_order(graph, params, ["slow", "fast"])
    assert order.tolist() == [3, 2, 0, 1]          # slow 10,10,10,20 - ties by fast


@pytest.mark.parametrize("backend", ["oracle", "generating"])
def test_unknown_sort_key_is_rejected_everywhere(backend, generating_backend):
    inputs, params = sweep()
    cls = CPUBackend if backend == "oracle" else generating_backend
    with pytest.raises(GraphError, match="no such param"):
        cls().compile(strategy()).run(inputs, params, sort_by="period")
