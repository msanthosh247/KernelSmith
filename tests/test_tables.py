"""Tables (reference data off the time axis) and series indexing (series[i]).

A table is any-length data shared by every parameter set and read only by
features; the graph's time axis stays the one its series live on. These tests
hold the three things that make that usable: the typing rules, the error
messages that teach the difference from an input, and every backend agreeing.
"""
import numpy as np
import pytest

from kernelsmith import CallFactory, F4, Graph, GraphError, I4, Shape
from kernelsmith.backends.cpu import CPUBackend, cpu_impl
from kernelsmith.errors import DslTypeError
from kernelsmith.features import sma
from conftest import tolerance

pytestmark = pytest.mark.filterwarnings("ignore:.*Grid size")


# a small feature reading a table: out[t] = table[t % len(table)] * scale
cycle = CallFactory("cycle_table", [F4.table, F4], [], [F4[:]])


@cpu_impl(cycle)
def _cycle(table, scale, *, n_bars):
    table = np.asarray(table, dtype=np.float64)
    return ((table[np.arange(n_bars) % table.size] * scale).astype(np.float32),)


def _cycle_kernel(table, scale, out):
    for t in range(out.shape[0]):
        out[t] = table[t % table.shape[0]] * scale


def _register_cycle_kernels():
    from numba import njit
    from kernelsmith.availability import cuda_importable
    from kernelsmith.backends.numba_cpu import NUMBA_CPU, register_numba_cpu
    if cycle not in NUMBA_CPU.entries:
        register_numba_cpu(cycle)(njit(_cycle_kernel))
    if cuda_importable():
        from numba import cuda
        from kernelsmith.backends.cuda import CUDA, register_cuda
        if cycle not in CUDA.entries:
            register_cuda(cycle)(cuda.jit(device=True)(_cycle_kernel))


# ---- typing -------------------------------------------------------------------

def test_table_signature():
    assert F4.table.shape is Shape.TABLE
    assert repr(F4.table) == "float32.table"
    assert F4.table != F4[:]


def test_register_table_returns_the_same_node():
    g = Graph()
    assert g.register_table("close") is g.register_table("close")
    assert g.tables["close"].shape is Shape.TABLE
    assert "close" not in g.inputs


def test_names_are_unique_across_inputs_and_tables():
    g = Graph()
    g.register_input("close")
    with pytest.raises(GraphError, match="already registered as an input"):
        g.register_table("close")
    g.register_table("history")
    with pytest.raises(GraphError, match="already registered as a table"):
        g.register_input("history")


def test_table_names_must_be_identifiers():
    with pytest.raises(GraphError, match="valid identifier"):
        Graph().register_table("close price")


# ---- the errors that teach the difference ---------------------------------------

def test_table_in_arithmetic_says_what_a_table_is():
    g = Graph()
    close = g.register_table("close")
    with pytest.raises(DslTypeError, match="'close' is a table.*register_input"):
        close * 2
    with pytest.raises(DslTypeError, match="'close' is a table"):
        -close


def test_series_where_a_table_is_expected_says_which_to_use():
    g = Graph()
    close = g.register_input("close")
    with pytest.raises(DslTypeError, match="expects a table.*'close'.*register_table"):
        cycle(close, 1.0)


def test_a_table_is_not_an_output():
    g = Graph()
    with pytest.raises(GraphError, match="a table is reference data"):
        g.register_output("close", g.register_table("close"))


def test_indexing_needs_a_series():
    g = Graph()
    with pytest.raises(DslTypeError, match="only a series can be indexed"):
        g.register_table("close")[0]
    with pytest.raises(DslTypeError, match="only a series can be indexed"):
        g.int_param("n")[0]


# ---- running: the time axis and the tables ----------------------------------------

def table_graph():
    """Only a table and a param: the time axis must come from n_bars."""
    _register_cycle_kernels()
    g = Graph()
    history = g.register_table("history")
    series = cycle(history, g.float_param("scale"))
    g.register_output("series", series)
    g.register_output("last", series[-1])
    g.register_output("mean3", sma(series, 3)[4])
    return g


HISTORY = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)


def test_table_graph_needs_n_bars():
    program = CPUBackend().compile(table_graph())
    with pytest.raises(GraphError, match="pass n_bars"):
        program.run({"history": HISTORY}, {"scale": [1.0]})


def test_missing_table_is_named():
    with pytest.raises(GraphError, match="missing table 'history'"):
        CPUBackend().compile(table_graph()).run({}, {"scale": [1.0]}, n_bars=5)


def test_n_bars_must_agree_with_series_inputs():
    _register_cycle_kernels()
    g = Graph()
    close = g.register_input("close")
    g.register_output("sum", close + cycle(g.register_table("history"), 1.0))
    program = CPUBackend().compile(g)
    data = {"close": np.ones(5, np.float32), "history": HISTORY}
    assert program.run(data, {}, n_bars=5)["sum"].shape == (1, 5)
    with pytest.raises(GraphError, match="contradicts the series inputs"):
        program.run(data, {}, n_bars=9)


def test_table_length_is_independent_of_the_time_axis():
    """A 7-long table, a 12-step axis: the table is never checked against it."""
    out = CPUBackend().compile(table_graph()).run({"history": HISTORY}, {"scale": [2.0]}, n_bars=12)
    np.testing.assert_array_equal(out["series"][0], 2 * HISTORY[np.arange(12) % 7])
    assert out["last"][0] == 2 * HISTORY[11 % 7]


def test_tables_on_every_backend(generating_backend):
    params = {"scale": [1.0, 0.5, -2.0]}
    expected = CPUBackend().compile(table_graph()).run({"history": HISTORY}, params, n_bars=10)
    actual = generating_backend().compile(table_graph()).run({"history": HISTORY}, params, n_bars=10)
    for name in expected:
        assert actual[name].shape == expected[name].shape, name
        np.testing.assert_allclose(actual[name], expected[name], err_msg=name,
                                   **tolerance(generating_backend))


def test_table_is_one_kernel_argument(generating_backend):
    source = generating_backend().compile(table_graph()).source
    assert "tab_history" in source.splitlines()[1]          # in the signature
    assert "cycle_table" in source and "(tab_history," in source


# ---- indexing -------------------------------------------------------------------

def indexed_graph():
    g = Graph()
    close = g.register_input("close")
    at = g.int_param("at")
    g.register_output("first", close[0])
    g.register_output("last", close[-1])
    g.register_output("chosen", close[at])
    g.register_output("past_end", close[99])
    g.register_output("before_start", close[-99])
    g.register_output("spread", close[-1] - close[0])        # a scalar feeding an expression
    g.register_output("mean", sma(close, 3)[-1])
    return g


def test_indexing_by_hand():
    close = np.arange(10, dtype=np.float32) + 100
    out = CPUBackend().compile(indexed_graph()).run({"close": close}, {"at": [2, -3]})
    assert out["first"].tolist() == [100, 100]
    assert out["last"].tolist() == [109, 109]
    assert out["chosen"].tolist() == [102, 107]
    assert np.isnan(out["past_end"]).all() and np.isnan(out["before_start"]).all()
    assert out["spread"].tolist() == [9, 9]
    assert out["mean"].tolist() == [108, 108]


def test_indexing_on_every_backend(generating_backend):
    close = (np.arange(10, dtype=np.float32) * 1.5) + 100
    params = {"at": [2, -3, 50]}
    expected = CPUBackend().compile(indexed_graph()).run({"close": close}, params)
    actual = generating_backend().compile(indexed_graph()).run({"close": close}, params)
    for name in expected:
        assert actual[name].shape == (3,), name
        np.testing.assert_allclose(actual[name], expected[name], equal_nan=True, err_msg=name,
                                   **tolerance(generating_backend))


def test_scalar_feature_output_is_passed_as_a_view(generating_backend):
    """The kernel writes out[0] of a one-element slice - an element would be a
    copy, and the write would be lost."""
    import re
    source = generating_backend().compile(indexed_graph()).source
    calls = [line.strip() for line in source.splitlines() if "element_" in line]
    # the last argument is a k:k+1 slice - [p, 3:4] on the CPU, [3:4, p] on the GPU
    view = re.compile(r"\w+\[(p, )?(\d+):(\d+)(, p)?\]\)$")
    assert calls
    for line in calls:
        match = view.search(line)
        assert match and int(match.group(3)) == int(match.group(2)) + 1, line


# ---- visualize and explain -------------------------------------------------------

def test_table_graph_plots_and_explains(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pytest.importorskip("networkx")
    from kernelsmith.ir.explain import explain_graph

    g = table_graph()
    assert g.visualize(str(tmp_path / "tables.png")) == str(tmp_path / "tables.png")
    assert (tmp_path / "tables.png").stat().st_size > 0
    assert "cycle_table" in explain_graph(g)
