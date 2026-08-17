import numpy as np
import pytest

pytest.importorskip("numba")

import kernelsmith.backends.numba_cpu as numba_backend  # noqa: E402
from kernelsmith import CallFactory, F4, Graph, GraphError, I4, KernelsmithError  # noqa: E402
from kernelsmith.backends.cpu import CpuBackend  # noqa: E402
from kernelsmith.backends.numba_cpu import (  # noqa: E402
    NUMBA_CPU_FN_REGISTER,
    NumbaCPU_Backend,
    register_numba_cpu,
)
from kernelsmith.features import rolling_min_max, sma  # noqa: E402


def prices(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(0, 1, n)) + 100).astype(np.float32)


def assert_parity(build, inputs, params):
    """The acceptance test: numba must agree with the numpy oracle exactly."""
    expected = CpuBackend().compile(build()).run(inputs, params)
    actual = NumbaCPU_Backend().compile(build()).run(inputs, params)

    assert set(expected) == set(actual)
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(actual[name]), expected[name], rtol=1e-5, equal_nan=True,
            err_msg=f"output '{name}' differs",
        )
    return actual


# ---- parity ----------------------------------------------------------------

def test_crossover_strategy_matches_the_oracle():
    def build():
        g = Graph()
        close, opn = g.register_input("close"), g.register_input("open")
        fast, slow = g.int_param("fast"), g.int_param("slow")
        med = (close + opn) / 2
        g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
        return g

    close = prices()
    assert_parity(
        build,
        {"close": close, "open": (close + 0.2).astype(np.float32)},
        {"fast": [5, 10, 20], "slow": [20, 30, 50]},
    )


def test_multi_output_and_dead_value():
    """Only the high is consumed; the low is dead but the call still runs."""
    def build():
        g = Graph()
        close, n = g.register_input("close"), g.int_param("n")
        _, high = rolling_min_max(close, n)
        g.register_output("h", high * 2)
        return g

    assert_parity(build, {"close": prices()}, {"n": [5, 10]})


def test_constants_and_unary_operators():
    def build():
        g = Graph()
        close = g.register_input("close")
        g.register_output("negated", -close)
        g.register_output("flag", ~(close > 100.0))
        g.register_output("scaled", (close * 2) - 1)
        return g

    assert_parity(build, {"close": prices()}, {})


def test_scalar_expression():
    def build():
        g = Graph()
        g.register_input("close")
        g.register_output("total", g.int_param("a") + g.int_param("b"))
        return g

    result = assert_parity(build, {"close": prices(20)}, {"a": [1, 2], "b": [10, 20]})
    assert result["total"].shape == (2,)


def test_output_that_is_a_bare_input():
    def build():
        g = Graph()
        close = g.register_input("close")
        g.register_output("passthrough", close)
        g.register_output("doubled", close * 2)
        return g

    assert_parity(build, {"close": prices(40)}, {})


def test_duplicate_removed_by_cse_still_matches():
    def build():
        g = Graph()
        close, n = g.register_input("close"), g.int_param("n")
        g.register_output("a", sma(close, n))
        g.register_output("b", sma(close, n))       # duplicate, also an output
        return g

    assert_parity(build, {"close": prices()}, {"n": [7]})


def test_single_parameter_set():
    def build():
        g = Graph()
        g.register_output("avg", sma(g.register_input("close"), g.int_param("n")))
        return g

    assert_parity(build, {"close": prices()}, {"n": 10})


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _crossover_graph(), id="crossover"),
        pytest.param(lambda: _two_consumer_graph(), id="two-consumers"),
        pytest.param(lambda: _chain_graph(), id="long-chain"),
        pytest.param(lambda: _output_feeding_expr_graph(), id="output-feeds-expr"),
    ],
)
def test_fusion_does_not_change_results(build, monkeypatch):
    """The acceptance test: fusing must be invisible in the results."""
    close = prices()
    inputs = {"close": close, "open": (close + 0.2).astype(np.float32)}
    params = {"fast": [5, 10], "slow": [20, 30]}

    fused = NumbaCPU_Backend().compile(build()).run(inputs, params)

    monkeypatch.setattr(numba_backend, "fuse", lambda ops, outputs=(), replace=None: (ops, {}))
    plain = NumbaCPU_Backend().compile(build()).run(inputs, params)

    assert set(fused) == set(plain)
    for name in fused:
        np.testing.assert_array_equal(
            np.asarray(fused[name]), np.asarray(plain[name]),
            err_msg=f"output '{name}' changed under fusion",
        )


def _crossover_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
    return g


def _two_consumer_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    g.int_param("fast"), g.int_param("slow")
    shared = close + opn
    g.register_output("doubled", shared * 2)
    g.register_output("tripled", shared * 3)
    return g


def _chain_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    g.int_param("fast"), g.int_param("slow")
    g.register_output("chain", ((((close + opn) / 2) * 3) - 1) > close)
    return g


def _output_feeding_expr_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("med", med)          # registered AND consumed below
    g.register_output("flag", med > close)
    return g


def test_repeated_runs_reuse_scratch_without_corrupting_results():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("avg", sma(close, n))
    program = NumbaCPU_Backend().compile(g)

    first = program.run({"close": prices(80)}, {"n": [5]})["avg"].copy()
    for _ in range(3):
        again = program.run({"close": prices(80)}, {"n": [5]})["avg"]
        np.testing.assert_array_equal(again, first)


def test_earlier_outputs_survive_a_later_run():
    """Scratch is recycled between runs; the arrays handed back must not be."""
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("avg", sma(close, n))
    program = NumbaCPU_Backend().compile(g)

    first = program.run({"close": prices(80, seed=1)}, {"n": [5]})["avg"]
    snapshot = first.copy()
    program.run({"close": prices(80, seed=2)}, {"n": [30]})
    np.testing.assert_array_equal(first, snapshot)


def test_changing_sweep_size_reallocates():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("avg", sma(close, n))
    program = NumbaCPU_Backend().compile(g)

    assert program.run({"close": prices(60)}, {"n": [5, 10]})["avg"].shape == (2, 60)
    assert program.run({"close": prices(90)}, {"n": [5]})["avg"].shape == (1, 90)


# ---- generated source ------------------------------------------------------

def compile_source(graph):
    return NumbaCPU_Backend().compile(graph).source


def test_source_shape():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    n = g.int_param("n")
    med = (close + opn) / 2
    g.register_output("signal", sma(med, n) > med)

    source = compile_source(g)

    assert "@njit(parallel=True, cache=False)" in source
    assert "for p in prange(n_params):" in source
    assert "sma_numba_cpu(" in source              # kernels called by derived name
    # two loops, not three: '+' and '/' fuse, and the sma between them and '>'
    # forces a second group
    assert source.count("for t in range(n_bars):") == 2
    assert "n_params, n_bars):" in source          # counts passed, never inferred


def test_only_used_pools_appear_in_the_signature():
    g = Graph()
    close = g.register_input("close")
    g.register_output("flag", close > 1.0)

    signature = compile_source(g).splitlines()[1]
    assert "out_b1_v" in signature
    assert "par_" not in signature                 # the graph declares no params
    assert "tmp_f32_v" not in signature            # nothing needs a float temp


def test_scratch_is_passed_positionally():
    scratchy = CallFactory("scratchy", [F4[:], I4], [F4[:]], [F4[:]])

    @register_numba_cpu(scratchy)
    def _scratchy(values, period, work, out):      # noqa: ARG001
        pass

    try:
        g = Graph()
        g.register_output("y", scratchy(g.register_input("close"), g.int_param("n")))
        call_line = [l for l in compile_source(g).splitlines() if "scratchy_numba_cpu" in l][0]
        # inputs..., scratch..., outputs...
        assert call_line.count("tmp_f32_v[p,") == 1
        assert call_line.count("out_f32_v[p,") == 1
    finally:
        NUMBA_CPU_FN_REGISTER.pop(scratchy, None)


def test_emission_is_deterministic():
    def build():
        g = Graph()
        close, opn = g.register_input("close"), g.register_input("open")
        fast, slow = g.int_param("fast"), g.int_param("slow")
        med = (close + opn) / 2
        g.register_output("signal", sma(med, fast) > sma(med, slow))
        return g

    assert compile_source(build()) == compile_source(build())


# ---- errors ----------------------------------------------------------------

def test_missing_kernel_is_named_at_compile():
    mystery = CallFactory("mystery", [F4[:], I4], [], [F4[:]])
    g = Graph()
    g.register_output("x", mystery(g.register_input("close"), g.int_param("n")))

    with pytest.raises(GraphError, match="mystery"):
        NumbaCPU_Backend().compile(g)


def test_eager_signature_must_match_the_feature():
    from numba import njit

    wrong = CallFactory("wrong", [F4[:], I4], [], [F4[:]])

    with pytest.raises(KernelsmithError, match="declares"):
        @register_numba_cpu(wrong)
        @njit("void(float32[:], float32, float32[:])")     # param typed float, not int
        def _wrong(values, period, out):
            pass


def test_missing_input_reuses_the_shared_message():
    g = Graph()
    g.register_output("avg", sma(g.register_input("close"), g.int_param("n")))
    program = NumbaCPU_Backend().compile(g)

    with pytest.raises(GraphError, match="missing input 'close'"):
        program.run({}, {"n": [5]})
    with pytest.raises(GraphError, match="missing param 'n'"):
        program.run({"close": prices(10)}, {})
