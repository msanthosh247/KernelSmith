import numpy as np
import pytest

import kernelsmith.backends.cpu as cpu_backend
from kernelsmith import CallFactory, F4, Graph, I4, SCAL, VEC
from kernelsmith.backends.cpu import CpuBackend, cpu_impl
from kernelsmith.features import rolling_min_max, sma
from kernelsmith.ir import cse


def op_names(ops):
    return sorted(op.name for op in ops)


# ---- structural ------------------------------------------------------------

def test_duplicate_feature_calls_collapse():
    g = Graph()
    close, opn = g.input("close"), g.input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    # sma(med, slow) written twice on purpose
    g.output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))

    ops = g.build()
    kept, replace = cse(ops)

    assert op_names(ops).count("sma") == 3
    assert op_names(kept).count("sma") == 2
    assert len(replace) == 1


def test_shared_subexpression_collapses_once():
    """The two chains dedupe bottom-up in a single pass: '+' first, then '/'."""
    g = Graph()
    a, b = g.input("a"), g.input("b")
    g.output("x", (a + b) / 2)
    g.output("y", (a + b) / 2)

    kept, _ = cse(g.build())
    assert op_names(kept) == ["+", "/"]


def test_commutative_operands_are_normalized():
    g = Graph()
    a, b = g.input("a"), g.input("b")
    g.output("x", a + b)
    g.output("y", b + a)

    assert len(cse(g.build())[0]) == 1


def test_non_commutative_operands_are_not_normalized():
    g = Graph()
    a, b = g.input("a"), g.input("b")
    g.output("x", a - b)
    g.output("y", b - a)

    assert len(cse(g.build())[0]) == 2


def test_nested_commutative_collapses():
    """(a+b)+c and c+(a+b): same tree shape, so the shared '+' dedupes first
    and the outer ops then match after sorting."""
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    g.output("x", (a + b) + c)
    g.output("y", c + (a + b))

    assert len(cse(g.build())[0]) == 2


def test_associativity_is_not_merged():
    """(a+b)+c and a+(b+c) build different values - deliberately left alone,
    since reassociating float arithmetic changes results."""
    g = Graph()
    a, b, c = g.input("a"), g.input("b"), g.input("c")
    g.output("x", (a + b) + c)
    g.output("y", a + (b + c))

    assert len(cse(g.build())[0]) == 4


def test_constants_dedupe_by_value_and_dtype():
    g = Graph()
    a = g.input("a")
    g.output("x", a * 2)
    g.output("y", a * 2.0)   # different dtype - must stay separate
    g.output("z", a * 2)

    kept, replace = cse(g.build())
    assert len(kept) == 2
    assert len(replace) == 1


def test_graph_without_duplicates_is_untouched():
    g = Graph()
    a = g.input("a")
    g.output("x", a * 3)

    ops = g.build()
    kept, replace = cse(ops)
    assert kept == ops
    assert replace == {}


def test_multi_output_call_maps_every_value():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    low_a, high_a = rolling_min_max(close, n)
    low_b, high_b = rolling_min_max(close, n)
    g.output("high", high_a)
    g.output("low", low_b)      # taken from the duplicate call

    kept, replace = cse(g.build())
    assert len(kept) == 1

    # whichever call survives (topological order is not source order), every
    # value of the dropped one maps to the survivor's value at the same index
    survivor = kept[0]
    dropped = [v for v in (low_a, high_a, low_b, high_b) if v in replace]
    assert len(dropped) == 2
    for value in dropped:
        assert replace[value] is survivor.outs[value.out_index]


def test_commutative_names_do_not_leak_into_features():
    """A feature named '+' must not have its arguments sorted - only Exprs are
    known to be commutative."""
    plus = CallFactory("+", [VEC(F4), VEC(F4)], [], [VEC(F4)])
    g = Graph()
    a, b = g.input("a"), g.input("b")
    g.output("x", plus(a, b))
    g.output("y", plus(b, a))

    assert len(cse(g.build())[0]) == 2


# ---- equivalence: the property that matters --------------------------------

def test_cse_does_not_change_results(monkeypatch):
    def build_graph():
        g = Graph()
        close, opn = g.input("close"), g.input("open")
        fast, slow = g.int_param("fast"), g.int_param("slow")
        med = (close + opn) / 2
        g.output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
        g.output("med_again", (close + opn) / 2)   # a duplicate that is itself an output
        low, high = rolling_min_max(close, slow)
        g.output("width", high - low)
        return g

    rng = np.random.default_rng(11)
    close = (np.cumsum(rng.normal(0, 1, 200)) + 100).astype(np.float32)
    opn = (close + rng.normal(0, 0.2, 200)).astype(np.float32)
    inputs = {"close": close, "open": opn}
    params = {"fast": [5, 10], "slow": [20, 30]}

    with_cse = CpuBackend().compile(build_graph()).run(inputs, params)

    monkeypatch.setattr(cpu_backend, "cse", lambda ops: (ops, {}))
    without_cse = CpuBackend().compile(build_graph()).run(inputs, params)

    assert set(with_cse) == set(without_cse)
    for name in with_cse:
        # assert_array_equal treats NaNs in matching positions as equal
        np.testing.assert_array_equal(with_cse[name], without_cse[name])


def test_compiled_program_uses_fewer_ops(monkeypatch):
    g = Graph()
    close = g.input("close")
    n = g.int_param("n")
    g.output("x", sma(close, n) > sma(close, n))

    assert len(CpuBackend().compile(g).ops) == 2      # one sma, one '>'
    monkeypatch.setattr(cpu_backend, "cse", lambda ops: (ops, {}))
    assert len(CpuBackend().compile(g).ops) == 3
