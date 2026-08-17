import pytest

from kernelsmith import Graph, Shape
from kernelsmith.features import rolling_min_max, sma
from kernelsmith.ir.cse import cse
from kernelsmith.ir.fuse import FusedExpr, consumers, fuse
from kernelsmith.ir.liveness import Liveness
from kernelsmith.ir.allocate import allocate


def plan(g):
    ops, replace = cse(g.build())
    fused, levels = fuse(ops, g.outputs.values(), replace)
    return ops, fused, levels, replace


def groups(ops):
    return [op for op in ops if isinstance(op, FusedExpr)]


# ---- grouping --------------------------------------------------------------

def test_a_chain_of_expressions_becomes_one_group():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    g.register_output("out", ((a + b) / 2) * 3)

    _, fused, _, _ = plan(g)
    assert len(fused) == 1
    assert [m.name for m in fused[0].members] == ["+", "/", "*"]


def test_group_spans_an_intervening_call():
    """The scheduler puts an sma between two comparisons; neither depends on
    it, so all three elementwise ops still belong in one loop."""
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))

    _, fused, _, _ = plan(g)
    sizes = sorted(len(group.members) for group in groups(fused))
    assert sizes == [2, 3]                  # (+, /) and (>, >, &)


def test_a_value_with_two_consumers_is_not_absorbed():
    """t feeds two expressions, so it must stay in a buffer - inlining it into
    one group would leave the other reading memory nobody wrote."""
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    t = a + b
    g.register_output("o1", t * 2)
    g.register_output("o2", t * 3)

    _, fused, _, _ = plan(g)
    assert groups(fused) == []
    assert len(fused) == 3


def test_a_registered_output_is_never_absorbed():
    """Consumed only inside the group, but the caller still reads it out."""
    g = Graph()
    x, y = g.register_input("x"), g.register_input("y")
    a = x + y
    g.register_output("a", a)
    g.register_output("b", a * 2)

    _, fused, _, _ = plan(g)
    assert groups(fused) == []


def test_scalars_and_vectors_never_share_a_group():
    g = Graph()
    close = g.register_input("close")
    scale = g.int_param("x") + g.int_param("y")     # scalar
    g.register_output("out", close * scale)          # vector

    _, fused, _, _ = plan(g)
    for group in groups(fused):
        shapes = {m.outs[0].shape for m in group.members}
        assert len(shapes) == 1


def test_calls_are_never_absorbed():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("out", sma(close, n) * 2)

    _, fused, _, _ = plan(g)
    assert groups(fused) == []                       # nothing to fuse with


# ---- the Op contract -------------------------------------------------------

def test_group_exposes_only_external_args():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    g.register_output("out", ((a + b) / 2) * 3)

    _, fused, _, _ = plan(g)
    group = fused[0]

    internal = set(group.produced)
    assert not (set(group.args) & internal)          # locals are not arguments
    assert group.outs == group.root.outs
    assert group.buffer_signature == ()


def test_group_args_are_deduped_and_ordered():
    g = Graph()
    x, y = g.register_input("x"), g.register_input("y")
    g.register_output("out", (x > y) & (x > (y * 2)))

    _, fused, _, _ = plan(g)
    group = groups(fused)[0]
    assert len(group.args) == len(set(group.args))    # x is read twice, listed once


def test_members_are_in_topological_order():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    g.register_output("out", ((a + b) / 2) * 3)

    _, fused, _, _ = plan(g)
    group = fused[0]
    seen = set()
    for member in group.members:
        for operand in member.args:
            assert operand.parent not in group._member_set or operand.parent in seen
        seen.add(member)


def test_levels_are_recomputed():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    n = g.int_param("n")
    med = (close + opn) / 2
    g.register_output("out", sma(med, n) > med)

    _, fused, levels, _ = plan(g)
    assert set(levels) == set(fused)
    for op in fused:
        for operand in op.args:
            producer = operand.parent
            if producer in levels:
                assert levels[op] > levels[producer]


# ---- rendering -------------------------------------------------------------

def test_formula_letters_the_external_inputs():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    g.register_output("out", (a + b) / 2)

    _, fused, _, _ = plan(g)
    assert fused[0].formula() == "((a + b) / c)"
    assert "((a + b) / c)" in repr(fused[0])


def test_formula_uses_readable_symbols_for_unary_ops():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    g.register_output("out", -(a + b))

    _, fused, _, _ = plan(g)
    assert fused[0].formula() == "(-(a + b))"


def test_formula_gives_up_past_the_letters():
    g = Graph()
    values = [g.register_input(f"i{k}") for k in range(30)]
    total = values[0]
    for value in values[1:]:
        total = total + value
    g.register_output("wide", total)

    _, fused, _, _ = plan(g)
    group = fused[0]
    assert group.formula() is None
    assert repr(group) == f"<FusedExpr {len(group.members)} ops>"
    assert group.name == f"fused[{len(group.members)}]"      # name still works


# ---- the payoff ------------------------------------------------------------

def test_fusion_frees_buffers():
    """The whole point: values that became locals stop needing slots."""
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))

    ops, fused, _, replace = plan(g)

    unfused_pools = allocate(ops, Liveness(ops, g.outputs, replace)).pool_size
    fused_pools = allocate(fused, Liveness(fused, g.outputs, replace)).pool_size

    assert sum(fused_pools.values()) < sum(unfused_pools.values())


def test_consumer_map_resolves_through_cse():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("a", sma(close, n) * 2)
    g.register_output("b", sma(close, n) * 3)        # duplicate sma

    ops, replace = cse(g.build())
    reads = consumers(ops, replace)
    for value, readers in reads.items():
        assert value not in replace                   # every key is canonical
        assert len(readers) == len(set(id(r) for r in readers))


def test_fusion_is_deterministic():
    def build():
        g = Graph()
        close, opn = g.register_input("close"), g.register_input("open")
        fast, slow = g.int_param("fast"), g.int_param("slow")
        med = (close + opn) / 2
        g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
        return g

    def shape():
        _, fused, _, _ = plan(build())
        return [
            (op.name, tuple(m.name for m in op.members)) if isinstance(op, FusedExpr) else (op.name, ())
            for op in fused
        ]

    assert shape() == shape()


# ---- visualization ---------------------------------------------------------

def test_fused_schedule_can_be_plotted(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pytest.importorskip("networkx")

    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
    low, high = rolling_min_max(close, slow)
    g.register_output("width", high - low)

    _, fused, levels, replace = plan(g)

    unfused_plot = tmp_path / "before.png"
    fused_plot = tmp_path / "after.png"
    g.visualize(savepath=str(unfused_plot))
    g.visualize(savepath=str(fused_plot), ops=fused, op_levels=levels, replace=replace)

    assert unfused_plot.stat().st_size > 0
    assert fused_plot.stat().st_size > 0
