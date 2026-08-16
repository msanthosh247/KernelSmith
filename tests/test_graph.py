import pytest

from kernelsmith import (
    Call,
    CallFactory,
    DType,
    DslTypeError,
    F4,
    Graph,
    GraphError,
    I4,
    Op,
    Shape,
    ValueNode,
    VarRole,
)


def make_graph():
    sma = CallFactory("sma", [F4[:], I4], [], [F4[:]])
    minmax = CallFactory("minmax", [F4[:], I4], [F4[:]], [F4[:], F4[:]])

    g = Graph()
    close = g.register_input("close")
    opn = g.register_input("open")
    fast = g.int_param("fast")
    slow = g.int_param("slow")

    med = (close + opn) / 2
    s1 = sma(med, fast)
    s2 = sma(med, slow)
    sig = (s1 > s2) & (close > s2)
    lo, hi = minmax(close, slow)
    rng_ok = (hi - lo) > 0.5

    g.register_output("signal", sig)
    g.register_output("range_ok", rng_ok)
    return g, sma


# ---- construction ----------------------------------------------------------

def test_expression_structure():
    g, _ = make_graph()
    sig = g.outputs["signal"]
    assert sig.dtype is DType.BOOL and sig.shape is Shape.VECTOR
    assert sig.parent.name == "&"


def test_ops_share_one_interface():
    """Expr and Call are both Ops: name, args, outs, buffer_signature."""
    g, sma = make_graph()
    for op in g.build():
        assert isinstance(op, Op)
        assert isinstance(op.name, str)
        assert isinstance(op.args, tuple) and isinstance(op.outs, tuple)
        assert all(isinstance(v, ValueNode) for v in op.args + op.outs)
        assert isinstance(op.buffer_signature, tuple)

    # Expr exposes its single output through outs as well as .output
    expr = g.outputs["signal"].parent
    assert expr.outs == (expr.output,)

    # dtype/shape live on values, never on ops - a multi-output op has no single dtype
    assert not hasattr(expr, "dtype")


def test_call_reports_factory_buffers():
    minmax = CallFactory("mm", [F4[:], I4], [F4[:], F4[:]], [F4[:]])
    g = Graph()
    out = minmax(g.register_input("close"), g.int_param("n"))
    assert len(out.parent.buffer_signature) == 2


def test_const_wrapping_and_division():
    g = Graph()
    close = g.register_input("close")
    med = (close + g.register_input("open")) / 2
    assert med.dtype is DType.FLOAT32 and med.shape is Shape.VECTOR
    two = med.parent.right
    assert two.role is VarRole.CONST and two.val == 2 and two.dtype is DType.INT32


def test_vector_scalar_broadcasts_to_vector():
    g = Graph()
    assert (g.register_input("close") + g.int_param("n")).shape is Shape.VECTOR


def test_inputs_memoized_params_checked():
    g = Graph()
    assert g.register_input("close") is g.register_input("close")
    g.int_param("n")
    with pytest.raises(GraphError, match="already exists"):
        g.float_param("n")


def test_nodes_hashable_despite_eq_overload():
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")
    assert isinstance(a == b, ValueNode)  # __eq__ builds a graph node
    assert len({a: 0, b: 1}) == 2         # identity hash keeps dicts working


def test_type_errors_at_build_time():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    with pytest.raises(DslTypeError):
        close & n
    with pytest.raises(DslTypeError):
        ~close


def test_truth_testing_is_rejected():
    """Comparisons build graph nodes, so a node has no truth value - saying so
    turns a silently wrong answer into a loud one."""
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")

    with pytest.raises(DslTypeError, match="ambiguous"):
        bool(a)
    with pytest.raises(DslTypeError, match="ambiguous"):
        if a:
            pass
    with pytest.raises(DslTypeError, match="ambiguous"):
        a and b
    with pytest.raises(DslTypeError, match="ambiguous"):
        assert a


def test_membership_needs_sets_not_sequences():
    """'in' over a list or tuple compares with '==', which builds a node - so it
    must raise rather than report a bogus hit. Sets and dicts hash by identity."""
    g = Graph()
    a, b = g.register_input("a"), g.register_input("b")

    with pytest.raises(DslTypeError, match="ambiguous"):
        a in (b,)

    assert a in {a}
    assert b not in {a}
    assert {a: 1}.get(a) == 1
    assert any(x is a for x in (a, b))     # the identity idiom passes use


# ---- factories -------------------------------------------------------------

def test_two_calls_are_independent():
    g, sma = make_graph()
    a = sma(g.register_input("close"), g.int_param("fast"))
    b = sma(g.register_input("open"), g.int_param("slow"))
    assert a.parent is not b.parent
    assert a.parent.args[0] is g.register_input("close")
    assert b.parent.args[0] is g.register_input("open")


def test_multi_output_unpacks_with_out_index():
    minmax = CallFactory("minmax", [F4[:], I4], [], [F4[:], F4[:]])
    g = Graph()
    lo, hi = minmax(g.register_input("close"), g.int_param("n"))
    assert lo.parent is hi.parent
    assert (lo.out_index, hi.out_index) == (0, 1)


def test_signature_errors():
    sma = CallFactory("sma", [F4[:], I4], [], [F4[:]])
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    with pytest.raises(DslTypeError, match="expects 2 arguments"):
        sma(close)
    with pytest.raises(DslTypeError, match="argument 0"):
        sma(n, n)


def test_output_lookup_is_explicit():
    g, _ = make_graph()
    signal = g.outputs["signal"]
    assert g.is_output(signal)
    assert g.output_names[signal] == "signal"
    assert not g.is_output(g.register_input("close"))


def test_output_registry_rules():
    g, _ = make_graph()
    minmax = CallFactory("mm", [F4[:], I4], [], [F4[:], F4[:]])
    pair = minmax(g.register_input("close"), g.int_param("slow"))
    with pytest.raises(GraphError, match="unpack"):
        g.register_output("pair", pair)
    with pytest.raises(GraphError, match="duplicate"):
        g.register_output("signal", g.outputs["signal"])


# ---- build -----------------------------------------------------------------

def test_build_topological_invariant():
    g, _ = make_graph()
    order = g.build()
    seen = set()
    for op in order:
        for arg in op.args:
            assert arg.parent is None or arg.parent in seen
        seen.add(op)
    assert len(order) == 10
    assert sorted(set(g.op_levels.values())) == [0, 1, 2, 3, 4]


def test_build_requires_outputs():
    with pytest.raises(GraphError, match="no outputs"):
        Graph().build()


# ---- visualize -------------------------------------------------------------

def test_visualize_smoke(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pytest.importorskip("networkx")

    g, _ = make_graph()
    out = tmp_path / "graph.png"
    assert g.visualize(savepath=str(out)) == str(out)
    assert out.stat().st_size > 0
