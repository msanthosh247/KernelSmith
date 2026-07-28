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
    SCAL,
    Shape,
    ValueNode,
    VarRole,
    VEC,
)


def make_graph():
    sma = CallFactory("sma", [VEC(F4), SCAL(I4)], [], [VEC(F4)])
    minmax = CallFactory("minmax", [VEC(F4), SCAL(I4)], [VEC(F4)], [VEC(F4), VEC(F4)])

    g = Graph()
    close = g.input("close")
    opn = g.input("open")
    fast = g.int_param("fast")
    slow = g.int_param("slow")

    med = (close + opn) / 2
    s1 = sma(med, fast)
    s2 = sma(med, slow)
    sig = (s1 > s2) & (close > s2)
    lo, hi = minmax(close, slow)
    rng_ok = (hi - lo) > 0.5

    g.output("signal", sig)
    g.output("range_ok", rng_ok)
    return g, sma


# ---- construction ----------------------------------------------------------

def test_expression_structure():
    g, _ = make_graph()
    sig = g.outputs["signal"]
    assert sig.dtype is DType.BOOL and sig.shape is Shape.VECTOR
    assert sig.parent.operation == "&"


def test_const_wrapping_and_division():
    g = Graph()
    close = g.input("close")
    med = (close + g.input("open")) / 2
    assert med.dtype is DType.FLOAT32 and med.shape is Shape.VECTOR
    two = med.parent.right
    assert two.role is VarRole.CONST and two.val == 2 and two.dtype is DType.INT32


def test_vector_scalar_broadcasts_to_vector():
    g = Graph()
    assert (g.input("close") + g.int_param("n")).shape is Shape.VECTOR


def test_inputs_memoized_params_checked():
    g = Graph()
    assert g.input("close") is g.input("close")
    g.int_param("n")
    with pytest.raises(GraphError, match="already exists"):
        g.float_param("n")


def test_nodes_hashable_despite_eq_overload():
    g = Graph()
    a, b = g.input("a"), g.input("b")
    assert isinstance(a == b, ValueNode)  # __eq__ builds a graph node
    assert len({a: 0, b: 1}) == 2         # identity hash keeps dicts working


def test_type_errors_at_build_time():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    with pytest.raises(DslTypeError):
        close & n
    with pytest.raises(DslTypeError):
        ~close


# ---- factories -------------------------------------------------------------

def test_two_calls_are_independent():
    g, sma = make_graph()
    a = sma(g.input("close"), g.int_param("fast"))
    b = sma(g.input("open"), g.int_param("slow"))
    assert a.parent is not b.parent
    assert a.parent.args[0] is g.input("close")
    assert b.parent.args[0] is g.input("open")


def test_multi_output_unpacks_with_out_index():
    minmax = CallFactory("minmax", [VEC(F4), SCAL(I4)], [], [VEC(F4), VEC(F4)])
    g = Graph()
    lo, hi = minmax(g.input("close"), g.int_param("n"))
    assert lo.parent is hi.parent
    assert (lo.out_index, hi.out_index) == (0, 1)


def test_signature_errors():
    sma = CallFactory("sma", [VEC(F4), SCAL(I4)], [], [VEC(F4)])
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    with pytest.raises(DslTypeError, match="expects 2 arguments"):
        sma(close)
    with pytest.raises(DslTypeError, match="argument 0"):
        sma(n, n)


def test_output_registry_rules():
    g, _ = make_graph()
    minmax = CallFactory("mm", [VEC(F4), SCAL(I4)], [], [VEC(F4), VEC(F4)])
    pair = minmax(g.input("close"), g.int_param("slow"))
    with pytest.raises(GraphError, match="unpack"):
        g.output("pair", pair)
    with pytest.raises(GraphError, match="duplicate"):
        g.output("signal", g.outputs["signal"])


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
