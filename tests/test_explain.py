from kernelsmith import CallFactory, F4, Graph, I4
from kernelsmith.features import rolling_min_max, sma
from kernelsmith.ir.explain import explain_graph


def test_explain_reports_the_whole_pipeline():
    scratchy = CallFactory("scratchy", [F4[:], I4], [F4[:]], [F4[:]])

    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
    g.register_output("scratchy", scratchy(med, fast))

    report = explain_graph(g)

    assert "2 input(s)   2 param(s)   2 output(s)" in report
    assert "sma" in report and "scratchy" in report
    assert "O:b1v[0]" in report          # the bool output and its buffer
    assert "pools:" in report
    assert report.count("sma") == 2      # CSE collapsed the duplicate


def test_explain_marks_dead_values():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    _, high = rolling_min_max(close, n)   # the low is never consumed
    g.register_output("h", high * 2)

    report = explain_graph(g)
    assert "*" in report
    assert "produced but never read" in report


def test_explain_omits_the_dead_legend_when_nothing_is_dead():
    g = Graph()
    close, n = g.register_input("close"), g.int_param("n")
    g.register_output("avg", sma(close, n))

    report = explain_graph(g)
    assert "produced but never read" not in report
