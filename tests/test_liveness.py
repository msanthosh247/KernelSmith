from kernelsmith import Graph
from kernelsmith.features import rolling_min_max, sma
from kernelsmith.ir.cse import cse
from kernelsmith.ir.liveness import Liveness


def dead_value_graph():
    """rolling_min_max produces two values but only the high is consumed."""
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    low, high = rolling_min_max(close, n)
    scaled = high * 2
    g.output("high2", scaled)
    return g, close, n, low, high, scaled


# ---- intervals -------------------------------------------------------------

def test_pre_existing_values_start_before_the_schedule():
    g, close, n, _, _, _ = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    assert live.define_index[close] == -1
    assert live.define_index[n] == -1
    assert live.last_use[close] == 0


def test_produced_value_interval():
    g, _, _, _, high, _ = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    assert live.define_index[high] == 0
    assert live.last_use[high] == 1


def test_last_use_is_the_last_consumer():
    g = Graph()
    close, opn = g.input("close"), g.input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2                       # consumed by three ops
    g.output("x", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))

    ops = g.build()
    live = Liveness(ops, g.outputs)

    consumers = [i for i, op in enumerate(ops) if any(a is med for a in op.args)]
    assert len(consumers) > 1
    assert live.last_use[med] == max(consumers)


def test_same_value_used_twice_in_one_op():
    g = Graph()
    x = g.input("x")
    g.output("sq", x * x)

    live = Liveness(g.build(), g.outputs)
    assert live.last_use[x] == 0


# ---- dead values -----------------------------------------------------------

def test_unconsumed_output_of_a_call_is_dead():
    """The call still runs - it is opaque - but the value needs no output buffer."""
    g, _, _, low, high, _ = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    assert live.dead == {low}
    assert high not in live.dead


def test_dead_values_get_a_degenerate_interval():
    g, _, _, low, _, _ = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    assert live.define_index[low] == live.last_use[low] == 0


def test_registered_outputs_are_never_dead():
    g, _, _, _, _, scaled = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    assert scaled not in live.dead
    assert not (live.dead & live.outputs)


# ---- output lifetime -------------------------------------------------------

def test_outputs_live_to_the_end():
    g, _, _, _, _, scaled = dead_value_graph()
    ops = g.build()
    live = Liveness(ops, g.outputs)

    assert live.last_use[scaled] == len(ops)


def test_a_later_consumer_cannot_shorten_an_output():
    """An output that also feeds another op must still survive the whole
    schedule, or the allocator would recycle it before it is copied out."""
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    avg = sma(close, n)
    g.output("avg", avg)
    g.output("doubled", avg * 2)

    ops = g.build()
    live = Liveness(ops, g.outputs)

    assert any(a is avg for a in ops[-1].args)     # genuinely consumed
    assert live.last_use[avg] == len(ops)


# ---- interaction with CSE --------------------------------------------------

def test_outputs_are_resolved_through_the_replacement_map():
    """After CSE a registered output can be a dropped duplicate; the survivor
    must inherit its output status."""
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    g.output("p", sma(close, n))
    g.output("q", sma(close, n))          # duplicate, also registered

    kept, replace = cse(g.build())
    live = Liveness(kept, g.outputs, replace)
    survivor = kept[0].outs[0]

    assert len(kept) == 1
    assert survivor in live.outputs
    assert survivor not in live.dead
    assert live.last_use[survivor] == len(kept)


# ---- invariants ------------------------------------------------------------

def test_maps_are_total_and_ordered():
    g = Graph()
    close, opn = g.input("close"), g.input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    low, high = rolling_min_max(close, slow)
    g.output("signal", sma(med, fast) > sma(med, slow))
    g.output("width", high - low)

    ops = g.build()
    live = Liveness(ops, g.outputs)

    for value in live.all_nodes:
        assert value in live.define_index
        assert value in live.last_use
        assert live.define_index[value] <= live.last_use[value]

    for i, op in enumerate(ops):
        for arg in op.args:
            assert live.define_index[arg] < i      # producers come first
            assert live.last_use[arg] >= i         # still alive when read


def test_live_at_selects_overlapping_intervals():
    g, _, _, _, high, scaled = dead_value_graph()
    live = Liveness(g.build(), g.outputs)

    at_one = live.live_at(1)
    assert high in at_one and scaled in at_one
    for value in at_one:
        assert live.define_index[value] <= 1 <= live.last_use[value]
