import random

import pytest

from kernelsmith import CallFactory, F4, Graph, I4, KernelsmithError, SCAL, VarRole, VEC
from kernelsmith.features import rolling_min_max, sma
from kernelsmith.ir.allocate import PoolKind, PoolTracker, allocate
from kernelsmith.ir.liveness import Liveness


def plan(g):
    ops = g.build()
    live = Liveness(ops, g.outputs)
    return ops, live, allocate(ops, live)


def temp_pools(alloc):
    return {k: v for k, v in alloc.pool_size.items() if k[0] is PoolKind.TEMP}


# ---- pool behaviour --------------------------------------------------------

def test_chain_needs_exactly_two_temp_slots():
    """Consecutive values in a chain overlap - op i reads t(i-1) while writing
    t(i) - so a chain alternates between two slots however long it gets."""
    sizes = []
    for length in (4, 8, 16):
        g = Graph()
        value = g.input("c")
        for k in range(length):
            value = value + k
        g.output("out", value)
        sizes.append(sum(temp_pools(plan(g)[2]).values()))

    assert sizes == [2, 2, 2]


def test_simultaneously_live_values_get_different_slots():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    fast, slow = sma(close, n), sma(close, g.int_param("m"))
    g.output("x", fast > slow)

    _, _, alloc = plan(g)
    assert alloc.slots[fast] != alloc.slots[slow]


def test_registered_outputs_use_the_output_pool():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    out = sma(close, n) * 2
    g.output("out", out)

    _, _, alloc = plan(g)
    assert alloc.slots[out][0][0] is PoolKind.OUTPUT


def test_outputs_are_never_recycled():
    """Two outputs of the same dtype and shape must not share a slot even
    though the first is 'finished' long before the schedule ends."""
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    first = sma(close, n)
    g.output("a", first)
    g.output("b", first * 2)

    _, _, alloc = plan(g)
    assert alloc.slots[g.outputs["a"]] != alloc.slots[g.outputs["b"]]


def test_dead_value_gets_a_temp_slot_and_frees_it():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    low, high = rolling_min_max(close, n)
    g.output("h", high * 2)

    ops, live, alloc = plan(g)
    assert low in live.dead
    assert alloc.slots[low][0][0] is PoolKind.TEMP
    # low dies at its own op, so its slot is available to later values
    assert sum(temp_pools(alloc).values()) == 2


def test_scratch_is_reused_across_ops():
    scratchy = CallFactory("scratchy", [VEC(F4), SCAL(I4)], [VEC(F4), VEC(F4)], [VEC(F4)])
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    g.output("y", scratchy(scratchy(close, n), n))

    ops, _, alloc = plan(g)
    first, second = alloc.scratch[ops[0]], alloc.scratch[ops[1]]
    assert len(first) == len(second) == 2
    assert set(i for _, i in first) == set(i for _, i in second)


def test_scratch_does_not_collide_with_the_ops_own_output():
    scratchy = CallFactory("scratchy", [VEC(F4), SCAL(I4)], [VEC(F4)], [VEC(F4)])
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    result = scratchy(close, n)
    g.output("y", result * 2)

    ops, _, alloc = plan(g)
    assert alloc.slots[result] not in alloc.scratch[ops[0]]


def test_repeated_argument_is_released_once():
    """'x * x' lists the same value twice; releasing twice would put one index
    in the free list twice and hand it to two live values."""
    g = Graph()
    x = g.input("x")
    squared = x * x
    g.output("out", squared * squared)

    ops, live, alloc = plan(g)
    assert len(ops[0].args) == 2 and ops[0].args[0] is ops[0].args[1]
    assert sum(temp_pools(alloc).values()) == 1


def test_provided_values_are_not_allocated():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    g.output("out", sma(close, n) + 1)

    _, live, alloc = plan(g)
    for value in live.all_nodes:
        if value.role is not VarRole.TEMP:
            assert value not in alloc.slots


def test_pool_size_is_the_high_water_mark():
    g = Graph()
    close, n = g.input("close"), g.int_param("n")
    low, high = rolling_min_max(close, n)
    g.output("w", high - low)

    _, _, alloc = plan(g)
    for key, size in alloc.pool_size.items():
        used = {index for slot_key, index in alloc.slots.values() if slot_key == key}
        used |= {index for slots in alloc.scratch.values() for k, index in slots if k == key}
        assert used <= set(range(size))
        assert size == max(used) + 1


# ---- the tracker itself ----------------------------------------------------

def test_tracker_reuses_released_slots():
    pool = PoolTracker()
    key = (PoolKind.TEMP, None, None)
    a, b = pool.take(key), pool.take(key)
    pool.release(b)
    assert pool.take(key) == b
    assert pool.sizes()[key] == 2


def test_tracker_rejects_double_release():
    pool = PoolTracker()
    key = (PoolKind.TEMP, None, None)
    slot = pool.take(key)
    pool.release(slot)
    with pytest.raises(KernelsmithError, match="released twice"):
        pool.release(slot)


def test_tracker_sizes_are_a_copy():
    pool = PoolTracker()
    key = (PoolKind.TEMP, None, None)
    pool.take(key)
    sizes = pool.sizes()
    sizes[key] = 99
    assert pool.sizes()[key] == 1


# ---- the invariant ---------------------------------------------------------

def random_graph(seed):
    rng = random.Random(seed)
    g = Graph()
    values = [g.input(f"i{k}") for k in range(rng.randint(1, 3))]
    for _ in range(rng.randint(3, 25)):
        left, right = rng.choice(values), rng.choice(values)
        op = rng.choice(["+", "-", "*"])
        values.append(left + right if op == "+" else left - right if op == "-" else left * right)
    for k, node in enumerate(rng.sample(values, rng.randint(1, min(3, len(values))))):
        g.output(f"o{k}", node)
    return g


@pytest.mark.parametrize("seed", range(60))
def test_live_values_never_share_a_slot(seed):
    """The property that makes an allocation correct: at every step, two values
    that are both live are in different buffers."""
    ops, live, alloc = plan(random_graph(seed))

    for i in range(len(ops)):
        occupied = {}
        for value in live.live_at(i):
            slot = alloc.slots.get(value)
            if slot is None:
                continue
            clash = occupied.get(slot)
            assert clash is None or clash is value, (
                f"step {i}: {value!r} and {clash!r} share {slot}"
            )
            occupied[slot] = value
