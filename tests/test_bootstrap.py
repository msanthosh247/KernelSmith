"""bootstrap_path: circular moving-block bootstrap of a price history.

The strongest check does not compare two implementations. The history below
has a distinct growth factor at every step, so each simulated step can be
traced back to the history return it used - and the block rules (start at
hash(seed, path, block number) mod n, then consecutive, wrapping) are asserted
on every backend directly.
"""
import numpy as np
import pytest

from kernelsmith import Graph
from kernelsmith.backends.cpu import CPUBackend
from kernelsmith.backends.numba_cpu import NumbaCPU_Backend
from kernelsmith.features import bootstrap_path
from kernelsmith.features.numpy_impl import hash3
from conftest import tolerance

pytestmark = pytest.mark.filterwarnings("ignore:.*Grid size")

SEED = 11


def traceable_history(n_factors=40):
    """Growth factors 0.98 .. 1.02, all distinct and >= 1e-3 apart."""
    factors = 1.0 + np.linspace(-0.02, 0.02, n_factors)
    np.random.default_rng(5).shuffle(factors)
    close = 100.0 * np.cumprod(np.concatenate([[1.0], factors]))
    growth = close[1:] / close[:-1]
    return close.astype(np.float32), growth


def graph():
    g = Graph()
    close = g.register_table("close")
    path = bootstrap_path(close, g.int_param("block"), g.int_param("path"), SEED)
    g.register_output("path", path)
    g.register_output("final", path[-1])
    return g


def run(backend, close, params, horizon):
    return backend().compile(graph()).run({"close": close}, params, n_bars=horizon)


def traced_indices(path, close, growth):
    """Which history factor each step used, from the step's ratio."""
    ratios = path.astype(np.float64) / np.concatenate([[close[-1]], path[:-1]]).astype(np.float64)
    return np.abs(ratios[:, None] - growth[None, :]).argmin(axis=1)


@pytest.mark.parametrize("block", [1, 3, 7, 50])
def test_block_structure_on_every_backend(generating_backend, block):
    close, growth = traceable_history()
    n = growth.size
    horizon = 30
    paths = [0, 1, 2, 1000]
    out = run(generating_backend, close, {"block": [block] * len(paths), "path": paths}, horizon)

    for row, path_id in enumerate(paths):
        used = traced_indices(out["path"][row], close, growth)
        for h in range(horizon):
            b, offset = divmod(h, block)
            start = int(hash3(SEED, path_id, [b])[0] % np.uint32(n))
            assert used[h] == (start + offset) % n, (block, path_id, h)


def test_matches_the_oracle(generating_backend):
    close, _ = traceable_history()
    params = {"block": [1, 5, 20, 0, 100], "path": [0, 1, 2, 3, 4]}
    expected = run(CPUBackend, close, params, 25)
    actual = run(generating_backend, close, params, 25)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], equal_nan=True, err_msg=name,
                                   **tolerance(generating_backend))


def test_degenerate_inputs_give_nan(generating_backend):
    close, _ = traceable_history()
    out = run(generating_backend, close, {"block": [0, -3], "path": [0, 0]}, 6)
    assert np.isnan(out["path"]).all() and np.isnan(out["final"]).all()
    single = run(generating_backend, close[:1], {"block": [2], "path": [0]}, 6)
    assert np.isnan(single["path"]).all()


def test_paths_start_from_the_last_close():
    close, growth = traceable_history()
    out = run(NumbaCPU_Backend, close, {"block": [4], "path": [9]}, 1)
    assert out["path"][0, 0] / close[-1] == pytest.approx(growth[traced_indices(out["path"][0], close, growth)[0]])


def test_same_path_id_same_path_and_ids_differ():
    close, _ = traceable_history()
    out = run(NumbaCPU_Backend, close, {"block": [5, 5, 5], "path": [3, 3, 4]}, 40)
    np.testing.assert_array_equal(out["path"][0], out["path"][1])
    assert not np.array_equal(out["path"][0], out["path"][2])
    again = run(NumbaCPU_Backend, close, {"block": [5], "path": [3]}, 40)
    np.testing.assert_array_equal(again["path"][0], out["path"][0])


def test_blocks_keep_volatility_clustering():
    """A calm first half and a wild second half. With long blocks a path stays
    mostly within one regime, so realised volatility differs more from path to
    path than when every step is drawn independently."""
    rng = np.random.default_rng(2)
    returns = np.concatenate([rng.normal(0, 0.002, 500), rng.normal(0, 0.03, 500)])
    close = (100 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))).astype(np.float32)
    paths = np.arange(2000)

    def realised_vol_spread(block):
        out = run(NumbaCPU_Backend, close, {"block": np.full(paths.size, block), "path": paths}, 40)
        logs = np.diff(np.log(np.concatenate([np.full((paths.size, 1), close[-1]), out["path"]], axis=1)), axis=1)
        return logs.std(axis=1).std()

    assert realised_vol_spread(40) > 2 * realised_vol_spread(1)


def test_sorting_paths_by_block_restores_order(generating_backend):
    close, _ = traceable_history()
    params = {"block": [20, 1, 5, 1, 20, 5], "path": [0, 1, 2, 3, 4, 5]}
    program = generating_backend().compile(graph())
    plain = program.run({"close": close}, params, n_bars=15)
    ordered = program.run({"close": close}, params, n_bars=15, sort_by="block")
    for name in plain:
        np.testing.assert_array_equal(ordered[name], plain[name])
