"""The examples run, and what they claim holds."""
import importlib.util
import pathlib

import numpy as np
import pytest

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_option_pricing_runs_and_prices_consistently(capsys):
    pytest.importorskip("numba")
    example = load("option_pricing")
    table = example.main(["--paths", "4000", "--backend", "numba", "--no-plot"])

    for (block, expiry), entry in table.items():
        corrected = entry["corrected"]
        # the martingale correction makes put-call parity hold to rounding ...
        assert corrected["parity"] < 1e-9, (block, expiry)
        # ... and prices are prices: non-negative, calls falling and puts rising in strike
        assert (corrected["calls"] >= 0).all() and (corrected["puts"] >= 0).all()
        assert (np.diff(corrected["calls"]) <= 1e-12).all()
        assert (np.diff(corrected["puts"]) >= -1e-12).all()
    assert "put-call parity" in capsys.readouterr().out


def test_option_pricing_backends_agree():
    """The same (seed, path) is the same path on the oracle and on numba."""
    pytest.importorskip("numba")
    example = load("option_pricing")
    from kernelsmith.backends.cpu import CPUBackend
    from kernelsmith.backends.numba_cpu import NumbaCPU_Backend

    history = example.garch_history(days=300)
    oracle, _, _ = example.simulate(CPUBackend, history, 50)
    compiled, _, _ = example.simulate(NumbaCPU_Backend, history, 50)
    for name in oracle:
        np.testing.assert_allclose(compiled[name], oracle[name], rtol=1e-6)
