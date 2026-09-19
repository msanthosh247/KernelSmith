"""CUDA backend: parity with the numpy oracle, plus the layout it emits.

Runs on a real GPU, or anywhere under the simulator:

    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_cuda.py

The simulator executes every thread in Python, so the sizes here stay small.
"""
import math
import re

import numpy as np
import pytest

from kernelsmith.availability import cuda_unavailable_reason, cuda_usable  # noqa: E402

# numba-cuda can be installed yet unimportable (no CUDA runtime): ask the import
if not cuda_usable():
    pytest.skip(f"no CUDA device and the simulator is not enabled"
                f" ({cuda_unavailable_reason or 'no device'})", allow_module_level=True)

from numba import cuda  # noqa: E402

import kernelsmith.backends.generated as generated  # noqa: E402
from kernelsmith import CallFactory, F4, Graph, I4, KernelsmithError  # noqa: E402
from kernelsmith.backends.cpu import CPUBackend  # noqa: E402
from kernelsmith.backends.cuda import (  # noqa: E402
    CUDA,
    DEFAULT_BLOCK_SIZE,
    CudaBackend,
    register_cuda,
)
from kernelsmith.features import rolling_min_max, sma  # noqa: E402
from kernelsmith.ir.fuse import FusedExpr  # noqa: E402
from conftest import tolerance  # noqa: E402

# small sweeps on purpose; numba rightly warns that they under-use the device
# (the message starts with a terminal bold code, hence the leading .*)
pytestmark = pytest.mark.filterwarnings("ignore:.*Grid size")


def prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(0, 1, n)) + 100).astype(np.float32)


def assert_parity(build, inputs, params, **backend_options):
    """The acceptance test: the GPU must agree with the numpy oracle."""
    expected = CPUBackend().compile(build()).run(inputs, params)
    actual = CudaBackend(**backend_options).compile(build()).run(inputs, params)

    assert set(expected) == set(actual)
    for name in expected:
        assert actual[name].shape == expected[name].shape, name
        np.testing.assert_allclose(
            actual[name], expected[name], equal_nan=True,
            err_msg=f"output '{name}' differs", **tolerance(CudaBackend),
        )
    return actual


# ---- graphs ------------------------------------------------------------------

def crossover():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output("signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow)))
    return g


def multi_output_with_dead_value():
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    _low, high = rolling_min_max(close, g.int_param("fast"))
    g.int_param("slow")
    g.register_output("h", high * 2)
    return g


def constants_and_unary():
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    g.int_param("fast"), g.int_param("slow")
    g.register_output("negated", -close)
    g.register_output("flag", ~(close > 100.0))
    g.register_output("scaled", (close * 2) - 1)
    return g


def scalar_expression():
    g = Graph()
    g.register_input("close"), g.register_input("open")
    g.register_output("total", g.int_param("fast") + g.int_param("slow"))
    return g


def bare_input_output():
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    g.int_param("fast"), g.int_param("slow")
    g.register_output("passthrough", close)
    g.register_output("doubled", close * 2)
    return g


def cse_duplicate():
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    fast, _ = g.int_param("fast"), g.int_param("slow")
    g.register_output("a", sma(close, fast))
    g.register_output("b", sma(close, fast))
    return g


def repeated_operand():
    g = Graph()
    close, fast = g.register_input("close"), g.int_param("fast")
    g.int_param("slow")
    dev = close - sma(close, fast)
    g.register_output("var", sma(dev * dev, fast))
    g.register_output("ratio", (close - 1) / (close - 1))
    return g


GRAPHS = [
    pytest.param(repeated_operand, id="repeated-operand"),
    pytest.param(crossover, id="crossover"),
    pytest.param(multi_output_with_dead_value, id="multi-output-dead-value"),
    pytest.param(constants_and_unary, id="constants-unary"),
    pytest.param(scalar_expression, id="scalar"),
    pytest.param(bare_input_output, id="bare-input"),
    pytest.param(cse_duplicate, id="cse-duplicate"),
]


def sweep(n_params=3, n_bars=60):
    close = prices(n_bars)
    inputs = {"close": close, "open": (close + 0.2).astype(np.float32)}
    fast = np.arange(n_params) % 7 + 3
    slow = np.arange(n_params) % 11 + 12
    return inputs, {"fast": fast, "slow": slow}


# ---- parity ------------------------------------------------------------------

@pytest.mark.parametrize("build", GRAPHS)
def test_matches_the_oracle(build):
    assert_parity(build, *sweep())


def test_single_parameter_set():
    assert_parity(crossover, *sweep(n_params=1))


def test_partial_last_block_computes_every_column():
    """37 parameter sets with 32-thread blocks: two blocks, the second mostly
    idle. Every real column must be computed and nothing past it touched."""
    inputs, params = sweep(n_params=37, n_bars=40)
    result = assert_parity(crossover, inputs, params)
    assert result["signal"].shape == (37, 40)


@pytest.mark.parametrize("block_size", [32, 64])
def test_block_size_does_not_change_results(block_size):
    assert_parity(crossover, *sweep(n_params=5), block_size=block_size)


def test_fusion_does_not_change_results(monkeypatch):
    inputs, params = sweep()
    fused = CudaBackend().compile(crossover()).run(inputs, params)

    monkeypatch.setattr(generated, "fuse", lambda ops, outputs=(), replace=None: (ops, {}))
    plain_program = CudaBackend().compile(crossover())
    assert not any(isinstance(op, FusedExpr) for op in plain_program.ops)
    plain = plain_program.run(inputs, params)

    np.testing.assert_array_equal(fused["signal"], plain["signal"])


# ---- the layout in the source -------------------------------------------------

def test_source_is_laid_out_parameter_innermost():
    source = CudaBackend().compile(crossover()).source

    assert source.startswith("@cuda.jit\n")
    assert "p = cuda.grid(1)" in source
    assert "if p < n_params:" in source
    assert "prange" not in source

    # temporaries: time first, parameter set last - adjacent threads, adjacent addresses
    assert "tmp_f32_v[t, " in source and ", p]" in source
    # params are (K, P), read one column per thread
    assert "par_i32[0, p]" in source
    # a feature call gets its parameter set's whole series as a strided column
    assert "sma_cuda(tmp_f32_v[:, " in source


def test_emission_is_deterministic():
    assert CudaBackend().compile(crossover()).source == CudaBackend().compile(crossover()).source


# ---- launch ------------------------------------------------------------------

def test_default_block_size_is_one_warp():
    assert DEFAULT_BLOCK_SIZE == 32
    assert CudaBackend().block_size == 32


@pytest.mark.parametrize("n_params", [1, 31, 32, 33, 1000])
def test_launch_covers_every_parameter_set(n_params):
    program = CudaBackend().compile(crossover())
    blocks, threads = program.launch_config(n_params)
    assert threads == 32
    assert blocks == math.ceil(n_params / 32)
    assert blocks * threads >= n_params


def test_block_size_is_validated():
    with pytest.raises(KernelsmithError, match="between 1 and 1024"):
        CudaBackend(block_size=0)
    with pytest.raises(KernelsmithError, match="between 1 and 1024"):
        CudaBackend(block_size=2048)


# ---- buffers -----------------------------------------------------------------

def test_repeated_runs_reuse_scratch_without_corrupting_results():
    program = CudaBackend().compile(crossover())
    inputs, params = sweep()
    first = program.run(inputs, params)["signal"].copy()
    for _ in range(2):
        np.testing.assert_array_equal(program.run(inputs, params)["signal"], first)


def test_earlier_outputs_survive_a_later_run():
    program = CudaBackend().compile(crossover())
    first = program.run(*sweep())["signal"]
    snapshot = first.copy()
    program.run(*sweep(n_params=2, n_bars=50))
    np.testing.assert_array_equal(first, snapshot)


def test_changing_sweep_size_reallocates():
    program = CudaBackend().compile(crossover())
    assert program.run(*sweep(n_params=2, n_bars=40))["signal"].shape == (2, 40)
    assert program.run(*sweep(n_params=3, n_bars=55))["signal"].shape == (3, 55)


# ---- registry ----------------------------------------------------------------

def test_every_feature_has_a_cuda_kernel():
    for feature in (sma, rolling_min_max):
        assert feature in CUDA


def test_missing_kernel_is_named_at_compile():
    mystery = CallFactory("mystery", [F4[:], I4], [], [F4[:]])
    g = Graph()
    g.register_output("x", mystery(g.register_input("close"), g.int_param("n")))
    with pytest.raises(Exception, match="no CUDA implementation for: mystery"):
        CudaBackend().compile(g)


def test_feature_names_must_be_unique_per_backend():
    first = CallFactory("cuda_twin", [F4[:], I4], [], [F4[:]])
    second = CallFactory("cuda_twin", [F4[:], I4], [], [F4[:]])

    def kernel(values, period, out):          # noqa: ARG001
        pass

    try:
        register_cuda(first)(kernel)
        with pytest.raises(KernelsmithError, match="unique"):
            register_cuda(second)(kernel)
    finally:
        CUDA.entries.pop(first, None)
        CUDA.entries.pop(second, None)


# ---- precision ---------------------------------------------------------------
# GeForce parts run FP64 at 1/64 of the FP32 rate, so the kernel must not touch
# it: not in the feature kernels, not in the generated arithmetic. The only
# reliable witness is the compiled code - one bare literal in a kernel widens
# everything downstream of it, and nothing fails, it is just slow.

def _every_feature_graph():
    from kernelsmith import features as F
    g = Graph()
    high, low, close = (g.register_input(name) for name in ("high", "low", "close"))
    history = g.register_table("history")
    n = g.int_param("n")
    outputs = [
        F.bootstrap_path(history, n, n, 7), close[-1], close[n],
        F.sma(close, n), F.ema(close, n), F.rma(close, n), F.wma(close, n),
        F.rolling_sum(close, n), F.rolling_std(close, n), F.mean_deviation(close, n),
        F.rsi(close, n), F.stochastic_k(high, low, close, n), F.true_range(high, low, close),
        *F.adx(high, low, close, n), *F.rolling_min_max(close, n),
        *F.rolling_arg_min_max(close, n), F.shift(close, n),
        (close + high) / 2 > close * 1.5,                  # int and float constants
    ]
    for i, value in enumerate(outputs):
        g.register_output(f"out{i}", value)
    return g


@pytest.mark.skipif(cuda.simulator_enabled if hasattr(cuda, "simulator_enabled")
                    else __import__("numba").config.ENABLE_CUDASIM,
                    reason="the simulator compiles nothing, so there is no PTX to read")
def test_compiled_kernel_has_no_double_precision():
    program = CudaBackend().compile(_every_feature_graph())
    close = prices(40)
    data = {"high": close + np.float32(1), "low": close - np.float32(1), "close": close,
            "history": prices(100)}
    program.run(data, {"n": [3, 5]})

    ptx = "\n".join(program.kernel.inspect_asm().values())
    doubles = sorted(set(re.findall(r"\b[a-z]+(?:\.[a-z0-9]+)*\.f64\b", ptx)))
    assert doubles == [], f"float64 instructions in the kernel: {doubles}"
    # the bootstrap's block counter replaced h // block: 64-bit division is emulated
    assert not re.search(r"(div|rem)\.[su]64", ptx), "64-bit division in the kernel"


def test_constants_are_emitted_with_their_dsl_type():
    g = Graph()
    close, fast = g.register_input("close"), g.int_param("fast")
    g.register_output("half", close / 2)               # int constant, float division
    g.register_output("big", close > 30)               # int constant, float compare
    g.register_output("next", fast + 1)                # int32 arithmetic
    source = CudaBackend().compile(g).source
    assert "np.float32(2.0)" in source and "np.float32(30.0)" in source
    assert "np.int32((par_i32[0, p]) + (np.int32(1)))" in source
    # every numeric literal sits inside a cast (cuda.grid(1) is not arithmetic)
    wrappers = {name for name, _ in re.findall(r"(\w+)\((\d+(?:\.\d+)?)\)", source)}
    assert wrappers <= {"float32", "int32", "grid"}, wrappers


def test_single_precision_sums_do_not_drift():
    """20,000 bars of a random walk around 1000. Measured: an uncompensated
    float32 rolling mean ends ~10 ulps out (7.7e-7 relative); the compensated
    one stays within one (7e-8)."""
    from kernelsmith.features import rolling_std
    close = (np.cumsum(np.random.default_rng(7).normal(0, 1, 20_000)) + 1000).astype(np.float32)

    def build():
        g = Graph()
        c, n = g.register_input("close"), g.int_param("n")
        g.register_output("mean", sma(c, n))
        g.register_output("std", rolling_std(c, n))
        return g

    expected = CPUBackend().compile(build()).run({"close": close}, {"n": [50]})
    actual = CudaBackend().compile(build()).run({"close": close}, {"n": [50]})
    tail = slice(-1000, None)
    np.testing.assert_allclose(actual["mean"][0, tail], expected["mean"][0, tail], rtol=2e-7, atol=0)
    np.testing.assert_allclose(actual["std"][0, tail], expected["std"][0, tail], rtol=1e-4)
