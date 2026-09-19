"""CUDA backend against the Numba CPU backend, on the same strategy.

Several CUDA figures, because they answer different questions:

  run()         what a caller pays: host->device copies, the kernel, and copying
                the outputs back.
  sorted        run(..., sort_by="slow"): warps get similar lookbacks, so their
                reads into a window coalesce. Helps window-heavy graphs, can
                cost a little elsewhere - which is why it is opt-in.
  kernel only   the device work alone, synchronised - what the layout and the
                code generation are responsible for.

Occupancy depends on the number of parameter sets, not the block size: there is
one thread per parameter set, so a small sweep leaves most of the GPU idle.
Sizes that will not fit in free device memory are skipped.

The sweep answers "how big a sweep does the GPU need, and is more VRAM
more speed?". It doubles the number of parameter sets until the device is out
of memory, for three graphs of rising arithmetic intensity, and plots time,
throughput (parameter-set x bar evaluations per second) and device memory
against it. Throughput climbs while more parameter sets means more resident
threads, and flattens once every SM is full - past that, a bigger sweep only
queues more waves, and filling VRAM buys capacity, not speed.

Run:  python benchmarks/cuda.py                 compare backends
      python benchmarks/cuda.py --block-sizes   sweep threads per block
      python benchmarks/cuda.py --sweep         scale the sweep; writes
                                                benchmarks/results/cuda_sweep.png
"""
import csv
import gc
import importlib.util
import pathlib
import sys
import time
import warnings

import numpy as np
from numba import config, cuda

from kernelsmith import Graph
from kernelsmith.backends.cuda import CudaBackend
from kernelsmith.backends.generated import NUMPY_DTYPE
from kernelsmith.backends.numba_cpu import NumbaCPU_Backend
from kernelsmith import features as F
from kernelsmith.features import sma

warnings.filterwarnings("ignore", message=".*Grid size")

N_BARS = 4000
SWEEPS = (8192, 32768)


def build_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output(
        "signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow))
    )
    return g


def data(n_params, n_bars=N_BARS):
    rng = np.random.default_rng(0)
    close = (np.cumsum(rng.normal(0, 1, n_bars)) + 100).astype(np.float32)
    inputs = {"close": close, "open": (close + 0.1).astype(np.float32)}
    params = {
        "fast": rng.integers(3, 30, n_params).astype(np.int32),
        "slow": rng.integers(30, 200, n_params).astype(np.int32),
    }
    return inputs, params


def best_of(call, repeats=5):
    call()                                   # warm: JIT, allocations, page faults
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def device_bytes(program, n_params, n_bars):
    """Every buffer the kernel needs resident on the device at once."""
    layout = program.layout
    total = 0
    for key, size in layout.allocation.pool_size.items():
        dims = layout.pool_shape(key, size, n_params, n_bars)
        total += int(np.prod(dims)) * np.dtype(NUMPY_DTYPE[key[1]]).itemsize
    return total


def kernel_only(program, inputs, params):
    """Time the launch alone, with everything already on the device."""
    arrays, _, n_params = program._prepare(inputs, params)
    ordered = [arrays[name] for name in program.param_names]
    return best_of(lambda: program._launch(ordered, n_params))


def compare():
    device = cuda.get_current_device()
    free, _ = cuda.current_context().get_memory_info()
    capacity = device.MULTIPROCESSOR_COUNT * device.MAX_THREADS_PER_MULTIPROCESSOR
    print(f"{device.name.decode() if isinstance(device.name, bytes) else device.name}"
          f"   {free / 2**30:.1f} GiB free   {capacity} resident threads"
          f"   CPU threads: {config.NUMBA_NUM_THREADS}")

    cpu_backend, gpu_backend = NumbaCPU_Backend(), CudaBackend()

    for n_params in SWEEPS:
        inputs, params = data(n_params)
        cpu = cpu_backend.compile(build_graph())
        gpu = gpu_backend.compile(build_graph())

        needed = device_bytes(gpu, n_params, N_BARS)
        if needed > 0.8 * free:
            print(f"\n{n_params} parameter sets: needs {needed / 2**30:.1f} GiB on the device - skipped")
            continue

        t_cpu = best_of(lambda: cpu.run(inputs, params))
        t_run = best_of(lambda: gpu.run(inputs, params))
        t_sorted = best_of(lambda: gpu.run(inputs, params, sort_by="slow"))
        t_kernel = kernel_only(gpu, inputs, params)

        mismatched = int(np.count_nonzero(
            cpu.run(inputs, params)["signal"] != gpu.run(inputs, params)["signal"]
        ))
        occupancy = min(1.0, n_params / capacity)

        print(f"\n{n_params} parameter sets x {N_BARS} bars"
              f"   (device {occupancy:.0%} occupied, {needed / 2**20:.0f} MiB,"
              f" signals differing from CPU: {mismatched})")
        print(f"  numba CPU run()        {t_cpu * 1e3:8.2f} ms")
        print(f"  CUDA run()             {t_run * 1e3:8.2f} ms   {t_cpu / t_run:5.2f}x CPU")
        print(f"  CUDA run(sorted)       {t_sorted * 1e3:8.2f} ms   {t_cpu / t_sorted:5.2f}x CPU")
        print(f"  CUDA kernel only       {t_kernel * 1e3:8.2f} ms   {t_cpu / t_kernel:5.2f}x CPU"
              f"   (copies: {(t_run - t_kernel) * 1e3:.2f} ms)")


def block_sizes(n_params=SWEEPS[-1]):
    inputs, params = data(n_params)
    print(f"{n_params} parameter sets x {N_BARS} bars, kernel only\n")
    baseline = None
    for size in (32, 64, 128, 256):
        program = CudaBackend(block_size=size).compile(build_graph())
        elapsed = kernel_only(program, inputs, params)
        baseline = baseline or elapsed
        print(f"  block {size:>4}   {elapsed * 1e3:8.2f} ms   {baseline / elapsed:5.2f}x block 32")


# ---- scaling the sweep -----------------------------------------------------------

RESULTS = pathlib.Path(__file__).parent / "results"


def channel_graph():
    """Rolling min/max breakout: ~period comparisons per bar."""
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    g.int_param("fast")
    low, high = F.rolling_min_max(close, g.int_param("slow"))
    g.register_output("signal", (close - low) > (high - close))
    return g


def rsi_bollinger_graph():
    """RSI crossing up through 30 below the lower Bollinger band."""
    g = Graph()
    close, _ = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    lower, _, _ = F.bollinger(close, slow, 2.0)
    g.register_output("signal", F.cross_over(F.rsi(close, fast), 30.0) & (close < lower))
    return g


SWEEP_GRAPHS = {
    "SMA crossover": build_graph,
    "channel (rolling min/max)": channel_graph,
    "RSI + Bollinger": rsi_bollinger_graph,
}


def host_available() -> float:
    """Free host RAM in bytes, or infinity when psutil is not installed."""
    if importlib.util.find_spec("psutil") is None:
        return float("inf")
    import psutil
    return psutil.virtual_memory().available


def release_device_memory(*programs):
    """Free everything a previous measurement left on the device.

    A program caches its scratch until the sweep size changes, and numba frees
    device memory lazily. Without this, the previous size's buffers are still
    resident when the next, larger size allocates - near the top of the sweep
    that pushed the footprint past free memory, and on a Windows (WDDM) driver
    the result was not an out-of-memory error but a 3x slower kernel, and a
    slow first point for whichever graph came next. Measured one set at a time,
    kernel time stays linear up to 70% of free memory."""
    for program in programs:
        program._scratch.clear()
    gc.collect()
    cuda.current_context().deallocations.clear()


def sweep_sizes(program, start):
    """Doubling sizes while they fit, then the largest multiple of 1024 that
    does - so the last point shows how far device memory actually goes.

    "Fits" is 70% of free memory: run() holds one set of outputs and the
    kernel-only timing prepares another.
    """
    budget = 0.7 * cuda.current_context().get_memory_info()[0]
    per_set = device_bytes(program, 1024, N_BARS) / 1024
    largest = int(budget / per_set) // 1024 * 1024
    sizes = []
    n_params = start
    while n_params <= largest:
        sizes.append(n_params)
        n_params *= 2
    if largest > sizes[-1]:
        sizes.append(largest)
    return sizes


def sweep(start=1024):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    resident = device.MULTIPROCESSOR_COUNT * device.MAX_THREADS_PER_MULTIPROCESSOR
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    print(f"{name}: {total / 2**30:.1f} GiB, {free / 2**30:.1f} free,"
          f" {resident} resident threads; {N_BARS} bars\n")

    rows = []
    for graph_name, build in SWEEP_GRAPHS.items():
        release_device_memory()
        cpu_program = NumbaCPU_Backend().compile(build())
        gpu_program = CudaBackend().compile(build())
        for n_params in sweep_sizes(gpu_program, start):
            release_device_memory(gpu_program)
            needed = device_bytes(gpu_program, n_params, N_BARS)
            inputs, params = data(n_params)
            t_run = best_of(lambda: gpu_program.run(inputs, params), repeats=3)
            t_kernel = kernel_only(gpu_program, inputs, params)
            t_cpu = float("nan")
            if needed < 0.4 * host_available():
                t_cpu = best_of(lambda: cpu_program.run(inputs, params), repeats=2)
            rows.append({
                "graph": graph_name, "n_params": n_params, "device_mib": needed / 2**20,
                "cpu_ms": t_cpu * 1e3, "run_ms": t_run * 1e3, "kernel_ms": t_kernel * 1e3,
            })
            r = rows[-1]
            print(f"{graph_name:<26} P={n_params:>7}  {r['device_mib']:7.0f} MiB"
                  f"  cpu {r['cpu_ms']:8.1f}  run {r['run_ms']:8.1f}  kernel {r['kernel_ms']:8.1f} ms"
                  f"  run {r['cpu_ms'] / r['run_ms']:5.2f}x cpu")
        del cpu_program, gpu_program
        print()

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "cuda_sweep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, len(SWEEP_GRAPHS), figsize=(5 * len(SWEEP_GRAPHS), 11),
                             sharex=True, squeeze=False)
    series = (("cpu_ms", "numba CPU run()", "tab:gray"),
              ("run_ms", "CUDA run()", "tab:blue"),
              ("kernel_ms", "CUDA kernel only", "tab:orange"))
    for column, graph_name in enumerate(SWEEP_GRAPHS):
        mine = [r for r in rows if r["graph"] == graph_name]
        sizes = np.array([r["n_params"] for r in mine])
        time_ax, rate_ax, memory_ax = axes[:, column]
        for key, label, color in series:
            ms = np.array([r[key] for r in mine])
            time_ax.plot(sizes, ms, "o-", color=color, label=label)
            # parameter-set x bar evaluations per second
            rate_ax.plot(sizes, sizes * N_BARS / (ms / 1e3) / 1e9, "o-", color=color, label=label)
        memory_ax.plot(sizes, [r["device_mib"] / 1024 for r in mine], "o-", color="tab:green",
                       label="device memory used")
        memory_ax.axhline(total / 2**30, color="tab:green", linestyle=":", label="device total")

        time_ax.set_title(graph_name)
        time_ax.set_ylabel("time (ms)")
        time_ax.set_yscale("log")
        rate_ax.set_ylabel("G evaluations / s")
        memory_ax.set_ylabel("GiB")
        memory_ax.set_xlabel("parameter sets (P)")
        for ax in (time_ax, rate_ax, memory_ax):
            ax.set_xscale("log", base=2)
            ax.axvline(resident, color="black", linestyle="--", linewidth=0.8)
            ax.grid(True, which="both", alpha=0.3)
        rate_ax.annotate("every SM full", (resident, rate_ax.get_ylim()[1]),
                         xytext=(4, -12), textcoords="offset points", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle(f"{name}: scaling the parameter sweep ({N_BARS} bars)")
    fig.tight_layout()
    fig.savefig(RESULTS / "cuda_sweep.png", dpi=120)
    print(f"wrote {RESULTS / 'cuda_sweep.png'} and cuda_sweep.csv")


if __name__ == "__main__":
    if not cuda.is_available():
        sys.exit("no CUDA device available")
    if "--block-sizes" in sys.argv:
        block_sizes()
    elif "--sweep" in sys.argv:
        sweep()
    else:
        compare()
