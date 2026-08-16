"""Is generated code as fast as code written by hand?

Three implementations of the same SMA-crossover strategy over a parameter
sweep: the numpy reference interpreter, kernelsmith's generated kernel, and a
hand-written njit function with the buffers a careful person would preallocate.

Run:  python benchmarks/numba_cpu.py            compare the three
      python benchmarks/numba_cpu.py --threads  thread-scaling curve
"""
import os
import sys
import time

import numpy as np
from numba import config, njit, prange, set_num_threads

from kernelsmith import Graph
from kernelsmith.backends.cpu import CpuBackend
from kernelsmith.backends.numba_cpu import NumbaCPU_Backend
from kernelsmith.features import sma
from kernelsmith.features.numba_cpu_impl import sma_numba_cpu


@njit(parallel=True, cache=True)
def hand_written(close, opn, fast, slow, med, buf_fast, buf_slow, out):
    n_params = fast.shape[0]
    n_bars = close.shape[0]
    for p in prange(n_params):
        for t in range(n_bars):
            med[p, t] = (close[t] + opn[t]) / 2
        sma_numba_cpu(med[p], fast[p], buf_fast[p])
        sma_numba_cpu(med[p], slow[p], buf_slow[p])
        for t in range(n_bars):                       # the three elementwise
            f = buf_fast[p, t]                        # ops fused by hand
            s = buf_slow[p, t]
            out[p, t] = (f > s) & (close[t] > s)


def build_graph():
    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    fast, slow = g.int_param("fast"), g.int_param("slow")
    med = (close + opn) / 2
    g.register_output(
        "signal", (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow))
    )
    return g


def best_of(call, repeats=7):
    call()                                            # warm the JIT and the pages
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    print(f"cores: {os.cpu_count()}   numba threads: {config.NUMBA_NUM_THREADS}")

    for n_params, n_bars in ((1024, 4000), (8192, 4000)):
        rng = np.random.default_rng(0)
        close = (np.cumsum(rng.normal(0, 1, n_bars)) + 100).astype(np.float32)
        opn = (close + 0.1).astype(np.float32)
        fast = rng.integers(3, 30, n_params).astype(np.int32)
        slow = rng.integers(30, 200, n_params).astype(np.int32)

        scratch = [np.zeros((n_params, n_bars), np.float32) for _ in range(3)]
        out = np.zeros((n_params, n_bars), np.bool_)

        inputs = {"close": close, "open": opn}
        params = {"fast": fast, "slow": slow}

        generated = NumbaCPU_Backend().compile(build_graph())
        interpreted = CpuBackend().compile(build_graph())

        t_hand = best_of(lambda: hand_written(close, opn, fast, slow, *scratch, out))
        t_generated = best_of(lambda: generated.run(inputs, params))
        t_numpy = best_of(lambda: interpreted.run(inputs, params), repeats=2)

        hand_written(close, opn, fast, slow, *scratch, out)
        agree = np.array_equal(generated.run(inputs, params)["signal"], out)

        print(f"\n{n_params} parameter sets x {n_bars} bars   (results agree: {agree})")
        print(f"  numpy interpreter    {t_numpy * 1e3:8.2f} ms")
        print(f"  hand-written numba   {t_hand * 1e3:8.2f} ms")
        print(f"  kernelsmith          {t_generated * 1e3:8.2f} ms"
              f"    {t_numpy / t_generated:5.1f}x vs numpy,"
              f" {t_generated / t_hand:.2f}x hand-written")


def scaling(n_params=8192, n_bars=4000):
    """How far does prange carry this workload?

    The curve flattens once memory bandwidth saturates - past that point the
    only way to go faster is to move less data, which is what fusion does.
    """
    from kernelsmith.backends.numba_cpu import kernel_parameters

    rng = np.random.default_rng(0)
    close = (np.cumsum(rng.normal(0, 1, n_bars)) + 100).astype(np.float32)
    opn = (close + 0.1).astype(np.float32)
    fast = rng.integers(3, 30, n_params).astype(np.int32)
    slow = rng.integers(30, 200, n_params).astype(np.int32)

    program = NumbaCPU_Backend().compile(build_graph())
    arrays = {
        "inp_f32": np.stack([close, opn], axis=1),
        "par_i32": np.stack([fast, slow], axis=1),
        **program._buffers(n_params, n_bars),
        "n_params": n_params,
        "n_bars": n_bars,
    }
    ordered = [arrays[n] for n in kernel_parameters(program.graph, program.allocation, program.binding)]

    print(f"{n_params} parameter sets x {n_bars} bars"
          f"   ({config.NUMBA_NUM_THREADS} threads available)\n")

    single = None
    for threads in (1, 2, 4, 8, 16, 32):
        if threads > config.NUMBA_NUM_THREADS:
            break
        set_num_threads(threads)
        elapsed = best_of(lambda: program.kernel(*ordered), repeats=5)
        single = single or elapsed
        print(f"  {threads:>3} thread(s)   {elapsed * 1e3:8.2f} ms   {single / elapsed:5.2f}x")

    set_num_threads(config.NUMBA_NUM_THREADS)


if __name__ == "__main__":
    if "--threads" in sys.argv:
        scaling()
    else:
        main()
