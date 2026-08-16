# KernelSmith

> Write trading strategies as Python expressions — compile them into fused CUDA kernels.

[![CI](https://github.com/msanthosh247/KernelSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/msanthosh247/KernelSmith/actions/workflows/ci.yml)

**Status: early development.** Phase 1 — the graph DSL and scheduling core — is under active work. The compiler passes and backends land next; the roadmap below is honest about what exists today.

## What it is

KernelSmith is a small compiler for Monte Carlo backtesting workloads: you describe a strategy once as a dataflow graph, and the compiler schedules it, plans its memory, and generates a fused GPU kernel that evaluates thousands of parameter combinations in parallel — one thread per parameter set, with buffer layouts chosen for coalesced access.

The graph is **value-centric SSA**: every value is created exactly once by its producer, which makes output overwrites and dependency cycles unrepresentable by construction — a whole class of framework bugs ruled out before any pass runs.

```python
from kernelsmith import Graph
from kernelsmith.backends.cpu import CpuBackend
from kernelsmith.features import sma

g = Graph()
close, opn = g.register_input("close"), g.register_input("open")
fast, slow = g.int_param("fast"), g.int_param("slow")

med    = (close + opn) / 2            # operators build typed graph nodes
signal = sma(med, fast) > sma(med, slow)
g.register_output("signal", signal)

program = CpuBackend().compile(g)     # schedules the graph, resolves implementations
out = program.run(
    inputs={"close": close_prices, "open": open_prices},
    params={"fast": [5, 10, 20], "slow": [20, 50, 100]},   # one thread per column, later
)
out["signal"].shape                   # (3, n_bars) - stacked along the parameter axis

g.visualize()                         # the plot below
```

![example strategy graph](assets/example_graph.png)

Type errors fail at graph-build time with messages that say what you probably meant:

```
DslTypeError: '&' requires bool operands, got float32 and float32 - did you mean a comparison?
```

## What the compiler decided

`explain_graph(g)` runs the whole pipeline and reports it — the schedule, each op's
dependency level, which buffer every result lives in, borrowed scratch, and values
that are computed but never read:

```
10 ops   2 input(s)   2 param(s)   3 output(s)

op  lvl  kind  name              outs                          scratch
--- ---- ----- ----------------- ----------------------------- -------------
0   0    expr  +                 T:f32v[0]                     -
1   1    expr  /                 T:f32v[1]                     -
2   2    call  stoch             O:f32v[0]                     f32v[0] f32v[2]
3   0    call  rolling_min_max   T:f32v[2] T:f32v[0]*          -
4   1    expr  -                 O:f32v[1]                     -
5   2    call  sma               T:f32v[2]                     -
6   3    expr  >                 T:b1v[0]                      -
7   2    call  sma               T:f32v[0]                     -
8   3    expr  >                 T:b1v[1]                      -
9   4    expr  &                 O:b1v[0]                      -

pools:
  output/bool/vector  x1
  output/float32/vector  x2
  temp/bool/vector  x2
  temp/float32/vector  x3

* produced but never read - scratch space, never copied back
```

Three `sma(med, slow)` calls were written; two survive, because the duplicate was
eliminated. Ten values share five temp buffers, because slots are recycled the
moment a value's live interval ends.

## Performance

The Numba CPU backend compiles a graph into one `@njit(parallel=True)` kernel,
`prange` over parameter sets. Measured against the same strategy written by hand
(`benchmarks/numba_cpu.py`, 16 cores, 8192 parameter sets × 4000 bars):

| | time | |
|---|---|---|
| numpy reference interpreter | 409 ms | the correctness oracle, not a fast path |
| hand-written njit, fused | 18.3 ms | |
| hand-written njit, *same structure as generated* | 21.1 ms | |
| **kernelsmith** | **20.8 ms** | 18× the interpreter, 1.14× hand-written |

Generated code matches hand-written code doing the same work — the residual gap is
one missing optimization, not codegen quality: kernelsmith currently emits five
elementwise passes where the fused version does two. Operator fusion is next.

`--threads` shows why that matters. The workload is memory-bandwidth-bound, so
parallelism saturates long before the core count:

```
  1 thread    72.9 ms   1.00x        past ~4 threads the cores are waiting
  2 threads   38.2 ms   1.91x        on memory, not computing - the only way
  4 threads   27.5 ms   2.66x        left to go faster is to move less data,
  8 threads   21.1 ms   3.46x        which is exactly what fusion does
 16 threads   20.4 ms   3.58x
```

## Architecture

Five layers, imports only point downward:

| Layer | Contents | Status |
|---|---|---|
| `dsl` | typed value nodes, operator overloading, call factories, `Graph` | ✅ working |
| `ir` | passes: topological scheduling ✅, CSE ✅, liveness ✅, buffer allocation ✅, fusion | 🔨 in progress |
| `backends` | CPU reference (test oracle) ✅, Numba parallel CPU ✅, CUDA, Triton | 🔨 in progress |
| `runtime` | memory planner, sessions, kernel cache | ⏳ |
| `backtest` | position sizers, portfolio sim, cost models — built *on* the compiler | ⏳ |

## Roadmap

- [x] Typed expression DSL (promotion lattice, build-time type errors)
- [x] Value-centric SSA graph with multi-output feature calls
- [x] Topological scheduling + dependency levels
- [x] Layered graph visualizer
- [x] CPU reference backend (every feature ships a numpy oracle; parity tests)
- [x] Common-subexpression elimination (commutative-aware, float-safe)
- [x] Liveness analysis and dead-value elimination
- [x] Numba parallel CPU backend — the graph compiles to one `@njit(parallel=True)`
      kernel, `prange` over parameter sets
- [ ] Operator fusion
- [x] Linear-scan buffer allocation (property-tested: live values never share a buffer)
- [ ] CUDA backend (numba) with coalesced `(T, F, P)` layout
- [ ] Benchmarks vs. multiprocessing CPU baseline
- [ ] Triton backend

## Provenance

This is a from-scratch redesign ("v2") of a CUDA backtesting compiler I built professionally at a proprietary trading firm, where v1 remains in production. v2 is a clean reimplementation that fixes v1's design mistakes — uncoalesced memory layout, codegen coupled to backtesting semantics, manual output-index bookkeeping. Example strategies in this repo are deliberately naive: the project is the compiler, not the alpha.

## Development

```bash
pip install -e .[dev]
pytest
python examples/sma_crossover.py          # opens the graph plot
```

## License

Apache-2.0
