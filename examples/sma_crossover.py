"""SMA-crossover demo: build a strategy graph, sweep parameters, plot the graph.

Run:  python examples/sma_crossover.py [savepath.png]
Without a savepath the graph plot opens in a window.
"""
import sys

import numpy as np

from kernelsmith import Graph
from kernelsmith.backends.cpu import CpuBackend
from kernelsmith.features import rolling_min_max, sma

# --- describe the strategy once -------------------------------------------
g = Graph()
close, opn = g.register_input("close"), g.register_input("open")
fast, slow = g.int_param("fast"), g.int_param("slow")

med = (close + opn) / 2
# note: sma(med, slow) is written twice, so the graph really does contain two
# identical calls - the CSE pass will collapse them into one
signal = (sma(med, fast) > sma(med, slow)) & (close > sma(med, slow))

low, high = rolling_min_max(close, slow)
range_ok = (high - low) > 0.5 # * close

g.register_output("signal", signal)
g.register_output("range_ok", range_ok)

print("topo order:", [op.name for op in g.build()])

# --- run it over a parameter sweep ----------------------------------------
rng = np.random.default_rng(7)
bars = 250
close_prices = (np.cumsum(rng.normal(0, 1, bars)) + 100.0).astype(np.float32)
open_prices = (close_prices + rng.normal(0, 0.2, bars)).astype(np.float32)

program = CpuBackend().compile(g)
print(f"after CSE: {len(g.ops)} ops -> {len(program.ops)} ops")

results = program.run(
    inputs={"close": close_prices, "open": open_prices},
    params={"fast": [5, 10, 20], "slow": [20, 50, 100]},
)

print("signal shape (params, bars):", results["signal"].shape)
for i, (f, s) in enumerate(zip([5, 10, 20], [20, 50, 100])):
    print(f"  fast={f:>3} slow={s:>4}  ->  {results['signal'][i].sum():>4} long bars")

# --- and look at it --------------------------------------------------------
g.visualize(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
