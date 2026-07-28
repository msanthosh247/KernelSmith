"""SMA-crossover demo: build a strategy graph and plot it.

Run:  python examples/sma_crossover.py [savepath.png]
Without a savepath the plot opens in a window.
"""
import sys

from kernelsmith import CallFactory, Call, Graph, F4, I4, SCAL, VEC

# feature specs - signatures only, the kernels come in later phases
sma = CallFactory("sma", input_signature=[VEC(F4), SCAL(I4)], buffer_signature=[], output_signature=[VEC(F4)])
minmax = CallFactory(
    "minmax",
    input_signature=[VEC(F4), SCAL(I4)],
    buffer_signature=[VEC(F4)],
    output_signature=[VEC(F4), VEC(F4)],
)

g = Graph()
close = g.input("close")
opn = g.input("open")
fast = g.int_param("fast")
slow = g.int_param("slow")

med = (close + opn) / 2
s1 = sma(med, fast)
s2 = sma(med, slow)
signal = (s1 > s2) & (close > s2)
lo, hi = minmax(close, slow)
range_ok = (hi - lo) > 0.5

g.output("signal", signal)
g.output("range_ok", range_ok)

order = g.build()
print("topo order:", [op.factory.func_name if isinstance(op, Call) else op.operation for op in order])

savepath = sys.argv[1] if len(sys.argv) > 1 else None
g.visualize(savepath=savepath)
