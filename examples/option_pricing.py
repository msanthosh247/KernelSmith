"""Option prices from a block-bootstrapped price history.

The idea: the future looks like the past, resampled. Take a history of daily
closes, cut its returns into blocks of ``block`` consecutive days, and glue
randomly chosen blocks together into many possible futures. Price European
calls and puts from where those futures end up.

What each part does, and where it runs:

  1. simulate      kernelsmith, one parameter set per simulated path. The
                   history is a *table* - read by every path, stored once -
                   and the graph's time axis is the future: n_bars = the
                   longest expiry. Each path is read at every expiry with
                   path[e - 1], so the long path never leaves the device.
  2. correct       numpy. Resampled returns carry the history's drift, so
                   the raw average is an expectation under the historical
                   measure, not a price. The empirical martingale correction
                   (Duan & Simonato, 1998) rescales each expiry's prices so
                   their mean is the forward S0 * exp(r * tau): a pure drift
                   change that keeps the resampled shape - volatility, skew,
                   fat tails. With it, put-call parity holds exactly.
  3. price         numpy. Sort each expiry's prices once; with prefix sums,
                   every strike's call and put is one binary search.
  4. smile         Black-Scholes implied volatility of each price.

Two things the prices are not: the correction is one risk-neutral measure out
of many (resampled returns do not make a complete market), and no volatility
risk premium is added - market options usually trade above realised
volatility, so these prices sit below them.

Blocks matter because volatility clusters, and in equities it rises after
falls. Drawing single days (block = 1) keeps each day's return but destroys
that link: every path mixes calm and wild days at random, and by a month out
the smile has collapsed to a flat line at the realised volatility - the
Black-Scholes world. Blocks of 20 days keep falls and the turbulence that
follows them together, and the smile keeps the downward skew equity options
show. The history is a simulated 20-year GJR-GARCH series, a standard model of
exactly that behaviour.

How much a bootstrap can say is bounded by its history: a month-long path is
little more than one resampled month, and 20 years hold ~250 of them. The
far wings, and the short expiries whose tails come from a few extreme days,
stay noisy however many paths are drawn - more paths do not add history.

Run:  python examples/option_pricing.py
      python examples/option_pricing.py --paths 50000 --backend numba
"""
import argparse
import importlib.util
import math
import pathlib
import time
import warnings

import numpy as np

from kernelsmith import Graph
from kernelsmith.features import bootstrap_path

warnings.filterwarnings("ignore", message=".*Grid size")

HERE = pathlib.Path(__file__).parent
TRADING_DAYS = 252
EXPIRIES = (5, 21, 63)              # trading days: one week, one month, one quarter
BLOCKS = (1, 20)
RATE = 0.04                         # continuously compounded risk-free rate
SEED = 2026
MONEYNESS = np.linspace(0.8, 1.2, 17)


# ---- data -----------------------------------------------------------------------

def garch_history(days=5040, annual_vol=0.20, drift=0.06, alpha=0.03, gamma=0.10,
                  beta=0.90, seed=7):
    """Daily closes from a GJR-GARCH(1,1), the standard model of equity volatility:
    it clusters, and a fall raises it more than a rise does (the leverage
    effect, ``gamma``) - which is what gives equity smiles their downward skew."""
    rng = np.random.default_rng(seed)
    target = annual_vol ** 2 / TRADING_DAYS
    omega = target * (1 - alpha - gamma / 2 - beta)
    variance, shock = target, 0.0
    returns = np.empty(days)
    for day in range(days):
        variance = omega + (alpha + gamma * (shock < 0)) * shock ** 2 + beta * variance
        shock = math.sqrt(variance) * rng.standard_normal()
        returns[day] = drift / TRADING_DAYS - variance / 2 + shock
    return (100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))).astype(np.float32)


# ---- the graph ------------------------------------------------------------------

def simulation_graph():
    g = Graph()
    history = g.register_table("close")             # the past: any length, stored once
    path = bootstrap_path(history, g.int_param("block"), g.int_param("path"), SEED)
    for expiry in EXPIRIES:
        g.register_output(f"S_{expiry}d", path[expiry - 1])
    return g


def simulate(backend, history, n_paths):
    """Every (block, path) pair is one parameter set - one GPU thread."""
    params = {
        "block": np.repeat(np.array(BLOCKS, dtype=np.int32), n_paths),
        "path": np.tile(np.arange(n_paths, dtype=np.int32), len(BLOCKS)),
    }
    program = backend().compile(simulation_graph())
    run = lambda: program.run({"close": history}, params, n_bars=max(EXPIRIES))   # noqa: E731
    run()                                            # compile and warm up
    start = time.perf_counter()
    result = run()
    elapsed = time.perf_counter() - start
    return result, params, elapsed


# ---- pricing (numpy: these reduce across paths) -----------------------------------

def martingale_correct(prices, forward):
    """Scale so the sample mean is exactly the forward: a pure drift change."""
    return prices * (forward / prices.mean())


def price_strikes(prices, strikes, discount):
    """Every strike from one sort: E[(S - K)+] and E[(K - S)+] by prefix sums."""
    ordered = np.sort(prices)
    n = ordered.size
    prefix = np.concatenate([[0.0], np.cumsum(ordered)])      # prefix[k] = sum of the k smallest
    below = np.searchsorted(ordered, strikes, side="right")   # how many finish at or below K
    calls = ((prefix[-1] - prefix[below]) - strikes * (n - below)) / n
    puts = (strikes * below - prefix[below]) / n
    return discount * calls, discount * puts


def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def black_scholes_call(spot, strikes, tau, vol):
    root = vol * math.sqrt(tau)
    d1 = (np.log(spot / strikes) + (RATE + 0.5 * vol ** 2) * tau) / root
    return spot * _norm_cdf(d1) - strikes * math.exp(-RATE * tau) * _norm_cdf(d1 - root)


def implied_vol(calls, spot, strikes, tau):
    """Bisection on Black-Scholes, vectorised over strikes; NaN outside its bounds."""
    low, high = np.full(strikes.shape, 1e-4), np.full(strikes.shape, 3.0)
    for _ in range(80):
        middle = 0.5 * (low + high)
        above = black_scholes_call(spot, strikes, tau, middle) > calls
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)
    intrinsic = np.maximum(spot - strikes * math.exp(-RATE * tau), 0.0)
    return np.where(calls > intrinsic + 1e-10, 0.5 * (low + high), np.nan)


def surface(result, params, spot):
    """(block, expiry) -> dict of prices, implied vols and the parity error."""
    table = {}
    for block in BLOCKS:
        rows = params["block"] == block
        for expiry in EXPIRIES:
            tau = expiry / TRADING_DAYS
            forward, discount = spot * math.exp(RATE * tau), math.exp(-RATE * tau)
            strikes = spot * MONEYNESS
            raw = result[f"S_{expiry}d"][rows].astype(np.float64)
            corrected = martingale_correct(raw, forward)
            entry = {"strikes": strikes, "raw_mean": raw.mean(), "forward": forward}
            for label, prices in (("raw", raw), ("corrected", corrected)):
                calls, puts = price_strikes(prices, strikes, discount)
                entry[label] = {
                    "calls": calls, "puts": puts,
                    "iv": implied_vol(calls, spot, strikes, tau),
                    # C - P must equal the discounted forward minus strike
                    "parity": np.abs(calls - puts - discount * (forward - strikes)).max(),
                }
            table[block, expiry] = entry
    return table


# ---- reporting -------------------------------------------------------------------

def report(table, spot, history):
    realised = np.diff(np.log(history.astype(np.float64))).std() * math.sqrt(TRADING_DAYS)
    print(f"history: {history.size} closes, last {spot:.2f}, realised vol {realised:.1%}")

    block, expiry = BLOCKS[-1], 21
    entry = table[block, expiry]
    print(f"\n{expiry}-day options, block {block}, martingale-corrected (r = {RATE:.0%})")
    print(f"  {'strike':>8} {'call':>8} {'put':>8} {'implied vol':>12}")
    for strike, call, put, vol in zip(entry["strikes"], entry["corrected"]["calls"],
                                      entry["corrected"]["puts"], entry["corrected"]["iv"]):
        # no simulated path reaches that far: a resampled history has bounded
        # support, so the far wing has no time value to imply a volatility from
        shown = "-" if np.isnan(vol) else f"{vol:.1%}"
        print(f"  {strike:8.2f} {call:8.3f} {put:8.3f} {shown:>12}")

    print("\nput-call parity  |C - P - e^-rt (F - K)|, worst strike:")
    for (block, expiry), entry in table.items():
        drift = entry["raw_mean"] / entry["forward"] - 1
        print(f"  block {block:>2}, {expiry:>2} days:  corrected {entry['corrected']['parity']:.1e}"
              f"   raw {entry['raw']['parity']:.3f}  (raw mean {drift:+.2%} off the forward)")


def plot(table, spot, history, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    realised = np.diff(np.log(history.astype(np.float64))).std() * math.sqrt(TRADING_DAYS)
    fig, axes = plt.subplots(1, len(EXPIRIES), figsize=(5 * len(EXPIRIES), 4.2), sharey=True)
    for ax, expiry in zip(axes, EXPIRIES):
        for block, color in zip(BLOCKS, ("tab:gray", "tab:blue")):
            entry = table[block, expiry]
            ax.plot(MONEYNESS, entry["corrected"]["iv"], "o-", color=color, label=f"block {block}")
        entry = table[BLOCKS[-1], expiry]
        ax.plot(MONEYNESS, entry["raw"]["iv"], "--", color="tab:red",
                label=f"block {BLOCKS[-1]}, no correction")
        ax.axhline(realised, color="black", linewidth=0.8, linestyle=":", label="realised vol")
        ax.set_title(f"{expiry} trading days")
        ax.set_xlabel("strike / spot")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("implied volatility")
    axes[0].legend(fontsize=8)
    fig.suptitle("Implied volatility from a block-bootstrapped history")
    fig.tight_layout()
    out_dir = pathlib.Path(out_dir)
    fig.savefig(out_dir / "option_pricing.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_dir / 'option_pricing.png'}")


# ---- main ------------------------------------------------------------------------

def available_backends(choice):
    from kernelsmith.backends.numba_cpu import NumbaCPU_Backend
    backends = {"numba": NumbaCPU_Backend}
    from kernelsmith.availability import cuda_usable
    if cuda_usable():
        from kernelsmith.backends.cuda import CudaBackend
        backends["cuda"] = CudaBackend
    if choice == "cpu":
        from kernelsmith.backends.cpu import CPUBackend
        return {"cpu": CPUBackend}
    if choice != "all":
        if choice not in backends:
            raise SystemExit(f"backend '{choice}' is not available here")
        return {choice: backends[choice]}
    return backends


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=int, default=200_000, help="simulated paths per block length")
    parser.add_argument("--backend", default="all", choices=["all", "numba", "cuda", "cpu"],
                        help="'all' runs every available generating backend and prices from the last")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-dir", default=str(HERE), help="where the plots are written")
    args = parser.parse_args(argv)

    history = garch_history()
    spot = float(history[-1])
    threads = args.paths * len(BLOCKS)
    print(f"{args.paths} paths x {len(BLOCKS)} block lengths = {threads} parameter sets,"
          f" {max(EXPIRIES)} steps each\n")

    result = params = None
    for name, backend in available_backends(args.backend).items():
        result, params, elapsed = simulate(backend, history, args.paths)
        steps = threads * max(EXPIRIES)
        print(f"  {name:<6} {elapsed * 1e3:8.1f} ms   {steps / elapsed / 1e9:6.2f} G path-steps/s")

    table = surface(result, params, spot)
    print()
    report(table, spot, history)
    if not args.no_plot and importlib.util.find_spec("matplotlib") is not None:
        plot(table, spot, history, args.out_dir)
    return table


if __name__ == "__main__":
    main()
