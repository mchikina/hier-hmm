#!/usr/bin/env python3
"""Choose `emission.calibration.refine_iters` on YOUR data, the way the shipped
default was chosen.

Calibration is a self-referential loop — each pass re-estimates a rate from the
calls the previous rates produced — so the number of passes has to be judged by
the calls, not by whether the rates look settled. This runs the loop, records
the rates after every pass, and scores the Level-2 calls at each of them as
excess over a permutation null.

    python examples/refine_sweep.py data1.npz [data2.npz ...] [-o outdir]

Every input is one window. Rates are calibrated on all of them pooled (that is
what a real run does); scoring is per input, and within each input per group if
the npz carries a `group` array — so a difference between groups cannot
masquerade as agreement between molecules. Pass several windows and the spread
between them is your error bar; with one window you get a curve and no way to
tell a peak from noise.

Reads `excess_top` (density-invariant) over `excess_coh` when they disagree.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hier_hmm import load_config                                # noqa: E402
from hier_hmm.calibrate import initial_rates, state_rates       # noqa: E402
from hier_hmm.data import load_npz                              # noqa: E402
from hier_hmm.metrics import concordance                        # noqa: E402
from hier_hmm.model import HierHMM                              # noqa: E402


class _Pooled:
    """Just enough of a Dataset for the calibration loop: every window's fibers."""
    def __init__(self, datasets):
        self.fibers = [f for d in datasets for f in d.fibers]


def rate_trace(model, pooled, cfg, n_iter):
    """Rates after 0, 1, ... n_iter refinements — the calibration loop, recorded."""
    classes = cfg["emission"]["classes"]
    order = list(classes)
    lo, hi = cfg["emission"]["calibration"]["clip"]
    rates = initial_rates(cfg, pooled)
    trace = [rates.copy()]
    print("  pass 0 (mixture init): "
          + "  ".join(f"{n}={r:.4f}" for n, r in zip(order, rates)), flush=True)
    for it in range(n_iter):
        counts = state_rates(model, pooled, rates)
        num, den = counts["num"], counts["den"]
        new = rates.copy()
        for i, name in enumerate(order):
            d = sum(den[s] for s in classes[name]["pool"])
            if d > 0:
                new[i] = np.clip(sum(num[s] for s in classes[name]["pool"]) / d, lo, hi)
        for i, name in enumerate(order):          # clamps after every rate moves
            tgt = classes[name].get("clamp_to")
            if tgt is not None:
                new[i] = min(new[i], new[order.index(tgt)])
        rates = new
        trace.append(rates.copy())
        print(f"  pass {it + 1}: " + "  ".join(f"{n}={r:.4f}" for n, r in zip(order, rates)),
              flush=True)
    return order, trace


def score(model, datasets, names, rates, n_perm, group_key, label):
    t0 = time.time()
    rows = [concordance(model, d, rates, n_perm=n_perm, seed=0, group_key=group_key)
            for d in datasets]
    out = {k: float(np.nanmean([r[k] for r in rows]))
           for k in ("excess_coh", "excess_top", "raw_coh", "n_obs", "n_null", "median_len")}
    out["per_window"] = {n: {"excess_top": round(r["excess_top"], 4),
                             "excess_coh": round(r["excess_coh"], 4),
                             "per_group": r["per_group"]}
                         for n, r in zip(names, rows)}
    print(f"  {label:14s} excess_top={out['excess_top']:+.4f}  "
          f"excess_coh={out['excess_coh']:+.4f}  raw={out['raw_coh']:.4f}  "
          f"calls/molecule obs/null={out['n_obs']:.2f}/{out['n_null']:.2f}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return out


def plot(results, order, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in results]
    windows = list(results[0]["per_window"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for name in order:
        axes[0].plot(ks, [r["rates"][name] for r in results], "o-", label=name)
    axes[0].set_yscale("log"); axes[0].set_ylabel("methylation rate")
    axes[0].set_title("calibrated rates"); axes[0].legend(fontsize=8)

    for ax, key in ((axes[1], "excess_top"), (axes[2], "excess_coh")):
        for w in windows:
            ax.plot(ks, [r["per_window"][w][key] for r in results], "-", lw=0.9,
                    alpha=0.55, color="0.5")
        ax.plot(ks, [r[key] for r in results], "o-", color="#e31a1c", lw=2, label="mean")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(key)
        ax.set_title(f"{key} over null (grey = per window)")
        ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("refinement pass k"); ax.grid(alpha=0.3)
    fig.suptitle("Level-2 call concordance vs number of emission-rate refinements", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help=".npz windows in the hier_hmm input format")
    ap.add_argument("-o", "--outdir", default="refine_sweep_out")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--group-by", default="group")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model = HierHMM(cfg)
    names = [os.path.splitext(os.path.basename(p))[0] for p in a.inputs]
    datasets = [load_npz(p, cfg) for p in a.inputs]
    for n, d in zip(names, datasets):
        print(f"{n}: {len(d.fibers)} fibers, {d.n_bins} bins of {d.bin_bp} bp")
    gk = a.group_by if all(a.group_by in d.meta for d in datasets) else None
    if a.group_by and gk is None:
        print(f"note: '{a.group_by}' is not in every input; scoring ungrouped")

    print("\ncalibration trace (pooled over windows):")
    order, trace = rate_trace(model, _Pooled(datasets), cfg, a.max_iter)

    print(f"\nconcordance at each pass ({a.n_perm} permutation nulls):")
    results = [{"k": k, "rates": dict(zip(order, r.tolist())),
                **score(model, datasets, names, r, a.n_perm, gk, f"k={k}")}
               for k, r in enumerate(trace)]

    best = max(results, key=lambda r: r["excess_top"])
    print(f"\nbest mean excess_top at k={best['k']} ({best['excess_top']:+.4f}).")
    if len(datasets) > 1:
        wins = sum(1 for w in results[0]["per_window"]
                   if max(results, key=lambda r: r["per_window"][w]["excess_top"])["k"]
                   == best["k"])
        print(f"{wins}/{len(datasets)} windows peak at the same k. With few windows a "
              f"1-pass difference is easily noise — check the grey curves in the figure "
              f"before treating the peak as sharp.")
    else:
        print("Only one window: there is no spread to judge the peak against. Run more "
              "windows before treating a 1-pass difference as real.")

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "refine_sweep.json"), "w") as fh:
        json.dump({"class_order": order, "results": results}, fh, indent=2)
    print(f"wrote {os.path.join(a.outdir, 'refine_sweep.json')}")
    plot(results, order, os.path.join(a.outdir, "refine_sweep.png"))


if __name__ == "__main__":
    main()
