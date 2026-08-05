"""Emission calibration: initialise, then refine from the segmentation itself.

The rates are not fitted jointly with the state path. They are initialised from
a 2-component binomial mixture over all pooled bins, and then re-estimated from
the positions the current segmentation assigns to each class — annotate,
re-estimate, repeat. Two passes is enough in practice; the numbers move in the
third decimal after that.

This is deliberately not an unsupervised mixture on a derived signal: the
mixture only has to get the model into the right basin, and the segmentation —
which knows about durations and grammar — decides what "accessible" and
"protected" actually mean on this data.
"""
from __future__ import annotations

import numpy as np

from .emission import binomial_mixture


def initial_rates(cfg: dict, dataset) -> np.ndarray:
    classes = cfg["emission"]["classes"]
    cal = cfg["emission"]["calibration"]
    order = list(classes)
    if cal["mixture_init"]:
        K = np.concatenate([f.k_bin for f in dataset.fibers])
        N = np.concatenate([f.n_bin for f in dataset.fibers])
        high, low = binomial_mixture(K, N, start=tuple(cal["mixture_start"]),
                                     iters=int(cal["mixture_iters"]),
                                     tol=float(cal["mixture_tol"]))
    else:
        high, low = None, None
    rates = np.empty(len(order))
    for i, name in enumerate(order):
        init = classes[name].get("init", "high")
        if isinstance(init, (int, float)):
            rates[i] = float(init)
        elif high is None:
            raise ValueError(f"emission.classes.{name}.init is {init!r} but "
                             f"emission.calibration.mixture_init is false; give a number")
        else:
            rates[i] = high if init == "high" else low
    lo, hi = cfg["emission"]["calibration"]["clip"]
    return np.clip(rates, lo, hi)


def state_rates(model, dataset, rates: np.ndarray) -> dict:
    """Annotate every fiber and pool (methylated, callable) by DECODED state."""
    inv = {v: k for k, v in model.labels.items()}
    num = {name: 0.0 for name in inv.values()}
    den = {name: 0.0 for name in inv.values()}
    for f in dataset.fibers:
        lab = model.annotate(f, rates)
        for value, name in inv.items():
            m = lab == value
            if m.any():
                num[name] += float(f.meth_bp[m].sum())
                den[name] += float(f.call_bp[m].sum())
    return {"num": num, "den": den}


def calibrate(model, dataset, verbose: bool = True) -> tuple[np.ndarray, dict]:
    """-> (rates in `emission.classes` order, a report dict)."""
    cfg = model.cfg
    classes = cfg["emission"]["classes"]
    order = list(classes)
    lo, hi = cfg["emission"]["calibration"]["clip"]
    rates = initial_rates(cfg, dataset)
    report = {"class_order": order, "init": dict(zip(order, rates.tolist())), "refine": []}
    if verbose:
        print("  init: " + "  ".join(f"{n}={r:.4f}" for n, r in zip(order, rates)))

    for it in range(int(cfg["emission"]["calibration"]["refine_iters"])):
        pooled = state_rates(model, dataset, rates)
        num, den = pooled["num"], pooled["den"]
        new = rates.copy()
        empty, frozen = [], []
        for i, name in enumerate(order):
            cap = classes[name].get("max_refine")
            if cap is not None and it >= int(cap):
                frozen.append(name)         # this class has had its passes; leave it
                continue
            d = sum(den[s] for s in classes[name]["pool"])
            if d <= 0:
                empty.append(name)          # nothing was decoded into this class
                continue
            new[i] = np.clip(sum(num[s] for s in classes[name]["pool"]) / d, lo, hi)
        for i, name in enumerate(order):    # clamps applied after every rate moves
            target = classes[name].get("clamp_to")
            if target is not None:
                new[i] = min(new[i], new[order.index(target)])
        rates = new
        row = dict(zip(order, rates.tolist()))
        row["per_state"] = {s: (num[s] / den[s] if den[s] > 0 else float("nan"))
                            for s in num}
        row["empty_classes"] = empty
        row["frozen_classes"] = frozen
        report["refine"].append(row)
        if verbose:
            print(f"  refine {it + 1}: "
                  + "  ".join(f"{n}={r:.4f}{'*' if n in frozen else ''}"
                              for n, r in zip(order, rates))
                  + "   [per-state "
                  + " ".join(f"{s}={v:.4f}" for s, v in row["per_state"].items()) + "]")
            if frozen:
                print(f"    * held at max_refine: {frozen}")
            if empty:
                print(f"    note: no positions decoded into {empty}; kept previous rate")
    report["final"] = dict(zip(order, rates.tolist()))
    return rates, report
