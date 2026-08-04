"""Synthetic data from the model's own priors.

Used by the tests (does the annotator recover a path it generated?) and by the
example, so the package can be run end to end without any private data. The
sampler walks the same phase-type chains the annotator builds, so a mismatch
between the two is a real bug and not a difference of conventions.
"""
from __future__ import annotations

import numpy as np

from .config import state_labels
from .phasetype import build_chain


def sample_path(chain, n: int, rng: np.random.Generator, start: int | None = None) -> np.ndarray:
    """Sample `n` steps of the expanded chain -> macro index per step."""
    T = np.exp(chain.log_T)
    p0 = np.exp(chain.log_start)
    s = int(start) if start is not None else int(rng.choice(len(p0), p=p0 / p0.sum()))
    out = np.empty(n, int)
    for t in range(n):
        out[t] = chain.macro[s]
        s = int(rng.choice(T.shape[0], p=T[s]))
    return out


DEFAULT_RATES = {"open": 0.62, "linker": 0.62, "nucleosome": 0.03, "footprint": 0.02}


def simulate(cfg: dict, n_fiber: int = 60, n_pos: int = 4000, rates: dict | None = None,
             callable_frac: float = 0.5, seed: int = 0,
             anchor: tuple | None = None, anchor_frac: float = 0.7,
             footprint_bp: int = 20, footprint_frac: float = 0.5,
             coverage_frac: float = 1.0, groups: int = 1) -> dict:
    """Generate a dataset in the package's input format.

    anchor          (start_bp, end_bp) window forced open on `anchor_frac` of
                    fibers, so the aggregate profile has something to show; None
                    leaves every fiber to the chain prior alone
    footprint_frac  fraction of anchored fibers carrying a `footprint_bp`
                    footprint at the middle of the anchor
    coverage_frac   mean fraction of the window each read spans (< 1 gives
                    ragged, contiguous read ends)
    """
    rng = np.random.default_rng(seed)
    labels = state_labels(cfg)
    rates = {**DEFAULT_RATES, **(rates or {})}
    missing = [s for s in labels if s not in rates]
    if missing:
        raise ValueError(f"no simulation rate given for state(s) {missing}")

    bin1 = int(cfg["binning"]["level1_bp"])
    n_bins = n_pos // bin1
    n_pos = n_bins * bin1
    l1 = build_chain(cfg["level1"]["states"], cfg["level1"]["transitions"], bin1,
                     cfg["level1"]["start"])
    l2 = build_chain(cfg["level2"]["states"], cfg["level2"]["transitions"],
                     int(cfg["binning"]["level2_bp"]), cfg["level2"]["start"])
    open_name = cfg["level2"]["applies_within"]
    call_name = cfg["level2"]["call_state"]

    truth = np.empty((n_fiber, n_pos), np.int8)
    m6a = np.full((n_fiber, n_pos), np.nan)
    group = np.empty(n_fiber, object)

    for i in range(n_fiber):
        macro = sample_path(l1, n_bins, rng)
        lab = np.repeat(np.array([labels[l1.macro_names[m]] for m in macro], np.int8), bin1)
        if anchor is not None and rng.random() < anchor_frac:
            a, z = int(anchor[0]), int(anchor[1])
            lab[a:z] = labels[open_name]
            if rng.random() < footprint_frac:
                mid = (a + z) // 2
                lab[mid - footprint_bp // 2: mid + footprint_bp // 2] = labels[call_name]
        else:
            # footprints inside whatever open runs the prior happened to make
            for a, z in _runs(lab == labels[open_name]):
                if z - a < 60:
                    continue
                mac2 = sample_path(l2, z - a, rng)
                hit = np.array([l2.macro_names[m] == call_name for m in mac2])
                lab[a:z][hit] = labels[call_name]
        truth[i] = lab

        p = np.array([rates[n] for n in labels])[lab]
        call = rng.random(n_pos) < callable_frac
        if coverage_frac < 1.0:
            span = max(int(coverage_frac * n_pos), 200)
            start = int(rng.integers(0, max(n_pos - span, 1)))
            keep = np.zeros(n_pos, bool)
            keep[start:start + span] = True
            call &= keep
            truth[i][~keep] = -1
        m6a[i][call] = (rng.random(n_pos)[call] < p[call]).astype(float)
        group[i] = f"g{i % groups}" if groups > 1 else "g0"

    return {"m6a": m6a, "truth": truth, "group": group.astype(str),
            "positions": np.arange(n_pos), "center": n_pos // 2,
            "rates": rates}


def _runs(mask: np.ndarray):
    d = np.concatenate([[0], mask.astype(np.int8), [0]])
    e = np.flatnonzero(np.diff(d) != 0)
    return list(zip(e[0::2], e[1::2]))
