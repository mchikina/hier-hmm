"""End-to-end run: prepare -> calibrate -> annotate -> results."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .calibrate import calibrate
from .config import as_yaml
from .model import NODATA, HierHMM


@dataclass
class Result:
    labels: np.ndarray                 # (n_fiber, n_bp) int8, NODATA outside the span
    posterior: np.ndarray              # (n_fiber, n_bp) float32, NaN where not searched
    label_map: dict                    # state name -> integer label
    rates: dict                        # emission class -> methylation rate
    calibration: dict                  # full calibration report
    coords: np.ndarray                 # per-bp x coordinate
    coord_label: str = "position (bp)"
    fiber_meta: dict = field(default_factory=dict)   # per-KEPT-fiber arrays
    cfg: dict = field(default_factory=dict)

    def fraction(self, *states) -> np.ndarray:
        """Per-position fraction of COVERING fibers in any of `states`."""
        want = [self.label_map[s] for s in states]
        cov = self.labels != NODATA
        hit = np.isin(self.labels, want)
        den = cov.sum(0)
        return np.where(den > 0, hit.sum(0) / np.maximum(den, 1), np.nan), den

    def state_fractions(self) -> dict:
        """Overall fraction of covered base pairs in each state."""
        cov = self.labels != NODATA
        total = int(cov.sum())
        return {name: float((self.labels == v).sum()) / total if total else float("nan")
                for name, v in self.label_map.items()}


def run(dataset, cfg: dict, verbose: bool = True) -> Result:
    if not dataset.fibers:
        raise ValueError("no fibers passed the data.min_bins_per_fiber / coverage filters")
    model = HierHMM(cfg)
    if verbose:
        for level, name in ((model.l1, "level1"), (model.l2, "level2")):
            dw = "  ".join(f"{s}: {level.chain.dwell_moments(s)[0] * _bin(cfg, name):.0f}"
                           f"+-{level.chain.dwell_moments(s)[1] * _bin(cfg, name):.0f} bp"
                           for s in level.macro_names)
            print(f"{name}: {level.chain.n_states} expanded states, "
                  f"decode={level.decode} tau={level.tau}   dwell {dw}")
        print(f"{len(dataset.fibers)} fibers, {dataset.n_bins} bins of "
              f"{dataset.bin_bp} bp")
    rates, report = calibrate(model, dataset, verbose=verbose)

    n_bp = dataset.n_bp
    L = np.full((len(dataset.fibers), n_bp), NODATA, np.int8)
    P = np.full((len(dataset.fibers), n_bp), np.nan, np.float32)
    for i, f in enumerate(dataset.fibers):
        lab, post = model.annotate(f, rates, want_post=True)
        L[i, f.lo_bp:f.hi_bp] = lab
        P[i, f.lo_bp:f.hi_bp] = post

    meta = {k: np.array([f.meta[k] for f in dataset.fibers]) for k in dataset.meta}
    meta["fiber_index"] = np.array([f.index for f in dataset.fibers])
    want = [model.labels[s] for s in cfg["plot"]["profile_states"]]
    meta["open_frac"] = np.array([
        float(np.isin(L[i][L[i] != NODATA], want).mean()) for i in range(len(L))])

    return Result(labels=L, posterior=P, label_map=model.labels,
                  rates=dict(zip(report["class_order"], rates.tolist())),
                  calibration=report, coords=dataset.coords,
                  coord_label=dataset.coord_label, fiber_meta=meta, cfg=cfg)


def _bin(cfg, level):
    return cfg["binning"]["level1_bp" if level == "level1" else "level2_bp"]


def save(result: Result, outdir: str, prefix: str = "annotation") -> dict:
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    arrays = dict(labels=result.labels, coords=result.coords,
                  label_names=np.array(list(result.label_map)),
                  label_values=np.array(list(result.label_map.values())))
    if result.cfg["output"]["save_posterior"]:
        arrays["posterior"] = result.posterior
    for k, v in result.fiber_meta.items():
        arrays[f"fiber_{k}"] = v
    paths["npz"] = os.path.join(outdir, f"{prefix}.npz")
    np.savez_compressed(paths["npz"], **arrays)

    paths["calibration"] = os.path.join(outdir, f"{prefix}.calibration.json")
    with open(paths["calibration"], "w") as fh:
        json.dump({"rates": result.rates, "report": result.calibration,
                   "state_fractions": result.state_fractions()}, fh, indent=2)
    if result.cfg["output"]["save_config"]:
        paths["config"] = os.path.join(outdir, f"{prefix}.config.yaml")
        with open(paths["config"], "w") as fh:
            fh.write(as_yaml(result.cfg))
    return paths
