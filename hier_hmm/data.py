"""Input contract and per-fiber preparation.

The model takes ONE (n_fiber, n_pos) float array:

    NaN  not callable at this position on this fiber — either the read does not
         cover it or the base is not assayable
    0    callable, unmethylated
    1    callable, methylated

An `.npz` may also carry, all optional:

    coverage   (n_fiber, n_pos) bool — read coverage, separate from callability.
               Only used to decide where a fiber's usable span starts and ends;
               without it the span is inferred from the first and last callable
               position, which is fine when callable sites are dense.
    positions  (n_pos,) genomic coordinate of each column
    center     scalar reference coordinate for the x axis
    <anything else of length n_fiber> is carried through as per-fiber metadata
               (group labels, cluster ids, read names, ...) and is available to
               `plot.order_by` and written back into the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Fiber:
    index: int
    lo_bin: int                       # first Level-1 bin of the usable span
    hi_bin: int                       # one past the last
    k_bin: np.ndarray                 # methylated count per Level-1 bin
    n_bin: np.ndarray                 # callable count per Level-1 bin
    meth_bp: np.ndarray               # per-bp methylated indicator over the span
    call_bp: np.ndarray               # per-bp callable indicator over the span
    meta: dict = field(default_factory=dict)

    @property
    def lo_bp(self) -> int:
        return self.lo_bin * self.bin_bp

    @property
    def hi_bp(self) -> int:
        return self.hi_bin * self.bin_bp

    bin_bp: int = 5


@dataclass
class Dataset:
    fibers: list
    n_pos: int
    n_bins: int
    bin_bp: int
    coords: np.ndarray                # per-bp x coordinate for plotting
    coord_label: str = "position (bp)"
    meta: dict = field(default_factory=dict)      # per-fiber arrays, full length
    extras: dict = field(default_factory=dict)    # positions / center / etc.

    @property
    def n_bp(self) -> int:
        return self.n_bins * self.bin_bp


def prepare(m6a: np.ndarray, cfg: dict, coverage: np.ndarray | None = None,
            meta: dict | None = None, extras: dict | None = None) -> Dataset:
    """Turn the raw array into per-fiber count tracks. No smoothing, no ratios."""
    m6a = np.asarray(m6a, float)
    if m6a.ndim != 2:
        raise ValueError(f"m6a must be (n_fiber, n_pos), got shape {m6a.shape}")
    n_fiber, n_pos = m6a.shape
    bin_bp = int(cfg["binning"]["level1_bp"])
    n_bins = n_pos // bin_bp
    if n_bins < 1:
        raise ValueError(f"n_pos={n_pos} is shorter than one bin ({bin_bp} bp)")

    dcfg = cfg["data"]
    thr = float(dcfg["meth_threshold"])
    if coverage is not None:
        coverage = np.asarray(coverage).astype(bool)
        if coverage.shape != m6a.shape:
            raise ValueError("coverage must have the same shape as m6a")

    fibers = []
    for i in range(n_fiber):
        row = m6a[i]
        call = np.isfinite(row)
        if coverage is not None:
            call &= coverage[i]
        meth = np.nan_to_num(row) > thr
        meth &= call

        cb = call[:n_bins * bin_bp].reshape(n_bins, bin_bp).sum(1).astype(float)
        kb = meth[:n_bins * bin_bp].reshape(n_bins, bin_bp).sum(1).astype(float)

        ok = cb >= float(dcfg["min_callable_per_bin"])
        if coverage is not None:
            covb = coverage[i][:n_bins * bin_bp].reshape(n_bins, bin_bp).mean(1)
            # Partially covered end bins otherwise produce read-end `open` and
            # `footprint` artefacts: they look accessible because most of the
            # bin simply has no data.
            ok &= covb >= float(dcfg["full_coverage_frac"])
        idx = np.flatnonzero(ok)
        if len(idx) < int(dcfg["min_bins_per_fiber"]):
            continue
        lo, hi = int(idx[0]), int(idx[-1]) + 1     # reads are contiguous
        fibers.append(Fiber(index=i, lo_bin=lo, hi_bin=hi,
                            k_bin=kb[lo:hi], n_bin=cb[lo:hi],
                            meth_bp=meth[lo * bin_bp:hi * bin_bp],
                            call_bp=call[lo * bin_bp:hi * bin_bp],
                            bin_bp=bin_bp,
                            meta={k: v[i] for k, v in (meta or {}).items()}))

    extras = dict(extras or {})
    positions = extras.get("positions")
    if positions is not None:
        positions = np.asarray(positions)
        centered = "center" in extras
        center = float(extras.get("center", 0.0))
        coords = positions[:n_bins * bin_bp].astype(float) - center
        label = "bp from center" if centered else "genomic position (bp)"
    else:
        coords = np.arange(n_bins * bin_bp, dtype=float)
        label = "position in window (bp)"
    return Dataset(fibers=fibers, n_pos=n_pos, n_bins=n_bins, bin_bp=bin_bp,
                   coords=coords, coord_label=label, meta=dict(meta or {}), extras=extras)


def load_npz(path: str, cfg: dict) -> Dataset:
    """Read the input contract described in this module's docstring."""
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    if "m6a" not in keys:
        raise KeyError(f"{path}: required array 'm6a' not found (has {sorted(keys)})")
    m6a = z["m6a"]
    coverage = z["coverage"] if "coverage" in keys else None
    n_fiber = m6a.shape[0]
    reserved = {"m6a", "coverage", "positions", "center"}
    meta = {k: z[k] for k in keys - reserved
            if z[k].ndim == 1 and len(z[k]) == n_fiber}
    extras = {k: z[k] for k in ("positions", "center") if k in keys}
    return prepare(m6a, cfg, coverage=coverage, meta=meta, extras=extras)
