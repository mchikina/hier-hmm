#!/usr/bin/env python3
"""End-to-end example on synthetic data: no private data needed.

    python examples/quickstart.py [outdir]

Simulates a locus from the model's own priors — a shared accessible core on
most molecules, a footprint in the middle of some of them, ragged read ends —
then annotates it, prints how well the annotation matches the known truth, and
writes the standard figure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hier_hmm import NODATA, load_config, prepare, run, save          # noqa: E402
from hier_hmm.plotting import plot_locus                              # noqa: E402
from hier_hmm.simulate import simulate                                # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "example_out"
N_POS = 4000

cfg = load_config()                       # packaged defaults; pass a path to override
# Two groups so the profile panel has something to compare, and ragged read ends
# so the no-data handling is exercised.
d = simulate(cfg, n_fiber=80, n_pos=N_POS, seed=0, groups=2, coverage_frac=0.85,
             anchor=(N_POS // 2 - 250, N_POS // 2 + 250), anchor_frac=0.75,
             footprint_frac=0.6, footprint_bp=24)

ds = prepare(d["m6a"], cfg, meta={"group": d["group"]},
             extras={"positions": d["positions"], "center": d["center"]})
res = run(ds, cfg)

os.makedirs(OUT, exist_ok=True)
paths = save(res, OUT, prefix="example")
paths["figure"] = plot_locus(res, os.path.join(OUT, "example.png"), group_key="group",
                             title="synthetic locus (truth known)")

# How close is the annotation to the path that generated the data? The
# accessible-vs-protected split is what the emission actually supports; the
# open/linker split is decided by the duration priors and the grammar.
truth, got = d["truth"], res.labels
covered = got != NODATA
acc = [res.label_map[s] for s in ("open", "linker", "footprint")]
print(f"\naccessible/protected agreement : "
      f"{(np.isin(truth, acc) == np.isin(got, acc))[covered].mean():.3f}")
print(f"exact 4-state agreement        : {(truth == got)[covered].mean():.3f}")
print("simulated rates                :", d["rates"])
print("calibrated rates               :", {k: round(v, 4) for k, v in res.rates.items()})
for k, v in paths.items():
    print(f"wrote {k}: {v}")
