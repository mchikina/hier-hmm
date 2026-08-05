"""Concordance must be positive when calls really do line up, and ~zero when the
only thing lining them up is the prior."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hier_hmm import HierHMM, load_config, prepare
from hier_hmm.calibrate import calibrate
from hier_hmm.metrics import (coherence, concordance, median_call_length,
                              permute_in_runs, top_share)
from hier_hmm.simulate import simulate


def test_raw_metrics_on_hand_made_rows():
    n = 200
    aligned = np.zeros((10, n), bool)
    aligned[:, 50:60] = True
    assert coherence(aligned) > 0.9
    assert top_share(aligned, frac=0.1) == 1.0
    assert median_call_length(aligned) == 10

    rng = np.random.default_rng(0)
    scattered = np.zeros((10, n), bool)
    for r in scattered:
        s = rng.integers(0, n - 10)
        r[s:s + 10] = True
    assert coherence(scattered) < 0.3
    assert top_share(scattered, frac=0.1) < 0.6


def test_permutation_preserves_the_count_inside_open_runs():
    cfg = load_config()
    d = simulate(cfg, n_fiber=6, n_pos=2000, seed=2)
    ds = prepare(d["m6a"], cfg)
    model = HierHMM(cfg)
    rates, _ = calibrate(model, ds, verbose=False)
    rng = np.random.default_rng(0)
    for f in ds.fibers:
        lab1 = model.level1(f, rates)
        perm = permute_in_runs(f, lab1, model, rng)
        assert perm.sum() == f.meth_bp.sum()          # count preserved overall
        assert not (perm & ~f.call_bp).any()          # only on callable positions
        outside = lab1.repeat(model.bin1) != model.labels[model.within]
        assert np.array_equal(perm[outside], f.meth_bp[outside])   # untouched elsewhere


def test_planted_footprints_beat_the_null():
    cfg = load_config()
    n_pos = 3000
    d = simulate(cfg, n_fiber=50, n_pos=n_pos, seed=4,
                 anchor=(n_pos // 2 - 250, n_pos // 2 + 250), anchor_frac=1.0,
                 footprint_frac=1.0, footprint_bp=30)
    ds = prepare(d["m6a"], cfg)
    model = HierHMM(cfg)
    rates, _ = calibrate(model, ds, verbose=False)
    r = concordance(model, ds, rates, n_perm=2, seed=0)
    assert r["excess_coh"] > 0.02, r
    assert r["excess_top"] > 0.05, r


def test_no_planted_signal_scores_near_zero():
    """Footprints scattered by the prior alone must not look concordant."""
    cfg = load_config()
    d = simulate(cfg, n_fiber=50, n_pos=3000, seed=6, anchor=None)
    ds = prepare(d["m6a"], cfg)
    model = HierHMM(cfg)
    rates, _ = calibrate(model, ds, verbose=False)
    r = concordance(model, ds, rates, n_perm=2, seed=0)
    assert abs(r["excess_coh"]) < 0.03, r


def test_max_refine_freezes_a_class():
    cfg = load_config(None, ["emission.calibration.refine_iters=3",
                             "emission.classes.footprint.max_refine=1"])
    d = simulate(cfg, n_fiber=30, n_pos=3000, seed=8)
    ds = prepare(d["m6a"], cfg)
    model = HierHMM(cfg)
    _, rep = calibrate(model, ds, verbose=False)
    fp = [row["footprint"] for row in rep["refine"]]
    assert fp[0] == fp[1] == fp[2], fp             # frozen after the first pass
    acc = [row["accessible"] for row in rep["refine"]]
    assert acc[0] != acc[1]                        # the others keep moving
    assert rep["refine"][1]["frozen_classes"] == ["footprint"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
