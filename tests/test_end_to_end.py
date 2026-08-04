"""Generate from the model's own priors and check the annotator gets it back."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hier_hmm import NODATA, load_config, prepare, run
from hier_hmm.data import load_npz
from hier_hmm.simulate import simulate


def _sim(cfg, **kw):
    kw.setdefault("n_fiber", 40)
    kw.setdefault("n_pos", 3000)
    kw.setdefault("seed", 1)
    return simulate(cfg, **kw)


def test_recovers_the_simulated_path():
    cfg = load_config()
    d = _sim(cfg)
    res = run(prepare(d["m6a"], cfg), cfg, verbose=False)
    truth, got = d["truth"], res.labels
    covered = got != NODATA
    # accessible-vs-protected is the call the emission actually supports; the
    # open/linker split is decided by durations and grammar, so score it coarsely
    acc = {res.label_map["open"], res.label_map["linker"], res.label_map["footprint"]}
    t_acc = np.isin(truth, list(acc))
    g_acc = np.isin(got, list(acc))
    agree = (t_acc == g_acc)[covered].mean()
    assert agree > 0.85, agree
    nuc = res.label_map["nucleosome"]
    exact = (truth == got)[covered].mean()
    assert exact > 0.70, exact
    assert (got == nuc).mean() > 0.3      # it must not collapse to one state


def test_calibration_recovers_the_simulated_rates():
    cfg = load_config()
    d = _sim(cfg)
    res = run(prepare(d["m6a"], cfg), cfg, verbose=False)
    true = d["rates"]
    assert abs(res.rates["accessible"] - true["open"]) < 0.06, res.rates
    assert abs(res.rates["protected"] - true["nucleosome"]) < 0.03, res.rates
    assert res.rates["protected"] < res.rates["accessible"]


def test_footprints_land_on_the_planted_site():
    cfg = load_config()
    n_pos = 3000
    a, z = n_pos // 2 - 250, n_pos // 2 + 250
    d = simulate(cfg, n_fiber=60, n_pos=n_pos, seed=3, anchor=(a, z),
                 anchor_frac=1.0, footprint_frac=1.0, footprint_bp=30)
    res = run(prepare(d["m6a"], cfg), cfg, verbose=False)
    fp = res.labels == res.label_map["footprint"]
    mid = n_pos // 2
    inside = fp[:, mid - 40:mid + 40].sum()
    outside = fp.sum() - inside
    assert inside > 0, "no footprint called at the planted site"
    # 80 bp of window against ~2900 bp of everything else
    assert inside / max(outside, 1) > 1.0, (inside, outside)


def test_no_call_is_shorter_than_the_configured_floor():
    cfg = load_config()
    d = _sim(cfg, n_fiber=30, seed=5)
    res = run(prepare(d["m6a"], cfg), cfg, verbose=False)
    floor = cfg["level2"]["min_call_bp"]
    fp = res.labels == res.label_map["footprint"]
    lengths = []
    for row in fp:
        pad = np.concatenate([[0], row.astype(np.int8), [0]])
        e = np.flatnonzero(np.diff(pad) != 0)
        lengths += list(e[1::2] - e[0::2])
    assert not lengths or min(lengths) >= floor, min(lengths)


def test_nodata_outside_the_covered_span():
    cfg = load_config()
    d = simulate(cfg, n_fiber=30, n_pos=3000, seed=7, coverage_frac=0.5)
    ds = prepare(d["m6a"], cfg)
    res = run(ds, cfg, verbose=False)
    assert (res.labels == NODATA).any()
    for i, f in enumerate(ds.fibers):
        row = res.labels[i]
        assert (row[:f.lo_bp] == NODATA).all()
        assert (row[f.hi_bp:] == NODATA).all()
        assert (row[f.lo_bp:f.hi_bp] != NODATA).all()


def test_bin_size_is_a_config_knob_only():
    """A different Level-1 bin must still run and still find the accessible core."""
    cfg = load_config(None, ["binning.level1_bp=10"])
    d = _sim(cfg, seed=11)
    res = run(prepare(d["m6a"], cfg), cfg, verbose=False)
    acc = [res.label_map[s] for s in ("open", "linker", "footprint")]
    covered = res.labels != NODATA
    agree = (np.isin(d["truth"], acc) == np.isin(res.labels, acc))[covered].mean()
    assert agree > 0.80, agree


def test_npz_roundtrip():
    import tempfile
    cfg = load_config()
    d = _sim(cfg, n_fiber=20, seed=13)
    out = os.path.join(tempfile.mkdtemp(), "in.npz")
    np.savez_compressed(out, m6a=d["m6a"], group=d["group"],
                        positions=d["positions"], center=d["center"])
    ds = load_npz(out, cfg)
    assert "group" in ds.meta and len(ds.fibers) == 20
    assert ds.coords[0] == d["positions"][0] - d["center"]
    res = run(ds, cfg, verbose=False)
    assert "group" in res.fiber_meta and "open_frac" in res.fiber_meta


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
