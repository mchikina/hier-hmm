"""The duration priors must be the ones the config asks for."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hier_hmm.config import load_config
from hier_hmm.phasetype import build_chain


def _chain(states, trans, bin_bp=1, start="uniform"):
    return build_chain(states, trans, bin_bp, start)


def test_dwell_mean_matches_config():
    cfg = load_config()
    for level, bin_key in (("level1", "level1_bp"), ("level2", "level2_bp")):
        bp = cfg["binning"][bin_key]
        ch = build_chain(cfg[level]["states"], cfg[level]["transitions"], bp,
                         cfg[level]["start"])
        for name, spec in cfg[level]["states"].items():
            mean_bins, _ = ch.dwell_moments(name)
            # exact up to the bin grid: min_bp is rounded to a whole number of bins
            assert abs(mean_bins * bp - spec["mean_bp"]) < bp + 1e-9, (
                level, name, mean_bins * bp, spec["mean_bp"])


def test_dwell_mean_by_simulation():
    """Walk the built matrix and compare empirical dwell to the config mean."""
    cfg = load_config()
    ch = build_chain(cfg["level1"]["states"], cfg["level1"]["transitions"], 1, "uniform")
    T = np.exp(ch.log_T)
    rng = np.random.default_rng(0)
    s = int(np.flatnonzero(ch.macro == ch.macro_index("nucleosome"))[0])
    lengths = []
    for _ in range(400):
        cur, n = s, 0
        while ch.macro[cur] == ch.macro_index("nucleosome"):
            n += 1
            cur = int(rng.choice(len(T), p=T[cur]))
        lengths.append(n)
    mean = np.mean(lengths)
    want = cfg["level1"]["states"]["nucleosome"]["mean_bp"]
    assert abs(mean - want) < 0.15 * want, (mean, want)
    assert min(lengths) >= cfg["level1"]["states"]["nucleosome"]["min_bp"], min(lengths)


def test_min_bp_is_a_hard_floor():
    states = {"a": {"mean_bp": 20.0, "k": 2, "min_bp": 10.0, "emission_class": "x"},
              "b": {"mean_bp": 5.0, "k": 1, "emission_class": "x"}}
    trans = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    ch = _chain(states, trans)
    a_states = np.flatnonzero(ch.macro == ch.macro_index("a"))
    # the first 10 stages must advance with probability 1
    assert np.allclose(np.exp(ch.log_T[a_states[:10], a_states[:10]]), 0.0)
    assert np.exp(ch.log_T[a_states[10], a_states[10]]) > 0.0


def test_refuses_impossible_advance_probability():
    """A bin size that cannot honour the mean must raise, not silently clip."""
    states = {"a": {"mean_bp": 15.0, "k": 3, "min_bp": 8.0, "emission_class": "x"},
              "b": {"mean_bp": 50.0, "k": 1, "emission_class": "x"}}
    trans = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    _chain(states, trans, bin_bp=1)                     # fine at 1 bp
    try:
        _chain(states, trans, bin_bp=5)                 # 3 / ((15-8)/5) = 2.1 > 0.95
    except ValueError as e:
        assert "advance probability" in str(e)
    else:
        raise AssertionError("expected a ValueError for an unattainable dwell mean")


def test_transition_rows_are_stochastic():
    cfg = load_config()
    for level in ("level1", "level2"):
        bp = cfg["binning"]["level1_bp" if level == "level1" else "level2_bp"]
        ch = build_chain(cfg[level]["states"], cfg[level]["transitions"], bp,
                         cfg[level]["start"])
        rows = np.exp(ch.log_T).sum(1)
        assert np.allclose(rows, 1.0), rows
        assert abs(np.exp(ch.log_start).sum() - 1.0) < 1e-12


def test_grammar_zeros_are_hard():
    """open -> nucleosome is the only exit from open; open -> linker must be 0."""
    cfg = load_config()
    ch = build_chain(cfg["level1"]["states"], cfg["level1"]["transitions"],
                     cfg["binning"]["level1_bp"], "uniform")
    T = np.exp(ch.log_T)
    o = np.flatnonzero(ch.macro == ch.macro_index("open"))
    lnk = np.flatnonzero(ch.macro == ch.macro_index("linker"))
    assert T[np.ix_(o, lnk)].sum() == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
