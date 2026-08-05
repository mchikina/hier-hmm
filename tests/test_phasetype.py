"""The length distributions must be the ones the config asks for — at any bin size."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hier_hmm.config import load_config
from hier_hmm.phasetype import (build_chain, sd_for_substates, sd_range,
                                substates_for_sd)


def _chain(states, trans, bin_bp=1, start="uniform"):
    return build_chain(states, trans, bin_bp, start)


def _level(cfg, level, bin_bp=None):
    bp = bin_bp or cfg["binning"]["level1_bp" if level == "level1" else "level2_bp"]
    return build_chain(cfg[level]["states"], cfg[level]["transitions"], bp,
                       cfg[level]["start"]), bp


def test_mean_matches_config():
    cfg = load_config()
    for level in ("level1", "level2"):
        ch, bp = _level(cfg, level)
        for name, spec in cfg[level]["states"].items():
            mean_bins, _ = ch.dwell_moments(name)
            assert abs(mean_bins * bp - spec["mean_bp"]) < 1e-6, (level, name)


def test_sd_matches_config():
    """sd_bp is honoured up to the integer rounding of the substate count."""
    cfg = load_config()
    for level in ("level1", "level2"):
        ch, bp = _level(cfg, level)
        for name, spec in cfg[level]["states"].items():
            if spec.get("sd_bp") is None:
                continue
            _, sd_bins = ch.dwell_moments(name)
            got, want = sd_bins * bp, spec["sd_bp"]
            assert abs(got - want) < 0.06 * want, (level, name, got, want)


def test_length_distribution_survives_a_bin_size_change():
    """The point of sd_bp: mean AND SD stay put when the bin size moves.

    With `k` pinned instead, the nucleosome SD runs 26.9 / 21.2 / 10.2 bp at
    1 / 5 / 10 bp bins — the state silently sharpens as bins get coarser.
    """
    cfg = load_config()
    for name in ("open", "linker", "nucleosome"):
        want_mean = cfg["level1"]["states"][name]["mean_bp"]
        want_sd = cfg["level1"]["states"][name]["sd_bp"]
        for bp in (1, 2, 5, 10):
            ch, _ = _level(cfg, "level1", bp)
            mean_bins, sd_bins = ch.dwell_moments(name)
            assert abs(mean_bins * bp - want_mean) < 1e-6, (name, bp)
            assert abs(sd_bins * bp - want_sd) < 0.15 * want_sd, \
                (name, bp, sd_bins * bp, want_sd)

    # and the pinned-k spelling really does drift, which is why sd_bp exists
    pinned = {"nucleosome": {"mean_bp": 129.0, "k": 6, "min_bp": 60.0,
                             "emission_class": "protected"}}
    sds = []
    for bp in (1, 5, 10):
        ch = _chain(pinned, {"nucleosome": {"nucleosome": 1.0}}, bp, "nucleosome")
        sds.append(ch.dwell_moments("nucleosome")[1] * bp)
    assert sds[0] > 2 * sds[-1], sds


def test_substate_count_solves_the_sd_relation():
    for tail, sd, bp in [(212.0, 101.0, 5), (69.0, 21.2, 5), (7.0, 3.1, 1)]:
        k = substates_for_sd(tail, sd, bp)
        assert abs(sd_for_substates(tail, k, bp) - sd) < 0.05 * sd, (tail, sd, bp, k)


def test_sd_too_sharp_for_the_bin_is_refused():
    """A 3 bp SD cannot be built out of 10 bp substates; say so, don't widen it."""
    states = {"a": {"mean_bp": 129.0, "sd_bp": 3.0, "emission_class": "x"},
              "b": {"mean_bp": 50.0, "sd_bp": 40.0, "emission_class": "x"}}
    trans = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    lo, hi = sd_range(129.0, 10.0)
    assert lo > 3.0
    try:
        _chain(states, trans, bin_bp=10)
    except ValueError as e:
        assert "not reachable" in str(e) and "sharpest available" in str(e)
    else:
        raise AssertionError("expected a ValueError for an unreachable SD")
    _chain(states, trans, bin_bp=1)               # fine at a finer bin


def test_sd_wider_than_one_substate_warns_and_clamps():
    import warnings
    states = {"a": {"mean_bp": 50.0, "sd_bp": 500.0, "emission_class": "x"},
              "b": {"mean_bp": 50.0, "sd_bp": 40.0, "emission_class": "x"}}
    trans = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = _chain(states, trans, bin_bp=1)
    assert any("flat hazard" in str(x.message) for x in w), [str(x.message) for x in w]
    assert int((ch.macro == ch.macro_index("a")).sum()) == 1


def test_state_too_short_for_the_bin_is_refused():
    """15 bp mean over an 8 bp floor is 1.4 bins at a 5 bp bin — no room."""
    cfg = load_config()
    try:
        _level(cfg, "level2", 5)
    except ValueError as e:
        assert "under one" in str(e) or "not reachable" in str(e), str(e)
    else:
        raise AssertionError("expected a ValueError for a state with no room")


def test_min_bp_is_a_hard_floor():
    states = {"a": {"mean_bp": 20.0, "sd_bp": 6.0, "min_bp": 10.0, "emission_class": "x"},
              "b": {"mean_bp": 5.0, "sd_bp": 4.0, "emission_class": "x"}}
    trans = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    ch = _chain(states, trans)
    a_states = np.flatnonzero(ch.macro == ch.macro_index("a"))
    assert np.allclose(np.exp(ch.log_T[a_states[:10], a_states[:10]]), 0.0)
    assert np.exp(ch.log_T[a_states[10], a_states[10]]) > 0.0


def test_mean_and_floor_by_simulation():
    cfg = load_config()
    ch, _ = _level(cfg, "level1", 1)
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
    spec = cfg["level1"]["states"]["nucleosome"]
    assert abs(np.mean(lengths) - spec["mean_bp"]) < 0.15 * spec["mean_bp"]
    assert min(lengths) >= spec["min_bp"]
    assert abs(np.std(lengths) - spec["sd_bp"]) < 0.35 * spec["sd_bp"]


def test_transition_rows_are_stochastic():
    cfg = load_config()
    for level in ("level1", "level2"):
        ch, _ = _level(cfg, level)
        assert np.allclose(np.exp(ch.log_T).sum(1), 1.0)
        assert abs(np.exp(ch.log_start).sum() - 1.0) < 1e-12


def test_grammar_zeros_are_hard():
    """open -> nucleosome is the only exit from open; open -> linker must be 0."""
    cfg = load_config()
    ch, _ = _level(cfg, "level1")
    T = np.exp(ch.log_T)
    o = np.flatnonzero(ch.macro == ch.macro_index("open"))
    lnk = np.flatnonzero(ch.macro == ch.macro_index("linker"))
    assert T[np.ix_(o, lnk)].sum() == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
