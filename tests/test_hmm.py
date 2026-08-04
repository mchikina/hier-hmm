"""The DP must agree with brute-force enumeration of every state path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import numpy as np

from hier_hmm import hmm


def _tiny(seed=0, n=6, S=3):
    rng = np.random.default_rng(seed)
    T = rng.random((S, S)) + 0.05
    T /= T.sum(1, keepdims=True)
    p0 = rng.random(S) + 0.05
    p0 /= p0.sum()
    log_E = np.log(rng.random((n, S)) + 0.05)
    return log_E, np.log(T), np.log(p0)


def _enumerate(log_E, log_T, log_start):
    n, S = log_E.shape
    best, best_path, tot = -np.inf, None, []
    post = np.zeros((n, S))
    for path in itertools.product(range(S), repeat=n):
        lp = log_start[path[0]] + log_E[0, path[0]]
        for t in range(1, n):
            lp += log_T[path[t - 1], path[t]] + log_E[t, path[t]]
        tot.append(lp)
        if lp > best:
            best, best_path = lp, path
        for t, s in enumerate(path):
            post[t, s] += np.exp(lp)
    Z = np.exp(np.array(tot)).sum()
    return best, np.array(best_path), post / Z, np.log(Z)


def test_viterbi_matches_brute_force():
    for seed in range(5):
        log_E, log_T, log_start = _tiny(seed)
        _, want, _, _ = _enumerate(log_E, log_T, log_start)
        got = hmm.viterbi(log_E, log_T, log_start)
        assert np.array_equal(got, want), (seed, got, want)


def test_posteriors_and_likelihood_match_brute_force():
    for seed in range(5):
        log_E, log_T, log_start = _tiny(seed)
        _, _, want_post, want_Z = _enumerate(log_E, log_T, log_start)
        got = np.exp(hmm.state_posteriors(log_E, log_T, log_start))
        assert np.allclose(got, want_post, atol=1e-12), seed
        assert abs(hmm.log_likelihood(log_E, log_T, log_start) - want_Z) < 1e-12


def test_macro_posteriors_sum_the_stages():
    log_E, log_T, log_start = _tiny(1, S=4)
    macro = np.array([0, 0, 1, 1])
    pm = hmm.macro_posteriors(log_E, log_T, log_start, macro, 2)
    ps = np.exp(hmm.state_posteriors(log_E, log_T, log_start))
    assert np.allclose(pm[:, 0], ps[:, :2].sum(1))
    assert np.allclose(pm.sum(1), 1.0)


def test_logsumexp_handles_all_neg_inf():
    A = np.full((3, 2), -np.inf)
    assert np.all(np.isneginf(hmm.logsumexp(A, 1)))
    assert np.all(np.isneginf(hmm.logsumexp(np.zeros((0, 4)), 0)))


def test_unreachable_states_do_not_poison_the_dp():
    """A -inf column (a state the grammar forbids) must not produce NaNs."""
    log_E, log_T, log_start = _tiny(2)
    log_T = log_T.copy()
    log_T[:, 2] = -np.inf
    log_start = log_start.copy()
    log_start[2] = -np.inf
    lp = hmm.state_posteriors(log_E, log_T, log_start)
    assert np.isfinite(np.exp(lp)).all()
    assert np.allclose(np.exp(lp).sum(1), 1.0)
    assert np.allclose(np.exp(lp)[1:, 2], 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
