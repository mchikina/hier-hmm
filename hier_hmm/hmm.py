"""Exact inference on a phase-type chain: Viterbi and forward-backward.

Both consume only `(log_E, log_T, log_start)`, so the emission model is fully
swappable — nothing about methylation appears in this module.
"""
from __future__ import annotations

import numpy as np


def logsumexp(A: np.ndarray, axis: int) -> np.ndarray:
    """log-sum-exp along one axis of a 2-D array, safe for all-(-inf) rows.

    Hand-rolled because forward-backward calls it twice per bin per fiber on
    small dense arrays, where scipy's general version costs several times more.
    """
    if A.shape[axis] == 0:
        return np.full(A.shape[1 - axis], -np.inf)
    m = np.max(A, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):      # an all-(-inf) row legitimately gives -inf
        return np.log(np.exp(A - m).sum(axis=axis)) + np.squeeze(m, axis)


def viterbi(log_E: np.ndarray, log_T: np.ndarray, log_start: np.ndarray) -> np.ndarray:
    """Single most probable state path. log_E is (n_bins, n_states)."""
    n = len(log_E)
    D = log_start + log_E[0]
    bp = np.zeros((n, log_T.shape[0]), np.int32)
    for t in range(1, n):
        c = D[:, None] + log_T
        bp[t] = np.argmax(c, 0)
        D = np.max(c, 0) + log_E[t]
    path = np.zeros(n, np.int32)
    path[-1] = int(np.argmax(D))
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def forward(log_E: np.ndarray, log_T: np.ndarray, log_start: np.ndarray) -> np.ndarray:
    n = len(log_E)
    la = np.empty_like(log_E)
    la[0] = log_start + log_E[0]
    for t in range(1, n):
        la[t] = logsumexp(la[t - 1][:, None] + log_T, 0) + log_E[t]
    return la


def backward(log_E: np.ndarray, log_T: np.ndarray) -> np.ndarray:
    n = len(log_E)
    lb = np.zeros_like(log_E)
    for t in range(n - 2, -1, -1):
        lb[t] = logsumexp(log_T + (lb[t + 1] + log_E[t + 1])[None, :], 1)
    return lb


def log_likelihood(log_E: np.ndarray, log_T: np.ndarray, log_start: np.ndarray) -> float:
    """log P(observations) under the chain, marginalising the state path."""
    return float(logsumexp(forward(log_E, log_T, log_start)[-1][:, None], 0)[0])


def state_posteriors(log_E: np.ndarray, log_T: np.ndarray,
                     log_start: np.ndarray) -> np.ndarray:
    """(n_bins, n_states) log posterior over expanded states."""
    lg = forward(log_E, log_T, log_start) + backward(log_E, log_T)
    return lg - logsumexp(lg, 1)[:, None]


def macro_posteriors(log_E: np.ndarray, log_T: np.ndarray, log_start: np.ndarray,
                     macro: np.ndarray, n_macro: int) -> np.ndarray:
    """(n_bins, n_macro) posterior over MACRO states, summing the dwell stages.

    Viterbi commits to one hard path and throws the confidence away; the
    posterior lets a call be thresholded and reported with an uncertainty.
    """
    lg = state_posteriors(log_E, log_T, log_start)
    cols = [logsumexp(lg[:, macro == m], axis=1) for m in range(n_macro)]
    return np.exp(np.stack(cols, 1))
