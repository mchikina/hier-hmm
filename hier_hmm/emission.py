"""Masked binomial emission on (methylated, callable) counts.

For a bin holding `n` callable positions of which `k` are methylated, a state
whose emission class has rate `mu` contributes

    log P(bin | class) = k log(mu) + (n - k) log(1 - mu).

Evidence therefore scales with the number of callable positions in the bin,
which is the point: a bin with one callable site and no methylation is much
weaker evidence than one with five. A bin with `n == 0` contributes 0 to every
state — it is uninformative and the duration prior carries it — so callable
density sets how confident a bin is allowed to be without biasing which state
it prefers.
"""
from __future__ import annotations

import numpy as np


def log_emission(k: np.ndarray, n: np.ndarray, rates: np.ndarray,
                 class_of_state: np.ndarray, clip=(1e-4, 1 - 1e-4)) -> np.ndarray:
    """(n_bins, n_states) log emission.

    k, n              per-bin methylated and callable counts
    rates             (n_classes,) methylation rate per emission class
    class_of_state    (n_states,) index into `rates` for each expanded state
    """
    mu = np.clip(np.asarray(rates, float), clip[0], clip[1])
    log_mu = np.log(mu)[class_of_state]
    log_1m = np.log1p(-mu)[class_of_state]
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    return k[:, None] * log_mu[None, :] + (n - k)[:, None] * log_1m[None, :]


def binomial_mixture(k: np.ndarray, n: np.ndarray, start=(0.60, 0.05),
                     iters: int = 200, tol: float = 1e-8) -> tuple[float, float]:
    """2-component binomial mixture EM on pooled per-bin counts -> (high, low).

    On counts rather than on a smoothed rate: with a handful of callable sites
    per bin, protected bins are usually exactly 0 methylated, and a Gaussian
    mixture on the rate parks that component at mu = 0 with a sigma as wide as
    the entire dynamic range. A binomial handles the zero mass natively —
    0 of 2 and 0 of 8 are different likelihoods.
    """
    m = n > 0
    k, n = np.asarray(k, float)[m], np.asarray(n, float)[m]
    if len(k) == 0:
        raise ValueError("no bins with callable sites; cannot calibrate")
    mu = np.clip(np.array(start, float), 1e-4, 1 - 1e-4)
    w = np.array([0.4, 0.6])
    for _ in range(iters):
        lg = (k[:, None] * np.log(mu)[None, :]
              + (n - k)[:, None] * np.log1p(-mu)[None, :]
              + np.log(w)[None, :])
        lg -= lg.max(1, keepdims=True)
        r = np.exp(lg)
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        new = (r * k[:, None]).sum(0) / np.maximum((r * n[:, None]).sum(0), 1e-9)
        new = np.clip(new, 1e-4, 1 - 1e-4)
        done = np.max(np.abs(new - mu)) < tol
        mu = new
        if done:
            break
    order = np.argsort(mu)
    return float(mu[order[1]]), float(mu[order[0]])
