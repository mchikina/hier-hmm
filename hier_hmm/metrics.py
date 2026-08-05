"""Cross-fiber concordance of Level-2 calls, scored against a permutation null.

Raw cross-fiber coherence is not a valid objective. A shared positional prior
raises it by construction, and so does anything that merely makes the calls more
numerous or more clustered — including a rate estimate that has drifted. Every
number here is therefore reported as EXCESS over a matched null: Level 1 is
frozen, and within each Level-1 open run that fiber's methylated positions are
re-placed at random among the run's callable positions, preserving the count.
The null is then called with the SAME rates as the observation, so a setting
that only inflates coherence inflates the null too and scores about zero.

Two excess metrics, and they can disagree:

    excess_coh   mean pairwise correlation of the per-fiber call indicators
    excess_top   share of call mass in the most recurrent `top_frac` of
                 positions — density-invariant, so it is the one to trust when
                 they disagree (coherence rises mechanically with the number of
                 calls, positional concentration does not).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .model import NODATA


# ---------------------------------------------------------------- raw metrics
def coherence(rows: np.ndarray) -> float:
    """Mean pairwise correlation of the per-fiber indicator vectors."""
    Z = np.asarray(rows, float)
    Z = Z[Z.std(1) > 1e-6]
    if len(Z) < 5:
        return float("nan")
    Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    C = Z @ Z.T / Z.shape[1]
    m = len(Z)
    return float((C.sum() - np.trace(C)) / (m * (m - 1)))


def top_share(rows: np.ndarray, frac: float = 0.02) -> float:
    """Share of the total call mass sitting in the `frac` most recurrent positions."""
    f = np.asarray(rows, float).mean(0)
    if f.sum() <= 0:
        return float("nan")
    k = max(int(len(f) * frac), 1)
    return float(np.sort(f)[-k:].sum() / f.sum())


def calls_per_fiber(rows: np.ndarray) -> float:
    rows = np.asarray(rows, bool)
    return float(np.mean([int(r[0]) + int(np.sum(r[1:] & ~r[:-1])) for r in rows]))


def median_call_length(rows: np.ndarray) -> float:
    lengths = []
    for r in np.asarray(rows, bool):
        d = np.concatenate([[0], r.astype(np.int8), [0]])
        e = np.flatnonzero(np.diff(d) != 0)
        lengths += list(e[1::2] - e[0::2])
    return float(np.median(lengths)) if lengths else float("nan")


# ---------------------------------------------------------------- the null
def permute_in_runs(fiber, lab1: np.ndarray, model, rng: np.random.Generator) -> np.ndarray:
    """Re-place a fiber's methylated positions within each searched interval.

    The intervals come from `model.search_intervals`, i.e. exactly the stretches
    Level 2 decodes — the interiors of open runs, margins already trimmed. That
    matters: permuting over the whole open run instead would let methylation
    cross the margin boundary, so the null would differ from the observation not
    only in WHERE methylation sits inside the searched region but in HOW MUCH is
    there at all (about 1.4% less, measured on T-cell data). Confining the
    permutation to the searched interval makes the count identical by
    construction, and the null then differs only in arrangement.
    """
    out = fiber.meth_bp.copy()
    for a, z in model.search_intervals(lab1):
        idx = np.flatnonzero(fiber.call_bp[a:z])
        n_meth = int(fiber.meth_bp[a:z].sum())
        if len(idx) >= 2 and n_meth > 0:
            out[a:z] = False
            out[a + rng.choice(idx, size=min(n_meth, len(idx)), replace=False)] = True
    return out


def indicators(model, dataset, rates: np.ndarray, n_perm: int = 4,
               seed: int = 0) -> tuple[np.ndarray, list]:
    """-> (observed indicator matrix, [null indicator matrices]) at bp resolution.

    Level 1 is computed once per fiber and reused for every null, so the two
    sides differ only in the Level-2 call.
    """
    rng = np.random.default_rng(seed)
    call = model.labels[model.call_state]
    n_bp = dataset.n_bp
    obs = np.zeros((len(dataset.fibers), n_bp), bool)
    nulls = [np.zeros_like(obs) for _ in range(n_perm)]
    for i, f in enumerate(dataset.fibers):
        lab1 = model.level1(f, rates)
        lab, _ = model.level2(f, lab1, rates)
        obs[i, f.lo_bp:f.hi_bp] = lab == call
        for p in range(n_perm):
            fp = dataclasses.replace(f, meth_bp=permute_in_runs(f, lab1, model, rng))
            lab_p, _ = model.level2(fp, lab1, rates)
            nulls[p][i, f.lo_bp:f.hi_bp] = lab_p == call
    return obs, nulls


def concordance(model, dataset, rates: np.ndarray, n_perm: int = 4, seed: int = 0,
                group_key: str | None = None, top_frac: float = 0.02) -> dict:
    """Excess coherence and excess top-share of the Level-2 calls.

    `group_key` names a per-fiber metadata field; metrics are computed within
    each group and averaged, so a difference between groups (cell types, say)
    cannot masquerade as agreement between fibers.
    """
    obs, nulls = indicators(model, dataset, rates, n_perm=n_perm, seed=seed)
    if group_key:
        g = np.array([f.meta[group_key] for f in dataset.fibers])
        groups = [(str(v), np.flatnonzero(g == v)) for v in sorted(set(g.tolist()))]
    else:
        groups = [("all", np.arange(len(obs)))]

    eco, eto, raw, n_obs, n_null, mlen, per_group = [], [], [], [], [], [], {}
    for name, idx in groups:
        if len(idx) < 10:
            continue
        o = (coherence(obs[idx]), top_share(obs[idx], top_frac),
             calls_per_fiber(obs[idx]), median_call_length(obs[idx]))
        nl = np.array([(coherence(n[idx]), top_share(n[idx], top_frac),
                        calls_per_fiber(n[idx])) for n in nulls])
        eco.append(o[0] - np.nanmean(nl[:, 0]))
        eto.append(o[1] - np.nanmean(nl[:, 1]))
        raw.append(o[0]); n_obs.append(o[2]); mlen.append(o[3])
        n_null.append(np.nanmean(nl[:, 2]))
        per_group[name] = {"excess_coh": round(eco[-1], 4), "excess_top": round(eto[-1], 4),
                           "n_obs": round(o[2], 2), "n_null": round(n_null[-1], 2)}
    return {"excess_coh": float(np.nanmean(eco)), "excess_top": float(np.nanmean(eto)),
            "raw_coh": float(np.nanmean(raw)), "n_obs": float(np.nanmean(n_obs)),
            "n_null": float(np.nanmean(n_null)), "median_len": float(np.nanmean(mlen)),
            "n_perm": n_perm, "per_group": per_group}
