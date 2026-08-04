"""Phase-type (Erlang) duration chains.

Each named macro state is expanded into `min_bins` advance-only stages followed
by `k` Erlang stages that each leave with probability `a`, so the dwell time is
`min_bins + NegBinom(k, a)` bins: mean `min_bins + k/a`, SD `~sqrt(k(1-a))/a`.
Choosing `a = k / (mean_bins - min_bins)` puts the mean at `mean_bins`.

A plain HMM gives every state a geometric dwell (mode at length 1), which is
badly wrong for a 147 bp nucleosome. This is the cheapest fix that keeps the
DP exact: the chain is still a Markov chain, just on more states.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ADVANCE = 0.95


@dataclass
class Chain:
    """An expanded phase-type chain.

    names      macro state name of each expanded state, in expansion order
    macro      integer macro index of each expanded state
    macro_names  macro names, indexed by the values in `macro`
    log_T      (S, S) log transition matrix
    log_start  (S,) log initial distribution
    """
    names: list
    macro: np.ndarray
    macro_names: list
    log_T: np.ndarray
    log_start: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.names)

    def macro_index(self, name: str) -> int:
        return self.macro_names.index(name)

    def dwell_moments(self, name: str) -> tuple[float, float]:
        """(mean, SD) dwell in bins for a macro, from the built matrix itself."""
        i = self.macro_names.index(name)
        sel = np.flatnonzero(self.macro == i)
        # self-loop / advance structure is a pure birth chain; walk it.
        mean = 0.0
        var = 0.0
        for s in sel:
            a = 1.0 - np.exp(self.log_T[s, s])
            mean += 1.0 / a
            var += (1.0 - a) / a ** 2
        return float(mean), float(np.sqrt(var))


def build_chain(states: dict, transitions: dict, bin_bp: float, start) -> Chain:
    """Expand a config `states`/`transitions` block into a Chain.

    `states` maps name -> {mean_bp, k[, min_bp]}; lengths are converted to bins
    with `bin_bp`. `start` is 'uniform' or a macro name.
    """
    names, advance, is_last, first = [], [], [], {}
    for name, spec in states.items():
        k = int(spec["k"])
        mean_bins = float(spec["mean_bp"]) / bin_bp
        min_bins = int(round(float(spec.get("min_bp", 0.0)) / bin_bp))
        first[name] = len(names)
        raw = k / max(mean_bins - min_bins, 1.0)
        # Clipping `a` here would silently break the dwell mean: the chain would
        # still run, just with a different duration prior than the config says.
        # Refuse instead — the fix is a finer bin, a smaller k, or a longer mean.
        if raw > MAX_ADVANCE:
            raise ValueError(
                f"state {name!r} needs a per-stage advance probability of {raw:.3f} "
                f"(> {MAX_ADVANCE}), so its mean dwell could not be honoured "
                f"(k={k}, mean={mean_bins:.2f} bins, min={min_bins} bins). "
                f"Use a finer bin size, a smaller k, or a longer mean_bp.")
        a = float(np.clip(raw, 1e-3, MAX_ADVANCE))
        for _ in range(min_bins):
            names.append(name); advance.append(1.0); is_last.append(False)
        for i in range(k):
            names.append(name); advance.append(a); is_last.append(i == k - 1)

    S = len(names)
    T = np.zeros((S, S))
    for s in range(S):
        a, name = advance[s], names[s]
        if not is_last[s]:
            if a >= 1.0:
                T[s, s + 1] = 1.0
            else:
                T[s, s] = 1.0 - a
                T[s, s + 1] = a
        else:
            T[s, s] = 1.0 - a
            row = transitions[name]
            total = float(sum(row.values()))
            if total <= 0:
                raise ValueError(f"transitions out of {name!r} sum to zero")
            for dst, w in row.items():
                T[s, first[dst]] += a * (w / total)

    macro_names = list(states)
    macro = np.array([macro_names.index(n) for n in names], int)
    log_start = np.full(S, -np.inf)
    if start == "uniform":
        log_start[:] = -np.log(S)
    else:
        log_start[first[start]] = 0.0
    with np.errstate(divide="ignore"):
        log_T = np.log(T)
    return Chain(names=names, macro=macro, macro_names=macro_names,
                 log_T=log_T, log_start=log_start)


def emission_class_of_state(chain: Chain, states: dict, classes: list) -> np.ndarray:
    """(S,) index into `classes` giving each expanded state's emission class."""
    return np.array([classes.index(states[n]["emission_class"]) for n in chain.names], int)
