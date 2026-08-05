"""State lengths, as chains of substates.

Each state is expanded into `min_bins` must-advance substates followed by `k`
substates that each exit with probability `a`, so its length is
`min_bins + NegBinom(k, a)` bins: mean `min_bins + k/a`, SD `sqrt(k(1-a))/a`.
Setting `a = k / (mean_bins - min_bins)` puts the mean exactly at `mean_bins`.

A plain HMM gives every state a geometric length (mode at 1 bp), which is badly
wrong for a 147 bp nucleosome. This is the cheapest fix that keeps the DP exact:
still a Markov chain, just on more states.

Length is specified in bp as (mean, SD, minimum) and `k` is SOLVED FOR at the
current bin size, so the same numbers describe the same distribution at any bin.
Pinning `k` instead is what makes a config bin-dependent: `a` scales with the
bin, so the mean survives but the SD shrinks as bins get coarser.
"""
from __future__ import annotations

import warnings
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


def substates_for_sd(tail_bp: float, sd_bp: float, bin_bp: float) -> int:
    """Number of substates giving a length SD of `sd_bp`, at this bin size.

    The tail of a state is negative-binomial: SD^2 = tail_bp^2/k - tail_bp*bin_bp
    (`tail_bp` = mean_bp - min_bp). Inverting for k and rounding is what makes a
    config bin-size-independent — hold the SD, let k follow.
    """
    k = tail_bp ** 2 / (sd_bp ** 2 + tail_bp * bin_bp)
    return max(int(round(k)), 1)


def sd_for_substates(tail_bp: float, k: int, bin_bp: float) -> float:
    """The length SD a k-substate chain actually delivers at this bin size."""
    return float(np.sqrt(max(tail_bp ** 2 / k - tail_bp * bin_bp, 0.0)))


def sd_range(tail_bp: float, bin_bp: float) -> tuple[float, float]:
    """(narrowest, widest) length SD reachable for this tail at this bin size."""
    k_max = max(int(MAX_ADVANCE * tail_bp / bin_bp), 1)
    return sd_for_substates(tail_bp, k_max, bin_bp), sd_for_substates(tail_bp, 1, bin_bp)


def build_chain(states: dict, transitions: dict, bin_bp: float, start) -> Chain:
    """Expand a config `states`/`transitions` block into a Chain.

    `states` maps name -> {mean_bp, sd_bp or k [, min_bp]}; lengths are in bp and
    converted to bins with `bin_bp`. `start` is 'uniform' or a macro name.

    Prefer `sd_bp`: the number of substates is then solved for at the current bin
    size, so the same config describes the same length distribution at any bin.
    Giving `k` directly pins the substate count instead, and the SD it produces
    shrinks as bins get coarser.
    """
    names, advance, is_last, first = [], [], [], {}
    for name, spec in states.items():
        mean_bins = float(spec["mean_bp"]) / bin_bp
        min_bins = int(round(float(spec.get("min_bp", 0.0)) / bin_bp))
        tail_bp = (mean_bins - min_bins) * bin_bp     # after rounding min to whole bins
        # No room at all: even a single substate would have to exit with
        # probability > MAX_ADVANCE. Catch this before saying anything about the
        # SD, which is meaningless for a state that cannot exist at this bin.
        if tail_bp < bin_bp / MAX_ADVANCE:
            raise ValueError(
                f"state {name!r}: a mean of {spec['mean_bp']:g} bp above a "
                f"{spec.get('min_bp', 0.0):g} bp minimum leaves {tail_bp:g} bp of length to "
                f"distribute, which is under one {bin_bp:g} bp bin — the state cannot be "
                f"built at this bin size. Use a finer bin for this level.")
        if spec.get("sd_bp") is not None:
            sd_want = float(spec["sd_bp"])
            k = substates_for_sd(tail_bp, sd_want, bin_bp)
            lo, hi = sd_range(tail_bp, bin_bp)
            # Too SHARP is genuinely impossible: substates are `bin_bp` wide and
            # there is a limit to how many fit in the mean. Refuse, rather than
            # quietly hand back a wider state than the config asked for.
            if sd_want < lo * (1 - 1e-6):
                raise ValueError(
                    f"state {name!r}: a length SD of {sd_want:g} bp is not reachable at a "
                    f"{bin_bp:g} bp bin. With mean {spec['mean_bp']:g} and minimum "
                    f"{spec.get('min_bp', 0.0):g} bp the sharpest available is {lo:.1f} bp "
                    f"(and the widest {hi:.1f}). Use a finer bin for this level.")
            # Too WIDE just means one substate — a flat hazard, the widest shape a
            # chain like this has. Say so and carry on.
            if sd_want > hi * (1 + 1e-3):
                warnings.warn(
                    f"state {name!r}: a length SD of {sd_want:g} bp is wider than a "
                    f"single-substate chain with mean {spec['mean_bp']:g} bp can be; "
                    f"using {hi:.1f} bp (flat hazard).", stacklevel=2)
        else:
            k = int(spec["k"])
        first[name] = len(names)
        raw = k / max(mean_bins - min_bins, 1.0)
        # Clipping `a` here would silently break the mean length: the chain would
        # still run, just with a different length prior than the config says.
        # Refuse instead — the fix is a finer bin, a smaller k, or a longer mean.
        if raw > MAX_ADVANCE:
            raise ValueError(
                f"state {name!r} needs a per-substate advance probability of {raw:.3f} "
                f"(> {MAX_ADVANCE}), so its mean length could not be honoured "
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
