"""The hierarchical annotator.

Level 1 runs over the whole fiber at `binning.level1_bp` and calls the coarse
states (open / linker / nucleosome by default). Level 2 runs *inside* Level-1
runs of `level2.applies_within`, at the finer `binning.level2_bp`, and overlays
its call state (footprint) on them. The two levels are separate chains with
separate duration priors, which is what lets a 147 bp nucleosome and a 15 bp
footprint coexist without one chain's prior distorting the other's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hmm
from .config import state_labels
from .emission import log_emission
from .phasetype import build_chain, emission_class_of_state

NODATA = -1


@dataclass
class Level:
    chain: object
    class_of_state: np.ndarray
    macro_names: list
    decode: str
    threshold_macro: int
    tau: float


class HierHMM:
    """Config in, per-fiber base-pair labels out."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.labels = state_labels(cfg)
        self.classes = list(cfg["emission"]["classes"])
        self.bin1 = int(cfg["binning"]["level1_bp"])
        self.bin2 = int(cfg["binning"]["level2_bp"])
        self.clip = tuple(cfg["emission"]["calibration"]["clip"])
        self.l1 = self._level("level1", self.bin1)
        self.l2 = self._level("level2", self.bin2)
        l2 = cfg["level2"]
        self.bg_state = l2["background_state"]
        self.call_state = l2["call_state"]
        self.within = l2["applies_within"]
        self.margin = float(l2["margin_bp"])
        self.min_open_run = float(l2["min_open_run_bp"])
        self.min_call = float(l2["min_call_bp"])

    def _level(self, key: str, bin_bp: int) -> Level:
        lv = self.cfg[key]
        chain = build_chain(lv["states"], lv["transitions"], bin_bp, lv["start"])
        return Level(chain=chain,
                     class_of_state=emission_class_of_state(chain, lv["states"], self.classes),
                     macro_names=chain.macro_names,
                     decode=lv["decode"],
                     threshold_macro=chain.macro_names.index(lv["threshold_state"]),
                     tau=float(lv["tau"]))

    # ------------------------------------------------------------------ level 1
    def _decode(self, level: Level, k: np.ndarray, n: np.ndarray,
                rates: np.ndarray, want_post: bool = False):
        log_E = log_emission(k, n, rates, level.class_of_state, self.clip)
        ch = level.chain
        if level.decode == "viterbi":
            path = hmm.viterbi(log_E, ch.log_T, ch.log_start)
            macro = ch.macro[path]
            post = np.zeros(len(k)) if want_post else None
            return macro, post
        pm = hmm.macro_posteriors(log_E, ch.log_T, ch.log_start,
                                  ch.macro, len(ch.macro_names))
        # argmax over the non-threshold states, then let the threshold state
        # override wherever its posterior clears tau. tau is the operating
        # point: it turns one fixed call into a recall curve.
        other = [i for i in range(pm.shape[1]) if i != level.threshold_macro]
        macro = np.asarray(other)[np.argmax(pm[:, other], 1)] if other else \
            np.zeros(len(k), int)
        macro = np.where(pm[:, level.threshold_macro] >= level.tau,
                         level.threshold_macro, macro)
        return macro.astype(int), (pm[:, level.threshold_macro] if want_post else None)

    def level1(self, fiber, rates: np.ndarray) -> np.ndarray:
        """-> per-BIN Level-1 label ints over the fiber's span."""
        macro, _ = self._decode(self.l1, fiber.k_bin, fiber.n_bin, rates)
        lut = np.array([self.labels[n] for n in self.l1.macro_names], np.int8)
        return lut[macro]

    # ------------------------------------------------------------------ level 2
    def level2(self, fiber, lab1: np.ndarray, rates: np.ndarray):
        """Overlay call-state runs on a Level-1 labelling, at `level2_bp`.

        Returns (bp labels, bp posterior); the posterior is NaN outside the
        interiors that were actually searched.
        """
        lab = np.repeat(lab1, self.bin1).astype(np.int8)
        prob = np.full(len(lab), np.nan)
        within = self.labels[self.within]
        call_lab = self.labels[self.call_state]
        b = self.bin2
        min_call_bins = int(round(self.min_call / b))
        i, n1 = 0, len(lab1)
        while i < n1:
            if lab1[i] != within:
                i += 1
                continue
            j = i
            while j < n1 and lab1[j] == within:
                j += 1
            a = int(i * self.bin1 + self.margin)
            z = int(j * self.bin1 - self.margin)
            if (j - i) * self.bin1 >= self.min_open_run and z - a >= 2 * b:
                k = _rebin(fiber.meth_bp[a:z], b)
                nn = _rebin(fiber.call_bp[a:z], b)
                macro, post = self._decode(self.l2, k, nn, rates, want_post=True)
                hit = macro == self.l2.chain.macro_names.index(self.call_state)
                hit = _drop_short(hit, min_call_bins)
                m = len(k) * b
                lab[a:a + m][np.repeat(hit, b)] = call_lab
                prob[a:a + m] = np.repeat(post, b)
            i = j
        return lab, prob

    def annotate(self, fiber, rates: np.ndarray, want_post: bool = False):
        lab, post = self.level2(fiber, self.level1(fiber, rates), rates)
        return (lab, post) if want_post else lab


def _rebin(x: np.ndarray, b: int) -> np.ndarray:
    if b == 1:
        return x.astype(float)
    n = (len(x) // b) * b
    return x[:n].reshape(-1, b).sum(1).astype(float)


def _drop_short(hit: np.ndarray, min_bins: int) -> np.ndarray:
    """Enforce the minimum call length on the DECODED runs.

    The state's `min_bp` constrains the hidden dwell, but thresholding the
    posterior clips the tapering edges of a call, so decoded runs come out
    shorter than the state that generated them. This makes the floor an actual
    guarantee rather than a prior.
    """
    if min_bins <= 1:
        return hit
    d = np.concatenate([[0], hit.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(d) != 0)
    out = hit.copy()
    for a, z in zip(edges[0::2], edges[1::2]):
        if z - a < min_bins:
            out[a:z] = False
    return out
