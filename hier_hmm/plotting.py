"""Single-molecule raster + aggregate profile.

Deliberately generic: it knows about states, groups and coordinates, and
nothing about any particular assay, factor or cell type. Annotations that are
specific to a locus go in through `marks` and `spans`.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

from .model import NODATA                                          # noqa: E402


def _group_colors(cfg: dict, groups: list) -> dict:
    given = dict(cfg["plot"].get("group_colors") or {})
    out = {}
    for i, g in enumerate(groups):
        out[g] = given.get(str(g), matplotlib.colormaps["tab10"](i % 10))
    return out


def _order(result, key: str | None) -> np.ndarray:
    n = len(result.labels)
    if not key or key == "none":
        return np.arange(n)
    if key not in result.fiber_meta:
        raise KeyError(f"plot.order_by={key!r} is not a per-fiber array "
                       f"(have {sorted(result.fiber_meta)})")
    return np.argsort(result.fiber_meta[key], kind="stable")


def plot_locus(result, path: str, group_key: str | None = None,
               title: str = "", marks=(), spans=()):
    """Write the standard two-panel figure.

    result     a runner.Result
    group_key  per-fiber metadata key to split the profile and colour the
               sidebar by (e.g. a cell type or a cluster id); None = one group
    marks      [(x, colour)] vertical lines drawn over the raster
    spans      [(x0, x1, colour)] shaded intervals
    """
    cfg = result.cfg
    pcfg = cfg["plot"]
    L, coords = result.labels, result.coords
    inv = {v: k for k, v in result.label_map.items()}

    rows = _order(result, pcfg["order_by"])
    if group_key:
        g = np.asarray(result.fiber_meta[group_key])
        groups = sorted(set(g.tolist()))
        rows = np.concatenate([rows[np.isin(rows, np.flatnonzero(g == gg))] for gg in groups])
    else:
        g, groups = None, []
    gcol = _group_colors(cfg, groups)

    fig = plt.figure(figsize=tuple(pcfg["figsize"]), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5])
    axp = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=axp)

    want = [result.label_map[s] for s in pcfg["profile_states"]]
    min_cov = int(pcfg["min_coverage"])
    series = [(None, np.arange(len(L)))] if g is None else \
             [(gg, np.flatnonzero(g == gg)) for gg in groups]
    for name, idx in series:
        if len(idx) == 0:
            continue
        sub = L[idx]
        cov = (sub != NODATA).sum(0)
        hit = np.isin(sub, want).sum(0)
        # Below min_coverage the aggregate is a handful of reads and reads as
        # noise — usually tall spikes at the window edges. Blank it instead.
        y = np.where(cov >= min_cov, hit / np.maximum(cov, 1), np.nan)
        axp.plot(coords, y, lw=1.6,
                 color=gcol.get(name, "#333333"),
                 label=f"{name} (n={len(idx)})" if name is not None else f"n={len(idx)}")
    axp.set_ylim(0, 1)
    axp.set_ylabel("fraction of covering fibers\n(" + "+".join(pcfg["profile_states"]) + ")")
    axp.grid(alpha=0.25)
    axp.legend(fontsize=9, loc="upper left")
    axp.tick_params(labelbottom=False)
    if title:
        axp.set_title(title, fontsize=11)

    # One imshow of the whole label matrix. Drawing per-segment rectangles is
    # ~10^5 patches per figure and takes minutes.
    names = list(result.label_map)
    colors = [pcfg["colors"].get("nodata", "#ffffff")] + \
             [pcfg["colors"].get(n, "#888888") for n in names]
    shifted = L.astype(np.int16)[rows] + 1
    axm.imshow(shifted, cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1,
               aspect="auto", interpolation="nearest", origin="lower",
               extent=[coords[0] - 0.5, coords[-1] + 0.5, -0.5, len(rows) - 0.5])

    span = coords[-1] - coords[0]
    if g is not None:
        bar_w = 0.012 * span
        x0 = coords[0] - 1.5 * bar_w
        for y, r in enumerate(rows):
            axm.barh(y, bar_w, left=x0, height=1.0, color=gcol[g[r]], edgecolor="none")
        axm.set_xlim(x0 - 0.5 * bar_w, coords[-1])
    for x, col in marks:
        axm.axvline(x, color=col, lw=0.7, alpha=0.6)
    for x0_, x1_, col in spans:
        axp.axvspan(x0_, x1_, color=col, alpha=0.15, lw=0)
        for x in (x0_, x1_):
            axm.axvline(x, color=col, lw=0.9, alpha=0.85)
    axm.set_ylim(-0.5, len(rows) - 0.5)
    axm.set_xlabel(getattr(result, "coord_label", "position (bp)"))
    axm.set_ylabel(f"fibers (ordered by {pcfg['order_by']})")

    legend = [Patch(facecolor=pcfg["colors"].get(n, "#888888"), label=n) for n in names]
    legend.append(Patch(facecolor=pcfg["colors"].get("nodata", "#ffffff"),
                        edgecolor="0.7", label="no data"))
    if g is not None:
        legend += [Patch(facecolor=gcol[gg], label=str(gg)) for gg in groups]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.savefig(path, dpi=int(pcfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return path
