"""Command line entry point: `hier-hmm <command>` (or `python -m hier_hmm`)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from .config import default_config_text, load_config
from .data import load_npz, prepare
from .runner import run, save


def _add_config_args(p):
    p.add_argument("-c", "--config", default=None,
                   help="YAML config overlaid on the packaged defaults")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override one config key, e.g. --set level1.tau=0.6 "
                        "(repeatable; applied after --config)")
    p.add_argument("-q", "--quiet", action="store_true")


def cmd_run(a) -> int:
    cfg = load_config(a.config, a.overrides)
    ds = load_npz(a.input, cfg)
    if not a.quiet:
        print(f"{a.input}: {ds.n_pos} positions, {len(ds.fibers)} fibers kept")
    result = run(ds, cfg, verbose=not a.quiet)
    paths = save(result, a.outdir, a.prefix)
    if not a.quiet:
        frac = result.state_fractions()
        print("state fractions over covered bp: "
              + "  ".join(f"{k}={v:.3f}" for k, v in frac.items()))
    if cfg["plot"]["enabled"] and not a.no_plot:
        from .plotting import plot_locus
        group = a.group_by if a.group_by in result.fiber_meta else None
        if a.group_by and group is None and not a.quiet:
            print(f"  note: --group-by {a.group_by!r} not in the input; plotting ungrouped")
        paths["figure"] = plot_locus(result, os.path.join(a.outdir, f"{a.prefix}.png"),
                                     group_key=group,
                                     title=a.title or os.path.basename(a.input))
    for k, v in paths.items():
        print(f"wrote {k}: {v}")
    return 0


def cmd_init_config(a) -> int:
    if os.path.exists(a.path) and not a.force:
        print(f"{a.path} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    with open(a.path, "w") as fh:
        fh.write(default_config_text())
    print(f"wrote {a.path} (the full commented default config)")
    return 0


def cmd_simulate(a) -> int:
    from .simulate import simulate
    cfg = load_config(a.config, a.overrides)
    d = simulate(cfg, n_fiber=a.n_fiber, n_pos=a.n_pos, seed=a.seed,
                 anchor=(a.n_pos // 2 - 250, a.n_pos // 2 + 250),
                 coverage_frac=a.coverage_frac, groups=a.groups)
    np.savez_compressed(a.out, m6a=d["m6a"], truth=d["truth"], group=d["group"],
                        positions=d["positions"], center=d["center"])
    print(f"wrote {a.out}: {d['m6a'].shape[0]} fibers x {d['m6a'].shape[1]} bp")
    return 0


def cmd_check(a) -> int:
    """Load a config and report the duration priors it actually implies."""
    from .model import HierHMM
    cfg = load_config(a.config, a.overrides)
    m = HierHMM(cfg)
    for name, level, bp in (("level1", m.l1, m.bin1), ("level2", m.l2, m.bin2)):
        print(f"{name}: bin={bp} bp, {level.chain.n_states} expanded states, "
              f"decode={level.decode}, tau={level.tau}")
        for s in level.macro_names:
            mean, sd = level.chain.dwell_moments(s)
            cls = cfg[name]["states"][s]["emission_class"]
            print(f"   {s:<12s} dwell {mean * bp:7.1f} +- {sd * bp:5.1f} bp "
                  f"(config mean {cfg[name]['states'][s]['mean_bp']:g})  class={cls}")
    print("emission classes: "
          + ", ".join(f"{k} <- {v['pool']}" for k, v in cfg["emission"]["classes"].items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hier-hmm",
                                 description="Hierarchical phase-type HMM for single-molecule "
                                             "methylation footprinting")
    ap.add_argument("--version", action="version", version=f"hier-hmm {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="annotate an .npz of fibers")
    p.add_argument("input", help=".npz with an (n_fiber, n_pos) 'm6a' array "
                                "(NaN = not callable, 0/1 = un/methylated)")
    p.add_argument("-o", "--outdir", default="hier_hmm_out")
    p.add_argument("--prefix", default="annotation")
    p.add_argument("--group-by", default="group",
                   help="per-fiber array in the input to split the profile by")
    p.add_argument("--title", default="")
    p.add_argument("--no-plot", action="store_true")
    _add_config_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("init-config", help="write a commented copy of the default config")
    p.add_argument("path", nargs="?", default="hier_hmm.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("check", help="print the duration priors a config implies")
    _add_config_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("simulate", help="write a synthetic dataset in the input format")
    p.add_argument("out", nargs="?", default="synthetic.npz")
    p.add_argument("--n-fiber", type=int, default=80)
    p.add_argument("--n-pos", type=int, default=4000)
    p.add_argument("--coverage-frac", type=float, default=1.0)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    _add_config_args(p)
    p.set_defaults(func=cmd_simulate)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
