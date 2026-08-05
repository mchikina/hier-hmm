"""Configuration loading, merging and validation.

`default_config.yaml` (shipped inside the package) is the single source of
truth. A user config is deep-merged onto it and every key the user supplies
must already exist in the default, so a misspelled setting raises instead of
being silently ignored. The fully resolved config is what the model reads, and
it is written next to the results so a run can be reproduced from its output.
"""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "default_config.yaml")

# Sub-trees whose keys are user-defined names rather than fixed settings.
# Unknown keys are allowed directly beneath these paths.
_OPEN_PATHS = frozenset({
    "level1.states", "level1.transitions",
    "level2.states", "level2.transitions",
    "emission.classes",
    "plot.colors", "plot.group_colors",
})


def default_config() -> dict:
    """The shipped defaults, as a fresh mutable dict."""
    with open(DEFAULT_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def default_config_text() -> str:
    """The shipped defaults verbatim, comments and all (for `init-config`)."""
    with open(DEFAULT_CONFIG_PATH) as fh:
        return fh.read()


def _merge(base: dict, over: dict, path: str = "") -> dict:
    for key, val in over.items():
        here = f"{path}.{key}" if path else str(key)
        open_here = path in _OPEN_PATHS or _parent_is_open(path)
        if key not in base and not open_here:
            raise KeyError(
                f"unknown config key {here!r}. Every key must already exist in "
                f"{os.path.basename(DEFAULT_CONFIG_PATH)}; run `hier-hmm init-config` "
                f"to get a commented copy of the valid keys.")
        if isinstance(base.get(key), dict) and isinstance(val, dict):
            _merge(base[key], val, here)
        else:
            base[key] = val
    return base


def _parent_is_open(path: str) -> bool:
    """True for paths *inside* an open sub-tree (e.g. emission.classes.myclass)."""
    parts = path.split(".")
    return any(".".join(parts[:i]) in _OPEN_PATHS for i in range(1, len(parts)))


def _set_path(cfg: dict, dotted: str, value) -> None:
    node, parts = cfg, dotted.split(".")
    for i, key in enumerate(parts[:-1]):
        sofar = ".".join(parts[:i + 1])
        if key not in node:
            if not (".".join(parts[:i]) in _OPEN_PATHS or _parent_is_open(".".join(parts[:i]))):
                raise KeyError(f"unknown config key {sofar!r}")
            node[key] = {}
        node = node[key]
        if not isinstance(node, dict):
            raise KeyError(f"config key {sofar!r} is not a section")
    leaf = ".".join(parts[:-1])
    if parts[-1] not in node and not (leaf in _OPEN_PATHS or _parent_is_open(leaf)):
        raise KeyError(f"unknown config key {dotted!r}")
    node[parts[-1]] = value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    """Defaults, overlaid with `path` (YAML), overlaid with `key.path=value` strings."""
    cfg = default_config()
    if path:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        _merge(cfg, user)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        _set_path(cfg, key.strip(), yaml.safe_load(raw))
    validate(cfg)
    return cfg


def dump_config(cfg: dict, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


def validate(cfg: dict) -> dict:
    """Structural checks that are cheap here and confusing if deferred."""
    for level in ("level1", "level2"):
        lv = cfg[level]
        states = lv["states"]
        if not states:
            raise ValueError(f"{level}.states is empty")
        for name, spec in states.items():
            for field in ("mean_bp", "emission_class"):
                if field not in spec:
                    raise ValueError(f"{level}.states.{name} is missing {field!r}")
            has_sd = spec.get("sd_bp") is not None
            has_k = spec.get("k") is not None
            if has_sd == has_k:
                raise ValueError(
                    f"{level}.states.{name}: give exactly one of sd_bp (preferred — the "
                    f"substate count is then solved for at the current bin size, so the "
                    f"config means the same thing at any bin) or k (pins the substate "
                    f"count, and the SD it gives shrinks as bins get coarser)")
            if has_sd and spec["sd_bp"] <= 0:
                raise ValueError(f"{level}.states.{name}.sd_bp must be > 0")
            if has_k and spec["k"] < 1:
                raise ValueError(f"{level}.states.{name}.k must be >= 1")
            if spec.get("min_bp", 0.0) >= spec["mean_bp"]:
                raise ValueError(
                    f"{level}.states.{name}: min_bp ({spec.get('min_bp')}) must be less "
                    f"than mean_bp ({spec['mean_bp']}) — the Erlang tail adds the rest")
            if spec["emission_class"] not in cfg["emission"]["classes"]:
                raise ValueError(f"{level}.states.{name}.emission_class "
                                 f"{spec['emission_class']!r} is not in emission.classes")
        for src, row in lv["transitions"].items():
            if src not in states:
                raise ValueError(f"{level}.transitions has unknown source state {src!r}")
            if not row:
                raise ValueError(f"{level}.transitions.{src} is empty; every state needs "
                                 f"at least one exit")
            for dst, w in row.items():
                if dst not in states:
                    raise ValueError(f"{level}.transitions.{src} -> unknown state {dst!r}")
                if w < 0:
                    raise ValueError(f"{level}.transitions.{src}.{dst} must be >= 0")
        for src in states:
            if src not in lv["transitions"]:
                raise ValueError(f"{level}.transitions has no row for state {src!r}")
        if lv["start"] != "uniform" and lv["start"] not in states:
            raise ValueError(f"{level}.start must be 'uniform' or one of {list(states)}")
        if lv["decode"] not in ("posterior", "viterbi"):
            raise ValueError(f"{level}.decode must be 'posterior' or 'viterbi'")
        if lv["threshold_state"] not in states:
            raise ValueError(f"{level}.threshold_state must be one of {list(states)}")
        # tau is compared directly against a posterior probability, so a value
        # outside [0, 1] does not error anywhere — it silently turns the
        # threshold state off (tau > 1) or on everywhere (tau < 0).
        if not 0.0 <= lv["tau"] <= 1.0:
            raise ValueError(f"{level}.tau is a posterior probability threshold and must be "
                             f"in [0, 1], got {lv['tau']}")

    l2 = cfg["level2"]
    for field in ("background_state", "call_state"):
        if l2[field] not in l2["states"]:
            raise ValueError(f"level2.{field} must be one of {list(l2['states'])}")
    if l2["background_state"] == l2["call_state"]:
        raise ValueError("level2.background_state and level2.call_state must differ")
    if l2["applies_within"] not in cfg["level1"]["states"]:
        raise ValueError(f"level2.applies_within must be a level1 state, got "
                         f"{l2['applies_within']!r}")

    known = set(cfg["level1"]["states"]) | set(cfg["level2"]["states"])
    for name, spec in cfg["emission"]["classes"].items():
        if not spec.get("pool"):
            raise ValueError(f"emission.classes.{name}.pool is empty")
        for st in spec["pool"]:
            if st not in known:
                raise ValueError(f"emission.classes.{name}.pool names unknown state {st!r}")
        init = spec.get("init", "high")
        if init not in ("high", "low") and not isinstance(init, (int, float)):
            raise ValueError(f"emission.classes.{name}.init must be 'high', 'low' or a number")
        clamp = spec.get("clamp_to")
        if clamp is not None and clamp not in cfg["emission"]["classes"]:
            raise ValueError(f"emission.classes.{name}.clamp_to names unknown class {clamp!r}")
        cap = spec.get("max_refine")
        if cap is not None and (not isinstance(cap, int) or cap < 0):
            raise ValueError(f"emission.classes.{name}.max_refine must be a "
                             f"non-negative integer or null, got {cap!r}")

    for b in ("level1_bp", "level2_bp"):
        if cfg["binning"][b] < 1 or int(cfg["binning"][b]) != cfg["binning"][b]:
            raise ValueError(f"binning.{b} must be a positive integer number of bp")
    if cfg["binning"]["level1_bp"] % cfg["binning"]["level2_bp"]:
        raise ValueError("binning.level1_bp must be a multiple of binning.level2_bp")

    for st in cfg["plot"]["profile_states"]:
        if st not in known:
            raise ValueError(f"plot.profile_states names unknown state {st!r}")
    return cfg


def state_labels(cfg: dict) -> dict:
    """Decoded state name -> integer label used in the output arrays.

    Level-1 states come first in config order, then any Level-2 states that are
    not already Level-1 states. `-1` is reserved for "no data".
    """
    names = list(cfg["level1"]["states"])
    for n in cfg["level2"]["states"]:
        if n not in names:
            names.append(n)
    return {n: i for i, n in enumerate(names)}


def as_yaml(cfg: dict) -> str:
    return yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False, default_flow_style=False)
