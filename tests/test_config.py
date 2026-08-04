"""The config is the contract: typos must fail, valid overrides must land."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

from hier_hmm.config import load_config, validate


def _write(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


def test_defaults_validate():
    cfg = load_config()
    assert cfg["binning"]["level1_bp"] == 5
    assert cfg["level1"]["states"]["nucleosome"]["min_bp"] == 60.0


def test_unknown_key_is_rejected():
    path = _write("level1:\n  taau: 0.6\n")
    try:
        load_config(path)
    except KeyError as e:
        assert "level1.taau" in str(e)
    else:
        raise AssertionError("a misspelled key must raise")
    finally:
        os.unlink(path)


def test_override_merges_without_clobbering_siblings():
    path = _write("level1:\n  tau: 0.8\n")
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)
    assert cfg["level1"]["tau"] == 0.8
    assert cfg["level1"]["states"]["open"]["mean_bp"] == 212.0     # untouched


def test_set_overrides():
    cfg = load_config(None, ["level1.tau=0.7", "binning.level1_bp=10",
                             "level1.states.nucleosome.mean_bp=140"])
    assert cfg["level1"]["tau"] == 0.7
    assert cfg["binning"]["level1_bp"] == 10
    assert cfg["level1"]["states"]["nucleosome"]["mean_bp"] == 140


def test_new_state_and_class_are_allowed():
    """`states` and `emission.classes` are open sub-trees: new names are fine."""
    cfg = load_config(None, [
        "emission.classes.linker.pool=[linker]",
        "emission.classes.linker.init=high",
        "level1.states.linker.emission_class=linker",
    ])
    assert cfg["level1"]["states"]["linker"]["emission_class"] == "linker"
    validate(cfg)


def test_validation_catches_bad_references():
    for override, needle in [
        ("level1.states.open.emission_class=nope", "not in emission.classes"),
        ("level1.start=nope", "must be 'uniform'"),
        ("level2.applies_within=nope", "must be a level1 state"),
        ("level1.decode=maybe", "must be 'posterior' or 'viterbi'"),
        ("level1.states.nucleosome.min_bp=200", "must be less"),
        ("plot.profile_states=[banana]", "unknown state"),
    ]:
        try:
            load_config(None, [override])
        except ValueError as e:
            assert needle in str(e), (override, str(e))
        else:
            raise AssertionError(f"{override} should not validate")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
