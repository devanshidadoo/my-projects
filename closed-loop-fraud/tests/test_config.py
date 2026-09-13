"""Configuration parsing.

The tests here look fussy. They exist because a config bug does not crash -- it quietly runs a
*different experiment* than the one the file describes, and the results look perfectly
plausible. Of the failures worth catching, silent ones are the expensive ones.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clfraud.config import ConfigError, ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(payload))
    return p


MINIMAL = {
    "name": "t",
    "data": {"source": "synthetic", "synthetic": {"n_transactions": 100}},
    "loop": {"n_cycles": 4},
    "arms": [{"name": "naive"}, {"name": "oracle", "oracle_labels": True}],
    "baseline_arm": "naive",
    "oracle_arm": "oracle",
}


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_parse(path):
    cfg = ExperimentConfig.load(path)
    assert cfg.arms
    assert cfg.baseline_arm in {a.name for a in cfg.arms}


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_control_arms_never_inherit_treatment_defaults(path):
    """Regression: an arm with no `exploration:` block must not explore.

    ``ExplorationConfig()`` and ``WeightConfig()`` default to the *proposed* behaviour, which is
    right for a caller building them in code and catastrophic for a YAML arm that never asked.
    Before this was fixed, `naive` and `oracle` silently ran 2 % cost-aware exploration with IPW
    -- so the control arms were treatment arms, the measured damage was understated, and nothing
    anywhere failed.
    """
    cfg = ExperimentConfig.load(path)
    raw_arms = {a["name"]: a for a in cfg.raw["arms"]}
    for arm in cfg.arms:
        spec = raw_arms[arm.name]
        if "exploration" not in spec:
            assert arm.exploration.mode == "none"
            assert arm.exploration.rate == 0.0
        if "weights" not in spec:
            assert arm.weights.scheme == "none"


def test_unknown_key_is_rejected(tmp_path):
    payload = dict(MINIMAL)
    payload["loop"] = {"n_cycles": 4, "cycle_dayz": 30}
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.load(_write(tmp_path, payload))


def test_unknown_arm_key_is_rejected(tmp_path):
    payload = dict(MINIMAL)
    payload["arms"] = [{"name": "naive", "oracle_lables": True}]
    payload["baseline_arm"] = "naive"
    payload["oracle_arm"] = None
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.load(_write(tmp_path, payload))


def test_duplicate_arm_names_are_rejected(tmp_path):
    payload = dict(MINIMAL)
    payload["arms"] = [{"name": "a"}, {"name": "a"}]
    payload["baseline_arm"] = "a"
    payload["oracle_arm"] = None
    with pytest.raises(ConfigError, match="duplicate"):
        ExperimentConfig.load(_write(tmp_path, payload))


def test_baseline_must_exist(tmp_path):
    payload = dict(MINIMAL)
    payload["baseline_arm"] = "nope"
    with pytest.raises(ConfigError, match="baseline_arm"):
        ExperimentConfig.load(_write(tmp_path, payload))


def test_unknown_source_is_rejected(tmp_path):
    payload = dict(MINIMAL)
    payload["data"] = {"source": "kaggle_magic"}
    with pytest.raises(ConfigError, match="unknown data source"):
        ExperimentConfig.load(_write(tmp_path, payload))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ExperimentConfig.load(tmp_path / "absent.yaml")


def test_to_dict_round_trips_the_resolved_config():
    cfg = ExperimentConfig.load(CONFIG_DIR / "headline.yaml")
    d = cfg.to_dict()
    assert d["loop"]["n_cycles"] == cfg.loop.n_cycles
    assert {a["name"] for a in d["arms"]} == {a.name for a in cfg.arms}
    # Resolved, not just echoed: defaults the YAML never mentioned must be present.
    assert "min_child_samples" in d["loop"]["model"]
