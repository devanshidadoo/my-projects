"""Configuration parsing. A typo'd key is an error, never a silent default."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deliveryrisk.config import ConfigError, ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path, raw) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_parses(path):
    cfg = ExperimentConfig.load(path)
    assert cfg.name
    assert 0 < cfg.train_share < 1
    assert cfg.to_dict()["name"] == cfg.name


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown config section"):
        ExperimentConfig.load(_write(tmp_path, {"name": "x", "modell": {}}))


def test_unknown_key_inside_a_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.load(_write(tmp_path, {"name": "x", "model": {"n_estimator": 10}}))


def test_unknown_synthetic_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.load(
            _write(tmp_path, {"name": "x", "data": {"synthetic": {"n_order": 10}}})
        )


def test_split_must_leave_a_test_block(tmp_path):
    with pytest.raises(ConfigError, match="test block"):
        ExperimentConfig.load(
            _write(tmp_path, {"name": "x", "split": {"train_share": 0.8, "valid_share": 0.3}})
        )


def test_unknown_source_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown data source"):
        ExperimentConfig.load(_write(tmp_path, {"name": "x", "data": {"source": "parquet"}}))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no such config"):
        ExperimentConfig.load(tmp_path / "nope.yaml")


def test_defaults_are_the_documented_ones(tmp_path):
    cfg = ExperimentConfig.load(_write(tmp_path, {"name": "bare"}))
    assert cfg.features.mode == "point_in_time"
    assert cfg.calibration_method == "isotonic"
    assert cfg.model.kind == "gbdt"
    assert cfg.economics.support_cost > 0


def test_headline_config_leaves_the_shrinkage_prior_unset():
    """It must be computed from the training window, never pinned in the config."""
    cfg = ExperimentConfig.load(CONFIG_DIR / "headline.yaml")
    assert cfg.features.prior_rate is None
