"""YAML-backed experiment configuration.

One file describes a whole run: how the warehouse is built, how features are extracted, how the
models are fitted and calibrated, what an order is worth, what an action costs, what the policy
believes an action achieves, and which decision rules to compare. Nothing in this project reads
a parameter that is not either in a config or a documented dataclass default, so every number in
``docs/RESULTS.md`` is reproducible from ``configs/headline.yaml`` plus a seed.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from deliveryrisk.data.synthetic import SyntheticConfig
from deliveryrisk.features.build import FeatureConfig
from deliveryrisk.models.base import ModelConfig
from deliveryrisk.policy.actions import ActionCosts, UpliftBelief
from deliveryrisk.policy.economics import CostModel


class ConfigError(ValueError):
    """Raised for an unusable experiment configuration."""


def _build(cls, data: dict[str, Any] | None):
    """Instantiate a dataclass from a mapping, rejecting unknown keys loudly.

    Silently ignoring a typo'd key is how a run ends up not being the run you described.
    """
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name for f in fields(cls) if f.init}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; valid keys: {sorted(known)}"
        )
    return cls(**data)


class ExperimentConfig:
    """Parsed experiment definition."""

    KNOWN_SECTIONS = {
        "name", "description", "output_dir", "data", "features", "split", "model",
        "calibration", "economics", "actions", "uplift", "policies", "seed",
        "calibration_reference",
    }

    def __init__(self, raw: dict[str, Any]) -> None:
        unknown = set(raw) - self.KNOWN_SECTIONS
        if unknown:
            raise ConfigError(f"unknown config section(s): {sorted(unknown)}")
        self.raw = raw
        self.name: str = raw.get("name", "unnamed")
        self.description: str = raw.get("description", "")
        self.output_dir = Path(raw.get("output_dir", f"results/{self.name}"))
        self.seed: int = int(raw.get("seed", 7))

        data = raw.get("data") or {}
        self.source: str = data.get("source", "synthetic")
        if self.source not in ("synthetic", "olist"):
            raise ConfigError(f"unknown data source {self.source!r}; expected synthetic | olist")
        self.database_url: str = data.get("database_url", "sqlite:///data/deliveryrisk.db")
        self.raw_dir: str | None = data.get("raw_dir")
        self.synthetic: SyntheticConfig = _build(SyntheticConfig, data.get("synthetic"))

        self.features: FeatureConfig = _build(FeatureConfig, raw.get("features"))
        split = raw.get("split") or {}
        unknown = set(split) - {"train_share", "valid_share"}
        if unknown:
            raise ConfigError(f"unknown key(s) in split: {sorted(unknown)}")
        self.train_share: float = float(split.get("train_share", 0.6))
        self.valid_share: float = float(split.get("valid_share", 0.15))
        if not 0 < self.train_share < 1 or not 0 < self.valid_share < 1:
            raise ConfigError("train_share and valid_share must lie in (0, 1)")
        if self.train_share + self.valid_share >= 1:
            raise ConfigError("train_share + valid_share must leave a test block")

        self.model: ModelConfig = _build(ModelConfig, raw.get("model"))
        cal = raw.get("calibration") or {}
        unknown = set(cal) - {"method", "n_bins"}
        if unknown:
            raise ConfigError(f"unknown key(s) in calibration: {sorted(unknown)}")
        self.calibration_method: str = cal.get("method", "isotonic")
        self.calibration_bins: int = int(cal.get("n_bins", 20))

        self.economics: CostModel = _build(CostModel, raw.get("economics"))
        self.actions: ActionCosts = _build(ActionCosts, raw.get("actions"))
        self.uplift: UpliftBelief = _build(UpliftBelief, raw.get("uplift"))

        pol = raw.get("policies") or {}
        unknown = set(pol) - {"fixed_rates", "budget_share", "budget_grid"}
        if unknown:
            raise ConfigError(f"unknown key(s) in policies: {sorted(unknown)}")
        self.fixed_rates: list[float] = list(pol.get("fixed_rates", [0.10, 0.20]))
        self.budget_share: float = float(pol.get("budget_share", 0.5))
        self.budget_grid: list[float] = list(
            pol.get("budget_grid", [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
        )

        self.calibration_reference: dict[str, float] = dict(raw.get("calibration_reference") or {})

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"no such config: {p}")
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} does not contain a mapping")
        cfg = cls(raw)
        cfg.path = p
        return cfg

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "name": self.name,
            "description": self.description,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "data": {"source": self.source, "database_url": self.database_url,
                     "synthetic": asdict(self.synthetic)},
            "features": asdict(self.features),
            "split": {"train_share": self.train_share, "valid_share": self.valid_share},
            "model": asdict(self.model),
            "calibration": {"method": self.calibration_method, "n_bins": self.calibration_bins},
            "economics": asdict(self.economics),
            "actions": asdict(self.actions),
            "uplift": asdict(self.uplift),
            "policies": {"fixed_rates": self.fixed_rates, "budget_share": self.budget_share,
                         "budget_grid": self.budget_grid},
        }
