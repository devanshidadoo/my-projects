"""YAML-backed experiment configuration.

One file describes a whole run: how the stream is generated, how features are built, how the
loop is timed, and which policy arms to compare. Everything in ``docs/RESULTS.md`` is
reproducible from ``configs/headline.yaml`` plus a seed, and nothing in this project reads a
parameter that is not either in the config or a documented default.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from clfraud.bias.reweighting import WeightConfig
from clfraud.data.synthetic import SyntheticConfig
from clfraud.evaluation.metrics import BusinessCosts
from clfraud.models.base import ModelConfig
from clfraud.simulator.labeling import LabelPipelineConfig
from clfraud.simulator.loop import LoopConfig, PolicyArm
from clfraud.simulator.policy import ExplorationConfig


class ConfigError(ValueError):
    """Raised for an unusable experiment configuration."""


def _build(cls, data: dict[str, Any] | None):
    """Instantiate a dataclass from a mapping, rejecting unknown keys loudly.

    Silently ignoring a typo'd key is how a run ends up not being the run you described. An
    unknown key is always an error here.
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


def _arm(spec: dict[str, Any]) -> PolicyArm:
    spec = dict(spec)
    name = spec.pop("name", None)
    if not name:
        raise ConfigError("every arm needs a `name`")

    # An omitted block means "this arm does not do that", NOT "use the dataclass default".
    # ExplorationConfig and WeightConfig default to the *proposed* behaviour (2 % cost-aware
    # exploration, IPW) because that is what a caller constructing them directly wants. Applying
    # those defaults to a YAML arm that never asked for them silently turned the `naive` control
    # into another treatment arm, which quietly destroys the whole comparison.
    exp_spec = spec.pop("exploration", None)
    w_spec = spec.pop("weights", None)
    exploration = (
        _build(ExplorationConfig, exp_spec)
        if exp_spec is not None
        else ExplorationConfig(mode="none", rate=0.0)
    )
    weights = _build(WeightConfig, w_spec) if w_spec is not None else WeightConfig(scheme="none")
    known = {"oracle_labels", "retrain", "estimate_propensity", "description"}
    unknown = set(spec) - known
    if unknown:
        raise ConfigError(f"unknown key(s) for arm {name!r}: {sorted(unknown)}")
    return PolicyArm(name=name, exploration=exploration, weights=weights, **spec)


class ExperimentConfig:
    """Parsed experiment definition."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.name: str = raw.get("name", "unnamed")
        self.description: str = raw.get("description", "")
        self.output_dir = Path(raw.get("output_dir", "results"))

        data = dict(raw.get("data", {}))
        self.source: str = data.pop("source", "synthetic")
        if self.source not in {"synthetic", "ieee_cis"}:
            raise ConfigError(f"unknown data source: {self.source}")
        self.raw_dir = data.pop("raw_dir", "data/raw")
        self.nrows = data.pop("nrows", None)
        self.stretch_months = data.pop("stretch_months", None)
        self.synthetic = _build(SyntheticConfig, data.pop("synthetic", None))
        if data:
            raise ConfigError(f"unknown key(s) under `data`: {sorted(data)}")

        loop = dict(raw.get("loop", {}))
        label = _build(LabelPipelineConfig, loop.pop("label", None))
        model = _build(ModelConfig, loop.pop("model", None))
        costs = _build(BusinessCosts, loop.pop("costs", None))
        self.loop = _build(LoopConfig, loop)
        self.loop.label, self.loop.model, self.loop.costs = label, model, costs

        arms = raw.get("arms") or []
        if not arms:
            raise ConfigError("at least one arm must be defined")
        self.arms = [_arm(a) for a in arms]
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate arm names: {names}")

        self.baseline_arm: str = raw.get("baseline_arm", names[0])
        self.oracle_arm: str | None = raw.get("oracle_arm")
        if self.baseline_arm not in names:
            raise ConfigError(f"baseline_arm {self.baseline_arm!r} is not among {names}")
        if self.oracle_arm and self.oracle_arm not in names:
            raise ConfigError(f"oracle_arm {self.oracle_arm!r} is not among {names}")

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config not found: {path}")
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(raw)

    def to_dict(self) -> dict[str, Any]:
        """Fully resolved configuration, defaults included -- written next to every result."""

        def unpack(obj):
            if is_dataclass(obj):
                return {k: unpack(v) for k, v in asdict(obj).items() if not k.startswith("_")}
            if isinstance(obj, (list, tuple)):
                return [unpack(v) for v in obj]
            if isinstance(obj, dict):
                return {k: unpack(v) for k, v in obj.items()}
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "synthetic": unpack(self.synthetic),
            "loop": unpack(self.loop),
            "arms": [unpack(a) for a in self.arms],
            "baseline_arm": self.baseline_arm,
            "oracle_arm": self.oracle_arm,
        }
