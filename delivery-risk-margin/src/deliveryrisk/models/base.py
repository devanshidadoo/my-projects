"""Model interface and the feature matrix both models share."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from deliveryrisk.features.sql import CATEGORICAL_COLUMNS, feature_columns

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Hyperparameters for both model families, in one object per experiment."""

    kind: str = "gbdt"  # gbdt | logistic
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 60
    subsample: float = 0.9
    colsample: float = 0.8
    reg_lambda: float = 1.0
    C: float = 0.5          # logistic regression only
    max_categories: int = 40  # rarer levels collapse to "__other__"
    seed: int = 13


class RiskModel(Protocol):
    """What the rest of the project needs from a scorer."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RiskModel: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FeatureSpace:
    """The columns a model was fitted on, and how categoricals were encoded.

    Held explicitly rather than inferred at predict time: a category level that appears only in
    the test window must be handled the same way in both places, and silently getting a
    different column order is the classic way to serve a model that is confidently wrong.
    """

    numeric: list[str]
    categorical: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical


OTHER = "__other__"


def build_feature_space(df: pd.DataFrame, cfg: ModelConfig) -> FeatureSpace:
    cols = [c for c in feature_columns() if c in df.columns]
    cat = [c for c in CATEGORICAL_COLUMNS if c in cols]
    num = [c for c in cols if c not in cat]
    levels = {}
    for c in cat:
        counts = df[c].astype(str).value_counts()
        levels[c] = sorted(counts.head(cfg.max_categories).index.tolist())
    return FeatureSpace(numeric=num, categorical=cat, levels=levels)


def apply_feature_space(df: pd.DataFrame, space: FeatureSpace) -> pd.DataFrame:
    out = df[space.columns].copy()
    for c in space.categorical:
        vals = out[c].astype(str)
        allowed = set(space.levels[c])
        out[c] = pd.Categorical(
            np.where(vals.isin(allowed), vals, OTHER),
            categories=[*space.levels[c], OTHER],
        )
    for c in space.numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def target(df: pd.DataFrame, name: str = "y_late") -> np.ndarray:
    return df[name].to_numpy(dtype=float)
