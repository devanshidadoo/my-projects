"""Model interface used by the simulator.

The loop only ever needs three things from a model: fit with per-sample weights (the hook that
selection-bias correction plugs into), predict a probability, and report feature importances so
the drift analysis can show *which* signals the closed loop erodes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Hyper-parameters for the gradient-boosted scorer.

    Defaults are deliberately modest. A larger model would paper over the feedback-loop effect
    by memorising the residual signal in whatever data survives censoring, which is exactly the
    dynamic under study -- so capacity is held fixed across every policy arm.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 48
    max_depth: int = -1
    min_child_samples: int = 40
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.75
    reg_lambda: float = 1.0
    scale_pos_weight: float | None = None
    random_state: int = 7
    n_jobs: int = 4
    early_stopping_rounds: int | None = None
    extra: dict = field(default_factory=dict)


class FraudScorer(ABC):
    """A fitted-or-fittable fraud probability model."""

    feature_names_: list[str]

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> FraudScorer:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) for each row, shape ``(n,)``."""

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Importances indexed by feature name, normalised to sum to one."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(n_features={len(getattr(self, 'feature_names_', []))})"
