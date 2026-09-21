"""Logistic regression baseline.

It is here to be beaten, and to be checked against: a linear model on the same features is the
cheapest available test that the gradient booster's advantage is real signal rather than a
target leak or a pipeline bug. When the two land far apart *in ranking* but together *in
calibration*, that is informative; when the booster is enormously better at everything, look for
the leak first.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deliveryrisk.models.base import (
    FeatureSpace,
    ModelConfig,
    apply_feature_space,
    build_feature_space,
)

log = logging.getLogger(__name__)


class LogisticRisk:
    name = "logistic"

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig(kind="logistic")
        self.space: FeatureSpace | None = None
        self.pipeline = None

    def _make_pipeline(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        assert self.space is not None
        # `add_indicator` keeps "this seller has no history" as its own signal rather than
        # pretending the median seller's history applies.
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)
        pre = ColumnTransformer(
            [("num", numeric, self.space.numeric), ("cat", categorical, self.space.categorical)]
        )
        return Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        C=self.cfg.C, max_iter=2000, class_weight=None, n_jobs=None
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogisticRisk:
        self.space = build_feature_space(X, self.cfg)
        Xt = apply_feature_space(X, self.space)
        for c in self.space.categorical:
            Xt[c] = Xt[c].astype(str)
        self.pipeline = self._make_pipeline()
        self.pipeline.fit(Xt, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None and self.space is not None, "fit() first"
        Xt = apply_feature_space(X, self.space)
        for c in self.space.categorical:
            Xt[c] = Xt[c].astype(str)
        return self.pipeline.predict_proba(Xt)[:, 1]


def make_model(cfg: ModelConfig):
    """Factory used by the pipeline and the configs."""
    from deliveryrisk.models.gbdt import GradientBoostedRisk

    if cfg.kind == "logistic":
        return LogisticRisk(cfg)
    if cfg.kind == "gbdt":
        return GradientBoostedRisk(cfg)
    raise ValueError(f"unknown model kind {cfg.kind!r}; expected 'gbdt' or 'logistic'")
