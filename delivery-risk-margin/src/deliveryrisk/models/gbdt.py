"""Gradient-boosted risk scorer.

LightGBM when it is installed, scikit-learn's ``HistGradientBoostingClassifier`` otherwise. Both
take native categorical features and native NaN, which matters here: ``seller_mean_handling`` is
*genuinely missing* for a seller's first order, and imputing it with a median tells the model
that a brand-new seller is an average seller, which is the one thing it is not.
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

try:  # pragma: no cover - import path depends on the environment
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False


class GradientBoostedRisk:
    name = "gbdt"

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig(kind="gbdt")
        self.space: FeatureSpace | None = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GradientBoostedRisk:
        self.space = build_feature_space(X, self.cfg)
        Xt = apply_feature_space(X, self.space)
        c = self.cfg
        if HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=c.n_estimators,
                learning_rate=c.learning_rate,
                num_leaves=c.num_leaves,
                min_child_samples=c.min_child_samples,
                subsample=c.subsample,
                subsample_freq=1,
                colsample_bytree=c.colsample,
                reg_lambda=c.reg_lambda,
                random_state=c.seed,
                n_jobs=-1,
                verbose=-1,
            )
            self.model.fit(Xt, y, categorical_feature=self.space.categorical)
        else:  # pragma: no cover - exercised only without the extra
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                max_iter=c.n_estimators,
                learning_rate=c.learning_rate,
                max_leaf_nodes=c.num_leaves,
                min_samples_leaf=c.min_child_samples,
                l2_regularization=c.reg_lambda,
                categorical_features=[
                    Xt.columns.get_loc(col) for col in self.space.categorical
                ],
                random_state=c.seed,
            )
            self.model.fit(Xt, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.space is not None and self.model is not None, "fit() first"
        Xt = apply_feature_space(X, self.space)
        return self.model.predict_proba(Xt)[:, 1]

    def importances(self) -> pd.DataFrame:
        assert self.space is not None
        if HAS_LGB and hasattr(self.model, "booster_"):
            gain = self.model.booster_.feature_importance(importance_type="gain")
        else:  # pragma: no cover
            gain = np.zeros(len(self.space.columns))
        return (
            pd.DataFrame({"feature": self.space.columns, "gain": gain})
            .sort_values("gain", ascending=False, ignore_index=True)
        )
