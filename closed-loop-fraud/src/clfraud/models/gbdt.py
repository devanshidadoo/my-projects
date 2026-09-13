"""Gradient-boosted scorer with a dependency-free fallback.

LightGBM is used when available; otherwise scikit-learn's ``HistGradientBoostingClassifier``
stands in. Both honour ``sample_weight``, which is the only interface the bias-correction layer
needs, so results are qualitatively identical either way (the reported numbers use LightGBM).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from clfraud.models.base import FraudScorer, ModelConfig

log = logging.getLogger(__name__)

try:  # pragma: no cover - import-time branch
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False


class GBDTScorer(FraudScorer):
    """LightGBM binary classifier over the point-in-time feature matrix."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.model_ = None
        self.feature_names_: list[str] = []
        self.backend_ = "lightgbm" if _HAS_LGB else "sklearn"

    def fit(self, X, y, sample_weight=None, eval_set=None) -> GBDTScorer:
        X = _as_frame(X)
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(X.columns)
        if len(np.unique(y)) < 2:
            # Degenerate cycles happen: a policy can censor away every positive in its training
            # window. Fall back to a constant predictor rather than crashing the loop -- and
            # make it loud, because it is itself a finding about the policy.
            log.warning("training set has a single class (n=%d, pos=%d)", len(y), int(y.sum()))
            self.model_ = float(y.mean())
            return self

        cfg = self.config
        if _HAS_LGB:
            params = dict(
                objective="binary",
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                max_depth=cfg.max_depth,
                min_child_samples=cfg.min_child_samples,
                subsample=cfg.subsample,
                subsample_freq=cfg.subsample_freq,
                colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
                verbose=-1,
                **cfg.extra,
            )
            if cfg.scale_pos_weight:
                params["scale_pos_weight"] = cfg.scale_pos_weight
            model = lgb.LGBMClassifier(**params)
            fit_kw: dict = {}
            if eval_set is not None and cfg.early_stopping_rounds:
                fit_kw["eval_set"] = [(_as_frame(eval_set[0]), np.asarray(eval_set[1]))]
                fit_kw["eval_metric"] = "average_precision"
                fit_kw["callbacks"] = [
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)
                ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y, sample_weight=sample_weight, **fit_kw)
        else:  # pragma: no cover - fallback path
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_iter=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_leaf_nodes=cfg.num_leaves,
                min_samples_leaf=cfg.min_child_samples,
                l2_regularization=cfg.reg_lambda,
                random_state=cfg.random_state,
                early_stopping=False,
            )
            model.fit(X, y, sample_weight=sample_weight)
        self.model_ = model
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        if isinstance(self.model_, float):
            return np.full(len(X), self.model_, dtype="float64")
        X = _as_frame(X)[self.feature_names_]
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        if isinstance(self.model_, float) or self.model_ is None:
            return pd.Series(dtype="float64")
        if _HAS_LGB and hasattr(self.model_, "booster_"):
            raw = self.model_.booster_.feature_importance(importance_type="gain")
        else:  # pragma: no cover
            from sklearn.inspection import permutation_importance  # noqa: F401

            raw = np.ones(len(self.feature_names_))
        s = pd.Series(raw, index=self.feature_names_, dtype="float64")
        total = s.sum()
        return (s / total) if total > 0 else s


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def make_scorer(config: ModelConfig | None = None) -> FraudScorer:
    """Factory used by the simulator so every policy arm gets an identical model class."""
    return GBDTScorer(config)
