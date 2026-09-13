"""Probability calibration, and the diagnostics that say whether it worked.

Why this module is load-bearing rather than a finishing touch
------------------------------------------------------------
The decision rule in :mod:`deliveryrisk.policy` intervenes when

    u_a * p * L_i  >  c_a          i.e.   p  >  c_a / (u_a * L_i)

The comparison is against a number on the **probability** axis. A model that ranks perfectly but
reports 0.32 where the truth is 0.11 will cross that line on the wrong orders, and no amount of
AUC fixes it. Ranking metrics are invariant to any monotone transform of the score; the
threshold is not. That asymmetry is the whole reason a calibration step sits between the model
and the policy, and it is measured in ``docs/RESULTS.md`` rather than assumed.

Calibrators are fitted on a **validation block the risk model never saw**. Fitting isotonic
regression on the training fold gives a calibrator fitted to the model's in-sample optimism, and
it will happily undo itself out of sample.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

METHODS = ("none", "platt", "isotonic")


@dataclass
class Calibrator:
    """Wraps a fitted monotone map from raw score to probability."""

    method: str = "isotonic"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r}; expected {METHODS}")
        self._model = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Calibrator:
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.method == "none":
            return self
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p_raw, y)
        else:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(C=1e6, max_iter=1000)
            self._model.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=float)
        if self.method == "none" or self._model is None:
            return p_raw
        if self.method == "isotonic":
            return np.clip(self._model.predict(p_raw), 1e-6, 1 - 1e-6)
        return np.clip(self._model.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ diagnostics
def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Observed frequency against predicted probability, in equal-count bins.

    Equal-count, not equal-width: 92 % of orders sit below p = 0.2, and equal-width bins would
    spend most of their resolution on a region that is nearly empty -- while the region the
    policy actually acts in gets one bar.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"),
             p_lo=("p", "min"), p_hi=("p", "max"))
        .reset_index()
    )
    out["gap"] = out.observed - out.predicted
    return out


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> dict[str, float]:
    """ECE, MCE, Brier and its decomposition, plus the calibration slope/intercept.

    The slope is the one to look at first: a slope below 1 means the scores are over-spread --
    the model is more confident than the data supports -- which is the usual state of an
    uncalibrated booster and exactly what over-intervening on the top decile looks like.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    tab = reliability_table(y, p, n_bins)
    w = tab.n / tab.n.sum()
    ece = float((w * tab.gap.abs()).sum())
    mce = float(tab.gap.abs().max())
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    reliability = float((w * (tab.predicted - tab.observed) ** 2).sum())
    resolution = float((w * (tab.observed - base) ** 2).sum())
    uncertainty = base * (1 - base)
    slope, intercept = _calibration_slope(y, p)
    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "brier_reliability": reliability,
        "brier_resolution": resolution,
        "brier_uncertainty": uncertainty,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted": float(p.mean()),
        "base_rate": base,
    }


def _calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Regress the outcome on the predicted log-odds. Perfect calibration gives (1, 0)."""
    from sklearn.linear_model import LogisticRegression

    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def fit_calibrated(
    model, X_train, y_train, X_valid, y_valid, method: str = "isotonic"
) -> tuple[object, Calibrator]:
    """Fit the risk model on train, then the calibrator on validation predictions."""
    model.fit(X_train, y_train)
    cal = Calibrator(method).fit(model.predict_proba(X_valid), y_valid)
    return model, cal
