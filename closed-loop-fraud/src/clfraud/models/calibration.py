"""Probability calibration.

Thresholds in this project are set by *quantile* on the score distribution, so calibration is
not load-bearing for the decline decision. It matters for two other things:

1. The doubly-robust off-policy estimator needs a reward model whose outputs are probabilities,
   not just rankings.
2. Selection-bias correction changes the *level* of predicted risk, not only its ordering, and
   the calibration drift over cycles is itself a reported diagnostic (``docs/RESULTS.md``).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Monotone, non-parametric mapping from raw score to probability."""

    def __init__(self) -> None:
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted_ = False

    def fit(self, scores: np.ndarray, y: np.ndarray, sample_weight=None) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype="float64")
        y = np.asarray(y, dtype="float64")
        if len(np.unique(y)) < 2:
            self.fitted_ = False
            return self
        self.iso_.fit(scores, y, sample_weight=sample_weight)
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype="float64")
        return self.iso_.predict(scores) if self.fitted_ else scores


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Binned |confidence - accuracy| gap, weighted by bin mass.

    Quantile bins rather than equal-width: with a 3.5 % base rate, equal-width bins put >99 % of
    the mass in the first bin and report a meaningless number near zero.
    """
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if len(y) == 0:
        return float("nan")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")
