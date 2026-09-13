"""Turning propensities into training weights.

The textbook weight is :math:`1/e(x)`. Used raw it is a variance disaster: with 2 % exploration
every explored row carries weight 50, so a few hundred explored transactions can outvote
hundreds of thousands of ordinary ones. Three standard stabilisers are implemented, and the
ablation in ``docs/RESULTS.md`` shows each one's contribution:

``clip``
    Truncate weights at a quantile of the weight distribution. Trades a little bias for a large
    variance reduction; the classic Ionides truncated-IPW estimator.
``self_normalise``
    Divide by the mean weight (Hajek / SNIPS). Removes the scale sensitivity that otherwise
    makes the effective learning rate depend on the exploration rate.
``stabilise``
    Multiply by the marginal observation rate, :math:`w = P(O{=}1)/e(x)`. Keeps the weighted
    sample size close to the raw one, which matters because the tree learner's
    ``min_child_samples`` is expressed in sample counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightConfig:
    """How to convert propensities into sample weights."""

    scheme: str = "ipw"          # {"none", "ipw"}
    #: Quantile truncation. Off by default, and that default is load-bearing: explored rows are
    #: well under 1 % of the training set, so *any* upper-quantile clip truncates precisely the
    #: rows the correction exists to amplify. Use `max_weight` (tied to the propensity floor)
    #: instead, and turn this on only for the variance ablation.
    clip_quantile: float | None = None
    #: Absolute cap, set to 1 / propensity_floor by default (0.004 -> 250).
    max_weight: float | None = 250.0
    self_normalise: bool = True
    stabilise: bool = True
    #: Extra multiplier on positives. Independent of bias correction; kept explicit so the
    #: class-imbalance knob and the selection-bias knob never get confused for each other.
    positive_boost: float = 1.0

    def __post_init__(self) -> None:
        if self.scheme not in {"none", "ipw"}:
            raise ValueError(f"unknown weighting scheme: {self.scheme}")
        if self.clip_quantile is not None and not 0 < self.clip_quantile <= 1:
            raise ValueError("clip_quantile must lie in (0, 1]")


def build_weights(
    propensity: np.ndarray,
    y: np.ndarray | None = None,
    config: WeightConfig | None = None,
) -> np.ndarray:
    """Build per-sample training weights from observation propensities.

    Parameters
    ----------
    propensity:
        ``e(x)`` for the *observed* rows only (these are the rows that reach training).
    y:
        Labels, used only for ``positive_boost``.
    """
    cfg = config or WeightConfig()
    e = np.asarray(propensity, dtype="float64")
    n = e.size
    if n == 0:
        return np.zeros(0, dtype="float64")

    if cfg.scheme == "none":
        w = np.ones(n, dtype="float64")
    else:
        w = 1.0 / np.clip(e, 1e-6, 1.0)
        if cfg.stabilise:
            # P(O=1) estimated on the observed sample via the weights themselves: the weighted
            # count is an estimate of the underlying population size, so n_obs / n_pop_hat.
            w = w * (n / w.sum())
        if cfg.clip_quantile is not None and n > 20:
            w = np.minimum(w, np.quantile(w, cfg.clip_quantile))
        if cfg.max_weight is not None:
            w = np.minimum(w, cfg.max_weight)
        if cfg.self_normalise:
            mean = w.mean()
            if mean > 0:
                w = w / mean

    if y is not None and cfg.positive_boost != 1.0:
        w = w * np.where(np.asarray(y) == 1, cfg.positive_boost, 1.0)
    return w


def effective_sample_size(w: np.ndarray) -> float:
    """Kish ESS: ``(sum w)^2 / sum w^2``. Equals ``n`` for uniform weights."""
    w = np.asarray(w, dtype="float64")
    denom = np.sum(w**2)
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0
