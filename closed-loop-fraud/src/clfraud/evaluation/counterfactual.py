r"""Off-policy evaluation: scoring a candidate policy from logs a different policy produced.

The practical question behind this module: *before* deploying a new threshold or a new model,
can you estimate what it would have cost, using only the decisions the old policy actually made?
Everything the logs contain was selected by the incumbent, so the naive answer -- average the
outcome over logged decisions -- measures the incumbent, not the candidate.

Notation. A logged record is :math:`(x, a, r, p)`: features, the action the logging policy took,
the reward observed for that action, and :math:`p = \pi_0(a \mid x)`, the probability the logging
policy took it. For a candidate :math:`\pi`, the three standard estimators are

.. math::
   \hat V_{\text{IPS}} &= \frac{1}{n}\sum_i \frac{\pi(a_i \mid x_i)}{p_i} r_i \\
   \hat V_{\text{SNIPS}} &= \Big(\sum_i \tfrac{\pi(a_i \mid x_i)}{p_i} r_i\Big) \Big/
                            \Big(\sum_i \tfrac{\pi(a_i \mid x_i)}{p_i}\Big) \\
   \hat V_{\text{DR}} &= \frac{1}{n}\sum_i \Big[\hat r(x_i, \pi) +
        \frac{\pi(a_i \mid x_i)}{p_i}\big(r_i - \hat r(x_i, a_i)\big)\Big]

IPS is unbiased but its variance blows up when the candidate visits regions the logger avoided.
SNIPS trades a small bias for a bounded estimate and is almost always the better default. DR
adds a reward model and is unbiased if *either* the propensities or the reward model are right,
which is the practical reason to prefer it: you rarely know which one you got wrong.

``experiments/run_all.py`` checks these against the simulator's oracle truth -- the one setting
where the counterfactual is actually knowable -- and reports the error of each.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OPEResult:
    """An off-policy estimate with a normal-approximation interval."""

    estimate: float
    std_error: float
    ess: float
    estimator: str
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.estimate - 1.96 * self.std_error, self.estimate + 1.96 * self.std_error)

    def error_vs(self, truth: float) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "truth": truth,
            "abs_error": abs(self.estimate - truth),
            "rel_error": abs(self.estimate - truth) / abs(truth) if truth else float("nan"),
            "covers_truth": float(self.ci95[0] <= truth <= self.ci95[1]),
            "ess": self.ess,
        }


def _importance_weights(
    target_prob: np.ndarray, logging_prob: np.ndarray, clip: float | None
) -> np.ndarray:
    w = np.asarray(target_prob, dtype="float64") / np.clip(
        np.asarray(logging_prob, dtype="float64"), 1e-9, None
    )
    return np.minimum(w, clip) if clip is not None else w


def ips_estimate(
    reward: np.ndarray,
    target_prob: np.ndarray,
    logging_prob: np.ndarray,
    clip: float | None = None,
) -> OPEResult:
    """Inverse propensity scoring. Unbiased; use ``clip`` when the overlap is poor."""
    r = np.asarray(reward, dtype="float64")
    w = _importance_weights(target_prob, logging_prob, clip)
    vals = w * r
    n = r.size
    ess = float(w.sum() ** 2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0
    return OPEResult(
        estimate=float(vals.mean()) if n else float("nan"),
        std_error=float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        ess=ess,
        estimator="ips",
        n=n,
    )


def snips_estimate(
    reward: np.ndarray,
    target_prob: np.ndarray,
    logging_prob: np.ndarray,
    clip: float | None = None,
) -> OPEResult:
    """Self-normalised IPS (Hajek). Bounded by the reward range; far lower variance than IPS."""
    r = np.asarray(reward, dtype="float64")
    w = _importance_weights(target_prob, logging_prob, clip)
    denom = w.sum()
    n = r.size
    if denom <= 0 or n == 0:
        return OPEResult(float("nan"), float("nan"), 0.0, "snips", n)
    est = float(np.sum(w * r) / denom)
    # Delta-method standard error for a ratio estimator.
    resid = w * (r - est)
    se = float(np.sqrt(np.sum(resid**2)) / denom) if n > 1 else float("nan")
    ess = float(denom**2 / np.sum(w**2))
    return OPEResult(est, se, ess, "snips", n)


def doubly_robust_estimate(
    reward: np.ndarray,
    target_prob: np.ndarray,
    logging_prob: np.ndarray,
    reward_hat_logged: np.ndarray,
    reward_hat_target: np.ndarray,
    clip: float | None = None,
) -> OPEResult:
    """Doubly robust estimator.

    Parameters
    ----------
    reward_hat_logged:
        Reward model evaluated at the action the logging policy took.
    reward_hat_target:
        Reward model's expectation under the *target* policy's action distribution.

    Consistent if either the propensities or the reward model is correct -- the estimator that
    survives being wrong about one thing.
    """
    r = np.asarray(reward, dtype="float64")
    w = _importance_weights(target_prob, logging_prob, clip)
    base = np.asarray(reward_hat_target, dtype="float64")
    corr = w * (r - np.asarray(reward_hat_logged, dtype="float64"))
    vals = base + corr
    n = r.size
    ess = float(w.sum() ** 2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0
    return OPEResult(
        estimate=float(vals.mean()) if n else float("nan"),
        std_error=float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        ess=ess,
        estimator="dr",
        n=n,
    )


def policy_value_from_logs(
    y_true_observed: np.ndarray,
    amounts: np.ndarray,
    approved_by_logger: np.ndarray,
    propensity: np.ndarray,
    candidate_declines: np.ndarray,
    clip: float | None = None,
) -> dict[str, OPEResult]:
    r"""Estimate a candidate decline policy's *fraud value blocked* from one cycle's logs.

    The quantity is a share of total fraud value:

    .. math::  V(\pi) = \frac{\sum_i \mathbb{1}[\pi \text{ declines } i]\, y_i a_i}
                              {\sum_i y_i a_i}

    Fraud is only observable on approved transactions, so both sums are estimated by weighting
    the observed rows by :math:`1/e_i`:

    .. math::
       \hat V_{\text{IPS}} &= \frac{1}{A}\sum_{i:\,O_i=1} \frac{d_i y_i a_i}{e_i}
       \qquad (A \text{ = true total fraud value, known only to the simulator}) \\
       \hat V_{\text{SNIPS}} &= \frac{\sum_{i:\,O_i=1} d_i y_i a_i / e_i}
                                       {\sum_{i:\,O_i=1} y_i a_i / e_i}

    SNIPS needs no oracle denominator at all -- it estimates the total fraud value from the same
    weighted logs -- which is what makes it the one an issuer could actually compute.

    A note on ``clip``, because getting it wrong silently halves the answer: with an exploration
    propensity floor of 0.004 the correct weight on an explored row is 250. Clipping at 50, a
    perfectly reasonable-looking default, discards four fifths of the evidence about the decline
    region and biases :math:`\hat V` downward by a factor of several. Clip above
    ``1 / propensity_floor`` or not at all.
    """
    obs = np.asarray(approved_by_logger, dtype=bool)
    y = np.asarray(y_true_observed, dtype="float64")
    amt = np.asarray(amounts, dtype="float64")
    e = np.asarray(propensity, dtype="float64")
    cand = np.asarray(candidate_declines, dtype=bool)

    total_fraud_value = float((y * amt).sum())
    m = obs & (e > 0)
    n = int(m.sum())
    if n == 0 or total_fraud_value <= 0:  # pragma: no cover - degenerate cycle
        nan = OPEResult(float("nan"), float("nan"), 0.0, "ips", 0)
        return {"ips": nan, "snips": OPEResult(float("nan"), float("nan"), 0.0, "snips", 0)}

    w = 1.0 / np.maximum(e[m], 1e-12)
    if clip is not None:
        w = np.minimum(w, clip)
    fraud_value = y[m] * amt[m]
    blocked = cand[m] * fraud_value

    ess = float(w.sum() ** 2 / np.sum(w**2))
    ips = float(np.sum(w * blocked) / total_fraud_value)
    ips_se = float(np.sqrt(np.sum((w * blocked) ** 2)) / total_fraud_value)

    denom = float(np.sum(w * fraud_value))
    if denom > 0:
        snips = float(np.sum(w * blocked) / denom)
        resid = w * (blocked - snips * fraud_value)
        snips_se = float(np.sqrt(np.sum(resid**2)) / denom)
    else:  # pragma: no cover
        snips, snips_se = float("nan"), float("nan")

    return {
        "ips": OPEResult(ips, ips_se, ess, "ips", n),
        "snips": OPEResult(snips, snips_se, ess, "snips", n),
    }
