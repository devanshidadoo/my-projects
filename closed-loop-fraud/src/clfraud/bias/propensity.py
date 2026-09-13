r"""Propensity of *label observation*, and what it buys.

The problem
-----------
Let :math:`A_i \in \{0,1\}` be "approved" and :math:`Y_i` the fraud label. A chargeback only
arrives for an approved transaction, so the label is observed iff :math:`A_i = 1`:

.. math::  O_i = A_i, \qquad  \mathbb{E}[\ell(f(X_i), Y_i) \mid O_i = 1]
           \neq \mathbb{E}[\ell(f(X_i), Y_i)]

Training on observed rows optimises the wrong expectation. Inverse-propensity weighting fixes
it when the *positivity* condition holds:

.. math::  e(x) = P(O = 1 \mid X = x) > 0 \quad \text{for all } x,

because then :math:`\mathbb{E}\big[\tfrac{O}{e(X)} \ell\big] = \mathbb{E}[\ell]`.

The catch that motivates the whole exploration arm
--------------------------------------------------
A deterministic threshold policy :math:`A = \mathbb{1}[s(X) < \tau]` gives
:math:`e(x) \in \{0, 1\}`. Positivity fails *exactly* in the region the model considers risky --
the region that matters most. No amount of cleverness recovers it from the logs: the data is not
missing at random, it is missing by construction.

Randomised exploration is what makes the estimand identifiable. Approving a small fraction
:math:`\epsilon` of would-be declines sets :math:`e(x) \ge \epsilon > 0` everywhere, so the IPW
estimator becomes unbiased and finite-variance. Exploration is not a heuristic bolted onto the
policy; it is the identification strategy.

Two propensity sources
----------------------
``known_propensity``
    The exact logging propensity. Available here because the policy is ours -- and available in
    a real issuer too, if decision randomness is logged. Always prefer it: it is exact, and
    estimation error in :math:`\hat e` shows up as bias in :math:`1/\hat e`.
``PropensityModel``
    Estimates :math:`\hat e(x)` from features. Needed when the logs record decisions but not the
    randomisation (the common case in legacy stacks). ``experiments/run_all.py`` reports both so
    the cost of estimating rather than logging the propensity is visible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

log = logging.getLogger(__name__)


def known_propensity(
    declined_by_policy: np.ndarray,
    explore_prob: np.ndarray,
) -> np.ndarray:
    """Exact ``P(label observed | x, policy)`` from the decision log.

    Parameters
    ----------
    declined_by_policy:
        Whether the *deterministic* part of the policy wanted to decline (before exploration).
    explore_prob:
        Per-transaction probability the exploration rule overrode that decline.

    Returns
    -------
    Propensity in ``[0, 1]``. Rows the policy would approve have propensity 1; rows it would
    decline have propensity equal to their exploration probability, which is 0 wherever
    exploration is switched off -- a positivity violation this function does not hide.
    """
    declined = np.asarray(declined_by_policy, dtype=bool)
    q = np.asarray(explore_prob, dtype="float64")
    return np.where(declined, q, 1.0)


@dataclass
class PropensityConfig:
    max_iter: int = 200
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    calibrate: bool = True
    random_state: int = 11


class PropensityModel:
    """Estimates ``e(x) = P(observed | x)`` from features available for *every* transaction.

    Note what is and is not available: the issuer computes features for declined transactions
    too -- it just never learns their outcome. So ``e(x)`` is estimable from a fully observed
    binary target, unlike the fraud label itself. That asymmetry is what makes the correction
    possible at all.
    """

    def __init__(self, config: PropensityConfig | None = None) -> None:
        self.config = config or PropensityConfig()
        self.model_ = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, observed: np.ndarray) -> PropensityModel:
        observed = np.asarray(observed).astype(int)
        self.feature_names_ = list(X.columns)
        cfg = self.config
        base = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        if len(np.unique(observed)) < 2:
            self.model_ = float(observed.mean())
            return self
        if cfg.calibrate:
            # Weights are 1/e_hat, so errors near e_hat ~ 0 explode. Isotonic calibration on a
            # held-out split keeps the small-propensity tail honest.
            self.model_ = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            self.model_ = base
        self.model_.fit(X, observed)
        return self

    def predict(self, X: pd.DataFrame, floor: float = 1e-3) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("propensity model is not fitted")
        if isinstance(self.model_, float):
            return np.full(len(X), max(self.model_, floor))
        p = self.model_.predict_proba(X[self.feature_names_])[:, 1]
        return np.clip(p, floor, 1.0)


def overlap_diagnostics(propensity: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Is the corrected estimator actually identified, and how much does it cost in variance?

    ``positivity_violation_rate``
        Share of rows with propensity 0. Any value above 0 means part of the feature space is
        unrecoverable no matter how the weights are built -- report it, never paper over it.
    ``ess_ratio``
        Kish effective sample size of the weighted training set divided by its raw size. IPW
        buys unbiasedness with variance; this says how much of the sample the weights actually
        leave you. Below ~0.2 the corrected fit is noisier than the biased one it replaces.
    ``max_weight_share``
        Share of total weight carried by the single heaviest row -- the practical failure mode,
        where one explored transaction dictates the decision boundary.
    """
    e = np.asarray(propensity, dtype="float64")
    obs = np.asarray(observed, dtype=bool)
    out = {
        "positivity_violation_rate": float(np.mean(e <= 0.0)),
        "min_propensity": float(e[e > 0].min()) if (e > 0).any() else 0.0,
        "median_propensity": float(np.median(e)),
        "observed_rate": float(obs.mean()),
    }
    w = np.where(obs & (e > 0), 1.0 / np.maximum(e, 1e-12), 0.0)
    pos = w[w > 0]
    if pos.size:
        out["ess"] = float(pos.sum() ** 2 / np.sum(pos**2))
        out["ess_ratio"] = out["ess"] / pos.size
        out["max_weight_share"] = float(pos.max() / pos.sum())
        out["weight_p99"] = float(np.quantile(pos, 0.99))
    else:  # pragma: no cover
        out.update({"ess": 0.0, "ess_ratio": 0.0, "max_weight_share": 0.0, "weight_p99": 0.0})
    return out
