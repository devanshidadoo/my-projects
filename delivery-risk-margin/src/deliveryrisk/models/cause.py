"""Why an order is at risk, not just whether.

The risk model answers ``P(late)``. The decision needs more than that, because the two actions
act on different mechanisms: a seller escalation compresses handling, buying express compresses
transit. So this model answers ``P(cause | late)`` over four disjoint classes:

    handling_only   only a nudge can still save it
    transit_only    only expediting can
    either          both mechanisms have enough slack that either action alone would do
    neither         it is going to be late whatever is bought

Factorising as ``P(late & cause) = P(late) * P(cause | late)`` -- rather than fitting a single
five-class model -- keeps the calibration problem where it belongs. Only ``P(late)`` is compared
against a currency threshold, so only ``P(late)`` has to be calibrated; the conditional model
supplies a *mix*, and a mix is a ratio, which is far less sensitive. It also means the
conditional model trains on the late orders alone (7.9 % of rows) without distorting the
marginal.

The labels are observable
-------------------------
This is the part that makes the whole cause-aware policy implementable on real data rather than
only in a simulator. A completed order records its handling time and its transit time separately
(``approved -> pickup -> delivered``), so "would this order have been on time had handling run at
the operational floor" is arithmetic on logged timestamps, not a counterfactual. What is *not*
identified from the log is whether an intervention would have achieved that compression -- that
is ``UpliftBelief``, and it is swept rather than asserted.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deliveryrisk.models.base import ModelConfig, apply_feature_space, build_feature_space
from deliveryrisk.models.gbdt import HAS_LGB
from deliveryrisk.policy.actions import CAUSES

log = logging.getLogger(__name__)


def cause_labels(df: pd.DataFrame) -> pd.Series:
    """Disjoint cause class for every late order; ``NaN`` for orders that were not late."""
    h = df.y_handling_fixable.astype("float")
    t = df.y_transit_fixable.astype("float")
    late = df.y_late.astype("float") > 0
    out = pd.Series(index=df.index, dtype=object)
    out[late & (h > 0) & (t == 0)] = "handling_only"
    out[late & (t > 0) & (h == 0)] = "transit_only"
    out[late & (h > 0) & (t > 0)] = "either"
    out[late & (h == 0) & (t == 0)] = "neither"
    return out


def population_mix(df: pd.DataFrame) -> dict[str, float]:
    """Share of each cause among late orders. The best a policy without a cause model can do."""
    lab = cause_labels(df).dropna()
    if lab.empty:
        return dict.fromkeys(CAUSES, 1.0 / len(CAUSES))
    counts = lab.value_counts(normalize=True)
    return {c: float(counts.get(c, 0.0)) for c in CAUSES}


class CauseModel:
    """Multiclass ``P(cause | late)``, fitted on late orders only."""

    name = "cause"

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig(kind="gbdt", n_estimators=250, num_leaves=15,
                                      min_child_samples=40)
        self.space = None
        self.model = None
        self.classes_: list[str] = list(CAUSES)
        self.fallback_mix: dict[str, float] = dict.fromkeys(CAUSES, 1.0 / len(CAUSES))

    def fit(self, df: pd.DataFrame) -> CauseModel:
        self.fallback_mix = population_mix(df)
        lab = cause_labels(df).dropna()
        X = df.loc[lab.index]
        present = [c for c in CAUSES if (lab == c).sum() >= 25]
        if len(present) < 2 or len(lab) < 200:
            log.warning(
                "cause model falling back to the population mix (%d labelled late orders, "
                "%d usable classes)", len(lab), len(present)
            )
            return self
        keep = lab.isin(present)
        lab, X = lab[keep], X[keep]
        self.classes_ = present
        self.space = build_feature_space(X, self.cfg)
        Xt = apply_feature_space(X, self.space)
        y = pd.Categorical(lab, categories=present).codes
        c = self.cfg
        if HAS_LGB:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=len(present),
                n_estimators=c.n_estimators,
                learning_rate=c.learning_rate,
                num_leaves=c.num_leaves,
                min_child_samples=c.min_child_samples,
                reg_lambda=c.reg_lambda,
                random_state=c.seed,
                n_jobs=-1,
                verbose=-1,
            )
            self.model.fit(Xt, y, categorical_feature=self.space.categorical)
        else:  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                max_iter=c.n_estimators, learning_rate=c.learning_rate,
                max_leaf_nodes=c.num_leaves, min_samples_leaf=c.min_child_samples,
                categorical_features=[Xt.columns.get_loc(col) for col in self.space.categorical],
                random_state=c.seed,
            ).fit(Xt, y)
        return self

    def predict_mix(self, df: pd.DataFrame) -> pd.DataFrame:
        """``P(cause | late)`` for every row, columns in :data:`CAUSES` order."""
        n = len(df)
        if self.model is None or self.space is None:
            return pd.DataFrame({c: np.full(n, self.fallback_mix[c]) for c in CAUSES},
                                index=df.index)
        proba = self.model.predict_proba(apply_feature_space(df, self.space))
        out = pd.DataFrame(0.0, index=df.index, columns=list(CAUSES))
        for j, cls in enumerate(self.classes_):
            out[cls] = proba[:, j]
        return out

    def predict_joint(self, df: pd.DataFrame, p_late: np.ndarray) -> dict[str, np.ndarray]:
        """``P(late & cause)`` -- what :class:`UpliftBelief` consumes."""
        mix = self.predict_mix(df)
        return {c: np.asarray(p_late, dtype=float) * mix[c].to_numpy() for c in CAUSES}
