"""The leakage audit: what a two-sided aggregate manufactures, and what it costs.

The leak that actually happens in this kind of work is not target encoding and it is not a
mislabelled column. It is one line::

    df["seller_late_rate"] = df.groupby("seller_id").y_late.transform("mean")

which is a *whole-table* aggregate: for an order approved in March it includes what that seller
did in April, and it includes the order's own outcome. The resulting column looks completely
ordinary. It has a sensible range, a sensible distribution, and a sensible name.

The audit fits one model per feature mode and scores the same held-out orders twice: once with
features that can see forward -- the number that goes in the deck -- and once with the causal
features a production system can actually compute at approval time. The difference between those
two rows is the part of the offline result that was never real.

It runs the subtler variant too. Once someone points out that ``transform('mean')`` includes the
row's own outcome, the natural fix is to subtract it: a leave-one-out aggregate. The audit exists
partly to show that this fix buys nothing here. The dominant leak is not the row's own outcome in
its own average; it is the centred window over a *shared* entity -- a carrier's +/-30-day late
rate, which reports the disruption the order is currently inside using parcels that had not yet
been delivered when the decision was made. Leave-one-out cannot touch that, because the rows
doing the damage belong to other orders.

Three things are measured rather than one:

1. **Ranking**, because that is what gets reported.
2. **Calibration**, because the policy threshold lives on the probability axis and the leak
   moves the probabilities, not just the order.
3. **Realised contribution**, because a leaky model does not merely look better than it is -- it
   makes worse decisions, and the money is the only unit in which "worse" is unambiguous.

The breakdown by seller history depth is the part worth reading twice. A two-sided aggregate
helps most exactly where an honest one has nothing: a seller's *first* orders. Those are the
orders a delivery-risk model exists to flag, and they are the ones the leak flatters hardest.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from deliveryrisk.config import ExperimentConfig
from deliveryrisk.data.db import Database
from deliveryrisk.evaluation.decisions import evaluate_policy
from deliveryrisk.evaluation.metrics import ranking_metrics
from deliveryrisk.features.build import build_features, labelled, split_by_time
from deliveryrisk.models.calibration import Calibrator, calibration_metrics
from deliveryrisk.pipeline import Prepared, build_policy_inputs, fit_models

log = logging.getLogger(__name__)

SCENARIOS = (
    ("honest (point-in-time)", "point_in_time", "point_in_time"),
    ("whole-table, as reported offline", "two_sided", "two_sided"),
    ("whole-table, as served", "two_sided", "point_in_time"),
    ("leave-one-out, as reported offline", "two_sided_loo", "two_sided_loo"),
    ("leave-one-out, as served", "two_sided_loo", "point_in_time"),
)


MODES_USED = ("point_in_time", "two_sided", "two_sided_loo")


def build_both_modes(db: Database, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """The same feature definitions under each window frame."""
    out = {}
    for mode in MODES_USED:
        fc = type(cfg.features)(**{**asdict(cfg.features), "mode": mode})
        out[mode] = build_features(db, fc, materialise=(mode == "point_in_time"))
    return out


def run_leak_audit(
    prep: Prepared, cfg: ExperimentConfig, *, model_kind: str = "gbdt"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (headline table, breakdown by seller history depth)."""
    frames = build_both_modes(prep.db, cfg)
    splits = {
        mode: split_by_time(labelled(df), cfg.train_share, cfg.valid_share)
        for mode, df in frames.items()
    }
    # The modes produce the same rows in the same order, so the splits align order for order --
    # which is what makes "same model, different features at serving time" a valid comparison.
    pit_ids = splits["point_in_time"][2].order_id.to_numpy()
    for mode in MODES_USED[1:]:
        assert (pit_ids == splits[mode][2].order_id.to_numpy()).all()

    fitted = {}
    for mode in MODES_USED:
        train, valid, _ = splits[mode]
        fitted[mode] = fit_models(cfg, train, valid, kind=model_kind)

    cost_model = cfg.economics.with_categories(prep.features.category.astype(str).unique())
    rows, breakdown = [], []
    for label, trained_on, scored_with in SCENARIOS:
        models = fitted[trained_on]
        test = splits[scored_with][2]
        valid = splits[scored_with][1]
        raw = models.risk.predict_proba(test)
        # Recalibrate on the block the scoring features come from: serving a leak-trained model
        # on causal features and then *not* recalibrating would confound the two effects.
        cal = Calibrator(cfg.calibration_method).fit(
            models.risk.predict_proba(valid), valid.y_late.to_numpy(dtype=float)
        )
        p = cal.transform(raw)
        y = test.y_late.to_numpy(dtype=float)
        inp = build_policy_inputs(cfg, models, test, prep.truth, p, cost_model)
        actions = inp.table_cause.best_action()
        outcome = evaluate_policy("margin_cause_aware", actions, inp.truth, inp.margin,
                                  inp.late_cost, inp.costs, seed=cfg.seed)
        rows.append(
            {
                "scenario": label,
                "trained_on": trained_on,
                "scored_with": scored_with,
                **ranking_metrics(y, p),
                **{k: v for k, v in calibration_metrics(y, p).items()
                   if k in ("ece", "calibration_slope", "brier")},
                "delta_per_1k": outcome.delta_per_1k,
                "treated_share": outcome.treated_share,
            }
        )
        breakdown.extend(_by_history_depth(label, test, p))
    return pd.DataFrame(rows), pd.DataFrame(breakdown)


#: A seller with fewer than this many prior deliveries at decision time has, in practice, no
#: history -- which is exactly the regime a whole-table aggregate invents one for.
NEW_SELLER_PRIORS = 5


def _by_history_depth(label: str, test: pd.DataFrame, p: np.ndarray) -> list[dict]:
    depth = test.seller_n_prior.fillna(0).to_numpy()
    buckets = {
        f"new seller (<{NEW_SELLER_PRIORS} prior)": depth < NEW_SELLER_PRIORS,
        "5-49 prior": (depth >= NEW_SELLER_PRIORS) & (depth < 50),
        "50+ prior": depth >= 50,
    }
    out = []
    y = test.y_late.to_numpy(dtype=float)
    for name, mask in buckets.items():
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            continue
        out.append({"scenario": label, "bucket": name, "n": int(mask.sum()),
                    **ranking_metrics(y[mask], p[mask])})
    return out
