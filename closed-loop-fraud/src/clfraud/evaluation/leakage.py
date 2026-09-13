"""Quantifying what a leaky feature pipeline buys you, and what it costs at launch.

The claim "these features are point-in-time" is only worth making if the alternative is
measurably different. So the audit runs the leak that actually happens in fraud work: velocity
aggregates computed over a **two-sided** window -- an unshifted ``groupby().rolling()``, a
``merge_asof`` pointing the wrong way, a centred window in a notebook. Every one of those lets a
transaction see the transactions that came after it.

The demonstration is framed as **train/serve skew**, which is how the failure actually arrives:

``honest``
    Fit on point-in-time features, score a future window with point-in-time features. What a
    correct pipeline reports and what it delivers -- the same number, which is the point.
``leaky (as reported offline)``
    Fit on two-sided features, score the future window with two-sided features. This is the
    number that goes in the deck, and it is the best of the three.
``leaky (as served)``
    The *same fitted model*, scoring the same transactions with the causal features a production
    system can actually compute. Nothing about the model changed; only the future disappeared.

The drop from the second row to the third is the part of the offline number that was never real.
It is also not recoverable by retraining, re-tuning, or a bigger model: the signal the model
learned to lean on does not exist at decision time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from clfraud.evaluation.metrics import ranking_metrics
from clfraud.features.point_in_time import (
    leaky_aggregate_baseline,
    leaky_future_window_features,
)
from clfraud.models.base import ModelConfig
from clfraud.models.gbdt import make_scorer

log = logging.getLogger(__name__)


def _episode_opening_mask(df: pd.DataFrame, quiet_hours: float = 24.0) -> np.ndarray:
    """Fraud transactions that open an episode: no other fraud on that card in the prior window.

    These are the ones worth catching. Stopping the twelfth authorisation of a card-testing burst
    is damage control; stopping the first is prevention. They are also where a two-sided window
    does its worst damage, because the opening transaction of a burst has *no* prior velocity and
    all of the burst in its forward window -- offline it looks like the most obvious fraud in the
    dataset, and at serving time it looks like nothing at all.
    """
    fraud = df["is_fraud"].to_numpy().astype(bool)
    card = df["card_id"].astype(str).to_numpy()
    ts = df["ts"].to_numpy(dtype="float64")
    last: dict[str, float] = {}
    opening = np.zeros(len(df), dtype=bool)
    for i in np.flatnonzero(fraud):
        prev = last.get(card[i])
        opening[i] = prev is None or (ts[i] - prev) > quiet_hours * 3_600.0
        last[card[i]] = ts[i]
    return opening


def _opening_recall(y: np.ndarray, opening: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Share of episode-opening fraud that lands in the declined top ``budget`` of scores."""
    n = y.size
    k = max(int(round(budget * n)), 1)
    top = np.zeros(n, dtype=bool)
    top[np.argsort(-scores, kind="stable")[:k]] = True
    total = int(opening.sum())
    return float((top & opening).sum() / total) if total else float("nan")


def run_leak_audit(
    df: pd.DataFrame,
    X_pit: pd.DataFrame,
    holdout_frac: float = 0.25,
    budget: float = 0.04,
    seed: int = 17,
    model_cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    """Train/serve-skew audit over a forward split.

    Parameters
    ----------
    X_pit:
        The strictly causal feature matrix, as built by :func:`build_features`.

    Returns
    -------
    One row per scenario, plus two rows quantifying target-encoding leakage under a random split
    for comparison. ``attrs["leak_inflation_roc"]`` holds the headline gap.
    """
    y = df["is_fraud"].to_numpy().astype(int)
    amounts = df["amount"].to_numpy(dtype="float64")
    n = len(df)
    cut = int(n * (1 - holdout_frac))
    tr, te = np.arange(cut), np.arange(cut, n)
    cfg = model_cfg or ModelConfig(n_estimators=250)

    passthrough = [c for c in X_pit.columns if c.startswith(("v", "raw_")) or c == "mcc"]
    log.info("building two-sided (leaky) velocity features for the audit")
    X_leak = leaky_future_window_features(df, passthrough=passthrough)[X_pit.columns]

    honest = make_scorer(cfg).fit(X_pit.iloc[tr], y[tr])
    leaky = make_scorer(cfg).fit(X_leak.iloc[tr], y[tr])

    opening = _episode_opening_mask(df)[te]
    scenarios = [
        ("honest (point-in-time)", "point_in_time", "point_in_time",
         honest.predict_proba(X_pit.iloc[te])),
        ("leaky, as reported offline", "two_sided_window", "two_sided_window",
         leaky.predict_proba(X_leak.iloc[te])),
        ("leaky, as served", "two_sided_window", "point_in_time",
         leaky.predict_proba(X_pit.iloc[te])),
    ]
    rows = []
    for scenario, trained, scored, sc in scenarios:
        rows.append(
            {
                "scenario": scenario,
                "trained_on": trained,
                "scored_with": scored,
                **ranking_metrics(y[te], sc, budget, amounts[te]),
                "opening_recall_at_budget": _opening_recall(y[te], opening, sc, budget),
            }
        )
    out = pd.DataFrame(rows)

    reported = float(out.loc[1, "recall_at_budget"])
    served = float(out.loc[2, "recall_at_budget"])
    honest_recall = float(out.loc[0, "recall_at_budget"])
    out.attrs["leak_inflation_recall"] = reported - served
    out.attrs["leak_inflation_roc"] = float(out.loc[1, "roc_auc"] - out.loc[2, "roc_auc"])
    out.attrs["honest_beats_served_leaky"] = honest_recall - served
    out.attrs["opening_inflation"] = float(
        out.loc[1, "opening_recall_at_budget"] - out.loc[2, "opening_recall_at_budget"]
    )
    log.info(
        "leak audit: reported %.4f -> served %.4f (%.1f%% of the claimed recall was not real); "
        "honest pipeline delivers %.4f; episode-opening recall %.4f -> %.4f",
        reported, served, 100 * (reported - served) / max(reported, 1e-9), honest_recall,
        out.loc[1, "opening_recall_at_budget"], out.loc[2, "opening_recall_at_budget"],
    )
    return out


def run_target_encoding_audit(
    df: pd.DataFrame,
    X_pit: pd.DataFrame,
    holdout_frac: float = 0.25,
    budget: float = 0.04,
    seed: int = 17,
) -> pd.DataFrame:
    """The other classic leak: whole-dataset ``groupby`` aggregates and target encoding.

    Kept separate from the velocity audit because it fails for a different reason and, on a
    stream where fraud is episodic rather than concentrated on repeat-offender cards, it turns
    out to buy much less than folklore suggests. Reporting that honestly is more useful than
    quietly dropping the comparison that did not cooperate.
    """
    y = df["is_fraud"].to_numpy().astype(int)
    amounts = df["amount"].to_numpy(dtype="float64")
    n = len(df)
    rng = np.random.default_rng(seed)
    cut = int(n * (1 - holdout_frac))

    leaky = leaky_aggregate_baseline(df)
    keep = [c for c in X_pit.columns if c.startswith(("v", "raw_")) or c == "mcc"]
    X_leaky = pd.concat([leaky.reset_index(drop=True), X_pit[keep].reset_index(drop=True)], axis=1)

    perm = rng.permutation(n)
    splits = {
        "random": (np.sort(perm[:cut]), np.sort(perm[cut:])),
        "forward": (np.arange(cut), np.arange(cut, n)),
    }
    rows = []
    for fname, feats in (("point_in_time", X_pit), ("leaky_target_encoding", X_leaky)):
        for sname, (tr, te) in splits.items():
            model = make_scorer(ModelConfig(n_estimators=250)).fit(feats.iloc[tr], y[tr])
            rows.append(
                {
                    "features": fname,
                    "split": sname,
                    **ranking_metrics(
                        y[te], model.predict_proba(feats.iloc[te]), budget, amounts[te]
                    ),
                }
            )
    return pd.DataFrame(rows)
