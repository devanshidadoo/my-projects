"""Ranking, capture and lift.

The decile table is the operational currency of this kind of work -- "what share of late orders
do we see if we look at the riskiest two deciles" is the question an ops lead actually asks. It
is reported here alongside PR-AUC because it is the metric that survives contact with a review,
and because it is the one that a leaky feature inflates most visibly.

It is also, on its own, not a decision. Everything in :mod:`deliveryrisk.policy` exists because
capture at a fixed review rate says nothing about whether the review was worth doing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Risk deciles, highest risk first, with per-decile and cumulative capture."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    # rank-then-cut so that ties in a coarse score do not collapse a decile to nothing
    df["rank"] = df.p.rank(method="first", ascending=False)
    df["decile"] = np.ceil(df["rank"] / (len(df) / n_bins)).clip(1, n_bins).astype(int)
    g = df.groupby("decile").agg(n=("y", "size"), n_late=("y", "sum"),
                                 mean_score=("p", "mean")).reset_index()
    total_late = float(df.y.sum())
    base = float(df.y.mean())
    g["late_rate"] = g.n_late / g.n
    g["lift"] = g.late_rate / base if base > 0 else np.nan
    g["capture"] = g.n_late / total_late if total_late > 0 else np.nan
    g["cum_capture"] = g.capture.cumsum()
    g["cum_share_of_orders"] = g.n.cumsum() / g.n.sum()
    return g


def capture_at_top(y: np.ndarray, p: np.ndarray, deciles: int = 2, n_bins: int = 10) -> float:
    """Share of all late orders that fall in the top ``deciles`` risk deciles."""
    tab = decile_table(y, p, n_bins)
    return float(tab.loc[tab.decile <= deciles, "capture"].sum())


def recall_at_rate(y: np.ndarray, p: np.ndarray, rate: float) -> float:
    """Share of late orders caught if the top ``rate`` of orders is reviewed."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(rate * len(p))))
    idx = np.argsort(-p, kind="stable")[:k]
    total = y.sum()
    return float(y[idx].sum() / total) if total > 0 else float("nan")


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {
        "n": float(len(y)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "capture_top1": capture_at_top(y, p, 1),
        "capture_top2": capture_at_top(y, p, 2),
        "capture_top3": capture_at_top(y, p, 3),
        "lift_top1": float("nan"),
        "recall_at_05": recall_at_rate(y, p, 0.05),
        "recall_at_10": recall_at_rate(y, p, 0.10),
    }
    tab = decile_table(y, p)
    out["lift_top1"] = float(tab.loc[tab.decile == 1, "lift"].iloc[0])
    return out


def bootstrap_ci(
    values: np.ndarray, statistic=np.mean, n_boot: int = 400, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap over orders.

    Used for the contribution numbers, where the per-order quantity is dominated by a small
    number of expensive late orders: a difference of a few units per thousand orders is not worth
    reporting without an interval around it.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = statistic(values[idx], axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
