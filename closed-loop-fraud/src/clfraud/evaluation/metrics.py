"""Metrics for a fraud model that has to make decisions under a budget.

ROC-AUC is close to useless here: at a 3.5 % base rate it is dominated by the ranking of
negatives against each other, and a model can gain AUC while getting worse at the only part of
the score distribution anyone acts on. Every headline number in this project is therefore
**recall at a fixed decline budget** -- of all fraud in the period, what share falls in the
top-k% of scores the policy can afford to decline. It is the quantity an issuer's fraud team
actually negotiates, and it is directly comparable across cycles because the budget is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_budget(
    y: np.ndarray,
    scores: np.ndarray,
    budget: float,
    amounts: np.ndarray | None = None,
) -> float:
    """Share of fraud captured by declining the top ``budget`` fraction of scores.

    With ``amounts``, returns the *value-weighted* recall (share of fraud dollars), which is
    what the loss line in a P&L responds to. Count-recall and value-recall diverge whenever a
    model is better at small fraud than large, so both are reported throughout.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype="float64")
    n = y.size
    if n == 0:
        return float("nan")
    w = np.ones(n) if amounts is None else np.asarray(amounts, dtype="float64")
    total = float((w * y).sum())
    if total <= 0:
        return float("nan")
    k = max(int(round(budget * n)), 1)
    # Descending order; ties broken by index for determinism across runs.
    order = np.argsort(-scores, kind="stable")[:k]
    return float((w[order] * y[order]).sum() / total)


def precision_at_budget(y: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Fraud share among the declined population: the false-positive tax on good customers."""
    y = np.asarray(y).astype(int)
    n = y.size
    if n == 0:
        return float("nan")
    k = max(int(round(budget * n)), 1)
    order = np.argsort(-np.asarray(scores, dtype="float64"), kind="stable")[:k]
    return float(y[order].mean())


def ranking_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    budget: float = 0.03,
    amounts: np.ndarray | None = None,
    extra_budgets: tuple[float, ...] = (0.005, 0.01, 0.05),
) -> dict[str, float]:
    """Full metric block for one cycle, evaluated against oracle labels."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype="float64")
    out: dict[str, float] = {
        "n": float(y.size),
        "n_fraud": float(y.sum()),
        "fraud_rate": float(y.mean()) if y.size else float("nan"),
    }
    if y.size == 0 or len(np.unique(y)) < 2:
        return out | {"roc_auc": float("nan"), "pr_auc": float("nan")}
    out["roc_auc"] = float(roc_auc_score(y, scores))
    out["pr_auc"] = float(average_precision_score(y, scores))
    out["recall_at_budget"] = recall_at_budget(y, scores, budget)
    out["precision_at_budget"] = precision_at_budget(y, scores, budget)
    if amounts is not None:
        out["value_recall_at_budget"] = recall_at_budget(y, scores, budget, amounts)
    for b in extra_budgets:
        out[f"recall_at_{b:g}"] = recall_at_budget(y, scores, b)
    return out


@dataclass
class BusinessCosts:
    """Unit economics for turning decisions into money.

    ``fp_cost_rate``
        Cost of declining a good transaction, as a share of its amount. Not the amount itself:
        a declined customer usually retries or shops elsewhere, so the issuer loses interchange
        plus some attrition risk, not the ticket. 1.5 % is a deliberately conservative stand-in
        and every cost figure scales linearly in it.
    ``recovery_rate``
        Share of fraud losses recovered downstream (representment, network rules). Applied to
        approved fraud so the loss line is net.
    """

    fp_cost_rate: float = 0.015
    recovery_rate: float = 0.15


def business_outcome(
    y: np.ndarray,
    amounts: np.ndarray,
    approved: np.ndarray,
    explored: np.ndarray | None = None,
    costs: BusinessCosts | None = None,
) -> dict[str, float]:
    """Realised P&L of one cycle's decisions.

    ``exploration_fraud_cost`` is isolated deliberately. Comparing an exploring arm's *total*
    loss against a non-exploring arm's conflates two opposite effects: exploration lets some
    fraud through today, and the better model it trains stops more fraud tomorrow. The direct
    cost of buying the labels is the first effect alone, and it is the number that has to be
    small for the trade to be worth making.
    """
    costs = costs or BusinessCosts()
    y = np.asarray(y).astype(int)
    amounts = np.asarray(amounts, dtype="float64")
    approved = np.asarray(approved).astype(bool)
    fraud = y == 1

    total_fraud_amt = float(amounts[fraud].sum())
    approved_fraud_amt = float(amounts[fraud & approved].sum())
    declined_legit_amt = float(amounts[~fraud & ~approved].sum())

    out = {
        "total_amount": float(amounts.sum()),
        "total_fraud_amount": total_fraud_amt,
        "fraud_amount_approved": approved_fraud_amt,
        "fraud_loss_net": approved_fraud_amt * (1.0 - costs.recovery_rate),
        "fraud_count_approved": float(np.sum(fraud & approved)),
        "fraud_value_blocked_share": (
            1.0 - approved_fraud_amt / total_fraud_amt if total_fraud_amt > 0 else float("nan")
        ),
        "declined_legit_amount": declined_legit_amt,
        "false_positive_cost": declined_legit_amt * costs.fp_cost_rate,
        "declined_legit_count": float(np.sum(~fraud & ~approved)),
        "approval_rate": float(approved.mean()) if approved.size else float("nan"),
    }
    out["total_cost"] = out["fraud_loss_net"] + out["false_positive_cost"]

    if explored is not None:
        explored = np.asarray(explored).astype(bool)
        expl_fraud_amt = float(amounts[fraud & explored].sum())
        out["exploration_fraud_amount"] = expl_fraud_amt
        out["exploration_fraud_count"] = float(np.sum(fraud & explored))
        out["exploration_fraud_cost_share"] = (
            expl_fraud_amt / total_fraud_amt if total_fraud_amt > 0 else 0.0
        )
        out["exploration_count"] = float(explored.sum())
        out["exploration_label_yield"] = (
            float(np.sum(fraud & explored) / max(explored.sum(), 1))
        )
    return out


def relative_change(new: float, old: float) -> float:
    """``(new - old) / old``; ``nan`` when the base is degenerate."""
    if old is None or not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    return float((new - old) / old)
