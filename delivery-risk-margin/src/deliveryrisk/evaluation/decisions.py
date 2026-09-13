"""Scoring a policy against what would actually have happened.

The counterfactual generator holds ``late(a)`` for every order and every action, so a policy's
realised value is not estimated here -- it is computed::

    contribution_i = margin_i - c_{a_i}(i) - L_i * 1{late_i(a_i)}

and the number that matters is the difference against doing nothing, per thousand orders.

Three things this deliberately does *not* do:

* It does not score the policy on its own beliefs. A policy that believes a nudge is twice as
  effective as it is will look excellent under its own expected-value table and mediocre here.
  That gap is the subject of ``docs/RESULTS.md`` §7.
* It does not reuse the model's probabilities. Realised lateness comes from the truth table.
* It does not report a point estimate alone. Late costs are heavy-tailed, so every headline
  carries a bootstrap interval over orders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from deliveryrisk.evaluation.metrics import bootstrap_ci
from deliveryrisk.policy.actions import ACTIONS

log = logging.getLogger(__name__)


@dataclass
class PolicyOutcome:
    name: str
    n_orders: int
    action_counts: dict[str, int]
    treated_share: float
    spend_per_1k: float
    late_rate: float
    late_rate_change_pct: float
    contribution_per_1k: float
    delta_per_1k: float
    delta_ci: tuple[float, float]
    benefit_per_1k: float
    roi: float
    detail: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | str]:
        row = {
            "policy": self.name,
            "treated_share": self.treated_share,
            "spend_per_1k": self.spend_per_1k,
            "late_rate": self.late_rate,
            "late_rate_change_pct": self.late_rate_change_pct,
            "contribution_per_1k": self.contribution_per_1k,
            "delta_per_1k": self.delta_per_1k,
            "delta_lo": self.delta_ci[0],
            "delta_hi": self.delta_ci[1],
            "roi": self.roi,
        }
        row.update({f"n_{a}": self.action_counts.get(a, 0) for a in ACTIONS})
        row.update(self.detail)
        return row


def align_truth(df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join the counterfactual outcome columns onto an evaluation frame, order for order."""
    cols = ["order_id", *[f"late_{a}" for a in ACTIONS]]
    out = df.merge(truth[cols], on="order_id", how="left", validate="one_to_one")
    missing = out[f"late_{ACTIONS[0]}"].isna().sum()
    if missing:
        raise ValueError(f"{missing} evaluation rows have no counterfactual truth")
    return out


def realised_contribution(
    actions: np.ndarray,
    truth: pd.DataFrame,
    margin: np.ndarray,
    late_cost: np.ndarray,
    action_costs: dict[str, np.ndarray],
) -> np.ndarray:
    """Per-order contribution under an assignment, using the true counterfactual outcome."""
    actions = np.asarray(actions, dtype=object)
    late = np.zeros(len(actions), dtype=float)
    cost = np.zeros(len(actions), dtype=float)
    for a in ACTIONS:
        m = actions == a
        if not m.any():
            continue
        late[m] = truth[f"late_{a}"].to_numpy(dtype=float)[m]
        cost[m] = np.asarray(action_costs[a], dtype=float)[m]
    return margin - cost - late_cost * late


def evaluate_policy(
    name: str,
    actions: np.ndarray,
    truth: pd.DataFrame,
    margin: np.ndarray,
    late_cost: np.ndarray,
    action_costs: dict[str, np.ndarray],
    *,
    baseline: np.ndarray | None = None,
    seed: int = 0,
) -> PolicyOutcome:
    contrib = realised_contribution(actions, truth, margin, late_cost, action_costs)
    if baseline is None:
        baseline = realised_contribution(
            np.array(["none"] * len(actions), dtype=object), truth, margin, late_cost, action_costs
        )
    delta = contrib - baseline
    n = len(actions)
    spend = np.zeros(n)
    late = np.zeros(n)
    for a in ACTIONS:
        m = actions == a
        if m.any():
            spend[m] = np.asarray(action_costs[a], dtype=float)[m]
            late[m] = truth[f"late_{a}"].to_numpy(dtype=float)[m]
    base_late = truth["late_none"].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(delta * 1000.0, seed=seed)
    counts = {a: int((actions == a).sum()) for a in ACTIONS}
    total_spend = float(spend.sum())
    benefit = float((late_cost * (base_late - late)).sum())
    return PolicyOutcome(
        name=name,
        n_orders=n,
        action_counts=counts,
        treated_share=float((actions != "none").mean()),
        spend_per_1k=1000.0 * total_spend / n,
        late_rate=float(late.mean()),
        late_rate_change_pct=float(late.mean() / base_late.mean() - 1.0) * 100.0,
        contribution_per_1k=1000.0 * float(contrib.mean()),
        delta_per_1k=1000.0 * float(delta.mean()),
        delta_ci=(lo, hi),
        benefit_per_1k=1000.0 * benefit / n,
        roi=float(benefit / total_spend) if total_spend > 0 else float("nan"),
    )


def compare_policies(
    policies: dict[str, np.ndarray],
    truth: pd.DataFrame,
    margin: np.ndarray,
    late_cost: np.ndarray,
    action_costs: dict[str, np.ndarray],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    baseline = realised_contribution(
        np.array(["none"] * len(truth), dtype=object), truth, margin, late_cost, action_costs
    )
    rows = [
        evaluate_policy(name, act, truth, margin, late_cost, action_costs,
                        baseline=baseline, seed=seed).as_row()
        for name, act in policies.items()
    ]
    return pd.DataFrame(rows).sort_values("delta_per_1k", ascending=False, ignore_index=True)


def oracle_actions(
    truth: pd.DataFrame, late_cost: np.ndarray, action_costs: dict[str, np.ndarray]
) -> np.ndarray:
    """The best assignment achievable with perfect foresight of every counterfactual.

    Not attainable, and not meant to be: it is the denominator that turns "the policy earned
    12 per thousand orders" into "the policy captured 71 % of what was there to capture".
    """
    values = np.stack(
        [
            -np.asarray(action_costs[a], dtype=float)
            - late_cost * truth[f"late_{a}"].to_numpy(dtype=float)
            for a in ACTIONS
        ],
        axis=1,
    )
    return np.array(ACTIONS, dtype=object)[values.argmax(axis=1)]
