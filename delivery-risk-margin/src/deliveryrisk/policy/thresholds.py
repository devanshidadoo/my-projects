"""Turning a probability into a decision.

Five ways to pick who gets an intervention, in increasing order of how much they know:

``fixed_rate``     treat the top k % by risk. The decile heuristic: no economics at all.
``f1``             the threshold that maximises F1 on validation. Classification, not decision.
``youden``         the threshold that maximises TPR - FPR. Same objection.
``margin_global``  one probability cut-off, chosen to maximise expected contribution.
``margin_per_order`` intervene when ``u_a * p_i * L_i > c_a(i)``, order by order.

and the action-assignment question on top of it, which only the last two can ask at all:
with more than one action available, *which* one, and for that the per-order expected
contribution is the only rule that answers correctly.

All of them are written against the same ``expected_gain`` table, so the comparison in
``docs/RESULTS.md`` is between decision rules, never between implementations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deliveryrisk.policy.actions import ACTIONS

log = logging.getLogger(__name__)


@dataclass
class DecisionTable:
    """Per-order expected value of every action, under the policy's own beliefs.

    Attributes
    ----------
    benefit:
        ``uplift_a * L_i`` -- expected lateness cost avoided.
    cost:
        ``c_a(i)`` -- what the action is billed at.
    gain:
        ``benefit - cost``. The quantity every policy here is trying to maximise.
    """

    benefit: pd.DataFrame
    cost: pd.DataFrame

    @property
    def gain(self) -> pd.DataFrame:
        return self.benefit - self.cost

    def best_action(self) -> np.ndarray:
        """Per-order argmax over actions, with ``none`` winning ties at zero."""
        g = self.gain.copy()
        g["none"] = 0.0
        # A tie between doing nothing and spending money is not a tie.
        best = g[list(ACTIONS)].to_numpy()
        idx = best.argmax(axis=1)
        chosen = np.array(ACTIONS, dtype=object)[idx]
        chosen[best.max(axis=1) <= 0] = "none"
        return chosen


def decision_table(
    uplifts: dict[str, np.ndarray], late_cost: np.ndarray, costs: dict[str, np.ndarray]
) -> DecisionTable:
    benefit = pd.DataFrame({a: uplifts[a] * late_cost for a in ACTIONS})
    cost = pd.DataFrame({a: np.asarray(costs[a], dtype=float) for a in ACTIONS})
    return DecisionTable(benefit=benefit, cost=cost)


# ------------------------------------------------------------------ policies
def assign_none(n: int) -> np.ndarray:
    return np.array(["none"] * n, dtype=object)


def assign_fixed_rate(p: np.ndarray, rate: float, action: str = "nudge") -> np.ndarray:
    """Treat the top ``rate`` share by predicted risk. The decile heuristic, made explicit."""
    n = len(p)
    k = int(round(rate * n))
    out = assign_none(n)
    if k > 0:
        idx = np.argsort(-np.asarray(p, dtype=float), kind="stable")[:k]
        out[idx] = action
    return out


def assign_threshold(p: np.ndarray, tau: float, action: str = "nudge") -> np.ndarray:
    out = assign_none(len(p))
    out[np.asarray(p, dtype=float) >= tau] = action
    return out


def assign_margin_per_order(table: DecisionTable) -> np.ndarray:
    """The rule this project argues for: per-order argmax of expected contribution."""
    return table.best_action()


def assign_margin_global(table: DecisionTable, p: np.ndarray, tau: float) -> np.ndarray:
    """One probability cut-off, then the best action for those above it.

    The middle ground people actually ship: a threshold a human can state, with the action choice
    still made per order.
    """
    chosen = table.best_action()
    chosen[np.asarray(p, dtype=float) < tau] = "none"
    return chosen


def threshold_f1(y: np.ndarray, p: np.ndarray, n_grid: int = 200) -> float:
    """The F1-optimal cut-off. Included because it is the default in every tutorial, and because
    it optimises a quantity nobody is paid on."""
    return _best_threshold(y, p, n_grid, _f1)


def threshold_youden(y: np.ndarray, p: np.ndarray, n_grid: int = 200) -> float:
    return _best_threshold(y, p, n_grid, _youden)


def threshold_expected_contribution(
    table: DecisionTable, p: np.ndarray, n_grid: int = 200
) -> float:
    """The best single global cut-off, found by sweeping it against realised expected gain.

    This is the strongest version of "one threshold for everyone": it is *chosen* on the
    objective the per-order rule optimises directly, so the gap between them in §5 is the cost of
    the constraint, not of a worse objective.
    """
    p = np.asarray(p, dtype=float)
    gain = table.gain.copy()
    gain["none"] = 0.0
    best_gain = gain[list(ACTIONS)].to_numpy().max(axis=1)
    best_gain = np.maximum(best_gain, 0.0)
    grid = np.quantile(p, np.linspace(0.5, 1.0, n_grid))
    totals = [float(best_gain[p >= t].sum()) for t in grid]
    return float(grid[int(np.argmax(totals))])


def implied_thresholds(table: DecisionTable, uplifts: dict[str, np.ndarray], p: np.ndarray,
                       action: str = "nudge") -> np.ndarray:
    """``c_a(i) / (u_a(i) * L_i)`` expressed back on the probability axis.

    Reporting the spread of this quantity is the cleanest way to show why a single threshold
    cannot be right: it is the threshold each order would need, and it ranges over an order of
    magnitude.
    """
    p = np.asarray(p, dtype=float)
    benefit_per_p = np.where(p > 0, table.benefit[action].to_numpy() / np.maximum(p, 1e-9), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return table.cost[action].to_numpy() / benefit_per_p


def _best_threshold(y, p, n_grid, score_fn) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    grid = np.quantile(p, np.linspace(0.5, 0.999, n_grid))
    scores = [score_fn(y, (p >= t).astype(float)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _f1(y, yhat) -> float:
    tp = float((y * yhat).sum())
    if tp == 0:
        return 0.0
    precision = tp / float(yhat.sum())
    recall = tp / float(y.sum())
    return 2 * precision * recall / (precision + recall)


def _youden(y, yhat) -> float:
    pos, neg = y.sum(), (1 - y).sum()
    tpr = float((y * yhat).sum()) / max(pos, 1)
    fpr = float(((1 - y) * yhat).sum()) / max(neg, 1)
    return tpr - fpr
