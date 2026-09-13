"""Spending a fixed budget as well as it can be spent.

The unconstrained rule takes every action with positive expected gain. Operations teams do not
get an unconstrained rule -- they get a weekly expedite budget and an escalation queue a team can
actually work. Under a cap, the right question stops being "is this worth doing" and becomes "is
this worth doing *with the marginal currency unit*", which is a knapsack.

The continuous relaxation of a knapsack over independent items is solved exactly by a single
price: pick, per order, the action maximising ``benefit - (1 + lambda) * cost`` and bisect
``lambda`` until the spend meets the cap. The result differs from greedy-by-ratio only on the
last item, and unlike greedy it hands back the shadow price -- the return on the next currency
unit of budget, which is the number worth taking to whoever owns the cap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from deliveryrisk.policy.actions import ACTIONS
from deliveryrisk.policy.thresholds import DecisionTable

log = logging.getLogger(__name__)


@dataclass
class Allocation:
    actions: np.ndarray
    spend: float
    expected_gain: float
    shadow_price: float
    iterations: int


def allocate_under_budget(
    table: DecisionTable, budget: float, *, tol: float = 1e-3, max_iter: int = 60
) -> Allocation:
    """Best assignment whose total action cost is within ``budget``."""
    benefit = table.benefit[list(ACTIONS)].to_numpy()
    cost = table.cost[list(ACTIONS)].to_numpy()

    def solve(lam: float) -> tuple[np.ndarray, float, float]:
        score = benefit - (1.0 + lam) * cost
        score[:, 0] = 0.0  # doing nothing is free and always available
        idx = score.argmax(axis=1)
        rows = np.arange(len(idx))
        chosen = np.array(ACTIONS, dtype=object)[idx]
        spend = float(cost[rows, idx].sum())
        gain = float((benefit[rows, idx] - cost[rows, idx]).sum())
        return chosen, spend, gain

    chosen, spend, gain = solve(0.0)
    if spend <= budget:
        return Allocation(chosen, spend, gain, 0.0, 0)

    # Bracket first: raise the price until even the cheapest sensible plan fits the budget.
    lo, hi = 0.0, 1.0
    iterations = 0
    while solve(hi)[1] > budget and hi < 1e6:
        hi *= 4.0
        iterations += 1
    for _ in range(max_iter - iterations):
        iterations += 1
        mid = 0.5 * (lo + hi)
        _, spend, _ = solve(mid)
        if spend > budget:
            lo = mid
        else:
            hi = mid
        if abs(spend - budget) <= tol * max(budget, 1.0):
            break
    # `hi` is always a feasible price, so the returned plan is always within budget.
    chosen, spend, gain = solve(hi)
    log.debug("budget %.0f -> spend %.0f at lambda %.3f", budget, spend, hi)
    return Allocation(chosen, spend, gain, hi, iterations)


def budget_frontier(table: DecisionTable, budgets: np.ndarray) -> list[Allocation]:
    return [allocate_under_budget(table, float(b)) for b in budgets]
