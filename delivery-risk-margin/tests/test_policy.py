"""The decision layer: the cost algebra, the assignment rule, and the budget.

These are the parts where a sign error costs money silently, so they are pinned arithmetically
rather than by eyeballing a results table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deliveryrisk.evaluation.decisions import (
    ACTIONS,
    evaluate_policy,
    oracle_actions,
    realised_contribution,
)
from deliveryrisk.policy.actions import CAUSES, ActionCosts, UpliftBelief
from deliveryrisk.policy.allocation import allocate_under_budget
from deliveryrisk.policy.economics import CostModel
from deliveryrisk.policy.thresholds import (
    assign_fixed_rate,
    decision_table,
    implied_thresholds,
    threshold_expected_contribution,
    threshold_f1,
)

RNG = np.random.default_rng(0)
N = 2000


@pytest.fixture
def table():
    p_late = RNG.beta(2, 18, N)
    mix = {"handling_only": 0.3, "transit_only": 0.3, "either": 0.35, "neither": 0.05}
    joint = {c: p_late * mix[c] for c in CAUSES}
    late_cost = 5 + RNG.gamma(3, 8, N)
    costs = {
        "none": np.zeros(N),
        "nudge": np.full(N, 1.2),
        "expedite": 10 + RNG.random(N) * 8,
        "both": 11.2 + RNG.random(N) * 8,
    }
    uplifts = UpliftBelief().uplifts(joint)
    return decision_table(uplifts, late_cost, costs), costs, late_cost, p_late


def test_gain_is_benefit_minus_cost(table):
    t, costs, late_cost, _ = table
    assert np.allclose(t.gain["nudge"], t.benefit["nudge"] - costs["nudge"])
    assert (t.cost["none"] == 0).all()
    assert (t.benefit["none"] == 0).all()


def test_best_action_never_spends_to_lose(table):
    t, costs, _, _ = table
    chosen = t.best_action()
    gain = t.gain
    for i, a in enumerate(chosen):
        if a == "none":
            assert gain.iloc[i].max() <= 0 or np.isclose(gain.iloc[i].max(), 0)
        else:
            assert gain.iloc[i][a] > 0
            assert gain.iloc[i][a] >= gain.iloc[i].max() - 1e-12


def test_uplift_composition_is_coherent():
    """`both` must dominate either action alone, and neither may exceed the risk it acts on."""
    joint = {c: np.array([0.05]) for c in CAUSES}
    u = UpliftBelief(rho_nudge=0.5, rho_expedite=0.8).uplifts(joint)
    assert u["both"] >= u["nudge"]
    assert u["both"] >= u["expedite"]
    p_late = sum(joint.values())
    for a in ACTIONS:
        assert (u[a] <= p_late + 1e-12).all(), a
    # A nudge cannot touch a transit-only order, and expediting cannot touch a handling-only one.
    only_transit = {c: np.array([0.1 if c == "transit_only" else 0.0]) for c in CAUSES}
    assert UpliftBelief().uplifts(only_transit)["nudge"] == 0.0
    only_handling = {c: np.array([0.1 if c == "handling_only" else 0.0]) for c in CAUSES}
    assert UpliftBelief().uplifts(only_handling)["expedite"] == 0.0


def test_risk_only_belief_uses_the_population_mix():
    """The ablation must be the strongest honest baseline, not a handicapped one."""
    mix = {"handling_only": 0.25, "transit_only": 0.25, "either": 0.4, "neither": 0.1}
    p = np.array([0.2])
    b = UpliftBelief(rho_nudge=0.5, rho_expedite=0.7)
    got = b.risk_only_uplifts(p, mix)
    assert got["nudge"] == pytest.approx(0.5 * (0.25 + 0.4) * 0.2)
    assert got["expedite"] == pytest.approx(0.7 * (0.25 + 0.4) * 0.2)


def test_implied_threshold_is_the_break_even_probability(table):
    t, costs, late_cost, p_late = table
    uplifts = {"nudge": t.benefit["nudge"].to_numpy() / late_cost}
    imp = implied_thresholds(t, uplifts, p_late, "nudge")
    # At exactly the break-even probability, expected benefit equals cost.
    u_per_p = t.benefit["nudge"].to_numpy() / np.maximum(p_late, 1e-9)
    assert np.allclose(imp * u_per_p, costs["nudge"], rtol=1e-6)


def test_budget_is_respected_and_value_is_monotone(table):
    t, _, _, _ = table
    unconstrained = allocate_under_budget(t, 1e12)
    budgets = [0.05, 0.2, 0.5, 1.0]
    prev = -np.inf
    for share in budgets:
        alloc = allocate_under_budget(t, share * unconstrained.spend)
        assert alloc.spend <= share * unconstrained.spend * 1.02 + 1e-6
        assert alloc.expected_gain >= prev - 1e-6
        prev = alloc.expected_gain
    assert unconstrained.shadow_price == 0.0


def test_shadow_price_falls_as_the_budget_grows(table):
    t, _, _, _ = table
    full = allocate_under_budget(t, 1e12).spend
    prices = [allocate_under_budget(t, s * full).shadow_price for s in (0.1, 0.3, 0.6, 1.0)]
    assert prices == sorted(prices, reverse=True)


def test_oracle_is_an_upper_bound(tiny_data):
    """No implementable policy can beat perfect foresight, by construction."""
    truth = tiny_data.truth
    n = len(truth)
    late_cost = np.full(n, 20.0)
    margin = np.full(n, 30.0)
    costs = {"none": np.zeros(n), "nudge": np.full(n, 1.2), "expedite": np.full(n, 14.0),
             "both": np.full(n, 15.2)}
    best = oracle_actions(truth, late_cost, costs)
    oracle_value = realised_contribution(best, truth, margin, late_cost, costs).sum()
    rng = np.random.default_rng(1)
    for _ in range(5):
        random_actions = rng.choice(np.array(ACTIONS, dtype=object), size=n)
        value = realised_contribution(random_actions, truth, margin, late_cost, costs).sum()
        assert oracle_value >= value - 1e-9


def test_evaluate_policy_reports_doing_nothing_as_exactly_zero(tiny_data):
    truth = tiny_data.truth
    n = len(truth)
    costs = {a: np.zeros(n) if a == "none" else np.full(n, 1.0) for a in ACTIONS}
    res = evaluate_policy("none", np.array(["none"] * n, dtype=object), truth,
                          np.full(n, 10.0), np.full(n, 20.0), costs)
    assert res.delta_per_1k == pytest.approx(0.0)
    assert res.spend_per_1k == 0.0
    assert res.treated_share == 0.0


def test_cost_model_scales_with_order_value():
    """`L` must be order-specific, or the per-order threshold collapses to a global one."""
    cost = CostModel().with_categories(["books", "furniture_decor"])
    df = pd.DataFrame(
        {"total_price": [50.0, 500.0], "total_freight": [10.0, 40.0],
         "category": ["books", "books"]}
    )
    late = cost.late_cost(df)
    assert late[1] > late[0] * 2
    assert (late > cost.support_cost).all()


def test_action_costs_follow_the_carrier_price_list():
    df = pd.DataFrame({"expedite_surcharge": [12.0, 24.0], "total_weight_g": [0.0, 10_000.0]})
    costs = ActionCosts(nudge_cost=1.5).per_order(df)
    assert (costs["nudge"] == 1.5).all()
    assert costs["expedite"][1] > costs["expedite"][0]
    assert np.allclose(costs["both"], costs["nudge"] + costs["expedite"])


def test_fixed_rate_treats_exactly_the_top_share():
    p = np.linspace(0, 1, 1000)
    a = assign_fixed_rate(p, 0.2, "nudge")
    assert (a == "nudge").sum() == 200
    assert (a[p > 0.81] == "nudge").all()


def test_expected_contribution_threshold_beats_f1_on_contribution(table):
    """Not a tautology worth skipping: it is the claim the headline rests on."""
    t, _, _, p_late = table
    y = (RNG.random(N) < p_late).astype(float)
    gain = t.gain.copy()
    gain["none"] = 0.0
    best = np.maximum(gain[list(ACTIONS)].to_numpy().max(axis=1), 0.0)
    tau_ec = threshold_expected_contribution(t, p_late)
    tau_f1 = threshold_f1(y, p_late)
    assert best[p_late >= tau_ec].sum() >= best[p_late >= tau_f1].sum()
