"""Metrics and off-policy estimators, checked against cases with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from clfraud.evaluation.counterfactual import (
    doubly_robust_estimate,
    ips_estimate,
    snips_estimate,
)
from clfraud.evaluation.metrics import (
    BusinessCosts,
    business_outcome,
    precision_at_budget,
    ranking_metrics,
    recall_at_budget,
    relative_change,
)
from clfraud.models.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
)


class TestRecallAtBudget:
    def test_perfect_ranking_captures_everything_the_budget_allows(self):
        y = np.array([1] * 10 + [0] * 90)
        scores = np.concatenate([np.ones(10), np.zeros(90)])
        assert recall_at_budget(y, scores, 0.10) == pytest.approx(1.0)
        # Budget below the fraud rate caps recall mechanically -- half the budget, half the fraud.
        assert recall_at_budget(y, scores, 0.05) == pytest.approx(0.5)

    def test_inverted_ranking_captures_nothing(self):
        y = np.array([1] * 10 + [0] * 90)
        scores = np.concatenate([np.zeros(10), np.ones(90)])
        assert recall_at_budget(y, scores, 0.10) == pytest.approx(0.0)

    def test_value_weighting_differs_from_count_recall(self):
        """A model good at small fraud and bad at large fraud scores well on counts, badly on $."""
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.11, 0.05])
        amounts = np.array([10.0, 5_000.0] + [50.0] * 8)
        assert recall_at_budget(y, scores, 0.1) == pytest.approx(0.5)
        assert recall_at_budget(y, scores, 0.1, amounts) < 0.01

    def test_no_fraud_is_nan_not_zero(self):
        assert np.isnan(recall_at_budget(np.zeros(10, dtype=int), np.arange(10.0), 0.1))

    def test_ties_are_broken_deterministically(self):
        y = np.array([0, 1, 0, 1])
        scores = np.full(4, 0.5)
        assert recall_at_budget(y, scores, 0.5) == recall_at_budget(y, scores, 0.5)

    def test_precision_at_budget(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        scores = np.linspace(1.0, 0.0, 10)
        assert precision_at_budget(y, scores, 0.2) == pytest.approx(1.0)

    def test_ranking_metrics_survive_a_single_class(self):
        out = ranking_metrics(np.zeros(50, dtype=int), np.random.default_rng(0).uniform(size=50))
        assert np.isnan(out["roc_auc"])
        assert out["n_fraud"] == 0.0


class TestBusinessOutcome:
    def test_isolates_the_direct_cost_of_exploration(self):
        """Exploration cost must be the fraud *exploration itself* let through, nothing else."""
        y = np.array([1, 1, 1, 0])
        amounts = np.array([100.0, 200.0, 300.0, 1_000.0])
        approved = np.array([True, False, True, True])
        explored = np.array([False, False, True, False])
        out = business_outcome(y, amounts, approved, explored, BusinessCosts(recovery_rate=0.0))
        assert out["fraud_amount_approved"] == pytest.approx(400.0)
        assert out["exploration_fraud_amount"] == pytest.approx(300.0)
        assert out["exploration_fraud_cost_share"] == pytest.approx(300.0 / 600.0)
        assert out["fraud_value_blocked_share"] == pytest.approx(200.0 / 600.0)

    def test_false_positive_cost_scales_with_the_assumed_rate(self):
        y = np.zeros(4, dtype=int)
        amounts = np.full(4, 100.0)
        approved = np.array([True, True, False, False])
        a = business_outcome(y, amounts, approved, costs=BusinessCosts(fp_cost_rate=0.015))
        b = business_outcome(y, amounts, approved, costs=BusinessCosts(fp_cost_rate=0.030))
        assert b["false_positive_cost"] == pytest.approx(2.0 * a["false_positive_cost"])

    def test_recovery_rate_reduces_net_loss(self):
        y = np.array([1])
        out = business_outcome(
            y, np.array([1_000.0]), np.array([True]), costs=BusinessCosts(recovery_rate=0.2)
        )
        assert out["fraud_loss_net"] == pytest.approx(800.0)

    def test_relative_change_guards_a_zero_base(self):
        assert np.isnan(relative_change(1.0, 0.0))
        assert relative_change(0.5, 1.0) == pytest.approx(-0.5)


class TestOffPolicyEstimators:
    """Estimate a known policy value from logs a *different* policy produced."""

    @staticmethod
    def _bandit_logs(n=200_000, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(size=n)
        # Logging policy acts with probability p0(x); the target acts with p1(x). The two are
        # deliberately anti-correlated and p0 gets small, so overlap is poor -- the regime where
        # the differences between these estimators actually show up.
        p0 = 0.05 + 0.90 * x
        p1 = 0.95 - 0.90 * x
        a = rng.uniform(size=n) < p0
        reward = np.where(a, 1.0 + 2.0 * x, 0.0)
        truth = float(np.mean(p1 * (1.0 + 2.0 * x)))
        prop = np.where(a, p0, 1.0 - p0)
        target = np.where(a, p1, 1.0 - p1)
        return reward, target, prop, truth, x, a, p1

    def test_ips_is_unbiased(self):
        reward, target, prop, truth, *_ = self._bandit_logs()
        est = ips_estimate(reward, target, prop)
        assert est.estimate == pytest.approx(truth, rel=0.02)
        assert est.ci95[0] <= truth <= est.ci95[1]

    def test_snips_has_lower_variance_than_ips(self):
        reward, target, prop, truth, *_ = self._bandit_logs(n=20_000, seed=5)
        ips = ips_estimate(reward, target, prop)
        snips = snips_estimate(reward, target, prop)
        assert snips.std_error < ips.std_error
        assert snips.estimate == pytest.approx(truth, rel=0.06)

    def test_doubly_robust_survives_a_wrong_reward_model(self):
        """DR needs only *one* of its two models to be right. Break the reward model and check."""
        reward, target, prop, truth, x, a, p1 = self._bandit_logs()
        # A reward model that is constant and far from the truth for every (x, action).
        wrong_logged = np.full_like(reward, 5.0)
        wrong_target = np.full_like(reward, 5.0)
        dr = doubly_robust_estimate(reward, target, prop, wrong_logged, wrong_target)
        assert dr.estimate == pytest.approx(truth, rel=0.05)

    def test_clipping_trades_bias_for_variance(self):
        reward, target, prop, _truth, *_ = self._bandit_logs(n=20_000, seed=9)
        loose = ips_estimate(reward, target, prop, clip=None)
        tight = ips_estimate(reward, target, prop, clip=1.2)
        assert tight.std_error < loose.std_error
        assert tight.ess > loose.ess

    def test_error_report_is_self_describing(self):
        reward, target, prop, truth, *_ = self._bandit_logs(n=5_000, seed=2)
        rep = snips_estimate(reward, target, prop).error_vs(truth)
        assert set(rep) == {"estimate", "truth", "abs_error", "rel_error", "covers_truth", "ess"}


class TestCalibration:
    def test_isotonic_improves_a_squashed_score(self):
        rng = np.random.default_rng(1)
        n = 20_000
        p = rng.uniform(0.0, 1.0, size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        squashed = p * 0.3                       # systematically under-confident
        cal = IsotonicCalibrator().fit(squashed, y)
        assert expected_calibration_error(y, cal.transform(squashed)) < expected_calibration_error(
            y, squashed
        )

    def test_ece_uses_quantile_bins_so_rare_events_are_visible(self):
        """Equal-width bins put 99 % of a 3.5 %-base-rate dataset in one bucket and report ~0."""
        rng = np.random.default_rng(2)
        n = 50_000
        y = (rng.uniform(size=n) < 0.035).astype(int)
        biased = np.clip(rng.uniform(size=n) * 0.02 + 0.20, 0, 1)  # wildly over-confident
        assert expected_calibration_error(y, biased) > 0.1

    def test_brier_rewards_the_truth(self):
        y = np.array([1, 0, 1, 0])
        assert brier_score(y, y.astype(float)) == pytest.approx(0.0)
        assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)

    def test_calibrator_is_a_noop_on_a_single_class(self):
        cal = IsotonicCalibrator().fit(np.linspace(0, 1, 10), np.zeros(10))
        np.testing.assert_allclose(cal.transform(np.array([0.3, 0.7])), [0.3, 0.7])


class TestPolicyValueFromLogs:
    """The estimator the simulator actually uses, on a case where the answer is known."""

    @staticmethod
    def _cycle(n=200_000, seed=3, explore_rate=0.02, floor=0.004):
        rng = np.random.default_rng(seed)
        score = rng.beta(1.5, 12.0, size=n)
        amt = np.exp(rng.normal(4.2, 0.9, size=n))
        # Fraud concentrated at high scores, as a working model would produce.
        y = (rng.uniform(size=n) < np.clip(0.004 + 2.2 * score**1.6, 0, 0.95)).astype(int)

        tau = np.quantile(score, 0.94)
        declined = score >= tau
        q = np.where(declined, max(explore_rate, floor), 0.0)
        explored = declined & (rng.uniform(size=n) < q)
        approved = (~declined) | explored
        e = np.where(declined, q, 1.0)

        # Candidate: decline twice the volume.
        cand = score >= np.quantile(score, 0.88)
        truth = float((amt[cand & (y == 1)]).sum() / amt[y == 1].sum())
        return y, amt, approved, e, cand, truth

    def test_recovers_the_true_blocked_share(self):
        from clfraud.evaluation.counterfactual import policy_value_from_logs

        y, amt, approved, e, cand, truth = self._cycle()
        est = policy_value_from_logs(y, amt, approved, e, cand, clip=None)
        assert est["ips"].estimate == pytest.approx(truth, rel=0.15)
        assert est["snips"].estimate == pytest.approx(truth, rel=0.15)

    def test_snips_needs_no_oracle_denominator(self):
        """SNIPS estimates the total fraud value from the same logs, so it stays a valid share."""
        from clfraud.evaluation.counterfactual import policy_value_from_logs

        y, amt, approved, e, cand, _ = self._cycle()
        est = policy_value_from_logs(y, amt, approved, e, cand, clip=None)
        assert 0.0 <= est["snips"].estimate <= 1.0

    def test_clipping_below_the_true_weight_biases_the_estimate_down(self):
        """Regression, and the reason ``clip`` defaults to None.

        With a propensity floor of 0.004 the correct weight on an explored row is 250. A clip of
        50 looks like sensible variance control and silently discards four fifths of the only
        evidence there is about the decline region -- which is exactly the region a candidate
        policy and the logging policy disagree about.
        """
        from clfraud.evaluation.counterfactual import policy_value_from_logs

        # Exploration at the floor, so the correct weight is 1/0.004 = 250 and a clip of 50 is
        # off by 5x on every explored row.
        y, amt, approved, e, cand, truth = self._cycle(explore_rate=0.004, floor=0.004)
        unclipped = policy_value_from_logs(y, amt, approved, e, cand, clip=None)["ips"].estimate
        clipped = policy_value_from_logs(y, amt, approved, e, cand, clip=50.0)["ips"].estimate
        assert clipped < 0.7 * truth
        assert abs(unclipped - truth) < abs(clipped - truth)

    def test_no_exploration_makes_the_decline_region_unrecoverable(self):
        """Without randomisation, the estimator cannot see past the threshold at all.

        Not an estimator failure -- an identification failure. The estimate collapses to the
        fraud the logging policy happened to approve, which is the whole point of the project.
        """
        from clfraud.evaluation.counterfactual import policy_value_from_logs

        y, amt, approved, e, cand, truth = self._cycle(explore_rate=0.0, floor=0.0)
        est = policy_value_from_logs(y, amt, approved, e, cand, clip=None)["ips"].estimate
        assert est < 0.5 * truth
