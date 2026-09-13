"""Policy mechanics, propensity bookkeeping, and weight construction.

The arithmetic here is the load-bearing part of the whole correction: if the logged propensity
does not match the probability the policy actually used, every weight downstream is wrong and
nothing complains.
"""
from __future__ import annotations

import numpy as np
import pytest

from clfraud.bias.propensity import known_propensity, overlap_diagnostics
from clfraud.bias.reweighting import WeightConfig, build_weights, effective_sample_size
from clfraud.simulator.policy import ExplorationConfig, OracleAccessPolicy, ThresholdPolicy


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 20_000
    scores = rng.beta(1.4, 12.0, size=n)
    amounts = np.exp(rng.normal(4.2, 0.9, size=n))
    return scores, amounts, rng


class TestThresholdPolicy:
    def test_threshold_spends_the_decline_budget(self, scored):
        scores, amounts, rng = scored
        p = ThresholdPolicy(decline_budget=0.03)
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        assert out.decline_rate == pytest.approx(0.03, abs=0.002)

    def test_decide_before_calibrate_is_an_error(self, scored):
        scores, amounts, rng = scored
        with pytest.raises(RuntimeError):
            ThresholdPolicy().decide(scores, amounts, rng)

    def test_no_exploration_means_deterministic_propensity(self, scored):
        scores, amounts, rng = scored
        p = ThresholdPolicy(0.03, ExplorationConfig(mode="none", rate=0.0))
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        assert set(np.unique(out.propensity)) <= {0.0, 1.0}
        assert not out.explored.any()
        # Positivity is violated exactly on the declines -- the fact that motivates exploration.
        assert overlap_diagnostics(out.propensity, out.approved)[
            "positivity_violation_rate"
        ] == pytest.approx(out.decline_rate, abs=0.002)

    @pytest.mark.parametrize("mode", ["uniform", "risk_tapered", "cost_aware"])
    def test_exploration_spends_its_budget(self, scored, mode):
        scores, amounts, rng = scored
        p = ThresholdPolicy(0.03, ExplorationConfig(mode=mode, rate=0.02))
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        share = out.explored.sum() / out.declined_by_policy.sum()
        assert share == pytest.approx(0.02, abs=0.012)

    @pytest.mark.parametrize("mode", ["uniform", "risk_tapered", "cost_aware"])
    def test_positivity_holds_wherever_exploration_is_on(self, scored, mode):
        scores, amounts, rng = scored
        p = ThresholdPolicy(0.03, ExplorationConfig(mode=mode, rate=0.02))
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        assert out.propensity.min() > 0.0

    def test_amount_capped_exploration_breaks_positivity_by_design(self, scored):
        """The cheap design is the one that gives up identification. Assert it, don't hide it."""
        scores, amounts, rng = scored
        p = ThresholdPolicy(
            0.03, ExplorationConfig(mode="amount_capped", rate=0.02, amount_cap=50.0,
                                    propensity_floor=0.0)
        )
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        blind = out.declined_by_policy & (amounts > 50.0)
        assert blind.any()
        assert (out.propensity[blind] == 0.0).all()

    def test_cost_aware_explores_cheap_transactions_more_often(self, scored):
        scores, amounts, rng = scored
        p = ThresholdPolicy(0.03, ExplorationConfig(mode="cost_aware", rate=0.05))
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        dec = out.declined_by_policy
        cheap = dec & (amounts < np.median(amounts[dec]))
        pricey = dec & (amounts >= np.median(amounts[dec]))
        assert out.explore_prob[cheap].mean() > 2.0 * out.explore_prob[pricey].mean()

    def test_cost_aware_covers_the_deep_tail_that_the_taper_abandons(self, scored):
        """The design flaw the cost-aware mode exists to fix.

        In the riskiest decile of the decline region -- where a closed loop forgets fastest,
        because those labels are the first to vanish -- a risk taper by construction spends
        only a small fraction of its budget, while cost-aware exploration still spends its
        nominal rate there. A propensity floor rescues the taper's *identifiability*; it does
        not rescue its coverage, and coverage is what the correction actually learns from.
        """
        scores, amounts, rng = scored
        p_cost = ThresholdPolicy(0.03, ExplorationConfig(mode="cost_aware", rate=0.02))
        p_taper = ThresholdPolicy(
            0.03, ExplorationConfig(mode="risk_tapered", rate=0.02, propensity_floor=0.0)
        )
        for p in (p_cost, p_taper):
            p.calibrate(scores)
        dec = scores >= p_cost.threshold_
        top = dec & (scores >= np.quantile(scores[dec], 0.9))
        cost_q = p_cost.decide(scores, amounts, rng).explore_prob[top].mean()
        taper_q = p_taper.decide(scores, amounts, rng).explore_prob[top].mean()
        assert cost_q == pytest.approx(0.02, abs=0.01)
        assert cost_q > 3.0 * taper_q

    def test_oracle_policy_observes_everything(self, scored):
        scores, amounts, rng = scored
        p = OracleAccessPolicy(0.03)
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        assert (out.propensity == 1.0).all()
        # It still declines: the counterfactual is about labels, not about actions.
        assert out.decline_rate == pytest.approx(0.03, abs=0.002)


class TestPropensityBookkeeping:
    def test_known_propensity_matches_the_policy(self, scored):
        scores, amounts, rng = scored
        p = ThresholdPolicy(0.03, ExplorationConfig(mode="uniform", rate=0.02))
        p.calibrate(scores)
        out = p.decide(scores, amounts, rng)
        np.testing.assert_allclose(
            known_propensity(out.declined_by_policy, out.explore_prob), out.propensity
        )

    def test_ipw_recovers_the_population_mean_from_a_biased_sample(self):
        """The actual estimator claim, on a case with a known answer.

        A sample selected on a covariate has a badly biased mean; weighting by the inverse of the
        (known) selection probability recovers the population mean. If this ever fails, the
        corrected arms in the simulator are not doing what their name says.
        """
        rng = np.random.default_rng(3)
        n = 200_000
        x = rng.uniform(size=n)
        y = (rng.uniform(size=n) < 0.1 + 0.6 * x).astype(float)
        e = np.where(x > 0.7, 0.02, 1.0)             # heavy selection against high x
        observed = rng.uniform(size=n) < e

        naive = y[observed].mean()
        w = build_weights(e[observed], config=WeightConfig(self_normalise=False, stabilise=False))
        corrected = float(np.sum(w * y[observed]) / np.sum(w))

        assert abs(naive - y.mean()) > 0.05
        assert corrected == pytest.approx(y.mean(), abs=0.01)


class TestWeights:
    def test_scheme_none_is_uniform(self):
        w = build_weights(np.array([0.02, 1.0, 1.0]), config=WeightConfig(scheme="none"))
        np.testing.assert_allclose(w, np.ones(3))

    def test_self_normalisation_keeps_mean_weight_at_one(self):
        e = np.concatenate([np.full(50, 0.02), np.ones(4_950)])
        w = build_weights(e, config=WeightConfig(self_normalise=True))
        assert w.mean() == pytest.approx(1.0)

    def test_max_weight_caps_the_tail(self):
        e = np.array([1e-6, 1.0, 1.0, 1.0])
        w = build_weights(
            e, config=WeightConfig(max_weight=10.0, self_normalise=False, stabilise=False)
        )
        assert w.max() == 10.0

    def test_rejects_unknown_scheme(self):
        with pytest.raises(ValueError):
            WeightConfig(scheme="magic")

    def test_ess_penalises_concentrated_weight(self):
        assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
        skewed = np.concatenate([np.full(99, 1.0), [1_000.0]])
        assert effective_sample_size(skewed) < 5.0

    def test_overlap_diagnostics_report_the_variance_cost(self):
        e = np.concatenate([np.full(20, 0.02), np.ones(1_980)])
        d = overlap_diagnostics(e, np.ones(2_000, dtype=bool))
        assert d["positivity_violation_rate"] == 0.0
        assert d["min_propensity"] == pytest.approx(0.02)
        assert 0.0 < d["ess_ratio"] < 1.0
