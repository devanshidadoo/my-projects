"""Closed-loop mechanics: censoring, maturation, and the arms actually differing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clfraud.bias.reweighting import WeightConfig
from clfraud.features.point_in_time import build_features
from clfraud.models.base import ModelConfig
from clfraud.simulator.labeling import LabelPipeline, LabelPipelineConfig
from clfraud.simulator.loop import ClosedLoopSimulator, LoopConfig, PolicyArm
from clfraud.simulator.policy import ExplorationConfig

DAY = 86_400.0


@pytest.fixture(scope="module")
def sim_inputs(small_stream):
    X = build_features(small_stream, passthrough=["v1", "v2", "v3", "v4", "mcc"])
    meta = small_stream[["ts", "amount", "is_fraud", "archetype"]]
    return X, meta


def _cfg(**kw) -> LoopConfig:
    base = dict(
        cycle_days=30.0, n_cycles=8, warmup_cycles=2, train_window_cycles=4,
        decline_budget=0.05, max_train_rows=20_000,
        model=ModelConfig(n_estimators=60, num_leaves=16),
        label=LabelPipelineConfig(delay_days=30.0),
    )
    base.update(kw)
    return LoopConfig(**base)


class TestLabelPipeline:
    def test_maturity_respects_the_chargeback_delay(self):
        lp = LabelPipeline(LabelPipelineConfig(delay_days=45.0))
        ts = np.array([0.0, 40 * DAY, 50 * DAY])
        np.testing.assert_array_equal(
            lp.is_mature(ts, as_of=46 * DAY), [True, False, False]
        )

    def test_unreported_fraud_only_removes_positives(self):
        lp = LabelPipeline(LabelPipelineConfig(unreported_fraud_rate=0.5, seed=3))
        y = np.ones(10_000, dtype=int)
        obs = lp.observed_labels(y)
        assert 0.45 < 1 - obs.mean() < 0.55

    def test_friendly_fraud_only_adds_positives(self):
        lp = LabelPipeline(LabelPipelineConfig(friendly_fraud_rate=0.1, seed=4))
        obs = lp.observed_labels(np.zeros(10_000, dtype=int))
        assert 0.08 < obs.mean() < 0.12

    def test_no_noise_is_the_identity(self):
        lp = LabelPipeline(LabelPipelineConfig())
        y = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(lp.observed_labels(y), y)

    def test_maturity_lag_is_reported_in_cycles(self):
        assert LabelPipeline(LabelPipelineConfig(delay_days=45)).maturity_lag_cycles(30) == 1.5


class TestClosedLoop:
    def test_rejects_misaligned_inputs(self, sim_inputs):
        X, meta = sim_inputs
        with pytest.raises(ValueError):
            ClosedLoopSimulator(X.iloc[:100], meta, _cfg())

    def test_naive_arm_loses_labels_and_the_oracle_does_not(self, sim_inputs):
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        naive = sim.run_arm(PolicyArm("naive"))
        oracle = sim.run_arm(PolicyArm("oracle", oracle_labels=True))

        assert naive.cycles["label_censoring_rate"].mean() == pytest.approx(0.05, abs=0.02)
        assert oracle.cycles["label_censoring_rate"].max() == 0.0
        # The mechanism under study: the oracle sees strictly more labelled fraud.
        assert (
            oracle.cycles["fraud_labels_gained"].sum()
            > naive.cycles["fraud_labels_gained"].sum()
        )
        assert naive.cycles["fraud_labels_censored"].sum() > 0

    def test_oracle_declines_exactly_like_the_naive_policy(self, sim_inputs):
        """The oracle arm is a *labelling* counterfactual, not a better decision rule.

        If it also declined differently, its advantage would confound "more labels" with
        "different actions" and the comparison would mean nothing.
        """
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        naive = sim.run_arm(PolicyArm("naive"))
        oracle = sim.run_arm(PolicyArm("oracle", oracle_labels=True))
        assert naive.cycles.loc[0, "realised_decline_rate"] == pytest.approx(
            oracle.cycles.loc[0, "realised_decline_rate"], abs=1e-9
        )

    def test_exploration_buys_labels_in_the_declined_region(self, sim_inputs):
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        naive = sim.run_arm(PolicyArm("naive"))
        explore = sim.run_arm(
            PolicyArm(
                "explore_ipw",
                exploration=ExplorationConfig(rate=0.05, mode="cost_aware"),
                weights=WeightConfig(scheme="ipw"),
            )
        )
        assert explore.cycles["explored_count"].sum() > 0
        assert explore.cycles["min_propensity"].min() > 0.0
        assert naive.cycles["min_propensity"].min() == 0.0

        # The exact invariant: censoring is the decline rate less whatever exploration bought
        # back. Naive censors every decline; the exploring arm censors strictly fewer.
        for r in (naive, explore):
            recovered = r.cycles["policy_decline_rate"] - r.cycles["label_censoring_rate"]
            np.testing.assert_allclose(
                recovered, r.cycles["explored_count"] / r.cycles["n"], atol=1e-9
            )
        assert (
            explore.cycles["label_censoring_rate"] < explore.cycles["policy_decline_rate"]
        ).all()
        np.testing.assert_allclose(
            naive.cycles["label_censoring_rate"], naive.cycles["policy_decline_rate"], atol=1e-12
        )

        # Deliberately *not* asserted: that the exploring arm censors fewer labels overall than
        # the naive one. It usually censors more. Each arm calibrates its threshold on its own
        # previous-cycle scores, so a better model declines a different -- often larger -- slice
        # of fraud, and 2-5 % of declines is a rounding error against that. The recovery does not
        # come from label volume; it comes from having any labelled evidence at all inside the
        # blocked region. See README, "The mechanism is visible in what the training set is
        # made of".

    def test_exploration_cost_is_bounded_by_its_rate(self, sim_inputs):
        """Approving 5 % of would-be declines cannot cost more than 5 % of fraud value.

        Two checks, because they fail for different reasons. The first is an identity that must
        hold every cycle: explored fraud is a subset of approved fraud. The second is the
        statistical claim -- uniform exploration at rate r costs r times the fraud value the
        policy was blocking -- and it holds *in aggregate*, not cycle by cycle. Fraud value is
        heavy-tailed, so one explored high-ticket transaction can dominate a single cycle's
        ratio; averaging ratios over cycles with small denominators is the wrong estimator.
        """
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        res = sim.run_arm(
            PolicyArm(
                "explore",
                exploration=ExplorationConfig(rate=0.05, mode="uniform"),
                weights=WeightConfig(scheme="ipw"),
            )
        )
        g = res.cycles
        assert (g["exploration_fraud_amount"] <= g["fraud_amount_approved"] + 1e-9).all()
        pooled = g["exploration_fraud_amount"].sum() / g["total_fraud_amount"].sum()
        assert pooled < 0.05

    def test_frozen_arm_never_retrains(self, sim_inputs):
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        frozen = sim.run_arm(PolicyArm("frozen", retrain=False))
        assert "train_rows" not in frozen.cycles.columns or frozen.cycles["train_rows"].isna().all()
        assert frozen.importances.empty

    def test_decline_budget_is_spent_every_cycle(self, sim_inputs):
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg(decline_budget=0.08))
        res = sim.run_arm(PolicyArm("naive"))
        # Thresholds come from the *previous* cycle, so the realised rate drifts a little.
        assert res.cycles["realised_decline_rate"].mean() == pytest.approx(0.08, abs=0.03)

    def test_results_are_reproducible(self, sim_inputs):
        X, meta = sim_inputs
        cfg = _cfg()
        a = ClosedLoopSimulator(X, meta, cfg).run_arm(
            PolicyArm("e", exploration=ExplorationConfig(rate=0.05), weights=WeightConfig("ipw"))
        )
        b = ClosedLoopSimulator(X, meta, cfg).run_arm(
            PolicyArm("e", exploration=ExplorationConfig(rate=0.05), weights=WeightConfig("ipw"))
        )
        pd.testing.assert_frame_equal(a.cycles, b.cycles)

    def test_evaluation_covers_the_whole_cycle_not_just_approvals(self, sim_inputs):
        """Metrics must be computed on declined transactions too, or the failure is invisible.

        This is the point of the simulator: every observable metric an issuer could compute
        looks fine while ground-truth recall falls, because the issuer only ever sees approvals.
        """
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        res = sim.run_arm(PolicyArm("naive"))
        cfg = _cfg()
        cycle_sizes = res.cycles["n"].to_numpy()
        # The simulator clips the cycle index at n_cycles - 1, so the final cycle absorbs any
        # transactions past the nominal horizon; mirror that here rather than off-by-one.
        cycle = np.clip(
            ((meta["ts"] - meta["ts"].min()) // (cfg.cycle_days * DAY)).astype(int),
            0, cfg.n_cycles - 1,
        )
        counts = meta.assign(cycle=cycle).groupby("cycle").size()
        np.testing.assert_array_equal(
            cycle_sizes, counts.loc[cfg.warmup_cycles : cfg.n_cycles - 1].to_numpy()
        )

    def test_composition_diagnostic_tracks_archetypes(self, sim_inputs):
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        res = sim.run_arm(PolicyArm("naive"))
        comp = res.train_composition
        assert not comp.empty
        shares = [c for c in comp.columns if c.startswith("pos_")]
        assert shares
        assert comp[shares].sum(axis=1).round(6).eq(1.0).all()


class TestReproducibility:
    """Same config, same numbers -- across processes, not just within one."""

    def test_arm_seed_does_not_depend_on_python_hash_randomisation(self):
        """Regression: arm RNGs were seeded with ``hash(arm.name)``.

        Python salts string hashing per process, so every run drew different exploration
        samples while the in-process reproducibility test passed happily. A published result
        nobody else can reproduce is worse than no result.
        """
        import inspect
        import zlib

        src = inspect.getsource(ClosedLoopSimulator.run_arm)
        assert "hash(arm.name)" not in src, "arm seed must not use Python's salted hash()"
        assert "crc32" in src, "arm seed must use a stable, unsalted hash"
        # crc32 is stable across processes and interpreter versions; hash() is not.
        assert zlib.crc32(b"explore_ipw") == 2_274_173_458

    def test_different_arm_names_get_different_exploration_draws(self, sim_inputs):
        """Two arms with identical settings but different names must not share a draw sequence.

        If they did, an apparent difference between arms could be an artefact of one shared
        random stream rather than of the policies themselves.
        """
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        kw = {
            "exploration": ExplorationConfig(rate=0.05, mode="cost_aware"),
            "weights": WeightConfig(scheme="ipw"),
        }
        ra = sim.run_arm(PolicyArm("arm_a", **kw)).cycles
        rb = sim.run_arm(PolicyArm("arm_b", **kw)).cycles
        assert not np.array_equal(
            ra["explored_count"].to_numpy(), rb["explored_count"].to_numpy()
        )
        # The draws differ; the budget does not. Pooled over cycles, because at this scale a
        # cycle explores single-digit transactions and per-cycle counts are pure Poisson noise.
        for r in (ra, rb):
            share = r["explored_count"].sum() / r["policy_decline_rate"].mul(r["n"]).sum()
            assert share == pytest.approx(0.05, abs=0.04)


class TestEstimatedPropensity:
    def test_estimated_propensity_models_the_decision_not_the_calendar(self, sim_inputs):
        """The estimated-propensity arm must model the approval rule, not label maturation.

        A version that trained on "is this row in the training pool" learned that recent
        transactions are unobserved -- true, but a fact about chargeback delay, not about the
        policy. The weights it produced described the calendar.
        """
        X, meta = sim_inputs
        sim = ClosedLoopSimulator(X, meta, _cfg())
        res = sim.run_arm(
            PolicyArm(
                "est",
                exploration=ExplorationConfig(rate=0.05, mode="cost_aware"),
                weights=WeightConfig(scheme="ipw"),
                estimate_propensity=True,
            )
        )
        trained = res.cycles.dropna(subset=["train_rows"])
        assert len(trained) > 0
        # Most traffic is approved, so a sane estimator puts the median propensity near the
        # approval rate -- far from the degenerate "everything recent is censored" regime.
        assert trained["ovl_median_propensity"].median() > 0.5
        # Typical cycles keep almost all of their effective sample. Individual cycles can still
        # dip when the estimator assigns one row a very small propensity, which is the known
        # variance cost of estimated weights rather than a modelling error -- hence the median.
        assert trained["train_ess_ratio"].median() > 0.5
