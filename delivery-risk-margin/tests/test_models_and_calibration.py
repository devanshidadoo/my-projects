"""Models, calibration, and the metrics the report is built on."""
from __future__ import annotations

import numpy as np
import pytest

from deliveryrisk.evaluation.metrics import (
    bootstrap_ci,
    capture_at_top,
    decile_table,
    ranking_metrics,
    recall_at_rate,
)
from deliveryrisk.features.build import labelled, split_by_time
from deliveryrisk.models.base import ModelConfig, apply_feature_space, build_feature_space
from deliveryrisk.models.calibration import (
    Calibrator,
    calibration_metrics,
    reliability_table,
)
from deliveryrisk.models.cause import CauseModel, cause_labels, population_mix

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ metrics
def test_decile_table_partitions_everything():
    y = (RNG.random(5000) < 0.1).astype(float)
    p = RNG.random(5000)
    tab = decile_table(y, p)
    assert len(tab) == 10
    assert tab.n.sum() == 5000
    assert tab.n_late.sum() == y.sum()
    assert tab.cum_capture.iloc[-1] == pytest.approx(1.0)
    assert tab.cum_capture.is_monotonic_increasing


def test_a_perfect_score_captures_everything_in_the_first_decile():
    y = np.zeros(1000)
    y[:80] = 1.0
    p = y + RNG.random(1000) * 1e-6
    assert capture_at_top(y, p, 1) == pytest.approx(1.0)
    assert recall_at_rate(y, p, 0.08) == pytest.approx(1.0)


def test_a_useless_score_captures_its_share_and_no_more():
    y = (RNG.random(20000) < 0.1).astype(float)
    p = RNG.random(20000)
    assert capture_at_top(y, p, 2) == pytest.approx(0.2, abs=0.03)
    m = ranking_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.02)
    assert m["lift_top1"] == pytest.approx(1.0, abs=0.15)


def test_ranking_metrics_are_invariant_to_monotone_rescaling():
    """The point the calibration section turns on: ranking cannot see a squashed score."""
    y = (RNG.random(5000) < 0.15).astype(float)
    p = RNG.beta(2, 8, 5000)
    a = ranking_metrics(y, p)
    b = ranking_metrics(y, p**3)
    for k in ("roc_auc", "pr_auc", "capture_top2"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k
    # ... while calibration certainly can.
    assert calibration_metrics(y, p)["ece"] < calibration_metrics(y, p**3)["ece"]


def test_bootstrap_interval_brackets_the_mean():
    x = RNG.normal(5.0, 2.0, 4000)
    lo, hi = bootstrap_ci(x, seed=1)
    assert lo < x.mean() < hi
    assert hi - lo < 1.0


# ------------------------------------------------------------------ calibration
def test_isotonic_fixes_a_stretched_score():
    p = RNG.beta(2, 20, 20000)
    y = (RNG.random(20000) < p).astype(float)
    stretched = np.clip(p * 2.5, 0, 1)
    before = calibration_metrics(y, stretched)["ece"]
    cal = Calibrator("isotonic").fit(stretched, y)
    after = calibration_metrics(y, cal.transform(stretched))["ece"]
    assert after < before / 3


def test_calibration_preserves_ranking():
    """A calibrator is a monotone map, so it must not reorder anyone."""
    p = RNG.beta(2, 15, 5000)
    y = (RNG.random(5000) < p).astype(float)
    cal = Calibrator("isotonic").fit(p, y)
    out = cal.transform(p)
    assert np.all(np.diff(out[np.argsort(p)]) >= -1e-12)


def test_none_method_is_the_identity():
    p = RNG.random(100)
    assert np.allclose(Calibrator("none").fit(p, (p > 0.5).astype(float)).transform(p), p)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_calibration_slope_detects_over_spread_scores():
    p = RNG.beta(2, 20, 20000)
    y = (RNG.random(20000) < p).astype(float)
    over = np.clip(p * 3.0, 1e-6, 1 - 1e-6)
    assert calibration_metrics(y, over)["calibration_slope"] < 1.0


def test_reliability_bins_are_equal_count():
    p = np.concatenate([RNG.beta(1, 50, 9000), RNG.random(1000)])
    y = (RNG.random(10000) < p).astype(float)
    tab = reliability_table(y, p, 10)
    assert tab.n.nunique() <= 2  # equal-count up to the remainder


# ------------------------------------------------------------------ models
@pytest.fixture(scope="module")
def splits(tiny_features):
    return split_by_time(labelled(tiny_features), 0.6, 0.15)


def test_split_is_chronological_and_exhaustive(splits, tiny_features):
    train, valid, test = splits
    assert train.approved_ts.max() <= valid.approved_ts.min()
    assert valid.approved_ts.max() <= test.approved_ts.min()
    assert len(train) + len(valid) + len(test) == len(labelled(tiny_features))


def test_feature_space_collapses_rare_levels_and_holds_column_order(tiny_features):
    space = build_feature_space(tiny_features, ModelConfig(max_categories=3))
    out = apply_feature_space(tiny_features, space)
    assert list(out.columns) == space.columns
    for c in space.categorical:
        assert len(space.levels[c]) <= 3
        assert set(out[c].cat.categories) <= set(space.levels[c]) | {"__other__"}


def test_unseen_category_at_predict_time_becomes_other(tiny_features):
    space = build_feature_space(tiny_features, ModelConfig())
    df = tiny_features.copy()
    df["category"] = "a_category_that_did_not_exist_at_training_time"
    out = apply_feature_space(df, space)
    assert (out.category.astype(str) == "__other__").all()


def test_models_beat_the_base_rate(splits):
    from deliveryrisk.models.linear import make_model

    train, valid, test = splits
    y = test.y_late.to_numpy(dtype=float)
    for kind in ("logistic", "gbdt"):
        model = make_model(ModelConfig(kind=kind, n_estimators=80, num_leaves=7,
                                       min_child_samples=20))
        model.fit(train, train.y_late.to_numpy(dtype=float))
        p = model.predict_proba(test)
        assert p.min() >= 0 and p.max() <= 1
        assert ranking_metrics(y, p)["roc_auc"] > 0.5, kind


def test_unknown_model_kind_is_rejected():
    from deliveryrisk.models.linear import make_model

    with pytest.raises(ValueError):
        make_model(ModelConfig(kind="random_forest_but_vibes"))


# ------------------------------------------------------------------ cause model
def test_cause_labels_are_disjoint_and_only_for_late_orders(tiny_features):
    lab = cause_labels(labelled(tiny_features))
    late = labelled(tiny_features).y_late > 0
    assert lab[~late.to_numpy()].isna().all()
    assert lab[late.to_numpy()].notna().all()
    assert set(lab.dropna().unique()) <= {"handling_only", "transit_only", "either", "neither"}


def test_population_mix_sums_to_one(tiny_features):
    mix = population_mix(labelled(tiny_features))
    assert sum(mix.values()) == pytest.approx(1.0)


def test_cause_model_returns_a_distribution(splits):
    train, _, test = splits
    model = CauseModel(ModelConfig(n_estimators=50, num_leaves=7, min_child_samples=10)).fit(train)
    mix = model.predict_mix(test)
    assert np.allclose(mix.sum(axis=1), 1.0, atol=1e-6)
    joint = model.predict_joint(test, np.full(len(test), 0.2))
    assert np.allclose(sum(joint.values()), 0.2, atol=1e-6)


def test_cause_model_falls_back_when_there_is_nothing_to_learn(splits):
    """Too few late orders must degrade to the population mix, not to a confident guess."""
    train, _, test = splits
    tiny = train.head(40)
    model = CauseModel().fit(tiny)
    mix = model.predict_mix(test)
    assert mix.nunique().max() == 1, "fallback should return one constant row"
