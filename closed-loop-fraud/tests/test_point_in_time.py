"""Unit tests for the accumulators and the feature builder's semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clfraud.features.point_in_time import NEVER_SEEN, FeatureBuilder, build_features, default_spec
from clfraud.features.windows import (
    CountSumWindow,
    DistinctWindow,
    RecencyTracker,
    WelfordTracker,
)
from tests.conftest import make_frame


class TestCountSumWindow:
    def test_window_boundary_is_inclusive_at_exactly_span(self):
        """An event exactly `span` seconds old is still inside the window.

        Pinned deliberately: the alternative convention is equally defensible, but silently
        flipping it would shift every velocity feature by a fraction of a percent and nobody
        would notice until a model comparison went inexplicably sideways.
        """
        w = CountSumWindow(100.0)
        w.update("a", 0.0, 10.0)
        assert w.query("a", 100.0) == (1, 10.0)
        assert w.query("a", 100.001) == (0, 0.0)

    def test_keys_are_independent(self):
        w = CountSumWindow(100.0)
        w.update("a", 0.0, 5.0)
        w.update("b", 0.0, 7.0)
        assert w.query("a", 1.0) == (1, 5.0)
        assert w.query("b", 1.0) == (1, 7.0)
        assert w.query("c", 1.0) == (0, 0.0)

    def test_evicted_keys_release_memory(self):
        """Active-key count must track live entities, not lifetime cardinality."""
        w = CountSumWindow(10.0)
        for i in range(500):
            w.update(f"k{i}", float(i), 1.0)
        assert w.n_active_keys == 500
        for i in range(500):
            w.query(f"k{i}", 10_000.0)
        assert w.n_active_keys == 0

    def test_rejects_nonpositive_span(self):
        with pytest.raises(ValueError):
            CountSumWindow(0.0)


class TestDistinctWindow:
    def test_counts_distinct_and_forgets(self):
        d = DistinctWindow(100.0)
        for t, v in [(0.0, "m1"), (1.0, "m2"), (2.0, "m1")]:
            d.update("card", t, v)
        assert d.query("card", 3.0) == 2
        # m1 at t=0 expires but m1 at t=2 survives.
        assert d.query("card", 101.0) == 2
        assert d.query("card", 103.0) == 0

    def test_repeat_values_do_not_double_count(self):
        d = DistinctWindow(1_000.0)
        for t in range(10):
            d.update("k", float(t), "same")
        assert d.query("k", 20.0) == 1


class TestRecencyTracker:
    def test_first_sight_is_infinite_gap(self):
        r = RecencyTracker()
        assert r.query("k", 5.0) == (float("inf"), 0)
        r.update("k", 5.0)
        assert r.query("k", 9.0) == (4.0, 1)


class TestWelfordTracker:
    def test_matches_numpy_on_prior_values(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(50, 12, size=200)
        w = WelfordTracker()
        for i, v in enumerate(vals):
            n, mean, std = w.query("k")
            assert n == i
            if i >= 2:
                assert mean == pytest.approx(vals[:i].mean(), rel=1e-9)
                assert std == pytest.approx(vals[:i].std(ddof=1), rel=1e-7)
            w.update("k", float(v))

    def test_zero_variance_is_not_a_division_by_zero(self):
        w = WelfordTracker()
        for _ in range(5):
            w.update("k", 3.0)
        _, mean, std = w.query("k")
        assert mean == pytest.approx(3.0)
        assert std == pytest.approx(0.0)


class TestFeatureBuilder:
    def test_first_transaction_has_empty_history(self):
        feats = build_features(make_frame([(0.0, 1, 1, 50.0, 0)]))
        assert feats["card_cnt_1h"].iloc[0] == 0
        assert feats["card_amt_24h"].iloc[0] == 0.0
        assert feats["card_since_last"].iloc[0] == NEVER_SEEN
        assert feats["card_amt_z"].iloc[0] == 0.0
        # No prior mean to divide by, so the ratio defaults to "typical", not to zero or inf.
        assert feats["card_amt_ratio"].iloc[0] == 1.0

    def test_velocity_accumulates_within_the_hour(self):
        rows = [(float(i) * 600.0, 1, i, 10.0, 0) for i in range(8)]
        feats = build_features(make_frame(rows))
        # 600 s apart, so at row i the prior hour holds min(i, 6) earlier transactions.
        assert feats["card_cnt_1h"].tolist() == [0, 1, 2, 3, 4, 5, 6, 6]

    def test_distinct_merchant_count_tracks_card_testing(self):
        rows = [(float(i) * 60.0, 1, i, 5.0, 0) for i in range(6)]
        feats = build_features(make_frame(rows))
        assert feats["card_nuniq_merchant"].tolist() == [0, 1, 2, 3, 4, 5]

    def test_amount_z_score_flags_an_outlier(self):
        rows = [(float(i) * 3_600.0, 1, 1, 20.0 + (i % 3), 0) for i in range(12)]
        rows.append((12 * 3_600.0, 1, 1, 4_000.0, 1))
        feats = build_features(make_frame(rows))
        assert feats["card_amt_z"].iloc[-1] > 10.0
        assert feats["card_amt_ratio"].iloc[-1] > 100.0

    def test_feature_names_match_matrix_columns(self):
        b = FeatureBuilder(default_spec())
        out = b.transform(make_frame([(0.0, 1, 1, 5.0, 0), (10.0, 1, 2, 6.0, 0)]))
        assert list(out.columns) == b.feature_names()

    def test_reset_clears_state(self):
        b = FeatureBuilder(default_spec())
        df = make_frame([(0.0, 1, 1, 5.0, 0), (10.0, 1, 1, 5.0, 0)])
        first = b.transform(df)
        again = b.reset().transform(df)
        pd.testing.assert_frame_equal(first, again)
        assert b.n_seen == len(df)

    def test_empty_input_returns_empty_matrix_with_schema(self):
        b = FeatureBuilder(default_spec())
        out = b.transform(make_frame([]).astype({"ts": "float64"}), validate=False)
        assert list(out.columns) == b.feature_names()
        assert len(out) == 0

    def test_passthrough_columns_are_carried_through_untouched(self, tiny_stream):
        cols = [c for c in tiny_stream.columns if c.startswith("v")][:4]
        feats = build_features(tiny_stream.head(300), passthrough=cols)
        for c in cols:
            np.testing.assert_allclose(
                feats[c].to_numpy(dtype="float64"),
                tiny_stream.head(300)[c].to_numpy(dtype="float64"),
                equal_nan=True, rtol=1e-6,
            )

    def test_matrix_has_no_infinities(self, tiny_stream):
        """Sentinels, not infinities: LightGBM tolerates inf, most downstream tooling does not."""
        feats = build_features(tiny_stream.head(2_000))
        assert np.isfinite(feats.to_numpy(dtype="float64")).all()
