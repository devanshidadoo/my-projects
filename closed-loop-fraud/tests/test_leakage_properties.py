"""Property-based tests that future data cannot reach past features.

A unit test checks one example. The leak these tests guard against does not show up in one
example -- it shows up when a particular ordering, a tie in timestamps, or a window boundary
lines up just so. So the invariants are stated as properties over *generated* streams and
Hypothesis is left to find the counterexample.

The four properties, in the order they matter:

``prefix_invariance``
    Features for the first *k* transactions do not change when the rest of the stream is
    appended. This is the definition of point-in-time correctness, and it is the one a batch
    ``groupby`` fails.
``future_independence``
    Arbitrarily mutating any transaction after position *i* -- amount, card, merchant, label --
    leaves row *i*'s features bit-identical.
``order_invariance``
    The result depends on the stream's time order, not on the order rows happen to sit in the
    input frame.
``self_exclusion``
    A transaction never sees itself, including when several transactions share a timestamp
    (IEEE-CIS records time to the second, so exact ties are common, not hypothetical).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clfraud.features.point_in_time import FeatureBuilder, build_features, default_spec
from tests.conftest import make_frame

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# Streams are drawn with a deliberately tiny entity pool: collisions on card and merchant are
# what exercise the accumulators, and a large pool would mostly generate singletons.
rows_strategy = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=60 * 86_400.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=4),      # card
        st.integers(min_value=0, max_value=3),      # merchant
        st.floats(min_value=0.01, max_value=5_000.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=1),      # label
    ),
    min_size=2,
    max_size=60,
)


def _sorted_frame(rows: list[tuple]) -> pd.DataFrame:
    df = make_frame(rows)
    return df.sort_values(["ts", "transaction_id"], kind="mergesort").reset_index(drop=True)


@given(rows=rows_strategy, cut=st.floats(min_value=0.1, max_value=0.9))
@SETTINGS
def test_prefix_invariance(rows, cut):
    """f(S)[:k] == f(S[:k]) -- appending the future changes nothing about the past."""
    df = _sorted_frame(rows)
    k = max(int(len(df) * cut), 1)
    full = build_features(df)
    prefix = build_features(df.iloc[:k])
    pd.testing.assert_frame_equal(
        full.iloc[:k].reset_index(drop=True), prefix.reset_index(drop=True)
    )


@given(rows=rows_strategy, data=st.data())
@SETTINGS
def test_future_independence(rows, data):
    """Mutating any strictly-later transaction leaves earlier feature rows identical."""
    df = _sorted_frame(rows)
    if len(df) < 3:
        return
    i = data.draw(st.integers(min_value=0, max_value=len(df) - 2))
    j = data.draw(st.integers(min_value=i + 1, max_value=len(df) - 1))

    mutated = df.copy()
    mutated.loc[j, "amount"] = float(data.draw(st.floats(0.01, 9_999.0, allow_nan=False)))
    mutated.loc[j, "card_id"] = str(data.draw(st.integers(0, 9)))
    mutated.loc[j, "merchant_id"] = str(data.draw(st.integers(0, 9)))
    mutated.loc[j, "is_fraud"] = np.int8(data.draw(st.integers(0, 1)))

    base = build_features(df)
    after = build_features(mutated)
    # Rows strictly before the mutation must be untouched. Rows at or after it may legitimately
    # differ -- that is the accumulator doing its job.
    pd.testing.assert_frame_equal(
        base.iloc[: j].reset_index(drop=True), after.iloc[: j].reset_index(drop=True)
    )


@given(rows=rows_strategy, seed=st.integers(0, 10_000))
@SETTINGS
def test_order_invariance(rows, seed):
    """Shuffling the input frame cannot change the features, because time order is canonical."""
    df = _sorted_frame(rows)
    shuffled = df.sample(frac=1.0, random_state=seed)
    base = build_features(df)
    other = build_features(shuffled)
    pd.testing.assert_frame_equal(base, other.reset_index(drop=True))


@given(
    n_tied=st.integers(min_value=2, max_value=8),
    amount=st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
)
@SETTINGS
def test_self_exclusion_under_timestamp_ties(n_tied, amount):
    """Transactions sharing a timestamp are ordered by id, and none of them sees itself.

    IEEE-CIS stores ``TransactionDT`` in whole seconds, so simultaneous transactions on one card
    are routine. A window implemented with a closed interval would count the current row (and
    its tied siblings) and quietly inflate every velocity feature on exactly the bursts that
    matter most.
    """
    rows = [(1_000.0, 0, 0, amount, 0) for _ in range(n_tied)]
    feats = build_features(make_frame(rows))
    counts = feats["card_cnt_1h"].to_numpy()
    # Strict prior ordering on ties: 0, 1, 2, ... never n_tied.
    assert counts.tolist() == list(range(n_tied))
    assert feats["card_amt_1h"].to_numpy()[0] == 0.0


def test_incremental_equals_batch(small_stream):
    """One pass over the whole stream == streaming it through in chunks.

    This is the train/serve-skew guarantee: the object that built the training matrix is the
    object that scores live traffic, so there is no second implementation to drift.
    """
    df = small_stream.head(5_000)
    batch = build_features(df)

    builder = FeatureBuilder(default_spec())
    chunks = [
        builder.transform(df.iloc[part])
        for part in np.array_split(np.arange(len(df)), 7)
    ]
    streamed = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(batch, streamed, check_exact=True)


def test_no_feature_is_a_function_of_the_label(small_stream):
    """Adversarial check: flipping every label must not move a single feature value.

    Label-derived features (target encoding, "merchant fraud rate") are the second classic leak
    and are easy to introduce accidentally through a shared helper. This makes it impossible to
    do so without a red test.
    """
    df = small_stream.head(4_000)
    flipped = df.copy()
    flipped["is_fraud"] = (1 - flipped["is_fraud"]).astype("int8")
    pd.testing.assert_frame_equal(build_features(df), build_features(flipped))


def test_velocity_features_match_a_brute_force_reference(small_stream):
    """Cross-check the O(1) accumulators against an O(n^2) definition of the same quantity.

    The accumulators are fast because they evict incrementally. The reference re-scans the whole
    prior stream for every row, which is obviously correct and obviously too slow to ship -- the
    right shape for a test oracle.
    """
    df = small_stream.head(2_500).reset_index(drop=True)
    feats = build_features(df)

    ts = df["ts"].to_numpy()
    card = df["card_id"].astype(str).to_numpy()
    amt = df["amount"].to_numpy()
    span = 86_400.0

    expected_cnt = np.zeros(len(df))
    expected_sum = np.zeros(len(df))
    for i in range(len(df)):
        prior = (np.arange(len(df)) < i) & (card == card[i]) & (ts >= ts[i] - span)
        expected_cnt[i] = prior.sum()
        expected_sum[i] = amt[prior].sum()

    np.testing.assert_allclose(feats["card_cnt_24h"].to_numpy(), expected_cnt, rtol=0, atol=1e-6)
    np.testing.assert_allclose(feats["card_amt_24h"].to_numpy(), expected_sum, rtol=1e-4)


def test_leaky_baseline_is_actually_leaky(small_stream):
    """Guard on the counterexample itself.

    ``leaky_aggregate_baseline`` exists to quantify how much offline AUC a whole-dataset groupby
    manufactures. If someone ever "fixed" it, the comparison in ``docs/RESULTS.md`` would
    silently become a comparison of nothing -- so assert that it still fails the property the
    real builder passes.
    """
    from clfraud.features.point_in_time import leaky_aggregate_baseline

    df = small_stream.head(3_000).reset_index(drop=True)
    k = 1_500
    full = leaky_aggregate_baseline(df)
    prefix = leaky_aggregate_baseline(df.iloc[:k])
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            full.iloc[:k].reset_index(drop=True), prefix.reset_index(drop=True)
        )
