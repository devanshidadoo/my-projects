"""The point-in-time guarantee, as executable properties.

The claim under test is exactly one sentence: *the feature row of an order is a function of
events strictly before its approval, and of its own pre-dispatch attributes.* Four properties
say the same thing in four ways that a real bug would break differently::

    prefix invariance      f(S)[:k] == f(S[:k])        appending the future changes nothing
    future independence    mutating order j > i leaves order i bit-identical
    order invariance       the physical row order of the input cannot matter
    self exclusion         an order never sees its own outcome, ties included

They are checked over Hypothesis-generated streams rather than one hand-written case, because
the interesting failures are at the boundaries -- simultaneous timestamps, a seller's first
order, an order that never delivers -- and those are exactly the cases a fixture author leaves
out.

The last test in the file is the negative control: the two-sided builder is asserted to *fail*
prefix invariance. Without it, someone could "fix" the leaky baseline into something harmless and
the entire leakage audit would quietly start measuring nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from factory import features_for, simple_spec
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deliveryrisk.features.sql import CATEGORICAL_COLUMNS, ENTITIES, feature_columns

SETTINGS = settings(
    max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

# `carrier_id` starts with an entity prefix but is a categorical attribute, not a history
# aggregate; comparing it as a float is how this list bites back.
HISTORY_COLUMNS = [
    c
    for c in feature_columns()
    if any(c.startswith(f"{e.name}_") for e in ENTITIES) and c not in CATEGORICAL_COLUMNS
]


@st.composite
def streams(draw, min_size: int = 4, max_size: int = 14):
    """A stream of orders with plausible-but-adversarial timings."""
    n = draw(st.integers(min_size, max_size))
    approved = sorted(
        draw(st.lists(st.integers(0, 40), min_size=n, max_size=n).map(lambda xs: [float(x) for x in xs]))
    )
    handling = draw(st.lists(st.floats(0.1, 8.0, allow_nan=False), min_size=n, max_size=n))
    transit = draw(st.lists(st.floats(0.5, 12.0, allow_nan=False), min_size=n, max_size=n))
    promise = draw(st.lists(st.floats(3.0, 20.0, allow_nan=False), min_size=n, max_size=n))
    sellers = draw(st.lists(st.sampled_from(["S0", "S1", "S2"]), min_size=n, max_size=n))
    delivered = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return simple_spec(approved, handling, transit, promise, sellers, delivered)


def _stable_ids(spec: pd.DataFrame) -> pd.DataFrame:
    """Pin each order's id to its identity, not to its row position.

    Without this, permuting the input renames the orders and the comparison becomes vacuous --
    it would pass for an implementation that ignored the permutation *and* for one that
    scrambled the results.
    """
    out = spec.copy()
    out["oid"] = [f"O{i:05d}" for i in range(len(out))]
    return out


def _compare(a: pd.DataFrame, b: pd.DataFrame, ids) -> None:
    a = a[a.order_id.isin(ids)].set_index("order_id").sort_index()
    b = b[b.order_id.isin(ids)].set_index("order_id").sort_index()
    for col in HISTORY_COLUMNS:
        left = a[col].astype(float).to_numpy()
        right = b[col].astype(float).to_numpy()
        assert np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True), col


@SETTINGS
@given(spec=streams())
def test_prefix_invariance(spec):
    """Truncating the future leaves every earlier feature row untouched."""
    k = max(2, len(spec) // 2)
    full = features_for(spec)
    prefix = features_for(spec.iloc[:k])
    _compare(full, prefix, prefix.order_id)


@SETTINGS
@given(spec=streams(min_size=5))
def test_future_independence(spec):
    """Changing what happens to a later order cannot move an earlier one."""
    k = len(spec) - 2
    mutated = spec.copy()
    mutated.loc[mutated.index[k:], "handling_days"] = 30.0
    mutated.loc[mutated.index[k:], "transit_days"] = 40.0
    mutated.loc[mutated.index[k:], "delivered"] = ~mutated.delivered.iloc[k:].to_numpy()
    base = features_for(spec)
    other = features_for(mutated)
    _compare(base, other, base.order_id.iloc[:k])


@SETTINGS
@given(spec=streams(), seed=st.integers(0, 10_000))
def test_order_invariance(spec, seed):
    """The physical order of rows in the input tables is not information."""
    spec = _stable_ids(spec)
    shuffled = spec.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    a = features_for(spec)
    b = features_for(shuffled)
    _compare(a, b, a.order_id)


def test_self_exclusion_under_ties():
    """Simultaneous events are never visible to a decision at the same instant.

    An order management system stamping whole seconds produces ties constantly, and a closed
    interval would let an order see a delivery confirmed in the same second it was approved --
    or, worse, its own queue entry.
    """
    # Order 0 is collected on day 1 and delivered at exactly day 4 -- the instant orders 1-3
    # are approved. A closed interval would hand all three a prior delivery they cannot know
    # about, and on a real ERP stamping whole seconds this happens thousands of times a day.
    spec = simple_spec(
        approved_days=[0.0, 4.0, 4.0, 4.0],
        handling_days=[1.0, 1.0, 1.0, 1.0],
        transit_days=[3.0, 3.0, 3.0, 3.0],
        promise_days=[2.0, 9.0, 9.0, 9.0],  # order 0 is late
        sellers=["S0"] * 4,
    )
    f = features_for(spec).set_index("order_id")
    assert f.loc["O00000", "y_late"] == 1, "the fixture failed to make order 0 late"
    for oid in ("O00001", "O00002", "O00003"):
        assert f.loc[oid, "seller_n_prior"] == 0, oid
        # Order 0 left the queue on day 1, and the three simultaneous orders do not see
        # each other's queue entries either.
        assert f.loc[oid, "seller_queue_depth"] == 0, oid

    # Now make the collection itself land on the tie: order 0 is still uncollected at day 4.
    spec.loc[0, "handling_days"] = 4.0
    g = features_for(spec).set_index("order_id")
    for oid in ("O00001", "O00002", "O00003"):
        assert g.loc[oid, "seller_queue_depth"] == 1, oid


def test_own_outcome_never_reaches_own_row():
    """Flipping an order's own lateness changes nothing about its own features."""
    base = simple_spec([0.0, 5.0, 10.0], promise_days=[20.0, 20.0, 20.0], sellers=["S0"] * 3)
    flipped = base.copy()
    flipped.loc[2, "promise_days"] = 1.0  # the last order is now catastrophically late
    a = features_for(base)
    b = features_for(flipped)
    _compare(a, b, ["O00002"])


def test_labels_do_not_touch_features():
    """Flipping *every* label must not move a single feature value.

    This is the cheapest possible test for accidental target encoding, and it is the one that
    catches a new aggregate added to the wrong SELECT block.
    """
    base = simple_spec(np.arange(0.0, 12.0), sellers=["S0", "S1"] * 6)
    flipped = base.copy()
    # Flip the labels through the *outcome*, not through the promise: the promise is known at
    # purchase and is legitimately a feature, so moving it would make this test trivially fail
    # for the right reason and hide the wrong one.
    flipped["transit_days"] = 40.0
    a = features_for(base).set_index("order_id")
    b = features_for(flipped).set_index("order_id")
    late_changed = (a.y_late.to_numpy() != b.y_late.to_numpy()).sum()
    assert late_changed > 0, "the fixture failed to flip any label"
    static = [c for c in feature_columns() if c not in HISTORY_COLUMNS]
    for col in static:
        if a[col].dtype.name in ("category", "object"):
            assert (a[col].astype(str) == b[col].astype(str)).all(), col
        else:
            assert np.allclose(a[col].astype(float), b[col].astype(float), equal_nan=True), col


def test_two_sided_builder_is_actually_leaky():
    """The negative control. If this ever passes, the leakage audit is measuring nothing."""
    spec = simple_spec(np.arange(0.0, 14.0), promise_days=6.0, sellers=["S0"] * 14)
    spec.loc[10:, "transit_days"] = 30.0  # the future goes wrong
    full = features_for(spec, mode="two_sided")
    prefix = features_for(spec.iloc[:8], mode="two_sided")
    with pytest.raises(AssertionError):
        _compare(full, prefix, prefix.order_id)
