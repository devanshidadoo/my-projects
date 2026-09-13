"""The fast SQL and the slow, obvious implementation must agree, exactly.

Window aggregates over an event stream are the kind of code that is either right or off by one,
and nothing in between is visible by reading it. So the same definitions are written a second
time in ``features/reference.py`` the way one would explain them at a whiteboard, and the two are
compared on every entity and every column.
"""
from __future__ import annotations

import numpy as np
import pytest

from deliveryrisk.features.build import FeatureConfig, build_features
from deliveryrisk.features.reference import reference_entity_features
from deliveryrisk.features.sql import ENTITIES

PRIOR_WEIGHT, PRIOR_RATE = 12.0, 0.09


@pytest.fixture(scope="module")
def built(tiny_db):
    features = build_features(
        tiny_db, FeatureConfig(prior_weight=PRIOR_WEIGHT, prior_rate=PRIOR_RATE)
    )
    facts = tiny_db.read_sql("SELECT * FROM order_facts")
    return features, facts


@pytest.mark.parametrize("spec", ENTITIES, ids=[e.name for e in ENTITIES])
def test_sql_matches_brute_force(built, spec):
    features, facts = built
    ref = reference_entity_features(
        facts, spec, prior_weight=PRIOR_WEIGHT, prior_rate=PRIOR_RATE
    )
    cols = [c for c in ref.columns if c != "order_id"]
    assert cols, f"{spec.name} produced no columns"
    for col in cols:
        a = features.set_index("order_id")[col].astype(float)
        b = ref.set_index("order_id")[col].astype(float)
        a, b = a.align(b, join="inner")
        assert len(a) == len(features), f"{col}: reference did not cover every order"
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), col


def test_queue_depth_is_never_negative(built):
    features, _ = built
    for spec in ENTITIES:
        if spec.queue is None:
            continue
        col = f"{spec.name}_queue_depth"
        assert (features[col].fillna(0) >= 0).all(), col


def test_new_entities_get_the_prior_not_a_nan(built):
    """A seller's first order has no history, and the shrunk rate must still be usable."""
    features, _ = built
    first = features[features.seller_n_prior == 0]
    assert len(first) > 0
    assert np.allclose(first.seller_late_rate, PRIOR_RATE)
    assert first.seller_mean_handling.isna().all(), (
        "an unknown mean must be missing, not imputed to the population median inside SQL"
    )
