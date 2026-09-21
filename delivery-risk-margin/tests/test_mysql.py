"""The MySQL path, exercised against a real server when one is available.

Skipped locally; the CI workflow starts a ``mysql:8`` service and sets
``DELIVERYRISK_MYSQL_URL``, so the claim that the same SQL runs on both engines is checked by a
machine rather than by the author's memory of the documentation.

What it checks is the only thing worth checking: that the *features come out identical*. Both
engines grew numeric ``RANGE`` frames and named ``WINDOW`` clauses independently, and "supported"
does not mean "agrees" -- particularly around frame peers, which is exactly where the
point-in-time guarantee lives.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from deliveryrisk.data.db import Database
from deliveryrisk.features.build import FeatureConfig, build_features
from deliveryrisk.features.sql import feature_columns

URL = os.environ.get("DELIVERYRISK_MYSQL_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set DELIVERYRISK_MYSQL_URL to run")


@pytest.fixture(scope="module")
def mysql_features(tiny_data):
    db = Database(URL)
    db.create_schema()
    db.load_frames(tiny_data.frames)
    return build_features(db, FeatureConfig(prior_weight=10.0, prior_rate=0.08))


def test_mysql_and_sqlite_agree_column_for_column(mysql_features, tiny_features):
    a = tiny_features.sort_values("order_id", ignore_index=True)
    b = mysql_features.sort_values("order_id", ignore_index=True)
    assert len(a) == len(b)
    assert (a.order_id.to_numpy() == b.order_id.to_numpy()).all()
    for col in feature_columns():
        if a[col].dtype.name in ("category", "object"):
            assert (a[col].astype(str) == b[col].astype(str)).all(), col
        else:
            assert np.allclose(
                a[col].astype(float), b[col].astype(float), rtol=1e-7, atol=1e-7, equal_nan=True
            ), col


def test_mysql_schema_round_trips(tiny_data):
    db = Database(URL)
    db.create_schema()
    db.load_frames(tiny_data.frames)
    counts = db.table_counts()
    for name, frame in tiny_data.frames.items():
        assert counts[name] == len(frame), name
