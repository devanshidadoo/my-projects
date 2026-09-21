from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def tiny_data():
    """A small but complete generated warehouse: 700 orders, real process, all nine tables."""
    from deliveryrisk.data.synthetic import SyntheticConfig, generate

    return generate(
        SyntheticConfig(
            n_orders=700, n_unique_customers=650, n_sellers=30, n_products=150,
            span_days=180, seed=23,
        )
    )


@pytest.fixture(scope="session")
def tiny_db(tiny_data):
    from deliveryrisk.data.db import Database

    db = Database("sqlite://:memory:")
    db.create_schema()
    db.load_frames(tiny_data.frames)
    return db


@pytest.fixture(scope="session")
def tiny_features(tiny_db):
    from deliveryrisk.features.build import FeatureConfig, build_features

    return build_features(tiny_db, FeatureConfig(prior_weight=10.0, prior_rate=0.08))
