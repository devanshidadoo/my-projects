"""Shared fixtures. Streams here are small on purpose: property tests run hundreds of times."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clfraud.data.synthetic import SyntheticConfig, generate_stream


@pytest.fixture(scope="session")
def tiny_stream() -> pd.DataFrame:
    return generate_stream(
        SyntheticConfig(
            n_transactions=4_000,
            n_months=4,
            n_cards=120,
            n_merchants=40,
            n_devices=90,
            n_addr=15,
            n_mcc=8,
            hot_cell_count=3,
            seed=1,
        )
    )


@pytest.fixture(scope="session")
def small_stream() -> pd.DataFrame:
    return generate_stream(
        SyntheticConfig(
            n_transactions=20_000,
            n_months=8,
            n_cards=500,
            n_merchants=120,
            n_devices=300,
            n_addr=40,
            n_mcc=12,
            hot_cell_count=5,
            seed=2,
        )
    )


def make_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a canonical frame from ``(ts, card, merchant, amount, fraud)`` tuples."""
    return pd.DataFrame(
        {
            "transaction_id": np.arange(1, len(rows) + 1, dtype="int64"),
            "ts": [float(r[0]) for r in rows],
            "card_id": pd.array([str(r[1]) for r in rows], dtype="string"),
            "merchant_id": pd.array([str(r[2]) for r in rows], dtype="string"),
            "device_id": pd.array([f"dev{r[1]}" for r in rows], dtype="string"),
            "email_domain": pd.array(["a.com"] * len(rows), dtype="string"),
            "addr_id": pd.array(["addr1"] * len(rows), dtype="string"),
            "product_cd": pd.array(["W"] * len(rows), dtype="string"),
            "amount": [float(r[3]) for r in rows],
            "is_fraud": np.array([int(r[4]) for r in rows], dtype="int8"),
        }
    )
