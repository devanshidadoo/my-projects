"""Build a minimal, valid nine-table warehouse from a compact order specification.

The property tests need to say "here is a stream of orders, now truncate it / permute it /
change the future" without also having to say anything about products, payments or reviews. This
builds the rest of the schema around whatever orders are asked for, so a test can be about the
one thing it is about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400.0


def make_frames(orders_spec: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Expand ``orders_spec`` into the nine tables.

    Expected columns: ``approved_days``, ``handling_days``, ``transit_days``, ``promise_days``,
    ``seller`` and ``customer``; everything else is filled with valid constants.
    """
    s = orders_spec.reset_index(drop=True)
    n = len(s)
    order_ids = (
        [str(o) for o in s.oid] if "oid" in s.columns else [f"O{i:05d}" for i in range(n)]
    )
    approved = s.approved_days.to_numpy(dtype=float) * DAY
    purchase = approved - 3600.0
    handling = s.handling_days.to_numpy(dtype=float)
    transit = s.transit_days.to_numpy(dtype=float)
    pickup = approved + handling * DAY
    delivered = pickup + transit * DAY
    promise = purchase + s.promise_days.to_numpy(dtype=float) * DAY
    delivered_flag = s.get("delivered", pd.Series(np.ones(n, dtype=bool))).to_numpy(dtype=bool)

    sellers = sorted(set(s.seller))
    customers = [f"C{i:05d}" for i in range(n)]
    uniques = [str(u) for u in s.get("customer", pd.Series(customers))]
    lanes = [f"L{si}" for si in range(max(1, len(sellers)))]

    frames = {
        "customers": pd.DataFrame(
            {
                "customer_id": customers,
                "customer_unique_id": uniques,
                "customer_state": "SP",
                "customer_zip_prefix": 1000,
                "first_seen_ts": purchase,
            }
        ),
        "sellers": pd.DataFrame(
            {
                "seller_id": sellers,
                "seller_state": "SP",
                "seller_zip_prefix": 2000,
                "onboarded_ts": 0.0,
                "fulfilment_mode": "merchant",
            }
        ),
        "products": pd.DataFrame(
            {
                "product_id": ["P0"],
                "category": ["books"],
                "weight_g": [500.0],
                "length_cm": [10.0],
                "height_cm": [10.0],
                "width_cm": [10.0],
            }
        ),
        "carriers": pd.DataFrame(
            {
                "carrier_id": ["CAR0"],
                "carrier_name": ["only"],
                "service_level": ["standard"],
                "reliability_index": [0.9],
                "expedite_surcharge": [12.0],
                "daily_capacity": [100],
            }
        ),
        "shipping_lanes": pd.DataFrame(
            {
                "lane_id": lanes,
                "origin_state": "SP",
                "dest_state": "SP",
                "carrier_id": "CAR0",
                "distance_km": 100.0,
                "base_transit_days": 3.0,
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": customers,
                "order_status": np.where(delivered_flag, "delivered", "shipped"),
                "purchase_ts": purchase,
                "approved_ts": approved,
                "pickup_ts": np.where(delivered_flag, pickup, np.nan),
                "delivered_ts": np.where(delivered_flag, delivered, np.nan),
                "estimated_delivery_ts": promise,
            }
        ),
        "order_items": pd.DataFrame(
            {
                "order_id": order_ids,
                "item_seq": 1,
                "product_id": "P0",
                "seller_id": s.seller.astype(str).to_numpy(),
                "lane_id": [lanes[sellers.index(k) % len(lanes)] for k in s.seller],
                "shipping_limit_ts": approved + 2 * DAY,
                "price": 100.0,
                "freight_value": 15.0,
            }
        ),
        "order_payments": pd.DataFrame(
            {
                "order_id": order_ids,
                "payment_seq": 1,
                "payment_type": "credit_card",
                "installments": 1,
                "payment_value": 115.0,
            }
        ),
    }
    live = delivered_flag
    frames["order_reviews"] = pd.DataFrame(
        {
            "review_id": [f"R{i:05d}" for i in range(int(live.sum()))],
            "order_id": np.array(order_ids)[live],
            "review_score": 5,
            "review_creation_ts": delivered[live] + DAY,
            "review_answer_ts": delivered[live] + 2 * DAY,
        }
    )
    return frames


def features_for(orders_spec: pd.DataFrame, mode: str = "point_in_time") -> pd.DataFrame:
    """Load a spec into an in-memory database and return the feature table."""
    from deliveryrisk.data.db import Database
    from deliveryrisk.features.build import FeatureConfig, build_features

    db = Database("sqlite://:memory:")
    db.create_schema()
    db.load_frames(make_frames(orders_spec))
    return build_features(
        db, FeatureConfig(mode=mode, prior_weight=10.0, prior_rate=0.1)
    ).sort_values("order_id", ignore_index=True)


def simple_spec(
    approved_days, handling_days=1.0, transit_days=3.0, promise_days=6.0, sellers=None,
    delivered=None,
) -> pd.DataFrame:
    n = len(approved_days)
    def col(v):
        return np.asarray(v, dtype=float) if np.ndim(v) else np.full(n, float(v))
    return pd.DataFrame(
        {
            "approved_days": np.asarray(approved_days, dtype=float),
            "handling_days": col(handling_days),
            "transit_days": col(transit_days),
            "promise_days": col(promise_days),
            "seller": sellers if sellers is not None else ["S0"] * n,
            "oid": [f"O{i:05d}" for i in range(n)],
            "delivered": (np.ones(n, dtype=bool) if delivered is None
                          else np.asarray(delivered, dtype=bool)),
        }
    )
