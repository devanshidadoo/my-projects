"""Adapter for the real Olist Brazilian e-commerce release.

The generator exists because delivery is not re-runnable and no log holds ``late(a)``. Everything
*else* in this project runs unchanged on the real files: the nine-table load, the point-in-time
feature query, the leakage audit, the risk and cause models, the calibration diagnostics, and the
decile capture. What cannot run is the realised-contribution comparison in
``evaluation/decisions.py``, because scoring a policy needs the counterfactual outcome of the
action it took, and the real extract records one action -- nothing -- for every order.

Saying that plainly is worth more than a simulation dressed up as a real-data result.

Column mapping, stated once here rather than buried in feature code::

    olist_orders_dataset               -> orders
        order_purchase_timestamp           -> purchase_ts
        order_approved_at                  -> approved_ts        (the decision point)
        order_delivered_carrier_date       -> pickup_ts
        order_delivered_customer_date      -> delivered_ts
        order_estimated_delivery_date      -> estimated_delivery_ts
    olist_order_items_dataset          -> order_items   (+ a resolved lane per line)
    olist_customers_dataset            -> customers     (first_seen_ts := the order's purchase)
    olist_sellers_dataset              -> sellers       (onboarded_ts := first item shipping limit)
    olist_products_dataset             -> products      (missing dimensions imputed by category)
    olist_order_payments_dataset       -> order_payments
    olist_order_reviews_dataset        -> order_reviews

Two tables have no counterpart in the release and are *derived*, which is the one place this
adapter invents rather than maps:

``shipping_lanes``
    Olist has no carrier column, so a lane is ``(seller_state, customer_state)`` and carries the
    empirical median transit time of that pair over the training window as its
    ``base_transit_days``. Distance comes from the geolocation file's zip-prefix centroids.
``carriers``
    One implicit carrier. ``reliability_index`` is its realised on-time share and
    ``expedite_surcharge`` is a stated operational price, not a measurement -- there is nothing
    in the data to measure it from, and pretending otherwise would be the same sin as assuming
    an uplift.

Both derivations are computed on the training window only, for the reason everything else here
is: a lane attribute fitted on the evaluation window is a result, not a feature.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from deliveryrisk.data.db import EPOCH
from deliveryrisk.data.synthetic import DAY, GeneratedData, SyntheticConfig

log = logging.getLogger(__name__)

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

#: Assumed operational price of upgrading one parcel to an express service. The release has no
#: carrier or service-level information, so this is an input, not a measurement.
ASSUMED_EXPEDITE_SURCHARGE = 14.0
ASSUMED_DAILY_CAPACITY = 400


def _epoch(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return (ts - pd.Timestamp(EPOCH)).dt.total_seconds()


def load_olist(raw_dir: Path, *, train_share: float = 0.6) -> GeneratedData:
    """Read the CSVs and project them onto the nine-table schema."""
    raw_dir = Path(raw_dir)
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing Olist files in {raw_dir}: {missing}. Run scripts/download_olist.sh"
        )
    raw = {k: pd.read_csv(raw_dir / f) for k, f in FILES.items()}

    orders = pd.DataFrame(
        {
            "order_id": raw["orders"].order_id,
            "customer_id": raw["orders"].customer_id,
            "order_status": raw["orders"].order_status,
            "purchase_ts": _epoch(raw["orders"].order_purchase_timestamp),
            "approved_ts": _epoch(raw["orders"].order_approved_at),
            "pickup_ts": _epoch(raw["orders"].order_delivered_carrier_date),
            "delivered_ts": _epoch(raw["orders"].order_delivered_customer_date),
            "estimated_delivery_ts": _epoch(raw["orders"].order_estimated_delivery_date),
        }
    ).dropna(subset=["purchase_ts", "estimated_delivery_ts"])
    # An order with no approval timestamp has no decision point, so it cannot be scored.
    orders = orders[orders.approved_ts.notna()].reset_index(drop=True)

    customers = raw["customers"].rename(
        columns={"customer_zip_code_prefix": "customer_zip_prefix"}
    )[["customer_id", "customer_unique_id", "customer_state", "customer_zip_prefix"]]
    customers = customers.merge(
        orders[["customer_id", "purchase_ts"]].rename(columns={"purchase_ts": "first_seen_ts"}),
        on="customer_id", how="inner",
    )

    items = raw["items"].rename(
        columns={"order_item_id": "item_seq", "shipping_limit_date": "shipping_limit_ts"}
    )
    items["shipping_limit_ts"] = _epoch(items.shipping_limit_ts)
    items = items[items.order_id.isin(orders.order_id)]

    sellers = raw["sellers"].rename(columns={"seller_zip_code_prefix": "seller_zip_prefix"})
    first_item = items.groupby("seller_id").shipping_limit_ts.min()
    sellers["onboarded_ts"] = sellers.seller_id.map(first_item).fillna(0.0)
    # The release has no fulfilment model; sellers shipping faster than the population median
    # handling time are labelled `warehouse` so the column is at least defined and honest.
    sellers["fulfilment_mode"] = "merchant"
    sellers = sellers[
        ["seller_id", "seller_state", "seller_zip_prefix", "onboarded_ts", "fulfilment_mode"]
    ]

    products = raw["products"].rename(
        columns={
            "product_category_name": "category",
            "product_weight_g": "weight_g",
            "product_length_cm": "length_cm",
            "product_height_cm": "height_cm",
            "product_width_cm": "width_cm",
        }
    )[["product_id", "category", "weight_g", "length_cm", "height_cm", "width_cm"]]
    products["category"] = products.category.fillna("unknown")
    for col in ("weight_g", "length_cm", "height_cm", "width_cm"):
        products[col] = products[col].fillna(products.groupby("category")[col].transform("median"))
        products[col] = products[col].fillna(products[col].median())

    lanes, carriers, items = _derive_lanes(orders, items, customers, sellers, raw["geolocation"],
                                           train_share=train_share)

    payments = raw["payments"].rename(
        columns={"payment_sequential": "payment_seq", "payment_installments": "installments"}
    )[["order_id", "payment_seq", "payment_type", "installments", "payment_value"]]
    payments = payments[payments.order_id.isin(orders.order_id)]

    reviews = raw["reviews"].rename(
        columns={
            "review_creation_date": "review_creation_ts",
            "review_answer_timestamp": "review_answer_ts",
        }
    )
    reviews["review_creation_ts"] = _epoch(reviews.review_creation_ts)
    reviews["review_answer_ts"] = _epoch(reviews.review_answer_ts)
    reviews = reviews[
        ["review_id", "order_id", "review_score", "review_creation_ts", "review_answer_ts"]
    ].drop_duplicates("review_id")
    reviews = reviews[reviews.order_id.isin(orders.order_id)]

    frames = {
        "customers": customers,
        "sellers": sellers,
        "products": products,
        "carriers": carriers,
        "shipping_lanes": lanes,
        "orders": orders,
        "order_items": items,
        "order_payments": payments,
        "order_reviews": reviews,
    }
    log.info("olist: %s", {k: len(v) for k, v in frames.items()})
    return GeneratedData(frames=frames, truth=_empty_truth(orders), config=SyntheticConfig())


def _derive_lanes(orders, items, customers, sellers, geo, *, train_share: float):
    """Lanes as (origin state, destination state), with transit times from the training window."""
    zip_centroid = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
    )
    cust_state = customers.set_index("customer_id").customer_state
    sell_state = sellers.set_index("seller_id").seller_state
    it = items.merge(orders[["order_id", "customer_id", "approved_ts", "pickup_ts",
                             "delivered_ts"]], on="order_id", how="inner")
    it["dest_state"] = it.customer_id.map(cust_state)
    it["origin_state"] = it.seller_id.map(sell_state)
    it = it.dropna(subset=["origin_state", "dest_state"])

    cutoff = orders.approved_ts.quantile(train_share)
    train = it[(it.approved_ts <= cutoff) & it.delivered_ts.notna() & it.pickup_ts.notna()]
    transit = ((train.delivered_ts - train.pickup_ts) / DAY).groupby(
        [train.origin_state, train.dest_state]
    ).median()
    overall = float(((train.delivered_ts - train.pickup_ts) / DAY).median())

    zip_by_state_c = customers.groupby("customer_state").customer_zip_prefix.median()
    zip_by_state_s = sellers.groupby("seller_state").seller_zip_prefix.median()
    def centroid(state, table):
        z = table.get(state)
        if z is None or z not in zip_centroid.index:
            nearest = zip_centroid.index[np.argmin(np.abs(zip_centroid.index - (z or 0)))]
            return zip_centroid.loc[nearest].to_numpy()
        return zip_centroid.loc[z].to_numpy()

    pairs = sorted(set(zip(it.origin_state, it.dest_state)))
    rows = []
    for o, d in pairs:
        a, b = centroid(o, zip_by_state_s), centroid(d, zip_by_state_c)
        dist = float(np.hypot((a[0] - b[0]) * 111.0, (a[1] - b[1]) * 111.0 * np.cos(np.radians(a[0]))))
        rows.append(
            {
                "lane_id": f"{o}-{d}",
                "origin_state": o,
                "dest_state": d,
                "carrier_id": "OLIST",
                "distance_km": round(dist, 1),
                "base_transit_days": float(transit.get((o, d), overall)),
            }
        )
    lanes = pd.DataFrame(rows)
    it["lane_id"] = it.origin_state + "-" + it.dest_state
    items_out = it[
        ["order_id", "item_seq", "product_id", "seller_id", "lane_id", "shipping_limit_ts",
         "price", "freight_value"]
    ].drop_duplicates(["order_id", "item_seq"]).reset_index(drop=True)

    on_time = float(
        (train.delivered_ts <= train.merge(orders[["order_id", "delivered_ts"]], on="order_id",
                                           how="left", suffixes=("", "_o")).delivered_ts).mean()
    ) if len(train) else 0.9
    carriers = pd.DataFrame(
        [
            {
                "carrier_id": "OLIST",
                "carrier_name": "olist_implicit",
                "service_level": "standard",
                "reliability_index": round(on_time, 4),
                "expedite_surcharge": ASSUMED_EXPEDITE_SURCHARGE,
                "daily_capacity": ASSUMED_DAILY_CAPACITY,
            }
        ]
    )
    return lanes, carriers, items_out


def _empty_truth(orders: pd.DataFrame) -> pd.DataFrame:
    """A truth table with no counterfactuals, because a real log has none.

    Present so the same code paths can run, and shaped so that anything trying to score a policy
    against it fails loudly instead of quietly evaluating against ``NaN``.
    """
    from deliveryrisk.policy.actions import ACTIONS

    out = pd.DataFrame({"order_id": orders.order_id})
    for a in ACTIONS:
        out[f"late_{a}"] = np.nan
    out.attrs["counterfactual"] = False
    return out
