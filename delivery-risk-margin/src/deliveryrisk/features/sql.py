"""Point-in-time feature SQL over the nine tables.

The guarantee
-------------
Every feature of order *i* is a function of facts whose timestamp is **strictly before**
``orders.approved_ts`` for order *i*, plus order *i*'s own pre-dispatch attributes. Nothing that
happened to order *i* after approval -- its pickup, its delivery, its review -- reaches its own
feature row, and neither does any other order's outcome that had not yet landed.

How it is enforced, rather than hoped for
-----------------------------------------
Aggregates are not ``GROUP BY``. Every history feature is a window aggregate over one event
stream per entity::

    kind 0  decision   at approved_ts        the row we are building features for
    kind 1  outcome    at delivered_ts       a completed delivery: was it late, how long
    kind 2  open       at approved/pickup    a parcel entering the seller's or carrier's queue
    kind 3  close      at pickup/delivered   and leaving it

    PARTITION BY entity_key ORDER BY ev_ts RANGE BETWEEN <lo> AND CURRENT ROW

with one trick doing the work::

    ev_ts = ts - EPS for decisions, ts for everything else

A ``RANGE`` frame includes every *peer* row -- rows whose ``ORDER BY`` value ties with the
current one. Shifting decisions back by an epsilon puts each decision strictly before every
real event at the same instant, so a decision at *t* sees events at *t* not at all: not another
order's delivery confirmed in the same second, and not its own queue-entry. Ties are where
point-in-time code usually breaks, and they are not rare here -- an ERP that stamps in whole
seconds produces thousands of them a day.

The same builder emits the leaky variants. ``mode="two_sided"`` changes the frames to
``UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`` and ``30 PRECEDING AND 30 FOLLOWING`` -- that is,
a seller late-rate computed over the whole table and a centred 30-day rolling window, which is
what ``df.groupby('seller_id').late.transform('mean')`` gives you. Same features, same names,
same model: only the frame moves.

``mode="two_sided_loo"`` is the version worth arguing about. It subtracts the order's own outcome
from every rate and mean it lands in -- the fix a careful person reaches for once someone points
out that ``transform('mean')`` includes the row itself. On this data it recovers **nothing**, and
``evaluation/leakage.py`` reports why: the dominant channel was never self-contamination. It is
the centred window on a *shared* entity. A carrier's +/-30-day late rate tells the model the
carrier was having a bad fortnight, measured partly from parcels that had not been delivered yet
-- including the disruption this very order is sitting inside. No leave-one-out touches that,
because the offending rows belong to other orders.

The recency columns (``*_days_since_outcome``, ``*_history_days``) are deliberately left
un-adjusted in that mode, for the same reason: nobody auditing for target leakage looks at a
column called "days since this seller last delivered". For a one-order customer that column is
the order's own delivery duration with a minus sign.

Portability
-----------
One dialect of SQL for SQLite and MySQL 8: numeric ``RANGE`` frames, named ``WINDOW`` clauses and
``CREATE TABLE AS SELECT`` behave the same on both. No engine-specific date functions -- which is
the other reason timestamps are stored as epoch doubles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

DAY = 86_400.0
#: Smaller than any real timestamp difference, large enough to survive double rounding at epoch
#: magnitudes (~1e8 seconds here, where a double resolves to ~1e-8).
EPS = 1e-4

MODES = ("point_in_time", "two_sided", "two_sided_loo")


@dataclass(frozen=True)
class EntitySpec:
    """One history stream: what it is keyed by, and what counts as being in its queue.

    Attributes
    ----------
    name:
        Feature prefix, e.g. ``seller`` gives ``seller_late_rate``.
    key:
        Column of ``order_facts`` holding the entity key.
    queue:
        ``(open_expr, close_expr)`` -- the interval during which an order occupies this entity's
        queue, or ``None`` if the entity has no queue. The seller's queue runs from approval to
        collection (orders it owes), the carrier's from collection to delivery (parcels it
        holds).
    outcomes:
        Whether completed deliveries of this entity feed its late-rate history. Always true
        today; kept explicit because a new entity type is the obvious place to get it wrong.
    """

    name: str
    key: str
    queue: tuple[str, str] | None = None
    outcomes: bool = True


#: An order not collected within 30 days is treated as gone from the seller's queue. Cancellations
#: carry no timestamp in this schema -- as in the source data -- so the alternative is a backlog
#: that only ever grows.
STALE_QUEUE_DAYS = 30.0

ENTITIES: tuple[EntitySpec, ...] = (
    EntitySpec("seller", "seller_id", queue=("approved_ts", f"COALESCE(pickup_ts, approved_ts + {STALE_QUEUE_DAYS * DAY})")),
    EntitySpec("lane", "lane_id", queue=("pickup_ts", "delivered_ts")),
    EntitySpec("carrier", "carrier_id", queue=("pickup_ts", "delivered_ts")),
    EntitySpec("category", "category"),
    EntitySpec("customer", "customer_unique_id"),
)

ENTITY_BY_NAME = {e.name: e for e in ENTITIES}


# --------------------------------------------------------------------------- order facts
ORDER_FACTS_SQL = """
DROP TABLE IF EXISTS order_primary;
CREATE TABLE order_primary AS
SELECT order_id, product_id, seller_id, lane_id, shipping_limit_ts
FROM (
    SELECT oi.order_id, oi.product_id, oi.seller_id, oi.lane_id, oi.shipping_limit_ts,
           ROW_NUMBER() OVER (
               PARTITION BY oi.order_id ORDER BY oi.price DESC, oi.item_seq ASC
           ) AS rn
    FROM order_items oi
) t
WHERE rn = 1;

DROP TABLE IF EXISTS order_rollup;
CREATE TABLE order_rollup AS
SELECT oi.order_id,
       COUNT(*)                          AS n_items,
       COUNT(DISTINCT oi.seller_id)      AS n_sellers,
       COUNT(DISTINCT oi.lane_id)        AS n_lanes,
       SUM(oi.price)                     AS total_price,
       SUM(oi.freight_value)             AS total_freight,
       MAX(oi.price)                     AS max_price,
       MIN(oi.shipping_limit_ts)         AS first_shipping_limit_ts,
       SUM(p.weight_g)                   AS total_weight_g,
       MAX(p.weight_g)                   AS max_weight_g,
       MAX(p.length_cm * p.height_cm * p.width_cm) AS max_volume_cm3
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
GROUP BY oi.order_id;

DROP TABLE IF EXISTS order_pay;
CREATE TABLE order_pay AS
SELECT order_id,
       COUNT(*)               AS n_payment_legs,
       SUM(payment_value)     AS payment_total,
       MAX(installments)      AS installments,
       MIN(payment_type)      AS payment_type
FROM order_payments
GROUP BY order_id;

DROP TABLE IF EXISTS order_facts;
CREATE TABLE order_facts AS
SELECT
    o.order_id,
    o.order_status,
    o.purchase_ts,
    o.approved_ts,
    o.pickup_ts,
    o.delivered_ts,
    o.estimated_delivery_ts,
    c.customer_unique_id,
    c.customer_state,
    c.customer_zip_prefix,
    op.seller_id,
    op.lane_id,
    op.product_id,
    op.shipping_limit_ts,
    pr.category,
    s.seller_state,
    s.fulfilment_mode,
    s.onboarded_ts,
    l.carrier_id,
    l.origin_state,
    l.dest_state,
    l.distance_km,
    l.base_transit_days,
    ca.service_level,
    ca.reliability_index,
    ca.expedite_surcharge,
    ca.daily_capacity,
    r.n_items, r.n_sellers, r.n_lanes, r.total_price, r.total_freight, r.max_price,
    r.total_weight_g, r.max_weight_g, r.max_volume_cm3, r.first_shipping_limit_ts,
    pay.n_payment_legs, pay.payment_total, pay.installments, pay.payment_type
FROM orders o
JOIN customers c       ON c.customer_id = o.customer_id
JOIN order_primary op  ON op.order_id = o.order_id
JOIN order_rollup r    ON r.order_id = o.order_id
JOIN products pr       ON pr.product_id = op.product_id
JOIN sellers s         ON s.seller_id = op.seller_id
JOIN shipping_lanes l  ON l.lane_id = op.lane_id
JOIN carriers ca       ON ca.carrier_id = l.carrier_id
LEFT JOIN order_pay pay ON pay.order_id = o.order_id
WHERE o.approved_ts IS NOT NULL;

-- Plain CREATE INDEX, not IF NOT EXISTS: MySQL has no such form, and the table was dropped two
-- statements ago, so the index cannot already exist.
CREATE INDEX ix_order_facts_approved ON order_facts (approved_ts);
"""


def _outcome_select(spec: EntitySpec) -> str:
    """Completed deliveries of this entity, as events at the moment they completed."""
    return f"""
SELECT '{spec.name}' AS entity_type,
       {spec.key} AS entity_key,
       delivered_ts AS ts,
       1 AS ev_kind,
       delivered_ts AS ev_ts,
       order_id,
       CASE WHEN delivered_ts > estimated_delivery_ts THEN 1 ELSE 0 END AS is_late,
       (delivered_ts - estimated_delivery_ts) / {DAY} AS days_over,
       (pickup_ts - approved_ts) / {DAY} AS handling_days,
       (delivered_ts - pickup_ts) / {DAY} AS transit_days,
       0 AS delta_open,
       0.0 AS open_ts_signed,
       0 AS own_has_outcome, 0 AS own_is_late, 0.0 AS own_handling_days,
       0.0 AS own_transit_days, 0.0 AS own_days_over, 0 AS own_in_recent
FROM order_facts
WHERE delivered_ts IS NOT NULL AND pickup_ts IS NOT NULL"""


def _decision_select(spec: EntitySpec) -> str:
    return f"""
SELECT '{spec.name}' AS entity_type,
       {spec.key} AS entity_key,
       approved_ts AS ts,
       0 AS ev_kind,
       approved_ts - {EPS} AS ev_ts,
       order_id,
       NULL AS is_late, NULL AS days_over, NULL AS handling_days, NULL AS transit_days,
       0 AS delta_open,
       0.0 AS open_ts_signed,
       -- The order's own outcome, carried on its decision row. Read by nothing except the
       -- leave-one-out adjustment in `mode="two_sided_loo"`; the point-in-time builder never
       -- touches these columns, and `_assert_no_outcome_features` keeps them out of the model.
       CASE WHEN delivered_ts IS NOT NULL AND pickup_ts IS NOT NULL THEN 1 ELSE 0 END
           AS own_has_outcome,
       CASE WHEN delivered_ts > estimated_delivery_ts THEN 1 ELSE 0 END AS own_is_late,
       COALESCE((pickup_ts - approved_ts) / {DAY}, 0.0)   AS own_handling_days,
       COALESCE((delivered_ts - pickup_ts) / {DAY}, 0.0)  AS own_transit_days,
       COALESCE((delivered_ts - estimated_delivery_ts) / {DAY}, 0.0) AS own_days_over,
       CASE WHEN delivered_ts IS NOT NULL AND pickup_ts IS NOT NULL
                 AND ABS(delivered_ts - approved_ts) <= :recent_span THEN 1 ELSE 0 END
           AS own_in_recent
FROM order_facts"""


def _queue_selects(spec: EntitySpec) -> list[str]:
    if spec.queue is None:
        return []
    open_expr, close_expr = spec.queue
    return [
        f"""
SELECT '{spec.name}' AS entity_type, {spec.key} AS entity_key, {open_expr} AS ts, 2 AS ev_kind,
       {open_expr} AS ev_ts, order_id,
       NULL AS is_late, NULL AS days_over, NULL AS handling_days, NULL AS transit_days,
       1 AS delta_open,
       {open_expr} AS open_ts_signed,
       0 AS own_has_outcome, 0 AS own_is_late, 0.0 AS own_handling_days,
       0.0 AS own_transit_days, 0.0 AS own_days_over, 0 AS own_in_recent
FROM order_facts WHERE {open_expr} IS NOT NULL""",
        f"""
SELECT '{spec.name}' AS entity_type, {spec.key} AS entity_key, {close_expr} AS ts, 3 AS ev_kind,
       {close_expr} AS ev_ts, order_id,
       NULL AS is_late, NULL AS days_over, NULL AS handling_days, NULL AS transit_days,
       -1 AS delta_open,
       -{open_expr} AS open_ts_signed,
       0 AS own_has_outcome, 0 AS own_is_late, 0.0 AS own_handling_days,
       0.0 AS own_transit_days, 0.0 AS own_days_over, 0 AS own_in_recent
FROM order_facts WHERE {close_expr} IS NOT NULL""",
    ]


def build_events_sql(
    entities: tuple[EntitySpec, ...] = ENTITIES, *, trailing_days: float = 30.0
) -> str:
    """The event table: one row per (entity, order, lifecycle moment)."""
    parts: list[str] = []
    for spec in entities:
        parts.append(_decision_select(spec))
        if spec.outcomes:
            parts.append(_outcome_select(spec))
        parts.extend(_queue_selects(spec))
    union = "\nUNION ALL\n".join(parts).replace(":recent_span", str(trailing_days * DAY))
    return (
        "DROP TABLE IF EXISTS pit_events;\n"
        f"CREATE TABLE pit_events AS{union};\n"
        "CREATE INDEX ix_pit_events_key "  # the table is dropped above; see order_facts
        "ON pit_events (entity_type, entity_key, ev_ts);\n"
    )


# --------------------------------------------------------------------------- window frames
def is_leaky(mode: str) -> bool:
    return mode != "point_in_time"


def frames_for(mode: str, trailing_days: float) -> tuple[str, str]:
    """The two window frames, for the honest builder and for the leaky one.

    ``point_in_time`` looks backwards only; ``two_sided`` is the unshifted rolling aggregate --
    the mistake this project exists to measure. Both are the *same* feature definitions.
    """
    if mode not in MODES:
        raise ValueError(f"unknown feature mode {mode!r}; expected one of {MODES}")
    span = trailing_days * DAY
    if mode == "point_in_time":
        return (
            "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            f"RANGE BETWEEN {span} PRECEDING AND CURRENT ROW",
        )
    # Both leaky modes use the same frames; they differ only in the leave-one-out adjustment.
    return (
        "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
        f"RANGE BETWEEN {span} PRECEDING AND {span} FOLLOWING",
    )


def _entity_cte(spec: EntitySpec, mode: str, trailing_days: float, prior_weight: float,
                prior_rate: float) -> str:
    frame_all, frame_trail = frames_for(mode, trailing_days)
    n = spec.name
    # Leave-one-out: strip the order's own outcome back out of every aggregate it landed in.
    # Applies to the counts, rates and means -- the recency and queue columns are untouched and
    # `evaluation/leakage.py` says so, because a leave-one-out fix that silently half-applied
    # would make the comparison meaningless.
    loo = mode == "two_sided_loo"
    a_n = " - own_has_outcome" if loo else ""
    a_late = " - own_is_late" if loo else ""
    a_h = " - own_handling_days" if loo else ""
    a_t = " - own_transit_days" if loo else ""
    a_do = " - own_days_over" if loo else ""
    a_nr = " - own_in_recent" if loo else ""
    a_lr = " - own_is_late * own_in_recent" if loo else ""
    a_tr = " - own_transit_days * own_in_recent" if loo else ""
    queue_cols = ""
    if spec.queue is not None:
        # `open_ts_signed` adds an entry's start time when it joins the queue and subtracts the
        # same start time when it leaves, so the running sum is the total start time of whatever
        # is *still* in the queue. Divided by the depth and subtracted from now, that is the mean
        # age of the backlog -- the difference between a seller with five fresh orders and a
        # seller with five orders they have been sitting on for a week.
        queue_cols = (
            f",\n           SUM(delta_open) OVER wall AS {n}_queue_depth"
            f",\n           SUM(open_ts_signed) OVER wall AS {n}_queue_open_ts_sum"
        )
    return f"""
{n}_w AS (
    SELECT order_id,
           ev_kind,
           ts,
           own_has_outcome, own_is_late, own_handling_days, own_transit_days, own_days_over,
           own_in_recent,
           SUM(CASE WHEN ev_kind = 1 THEN 1 ELSE 0 END)            OVER wall AS {n}_n_prior,
           SUM(CASE WHEN ev_kind = 1 THEN is_late ELSE 0 END)      OVER wall AS {n}_n_late,
           SUM(CASE WHEN ev_kind = 1 THEN handling_days ELSE 0 END) OVER wall AS {n}_sum_handling,
           SUM(CASE WHEN ev_kind = 1 THEN transit_days ELSE 0 END)  OVER wall AS {n}_sum_transit,
           SUM(CASE WHEN ev_kind = 1 THEN days_over ELSE 0 END)     OVER wall AS {n}_sum_days_over,
           MAX(CASE WHEN ev_kind = 1 THEN ts END)                  OVER wall AS {n}_last_outcome_ts,
           MIN(ts)                                                 OVER wall AS {n}_first_ts,
           SUM(CASE WHEN ev_kind = 1 THEN 1 ELSE 0 END)            OVER wtrail AS {n}_n_prior_recent,
           SUM(CASE WHEN ev_kind = 1 THEN is_late ELSE 0 END)      OVER wtrail AS {n}_n_late_recent,
           SUM(CASE WHEN ev_kind = 1 THEN transit_days ELSE 0 END) OVER wtrail AS {n}_sum_transit_recent{queue_cols}
    FROM pit_events
    WHERE entity_type = '{n}'
    WINDOW wall   AS (PARTITION BY entity_key ORDER BY ev_ts {frame_all}),
           wtrail AS (PARTITION BY entity_key ORDER BY ev_ts {frame_trail})
),
{n}_f AS (
    SELECT order_id,
           ({n}_n_prior{a_n}) AS {n}_n_prior,
           ({n}_n_prior_recent{a_nr}) AS {n}_n_prior_recent,
           ({n}_n_late{a_late} + {prior_weight} * {prior_rate})
               / ({n}_n_prior{a_n} + {prior_weight}) AS {n}_late_rate,
           ({n}_n_late_recent{a_lr} + {prior_weight} * {prior_rate})
               / ({n}_n_prior_recent{a_nr} + {prior_weight}) AS {n}_late_rate_recent,
           CASE WHEN ({n}_n_prior{a_n}) > 0
                THEN ({n}_sum_handling{a_h}) / ({n}_n_prior{a_n}) END AS {n}_mean_handling,
           CASE WHEN ({n}_n_prior{a_n}) > 0
                THEN ({n}_sum_transit{a_t}) / ({n}_n_prior{a_n}) END AS {n}_mean_transit,
           CASE WHEN ({n}_n_prior{a_n}) > 0
                THEN ({n}_sum_days_over{a_do}) / ({n}_n_prior{a_n}) END AS {n}_mean_days_over,
           CASE WHEN ({n}_n_prior_recent{a_nr}) > 0
                THEN ({n}_sum_transit_recent{a_tr}) / ({n}_n_prior_recent{a_nr})
                END AS {n}_mean_transit_recent,
           (ts - {n}_last_outcome_ts) / {DAY} AS {n}_days_since_outcome,
           (ts - {n}_first_ts) / {DAY}        AS {n}_history_days{
        (
            ',' + chr(10) + ' ' * 11 + f'{n}_queue_depth,' + chr(10) + ' ' * 11
            + f'CASE WHEN {n}_queue_depth > 0 THEN '
              f'(ts - {n}_queue_open_ts_sum / {n}_queue_depth) / {DAY} END '
              f'AS {n}_queue_mean_age_days'
        ) if spec.queue is not None else ''}
    FROM {n}_w
    WHERE ev_kind = 0
)"""


def entity_feature_columns(spec: EntitySpec) -> list[str]:
    n = spec.name
    cols = [
        f"{n}_n_prior", f"{n}_n_prior_recent", f"{n}_late_rate", f"{n}_late_rate_recent",
        f"{n}_mean_handling", f"{n}_mean_transit", f"{n}_mean_days_over",
        f"{n}_mean_transit_recent", f"{n}_days_since_outcome", f"{n}_history_days",
    ]
    if spec.queue is not None:
        cols += [f"{n}_queue_depth", f"{n}_queue_mean_age_days"]
    return cols


#: Features known at approval from the order itself. Nothing here reads a post-approval column.
STATIC_FEATURE_SQL = f"""
    (f.estimated_delivery_ts - f.purchase_ts) / {DAY}      AS promise_days,
    (f.estimated_delivery_ts - f.approved_ts) / {DAY}      AS slack_days,
    (f.approved_ts - f.purchase_ts) / 3600.0               AS approval_lag_hours,
    (f.first_shipping_limit_ts - f.approved_ts) / {DAY}    AS handover_deadline_days,
    (f.approved_ts - f.onboarded_ts) / {DAY}               AS seller_tenure_days,
    f.n_items, f.n_sellers, f.n_lanes,
    f.total_price, f.total_freight, f.max_price,
    f.total_freight / (f.total_price + 1.0)                AS freight_ratio,
    f.total_weight_g, f.max_weight_g, f.max_volume_cm3,
    f.n_payment_legs, f.payment_total, f.installments,
    f.distance_km, f.base_transit_days, f.reliability_index, f.daily_capacity,
    (f.estimated_delivery_ts - f.approved_ts) / ({DAY} * f.base_transit_days) AS slack_transit_ratio,
    CASE WHEN f.origin_state = f.dest_state THEN 1 ELSE 0 END AS same_state,
    CAST(f.approved_ts / {DAY} AS INTEGER) % 7             AS approved_dow,
    CAST(f.approved_ts % {DAY} / 3600.0 AS INTEGER)        AS approved_hour,
    CASE WHEN CAST(f.approved_ts / {DAY} AS INTEGER) % 7 IN (5, 6) THEN 1 ELSE 0 END
        AS approved_weekend,
    CAST(CAST(f.purchase_ts / {DAY} AS INTEGER) % 365 / 7 AS INTEGER) AS purchase_week_of_year,
    f.payment_type, f.service_level, f.fulfilment_mode, f.category,
    f.origin_state, f.dest_state, f.carrier_id
"""

STATIC_FEATURE_COLUMNS = [
    "promise_days", "slack_days", "approval_lag_hours", "handover_deadline_days",
    "seller_tenure_days", "n_items", "n_sellers", "n_lanes", "total_price", "total_freight",
    "max_price", "freight_ratio", "total_weight_g", "max_weight_g", "max_volume_cm3",
    "n_payment_legs", "payment_total", "installments", "distance_km", "base_transit_days",
    "reliability_index", "daily_capacity", "slack_transit_ratio", "same_state", "approved_dow",
    "approved_hour", "approved_weekend", "purchase_week_of_year",
]

CATEGORICAL_COLUMNS = [
    "payment_type", "service_level", "fulfilment_mode", "category",
    "origin_state", "dest_state", "carrier_id",
]

#: Post-decision columns carried for evaluation and for the cost model. They are targets and
#: economics inputs, never features; ``feature_columns()`` excludes them and a test pins it.
OUTCOME_COLUMNS = [
    "order_status", "approved_ts", "purchase_ts", "pickup_ts", "delivered_ts",
    "estimated_delivery_ts", "observed_handling_days", "observed_transit_days", "days_over",
    "y_late", "y_handling_fixable", "y_transit_fixable", "seller_id", "lane_id", "carrier_id_key",
    "customer_unique_id",
]


def feature_columns(entities: tuple[EntitySpec, ...] = ENTITIES) -> list[str]:
    """Every column a model is allowed to see."""
    cols = list(STATIC_FEATURE_COLUMNS)
    for spec in entities:
        cols.extend(entity_feature_columns(spec))
    return cols + list(CATEGORICAL_COLUMNS)


def build_entity_sql(
    spec: EntitySpec,
    *,
    mode: str = "point_in_time",
    trailing_days: float = 30.0,
    prior_weight: float = 40.0,
    prior_rate: float = 0.08,
) -> str:
    """Materialise one entity's history features into ``feat_<entity>``, indexed on order_id.

    Kept as its own table rather than a CTE in the final query for a boring reason that costs an
    hour to rediscover: both engines happily plan a join against an un-indexed inline subquery as
    a nested scan, which turns five joins over 100 K orders into a run you give up waiting for.
    """
    cte = _entity_cte(spec, mode, trailing_days, prior_weight, prior_rate)
    n = spec.name
    return (
        f"DROP TABLE IF EXISTS feat_{n};\n"
        f"CREATE TABLE feat_{n} AS WITH{cte} SELECT * FROM {n}_f;\n"
        f"CREATE INDEX ix_feat_{n}_order ON feat_{n} (order_id);\n"
    )


def build_features_sql(
    *,
    mode: str = "point_in_time",
    trailing_days: float = 30.0,
    prior_weight: float = 40.0,
    prior_rate: float = 0.08,
    handling_floor_days: float = 0.35,
    transit_floor_share: float = 0.34,
    entities: tuple[EntitySpec, ...] = ENTITIES,
) -> str:
    """The one query that produces the modelling table.

    Assumes ``feat_<entity>`` tables already exist (see :func:`build_entity_sql`).

    ``prior_weight`` / ``prior_rate`` are the empirical-Bayes shrinkage on every entity late
    rate. They matter more than they look: a seller with two prior deliveries and one late one
    has a raw late rate of 0.5, and a tree will happily carve that out. ``prior_rate`` must come
    from the training window only -- ``pipeline.py`` computes it there and passes it in -- or the
    shrinkage target itself is a summary of the future.
    """
    joins = "\n".join(
        f"LEFT JOIN feat_{spec.name} ON feat_{spec.name}.order_id = f.order_id"
        for spec in entities
    )
    entity_cols = ",\n    ".join(
        col for spec in entities for col in entity_feature_columns(spec)
    )
    return f"""
SELECT
    f.order_id,
    f.order_status,
    f.purchase_ts,
    f.approved_ts,
    f.pickup_ts,
    f.delivered_ts,
    f.estimated_delivery_ts,
    f.seller_id,
    f.lane_id,
    f.carrier_id AS carrier_id_key,
    f.customer_unique_id,
    f.expedite_surcharge,
    (f.pickup_ts - f.approved_ts) / {DAY}    AS observed_handling_days,
    (f.delivered_ts - f.pickup_ts) / {DAY}   AS observed_transit_days,
    (f.delivered_ts - f.estimated_delivery_ts) / {DAY} AS days_over,
    CASE WHEN f.delivered_ts IS NULL THEN NULL
         WHEN f.delivered_ts > f.estimated_delivery_ts THEN 1 ELSE 0 END AS y_late,
    CASE WHEN f.delivered_ts IS NULL OR f.pickup_ts IS NULL THEN NULL
         WHEN f.delivered_ts > f.estimated_delivery_ts
              AND {handling_floor_days} + (f.delivered_ts - f.pickup_ts) / {DAY}
                  <= (f.estimated_delivery_ts - f.approved_ts) / {DAY}
         THEN 1 ELSE 0 END AS y_handling_fixable,
    CASE WHEN f.delivered_ts IS NULL OR f.pickup_ts IS NULL THEN NULL
         WHEN f.delivered_ts > f.estimated_delivery_ts
              AND (f.pickup_ts - f.approved_ts) / {DAY}
                  + {transit_floor_share} * f.base_transit_days
                  <= (f.estimated_delivery_ts - f.approved_ts) / {DAY}
         THEN 1 ELSE 0 END AS y_transit_fixable,
{STATIC_FEATURE_SQL},
    {entity_cols}
FROM order_facts f
{joins}
ORDER BY f.approved_ts, f.order_id
"""
