"""The nine operational tables, and the DDL for them.

This is a normalised order-management schema of the shape a marketplace actually runs on --
nine tables, foreign keys, one row per business fact -- rather than the single wide CSV that
most delivery-risk write-ups start from. That distinction is the point of the project: a wide
CSV has already made every join decision for you, and the join decisions are where the
point-in-time bugs live.

    customers        one row per shipping identity
    sellers          one row per merchant
    products         catalogue: category and parcel dimensions
    orders           the order header and its four lifecycle timestamps
    order_items      one row per (order, line): product, seller, price, freight
    order_payments   one row per payment leg
    order_reviews    post-delivery satisfaction, used for the cost of lateness, never as a feature
    shipping_lanes   origin state x destination state x carrier: the transit unit
    carriers         service levels and the price of expediting on each

Timestamps
----------
Every ``*_ts`` column is **epoch seconds stored as a double**, in both dialects. The feature SQL
does strict ordering and interval arithmetic on these columns inside window frames; doubles
behave identically on SQLite and MySQL, whereas ``DATETIME`` comparison, rounding and
``NULL``-in-frame semantics do not. Portability of the *feature* definition is worth more here
than readability of a column dump, and :func:`deliveryrisk.data.db.to_datetime` converts on the
way out.

The lifecycle, and where the decision sits::

    purchase_ts ---- approved_ts ---- pickup_ts ---- delivered_ts
                          ^
                          |  the model scores here: payment cleared, parcel not yet dispatched.
                          |  Everything after this point is unobservable and unusable.
                     estimated_delivery_ts (the promise, set at purchase)

``pickup_ts`` and ``delivered_ts`` are the outcome. They are in the schema because the *history*
of other orders' outcomes is the strongest feature there is; they are unusable for the order
being scored. Keeping both facts true at once is what :mod:`deliveryrisk.features` is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Logical column types, mapped per dialect below.
TYPES = {
    "sqlite": {
        "id": "TEXT",
        "text": "TEXT",
        "int": "INTEGER",
        "float": "REAL",
        "ts": "REAL",
        "bool": "INTEGER",
    },
    "mysql": {
        "id": "VARCHAR(40)",
        "text": "VARCHAR(64)",
        "int": "INT",
        "float": "DOUBLE",
        "ts": "DOUBLE",
        "bool": "TINYINT(1)",
    },
}

DIALECTS = tuple(TYPES)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    null: bool = True
    comment: str = ""

    def ddl(self, dialect: str) -> str:
        sql = f"  {self.name} {TYPES[dialect][self.type]}"
        if not self.null:
            sql += " NOT NULL"
        if self.comment and dialect == "mysql":
            sql += f" COMMENT '{self.comment}'"
        return sql


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[tuple[str, str, str], ...] = ()  # (column, ref_table, ref_column)
    indexes: tuple[tuple[str, ...], ...] = ()
    doc: str = ""

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def ddl(self, dialect: str, *, foreign_keys: bool = True) -> str:
        if dialect not in TYPES:
            raise ValueError(f"unknown dialect {dialect!r}; expected one of {sorted(TYPES)}")
        lines = [c.ddl(dialect) for c in self.columns]
        lines.append(f"  PRIMARY KEY ({', '.join(self.primary_key)})")
        if foreign_keys:
            lines += [
                f"  FOREIGN KEY ({col}) REFERENCES {ref_t}({ref_c})"
                for col, ref_t, ref_c in self.foreign_keys
            ]
        body = ",\n".join(lines)
        suffix = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if dialect == "mysql" else ""
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n){suffix};"

    def index_ddl(self, dialect: str) -> list[str]:
        out = []
        for cols in self.indexes:
            idx = f"ix_{self.name}_{'_'.join(cols)}"
            # MySQL has no CREATE INDEX IF NOT EXISTS; the loader drops the schema first.
            guard = "" if dialect == "mysql" else "IF NOT EXISTS "
            out.append(f"CREATE INDEX {guard}{idx} ON {self.name} ({', '.join(cols)});")
        return out


def _c(name: str, type_: str, null: bool = True, comment: str = "") -> Column:
    return Column(name, type_, null, comment)


CUSTOMERS = Table(
    name="customers",
    doc="One row per shipping identity. `customer_unique_id` links repeat buyers.",
    columns=(
        _c("customer_id", "id", False),
        _c("customer_unique_id", "id", False, "stable across orders"),
        _c("customer_state", "text", False),
        _c("customer_zip_prefix", "int", False),
        _c("first_seen_ts", "ts", False, "acquisition time; safe as a feature"),
    ),
    primary_key=("customer_id",),
    indexes=(("customer_unique_id",),),
)

SELLERS = Table(
    name="sellers",
    doc="Merchants. Handling time is a seller property and the largest controllable delay term.",
    columns=(
        _c("seller_id", "id", False),
        _c("seller_state", "text", False),
        _c("seller_zip_prefix", "int", False),
        _c("onboarded_ts", "ts", False),
        _c("fulfilment_mode", "text", False, "merchant | warehouse"),
    ),
    primary_key=("seller_id",),
    indexes=(("seller_state",),),
)

PRODUCTS = Table(
    name="products",
    doc="Catalogue. Dimensions drive freight class, which drives lane choice.",
    columns=(
        _c("product_id", "id", False),
        _c("category", "text", False),
        _c("weight_g", "float", False),
        _c("length_cm", "float", False),
        _c("height_cm", "float", False),
        _c("width_cm", "float", False),
    ),
    primary_key=("product_id",),
    indexes=(("category",),),
)

ORDERS = Table(
    name="orders",
    doc=(
        "The order header. `approved_ts` is the decision point; `pickup_ts` and `delivered_ts` "
        "are outcomes of this order and must never reach its own feature row."
    ),
    columns=(
        _c("order_id", "id", False),
        _c("customer_id", "id", False),
        _c("order_status", "text", False, "delivered | shipped | cancelled"),
        _c("purchase_ts", "ts", False),
        _c("approved_ts", "ts", True, "DECISION POINT"),
        _c("pickup_ts", "ts", True, "carrier collection; outcome"),
        _c("delivered_ts", "ts", True, "outcome"),
        _c("estimated_delivery_ts", "ts", False, "the promise made at purchase"),
    ),
    primary_key=("order_id",),
    foreign_keys=(("customer_id", "customers", "customer_id"),),
    indexes=(("approved_ts",), ("customer_id",), ("delivered_ts",)),
)

ORDER_ITEMS = Table(
    name="order_items",
    doc="One row per line. An order with two sellers has two handling clocks, not one.",
    columns=(
        _c("order_id", "id", False),
        _c("item_seq", "int", False),
        _c("product_id", "id", False),
        _c("seller_id", "id", False),
        _c("lane_id", "id", False, "resolved shipping lane for this line"),
        _c("shipping_limit_ts", "ts", False, "contractual handover deadline for the seller"),
        _c("price", "float", False),
        _c("freight_value", "float", False),
    ),
    primary_key=("order_id", "item_seq"),
    foreign_keys=(
        ("order_id", "orders", "order_id"),
        ("product_id", "products", "product_id"),
        ("seller_id", "sellers", "seller_id"),
        ("lane_id", "shipping_lanes", "lane_id"),
    ),
    indexes=(("seller_id",), ("lane_id",), ("product_id",)),
)

ORDER_PAYMENTS = Table(
    name="order_payments",
    doc="Payment legs. Installment counts and payment type both shift approval latency.",
    columns=(
        _c("order_id", "id", False),
        _c("payment_seq", "int", False),
        _c("payment_type", "text", False),
        _c("installments", "int", False),
        _c("payment_value", "float", False),
    ),
    primary_key=("order_id", "payment_seq"),
    foreign_keys=(("order_id", "orders", "order_id"),),
)

ORDER_REVIEWS = Table(
    name="order_reviews",
    doc=(
        "Post-delivery satisfaction. Used only to price the cost of a late delivery "
        "(policy/economics.py); it is structurally post-decision and is never a feature."
    ),
    columns=(
        _c("review_id", "id", False),
        _c("order_id", "id", False),
        _c("review_score", "int", False),
        _c("review_creation_ts", "ts", False),
        _c("review_answer_ts", "ts", True),
    ),
    primary_key=("review_id",),
    foreign_keys=(("order_id", "orders", "order_id"),),
    indexes=(("order_id",),),
)

SHIPPING_LANES = Table(
    name="shipping_lanes",
    doc=(
        "Origin state x destination state x carrier. The lane is the unit transit time is a "
        "property of; a seller's late rate is not portable across lanes."
    ),
    columns=(
        _c("lane_id", "id", False),
        _c("origin_state", "text", False),
        _c("dest_state", "text", False),
        _c("carrier_id", "id", False),
        _c("distance_km", "float", False),
        _c("base_transit_days", "float", False, "carrier's published transit time"),
    ),
    primary_key=("lane_id",),
    foreign_keys=(("carrier_id", "carriers", "carrier_id"),),
    indexes=(("carrier_id",), ("origin_state", "dest_state")),
)

CARRIERS = Table(
    name="carriers",
    doc="Service levels, and what it costs to buy a faster one on the day.",
    columns=(
        _c("carrier_id", "id", False),
        _c("carrier_name", "text", False),
        _c("service_level", "text", False, "economy | standard | express"),
        _c("reliability_index", "float", False, "long-run share of shipments inside promise"),
        _c("expedite_surcharge", "float", False, "currency cost of upgrading one parcel"),
        _c("daily_capacity", "int", False),
    ),
    primary_key=("carrier_id",),
)

#: Declaration order is load order: parents before children, so foreign keys hold at every step.
TABLES: tuple[Table, ...] = (
    CUSTOMERS,
    SELLERS,
    PRODUCTS,
    CARRIERS,
    SHIPPING_LANES,
    ORDERS,
    ORDER_ITEMS,
    ORDER_PAYMENTS,
    ORDER_REVIEWS,
)

TABLES_BY_NAME = {t.name: t for t in TABLES}


def schema_ddl(dialect: str = "sqlite", *, foreign_keys: bool = True) -> str:
    """Full ``CREATE TABLE``/``CREATE INDEX`` script for one dialect."""
    parts = [f"-- deliveryrisk operational schema ({dialect}); {len(TABLES)} tables"]
    for t in TABLES:
        parts.append(f"\n-- {t.name}: {t.doc}")
        parts.append(t.ddl(dialect, foreign_keys=foreign_keys))
        parts.extend(t.index_ddl(dialect))
    return "\n".join(parts) + "\n"


def drop_ddl(dialect: str = "sqlite") -> str:
    """Drop script, children first."""
    stmts = [f"DROP TABLE IF EXISTS {t.name};" for t in reversed(TABLES)]
    if dialect == "mysql":
        stmts = ["SET FOREIGN_KEY_CHECKS=0;", *stmts, "SET FOREIGN_KEY_CHECKS=1;"]
    return "\n".join(stmts) + "\n"


@dataclass(frozen=True)
class SchemaViolation:
    table: str
    problem: str
    detail: str = ""


def validate_tables(frames: dict[str, "object"]) -> list[SchemaViolation]:
    """Check a dict of DataFrames against the declared schema before it reaches a database.

    Catching a missing column or a null primary key here produces a message naming the table and
    the column; catching it in the loader produces a driver-level integrity error naming neither.
    """
    out: list[SchemaViolation] = []
    for table in TABLES:
        df = frames.get(table.name)
        if df is None:
            out.append(SchemaViolation(table.name, "missing table"))
            continue
        missing = [c for c in table.column_names if c not in df.columns]
        if missing:
            out.append(SchemaViolation(table.name, "missing columns", ", ".join(missing)))
            continue
        for col in table.columns:
            if not col.null and df[col.name].isna().any():
                n = int(df[col.name].isna().sum())
                out.append(SchemaViolation(table.name, f"nulls in NOT NULL column {col.name}", str(n)))
        if df.duplicated(subset=list(table.primary_key)).any():
            n = int(df.duplicated(subset=list(table.primary_key)).sum())
            out.append(SchemaViolation(table.name, "duplicate primary key", str(n)))
    return out


#: Columns that are outcomes of the order being scored. Any feature query that reads one of these
#: for the row it is describing is, by definition, leaking. `features/sql.py` asserts against it.
POST_DECISION_COLUMNS: frozenset[str] = frozenset(
    {"orders.pickup_ts", "orders.delivered_ts", "order_reviews.review_score",
     "order_reviews.review_creation_ts", "order_reviews.review_answer_ts", "orders.order_status"}
)
