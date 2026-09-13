"""Run the feature SQL against a database and hand back the modelling table."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from deliveryrisk.data.db import Database
from deliveryrisk.features.sql import (
    CATEGORICAL_COLUMNS,
    ENTITIES,
    ORDER_FACTS_SQL,
    OUTCOME_COLUMNS,
    EntitySpec,
    build_entity_sql,
    build_events_sql,
    build_features_sql,
    feature_columns,
)

log = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """How the feature table is built. One object, so an experiment can state it in YAML."""

    mode: str = "point_in_time"
    trailing_days: float = 30.0
    prior_weight: float = 40.0
    prior_rate: float | None = None  # computed from the training window when None
    handling_floor_days: float = 0.35
    transit_floor_share: float = 0.34


def materialise_facts(
    db: Database, entities: tuple[EntitySpec, ...] = ENTITIES, *, trailing_days: float = 30.0
) -> None:
    """Build ``order_facts`` and ``pit_events``. Independent of the feature mode, so the leak
    audit's two runs share the work."""
    t0 = time.time()
    db.executescript(ORDER_FACTS_SQL)
    db.executescript(build_events_sql(entities, trailing_days=trailing_days))
    n_ev = int(db.read_sql("SELECT COUNT(*) AS n FROM pit_events")["n"].iloc[0])
    log.info("materialised order_facts + %d events in %.1fs", n_ev, time.time() - t0)


def build_features(
    db: Database,
    cfg: FeatureConfig | None = None,
    *,
    entities: tuple[EntitySpec, ...] = ENTITIES,
    materialise: bool = True,
) -> pd.DataFrame:
    """Execute the feature query and return one row per approved order."""
    cfg = cfg or FeatureConfig()
    if materialise:
        materialise_facts(db, entities, trailing_days=cfg.trailing_days)
    prior_rate = cfg.prior_rate
    if prior_rate is None:
        prior_rate = _training_prior_rate(db)
    t0 = time.time()
    for spec in entities:
        db.executescript(
            build_entity_sql(
                spec,
                mode=cfg.mode,
                trailing_days=cfg.trailing_days,
                prior_weight=cfg.prior_weight,
                prior_rate=prior_rate,
            )
        )
        log.debug("materialised feat_%s", spec.name)
    sql = build_features_sql(
        mode=cfg.mode,
        trailing_days=cfg.trailing_days,
        prior_weight=cfg.prior_weight,
        prior_rate=prior_rate,
        handling_floor_days=cfg.handling_floor_days,
        transit_floor_share=cfg.transit_floor_share,
        entities=entities,
    )
    df = db.read_sql(sql)
    log.info("built %d x %d feature rows (%s) in %.1fs", len(df), df.shape[1], cfg.mode,
             time.time() - t0)
    _assert_no_outcome_features(df, entities)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _training_prior_rate(db: Database, train_share: float = 0.6) -> float:
    """Late rate over the earliest ``train_share`` of approved orders.

    Deliberately not the overall rate: the shrinkage target is a parameter of the feature
    definition, and a feature definition that has read the evaluation window is not a feature
    definition, it is a result.
    """
    q = """
    SELECT delivered_ts, estimated_delivery_ts, approved_ts
    FROM order_facts WHERE delivered_ts IS NOT NULL ORDER BY approved_ts
    """
    df = db.read_sql(q)
    cut = int(len(df) * train_share)
    head = df.iloc[:cut] if cut > 100 else df
    rate = float((head.delivered_ts > head.estimated_delivery_ts).mean())
    log.info("shrinkage prior rate from the first %.0f%% of orders: %.4f", train_share * 100, rate)
    return rate


def _assert_no_outcome_features(df: pd.DataFrame, entities: tuple[EntitySpec, ...]) -> None:
    """The feature list and the outcome list must not intersect. Cheap, and it has caught a
    rename more than once: adding ``days_over`` to a feature block is a two-character mistake
    that produces a 0.99 AUC and a plausible-looking model."""
    feats = set(feature_columns(entities))
    leaked = sorted(feats & set(OUTCOME_COLUMNS))
    if leaked:
        raise AssertionError(f"outcome columns declared as features: {leaked}")
    missing = sorted(feats - set(df.columns))
    if missing:
        raise AssertionError(f"feature query did not produce: {missing}")


def split_by_time(
    df: pd.DataFrame, train_share: float = 0.6, valid_share: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split on ``approved_ts``.

    Never a random split. Every entity history feature is a running aggregate, so a random split
    puts an order's own future neighbours in the training set and reports a number the calendar
    will never reproduce. The validation block exists to fit the calibrator on data the risk
    model did not see, which is the whole reason calibration works at all.
    """
    df = df.sort_values(["approved_ts", "order_id"], ignore_index=True)
    n = len(df)
    i, j = int(n * train_share), int(n * (train_share + valid_share))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with an outcome: delivered orders only.

    Cancellations never deliver and in-flight orders have not yet. Both stay in the *event*
    stream -- a cancelled order still occupied its seller's queue -- and both drop out of
    training and evaluation, which is the same right-censoring a live extract has.
    """
    return df[df.y_late.notna()].copy()
