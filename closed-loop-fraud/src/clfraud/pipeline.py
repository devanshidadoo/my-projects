"""Dataset preparation: source -> canonical stream -> cached point-in-time feature matrix.

Feature construction is deliberately *policy-independent* and therefore cached once and shared
by every arm. That is not just an optimisation, it is the modelling claim: an issuer computes
features for declined transactions too. It never learns their outcome, but it always knows what
they looked like. Selection acts on labels, not on features -- which is exactly why estimating
``P(observed | x)`` is possible at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from clfraud.config import ExperimentConfig
from clfraud.data.schema import raw_feature_columns
from clfraud.data.synthetic import generate_stream
from clfraud.features.point_in_time import FeatureSpec, build_features, default_spec

log = logging.getLogger(__name__)

META_COLUMNS = ["ts", "amount", "is_fraud"]


def _fingerprint(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_stream(cfg: ExperimentConfig, cache_dir: Path | None = None) -> pd.DataFrame:
    """Materialise the canonical transaction stream described by ``cfg``."""
    if cfg.source == "synthetic":
        key = _fingerprint(
            {k: v for k, v in asdict(cfg.synthetic).items() if not k.startswith("_")}
        )
        path = (cache_dir / f"stream_{key}.parquet") if cache_dir else None
        if path and path.exists():
            log.info("loading cached stream %s", path)
            return pd.read_parquet(path)
        df = generate_stream(cfg.synthetic)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        return df

    from clfraud.data.ieee_cis import load_ieee_cis, stretch_to_horizon

    df = load_ieee_cis(cfg.raw_dir, nrows=cfg.nrows)
    if cfg.stretch_months:
        df = stretch_to_horizon(df, cfg.stretch_months)
    return df


def build_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(features, meta)`` for a canonical stream.

    ``meta`` carries the columns the simulator needs but the model must never see: timestamps,
    amounts, ground-truth labels, and (synthetic only) the fraud archetype used for the
    composition diagnostics.
    """
    spec = spec or default_spec()
    passthrough = raw_feature_columns(df)
    meta_cols = META_COLUMNS + (["archetype"] if "archetype" in df.columns else [])
    meta = df[meta_cols].reset_index(drop=True)

    path = (cache_dir / f"features_{cache_key}.parquet") if (cache_dir and cache_key) else None
    if path and path.exists():
        log.info("loading cached features %s", path)
        return pd.read_parquet(path), meta

    log.info("building point-in-time features for %d transactions", len(df))
    X = build_features(df, spec, passthrough=passthrough)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(path, index=False)
    return X, meta


def prepare(cfg: ExperimentConfig, cache_dir: Path | None = None):
    """End-to-end preparation: stream + feature matrix + meta, with caching."""
    df = load_stream(cfg, cache_dir)
    key = _fingerprint(
        {
            "n": len(df),
            "source": cfg.source,
            "syn": {k: v for k, v in asdict(cfg.synthetic).items() if not k.startswith("_")}
            if cfg.source == "synthetic"
            else cfg.raw_dir,
        }
    )
    X, meta = build_matrix(df, cache_dir=cache_dir, cache_key=key)
    return df, X, meta


def dataset_summary(df: pd.DataFrame) -> dict[str, float]:
    """Marginals reported in the run header, comparable to IEEE-CIS's own."""
    span_days = float((df["ts"].max() - df["ts"].min()) / 86_400.0)
    out = {
        "n_transactions": float(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "n_cards": float(df["card_id"].nunique()),
        "n_merchants": float(df["merchant_id"].nunique()),
        "span_days": span_days,
        "amount_median": float(df["amount"].median()),
        "amount_mean": float(df["amount"].mean()),
        "amount_p99": float(df["amount"].quantile(0.99)),
        "fraud_amount_share": float(
            df.loc[df["is_fraud"] == 1, "amount"].sum() / df["amount"].sum()
        ),
    }
    return out
