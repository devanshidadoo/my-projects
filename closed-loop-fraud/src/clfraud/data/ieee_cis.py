"""Adapter mapping the Kaggle IEEE-CIS Fraud Detection files onto the canonical schema.

Usage
-----
Download ``train_transaction.csv`` and ``train_identity.csv`` from the competition
(`ieee-fraud-detection`) into ``data/raw/`` and run::

    clfraud ingest-ieee --raw-dir data/raw --out data/processed/ieee.parquet

Entity mapping
--------------
IEEE-CIS is anonymised: there is no merchant column and no account id. The community-standard
proxies are used, and each is *declared here rather than buried in feature code* so the
assumption is auditable:

===============  =======================================================================
canonical        IEEE-CIS source
===============  =======================================================================
``card_id``      ``card1`` + ``card2`` + ``addr1`` (the usual "uid" proxy for an account)
``merchant_id``  ``ProductCD`` + ``card4`` + ``R_emaildomain`` (acceptor proxy)
``device_id``    ``DeviceInfo`` + ``id_30`` + ``id_31``, else ``DeviceType``
``email_domain`` ``P_emaildomain``
``addr_id``      ``addr1``
``ts``           ``TransactionDT`` (seconds from an unspecified origin)
===============  =======================================================================

The ``V*``/``C*``/``D*``/``M*`` blocks pass through as model features. Note that IEEE-CIS's own
``C*`` and ``D*`` columns are *already* velocity/recency aggregates computed by Vesta with an
undocumented window; :mod:`clfraud.features.point_in_time` recomputes its own aggregates from
``ts`` alone so that the point-in-time guarantee is one this codebase can actually test.

Because a real log has no counterfactual labels for declined transactions, the closed-loop
simulator cannot run end-to-end on IEEE-CIS. What this adapter *is* for: validating that the
feature engine, leakage tests, and static model benchmark reproduce on real data. See
``docs/METHODOLOGY.md`` §"Why a simulator".
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from clfraud.data.schema import validate_frame

log = logging.getLogger(__name__)

#: Raw IEEE-CIS column families carried through to the model matrix.
PASSTHROUGH_PREFIXES = ("V", "C", "D", "M", "id_")
#: Columns consumed to build canonical entity keys; not passed through as features.
_CONSUMED = {
    "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
    "card1", "card2", "card4", "addr1", "ProductCD",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType", "id_30", "id_31",
}


def _concat_keys(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([f"{prefix}_unknown"] * len(df), index=df.index, dtype="string")
    parts = [df[c].astype("string").fillna("na") for c in present]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return (prefix + "_" + key).astype("string")


def load_ieee_cis(
    raw_dir: str | Path,
    *,
    transaction_file: str = "train_transaction.csv",
    identity_file: str | None = "train_identity.csv",
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and canonicalise IEEE-CIS.

    Parameters
    ----------
    raw_dir:
        Directory holding the extracted competition CSVs.
    nrows:
        Optional row cap, useful for smoke tests on a laptop.
    """
    raw_dir = Path(raw_dir)
    tx_path = raw_dir / transaction_file
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Download the IEEE-CIS Fraud Detection dataset "
            "(kaggle competitions download -c ieee-fraud-detection) and extract it there."
        )
    log.info("reading %s", tx_path)
    tx = pd.read_csv(tx_path, nrows=nrows, low_memory=False)

    if identity_file:
        id_path = raw_dir / identity_file
        if id_path.exists():
            ident = pd.read_csv(id_path, low_memory=False)
            # The public test-identity file uses hyphens (`id-01`); normalise both spellings.
            ident.columns = [c.replace("-", "_") for c in ident.columns]
            tx = tx.merge(ident, on="TransactionID", how="left")
            log.info("joined identity: %d columns", ident.shape[1] - 1)
        else:
            log.warning("%s not found; continuing without identity features", id_path)

    device = _concat_keys(tx, ["DeviceInfo", "id_30", "id_31"], "dev")
    if "DeviceType" in tx.columns:
        fallback = ("devt_" + tx["DeviceType"].astype("string").fillna("na")).astype("string")
        device = device.where(~device.eq("dev_na|na|na"), fallback)

    out = pd.DataFrame(
        {
            "transaction_id": tx["TransactionID"].astype("int64"),
            "ts": tx["TransactionDT"].astype("float64"),
            "amount": tx["TransactionAmt"].astype("float64"),
            "card_id": _concat_keys(tx, ["card1", "card2", "addr1"], "card"),
            "merchant_id": _concat_keys(tx, ["ProductCD", "card4", "R_emaildomain"], "mer"),
            "device_id": device,
            "email_domain": tx.get(
                "P_emaildomain", pd.Series(pd.NA, index=tx.index)
            ).astype("string").fillna("unknown"),
            "addr_id": ("addr_" + tx.get("addr1", pd.Series(pd.NA, index=tx.index))
                        .astype("string").fillna("na")).astype("string"),
            "product_cd": tx.get("ProductCD", pd.Series("W", index=tx.index)).astype("string"),
            "is_fraud": tx["isFraud"].astype("int8"),
        }
    )

    keep = [
        c for c in tx.columns
        if c not in _CONSUMED and c.startswith(PASSTHROUGH_PREFIXES)
    ]
    extra = tx[keep].copy()
    # M1..M9 are T/F strings; map to a numeric ternary so every passthrough column is numeric.
    for c in extra.columns:
        if extra[c].dtype == object:
            mapped = extra[c].map({"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0})
            extra[c] = mapped if mapped.notna().any() else pd.factorize(extra[c])[0].astype(float)
        extra[c] = pd.to_numeric(extra[c], errors="coerce").astype("float32")
    extra.columns = [f"raw_{c}" for c in extra.columns]

    out = pd.concat([out, extra], axis=1)
    log.info(
        "canonicalised %d rows, fraud rate %.4f, %d passthrough columns",
        len(out), out["is_fraud"].mean(), extra.shape[1],
    )
    return validate_frame(out)


def summarise(df: pd.DataFrame) -> dict[str, float]:
    """Marginals used to calibrate the synthetic generator against the real dataset."""
    span = (df["ts"].max() - df["ts"].min()) / 86_400.0
    return {
        "n_transactions": float(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "n_cards": float(df["card_id"].nunique()),
        "n_merchants": float(df["merchant_id"].nunique()),
        "amount_median": float(df["amount"].median()),
        "amount_mean": float(df["amount"].mean()),
        "amount_p99": float(df["amount"].quantile(0.99)),
        "span_days": float(span),
        "txn_per_day": float(len(df) / max(span, 1e-9)),
        "mean_txn_per_card": float(len(df) / max(df["card_id"].nunique(), 1)),
        "fraud_amount_share": float(
            df.loc[df["is_fraud"] == 1, "amount"].sum() / max(df["amount"].sum(), 1e-9)
        ),
    }


def stretch_to_horizon(df: pd.DataFrame, months: int = 18) -> pd.DataFrame:
    """Rescale ``ts`` so a short real window spans the simulator's retraining horizon.

    IEEE-CIS covers ~182 days. The closed-loop question is about *retraining cycles*, so the
    default experiment needs 18 of them. Rather than silently pretend the data is longer, this
    helper makes the transformation explicit and reversible: inter-arrival gaps are scaled by a
    single constant, so every velocity feature is scaled by the same constant too.

    This is a presentation device for the real dataset only; the synthetic stream is generated
    at 18 months natively and needs no stretching.
    """
    span = df["ts"].max() - df["ts"].min()
    target = months * 30.0 * 86_400.0
    factor = target / span
    out = df.copy()
    out["ts"] = (out["ts"] - out["ts"].min()) * factor + 86_400.0
    log.info("stretched %.1f days -> %.1f days (factor %.3f)", span / 86_400, target / 86_400, factor)
    out.attrs["time_stretch_factor"] = factor
    return out
