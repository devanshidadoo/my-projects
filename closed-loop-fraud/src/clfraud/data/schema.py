"""Canonical transaction schema shared by the synthetic generator and the IEEE-CIS adapter.

Everything downstream (features, simulator, evaluation) speaks this schema only. That is what
lets the exact same code path run on a synthetic 18-month stream and on the real Kaggle
IEEE-CIS files without a branch anywhere outside :mod:`clfraud.data`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Columns every transaction stream must expose.
CANONICAL_COLUMNS: dict[str, str] = {
    "transaction_id": "int64",
    # Seconds from an arbitrary epoch. IEEE-CIS uses `TransactionDT`, whose origin is unknown;
    # only differences are meaningful, which is all the feature layer ever uses.
    "ts": "float64",
    "amount": "float64",
    "card_id": "string",
    "merchant_id": "string",
    "device_id": "string",
    "email_domain": "string",
    "addr_id": "string",
    "product_cd": "string",
    "is_fraud": "int8",
}

#: Entity keys over which velocity / aggregate features are accumulated.
ENTITY_KEYS: tuple[str, ...] = (
    "card_id",
    "merchant_id",
    "device_id",
    "email_domain",
    "addr_id",
    "card_merchant",  # composite, synthesised in the feature layer
)


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical transaction contract."""


@dataclass(frozen=True)
class TransactionSchema:
    """Contract for a transaction stream.

    Attributes
    ----------
    required:
        Column -> dtype mapping that must be present.
    passthrough:
        Extra raw model columns (IEEE-CIS ``V*``/``C*``/``D*``, synthetic ``v*``) that are
        carried into the model matrix untouched. These are *current-transaction* attributes
        only; nothing in this project lets them see the future.
    """

    required: dict[str, str] = field(default_factory=lambda: dict(CANONICAL_COLUMNS))
    passthrough: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return list(self.required) + list(self.passthrough)


def validate_frame(df: pd.DataFrame, *, require_labels: bool = True) -> pd.DataFrame:
    """Validate and normalise a transaction frame.

    Checks structure, not statistics: presence of required columns, monotone non-decreasing
    time, unique ids, non-negative amounts, binary labels. Returns a time-sorted copy with a
    deterministic tie-break on ``transaction_id`` so that two runs over the same data always
    produce byte-identical feature matrices.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if require_labels is False:
        missing = [c for c in missing if c != "is_fraud"]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    out = df.copy()
    if out["transaction_id"].duplicated().any():
        raise SchemaError("transaction_id must be unique")
    if not np.isfinite(out["ts"].to_numpy(dtype="float64")).all():
        raise SchemaError("ts contains non-finite values")
    if (out["amount"] < 0).any():
        raise SchemaError("amount must be non-negative")
    if require_labels:
        bad = set(np.unique(out["is_fraud"].to_numpy())) - {0, 1}
        if bad:
            raise SchemaError(f"is_fraud must be binary, found {sorted(bad)}")

    # Deterministic total order. Ties in `ts` are common in IEEE-CIS (second resolution) and
    # would otherwise make windowed features depend on input row order.
    out = out.sort_values(["ts", "transaction_id"], kind="mergesort").reset_index(drop=True)
    return out


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric passthrough columns: everything numeric that is not schema metadata or a label."""
    blocked = set(CANONICAL_COLUMNS) | {"cycle", "is_fraud"}
    return [
        c
        for c in df.columns
        if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
    ]
