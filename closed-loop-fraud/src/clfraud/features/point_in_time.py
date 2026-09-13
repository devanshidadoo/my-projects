"""Point-in-time feature builder.

Guarantee
---------
For a stream ordered by ``(ts, transaction_id)``, the feature row for transaction *i* is a
function of transactions ``{j : (ts_j, id_j) < (ts_i, id_i)}`` and of transaction *i*'s own
immutable attributes -- nothing else. Formally, with ``f`` the builder and ``S`` the stream::

    f(S)[i] == f(S[:i+1])[i]                       # prefix invariance
    f(S)[i] == f(S')[i]  whenever S'[:i+1] == S[:i+1]   # future independence

Both are executable properties, not comments: see ``tests/test_leakage_properties.py``.

The builder is also *stateful by design*. ``transform`` runs one pass and leaves the
accumulators warm, so a production deployment can score a live transaction with the same object
that built the training matrix -- no train/serve skew from a second implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from clfraud.data.schema import validate_frame
from clfraud.features.windows import (
    DAY,
    HOUR,
    CountSumWindow,
    DistinctWindow,
    RecencyTracker,
    WelfordTracker,
)

log = logging.getLogger(__name__)

#: Sentinel for "this key has never been seen". Chosen finite and large so tree models can split
#: on it, and distinguishable from any real gap (about 3.2 years in seconds).
NEVER_SEEN = 1e8


@dataclass(frozen=True)
class EntitySpec:
    """One entity to accumulate over.

    Attributes
    ----------
    name:
        Short prefix used in feature names.
    column:
        Canonical column holding the key (or ``"card_merchant"`` for the composite).
    windows:
        Trailing windows (seconds) for count/amount aggregates.
    distinct_of:
        Optional secondary column whose distinct count is tracked, with its window.
    amount_stats:
        Whether to track prior-only mean/std of amount for the z-score feature.
    """

    name: str
    column: str
    windows: tuple[tuple[str, float], ...]
    distinct_of: tuple[str, float] | None = None
    amount_stats: bool = False


@dataclass(frozen=True)
class FeatureSpec:
    """Full feature configuration."""

    entities: tuple[EntitySpec, ...]
    include_calendar: bool = True
    include_amount_shape: bool = True
    passthrough: tuple[str, ...] = field(default_factory=tuple)

    def with_passthrough(self, cols: list[str]) -> FeatureSpec:
        return replace(self, passthrough=tuple(cols))


def default_spec() -> FeatureSpec:
    """The spec used by every experiment in ``docs/RESULTS.md``.

    Entity coverage is deliberately asymmetric: the card gets the full window ladder because
    card-level velocity is where card-testing bursts show up, the merchant gets weekly/monthly
    aggregates because breaches unfold over days, and the card x merchant composite gets short
    windows because repeated hits on one acceptor within an hour is its own signature.
    """
    return FeatureSpec(
        entities=(
            EntitySpec(
                name="card",
                column="card_id",
                windows=(("1h", HOUR), ("24h", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY)),
                distinct_of=("merchant_id", DAY),
                amount_stats=True,
            ),
            EntitySpec(
                name="merch",
                column="merchant_id",
                windows=(("24h", DAY), ("7d", 7 * DAY)),
                distinct_of=("card_id", 7 * DAY),
                amount_stats=True,
            ),
            EntitySpec(
                name="dev",
                column="device_id",
                windows=(("24h", DAY), ("7d", 7 * DAY)),
                distinct_of=("card_id", 7 * DAY),
            ),
            EntitySpec(
                name="cardmerch",
                column="card_merchant",
                windows=(("1h", HOUR), ("7d", 7 * DAY)),
            ),
            EntitySpec(name="email", column="email_domain", windows=(("24h", DAY),)),
            EntitySpec(name="addr", column="addr_id", windows=(("24h", DAY),)),
        )
    )


class FeatureBuilder:
    """Single-pass, strictly causal feature construction."""

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or default_spec()
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self._windows: dict[tuple[str, str], CountSumWindow] = {}
        self._distinct: dict[str, DistinctWindow] = {}
        self._recency: dict[str, RecencyTracker] = {}
        self._welford: dict[str, WelfordTracker] = {}
        for ent in self.spec.entities:
            for wname, secs in ent.windows:
                self._windows[(ent.name, wname)] = CountSumWindow(secs)
            if ent.distinct_of is not None:
                self._distinct[ent.name] = DistinctWindow(ent.distinct_of[1])
            self._recency[ent.name] = RecencyTracker()
            if ent.amount_stats:
                self._welford[ent.name] = WelfordTracker()
        self._n_seen = 0

    def reset(self) -> FeatureBuilder:
        """Forget all accumulated history. Returns self for chaining."""
        self._reset()
        return self

    @property
    def n_seen(self) -> int:
        """Number of transactions absorbed so far."""
        return self._n_seen

    # ------------------------------------------------------------------ names
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for ent in self.spec.entities:
            for wname, _ in ent.windows:
                names += [f"{ent.name}_cnt_{wname}", f"{ent.name}_amt_{wname}"]
            if ent.distinct_of is not None:
                names.append(f"{ent.name}_nuniq_{ent.distinct_of[0].split('_')[0]}")
            names += [f"{ent.name}_since_last", f"{ent.name}_lifetime_cnt"]
            if ent.amount_stats:
                names += [f"{ent.name}_amt_z", f"{ent.name}_amt_ratio"]
        if self.spec.include_calendar:
            names += ["hour_of_day", "day_of_week", "is_night"]
        if self.spec.include_amount_shape:
            names += ["log_amount", "amount_cents", "is_round_amount"]
        names += list(self.spec.passthrough)
        return names

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
        """Compute the point-in-time feature matrix for ``df``.

        ``df`` is processed in stream order. The builder's state carries over between calls,
        so ``transform(a)`` followed by ``transform(b)`` is equivalent to ``transform(a+b)``
        restricted to ``b`` -- the incremental-scoring path.

        Absolute stream time is deliberately absent from the output. Calendar *shape* (hour of
        day, day of week) generalises; ``days_since_start`` only lets a tree memorise which
        stretch of the timeline it was trained on, which is worthless once deployed forward.
        """
        if validate:
            df = validate_frame(df, require_labels=False)
        n = len(df)
        if n == 0:
            return pd.DataFrame(columns=self.feature_names(), dtype="float32")

        ts_arr = df["ts"].to_numpy(dtype="float64")
        amt_arr = df["amount"].to_numpy(dtype="float64")

        # Materialise key columns once as object arrays; per-row .iat lookups dominate runtime.
        key_cols: dict[str, np.ndarray] = {}
        needed = {e.column for e in self.spec.entities}
        needed |= {e.distinct_of[0] for e in self.spec.entities if e.distinct_of}
        for col in needed:
            if col == "card_merchant":
                key_cols[col] = (
                    df["card_id"].astype(str) + "\x1f" + df["merchant_id"].astype(str)
                ).to_numpy()
            else:
                key_cols[col] = df[col].astype(str).to_numpy()

        names = self.feature_names()
        n_dyn = len(names) - len(self.spec.passthrough)
        out = np.empty((n, n_dyn), dtype="float64")

        # Bind everything into flat lists so the hot loop does no dict lookups on names.
        plan = []
        for ent in self.spec.entities:
            plan.append(
                (
                    key_cols[ent.column],
                    [self._windows[(ent.name, w)] for w, _ in ent.windows],
                    self._distinct.get(ent.name),
                    key_cols[ent.distinct_of[0]] if ent.distinct_of else None,
                    self._recency[ent.name],
                    self._welford.get(ent.name),
                )
            )

        for i in range(n):
            ts = ts_arr[i]
            amt = amt_arr[i]
            c = 0
            for keys, wins, dist, dist_keys, rec, wel in plan:
                k = keys[i]
                for w in wins:
                    cnt, tot = w.query(k, ts)           # --- query BEFORE update: the guarantee
                    out[i, c] = cnt
                    out[i, c + 1] = tot
                    c += 2
                if dist is not None:
                    out[i, c] = dist.query(k, ts)
                    c += 1
                gap, life = rec.query(k, ts)
                out[i, c] = NEVER_SEEN if gap == float("inf") else gap
                out[i, c + 1] = life
                c += 2
                if wel is not None:
                    nprev, mean, std = wel.query(k)
                    if nprev >= 2 and std > 1e-9:
                        out[i, c] = (amt - mean) / std
                    else:
                        out[i, c] = 0.0
                    out[i, c + 1] = amt / mean if nprev >= 1 and mean > 1e-9 else 1.0
                    c += 2

                # --- state advance: everything below is invisible to row i
                for w in wins:
                    w.update(k, ts, amt)
                if dist is not None:
                    dist.update(k, ts, dist_keys[i])
                rec.update(k, ts)
                if wel is not None:
                    wel.update(k, amt)

            if self.spec.include_calendar:
                tod = ts % DAY
                out[i, c] = tod / HOUR
                out[i, c + 1] = (ts // DAY) % 7
                out[i, c + 2] = 1.0 if tod < 6 * HOUR else 0.0
                c += 3
            if self.spec.include_amount_shape:
                cents = round((amt - int(amt)) * 100)
                out[i, c] = np.log1p(amt)
                out[i, c + 1] = cents
                out[i, c + 2] = 1.0 if (cents == 0 and int(amt) % 5 == 0) else 0.0
                c += 3

        self._n_seen += n
        feats = pd.DataFrame(out, columns=names[:n_dyn], index=df.index).astype("float32")
        if self.spec.passthrough:
            extra = df[list(self.spec.passthrough)].astype("float32")
            feats = pd.concat([feats, extra], axis=1)
        return feats

    # ------------------------------------------------------------------ diagnostics
    def state_size(self) -> dict[str, int]:
        """Number of live keys per accumulator -- memory profile of a long-running deployment."""
        return {f"{a}_{b}": w.n_active_keys for (a, b), w in self._windows.items()}


def build_features(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    *,
    passthrough: list[str] | None = None,
) -> pd.DataFrame:
    """One-shot helper: build the feature matrix for a whole stream with a fresh builder."""
    spec = spec or default_spec()
    if passthrough:
        spec = spec.with_passthrough(passthrough)
    return FeatureBuilder(spec).transform(df)


def leaky_future_window_features(
    df: pd.DataFrame, spec: FeatureSpec | None = None, *, passthrough: list[str] | None = None
) -> pd.DataFrame:
    """The **wrong** velocity features: a two-sided window instead of a backward one.

    This is the leak that actually bites in fraud work, and it is far more insidious than target
    encoding because the resulting columns look completely ordinary. It arrives through a
    ``groupby(...).rolling('24h')`` that was never shifted, a ``merge_asof`` with the wrong
    ``direction``, or a centred window in a notebook -- and every one of those computes, for each
    transaction, aggregates over a window that includes transactions that had not happened yet.

    For card-testing bursts the effect is devastating and invisible: the *first* authorisation of
    a 12-transaction burst gets a card velocity of 12, because the other eleven are in its
    window. At serving time that transaction has a velocity of zero, because the burst has not
    happened. The model is superb offline and useless live.

    The implementation is deliberately a thin wrapper over the real builder: the same
    accumulators run a second time over the time-reversed stream, and the two passes are
    combined. Counts and amounts add (a symmetric window is the union of both halves);
    ``since_last`` becomes the distance to the *nearest* neighbour in either direction; lifetime
    counts become whole-dataset totals. Distinct counts take the larger of the two passes, which
    understates the leak -- a true centred window would count the union -- so this baseline is a
    conservative one.
    """
    spec = spec or default_spec()
    if passthrough:
        spec = spec.with_passthrough(passthrough)

    forward = FeatureBuilder(spec).transform(df)
    forward.index = df["transaction_id"].to_numpy()

    rev = df.copy()
    rev["ts"] = -rev["ts"].to_numpy()
    rev["transaction_id"] = -rev["transaction_id"].to_numpy()
    rev = rev.sort_values(["ts", "transaction_id"], kind="mergesort")
    backward = FeatureBuilder(spec).transform(rev, validate=False)
    backward.index = -rev["transaction_id"].to_numpy()
    backward = backward.loc[forward.index]

    out = forward.copy()
    for col in forward.columns:
        if col.endswith("_since_last"):
            out[col] = np.minimum(forward[col], backward[col])
        elif col.endswith("_lifetime_cnt") or "_cnt_" in col or "_amt_" in col and "_amt_z" not in col:
            if col.endswith("_amt_ratio"):
                continue
            out[col] = forward[col] + backward[col]
        elif "_nuniq_" in col:
            out[col] = np.maximum(forward[col], backward[col])
    out.index = df.index
    return out.astype("float32")


def leaky_aggregate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """The **wrong** way to build these features, kept as a measurable counterexample.

    This is the pattern that appears in most public fraud notebooks: whole-dataset ``groupby``
    aggregates joined back onto every row. Each row's card-level mean amount is computed from
    the card's *entire* history, including transactions that had not happened yet -- and, worse,
    the target-encoded columns below let a card's future fraud outcomes bleed into its past
    rows. ``experiments/run_all.py`` fits the same model on these features to quantify exactly
    how much offline AUC the leak manufactures.
    """
    out = pd.DataFrame(index=df.index)
    for col in ("card_id", "merchant_id", "device_id"):
        grp = df.groupby(col, observed=True)["amount"]
        out[f"leak_{col}_mean_amt"] = grp.transform("mean").astype("float32")
        out[f"leak_{col}_cnt"] = grp.transform("count").astype("float32")
        out[f"leak_{col}_max_amt"] = grp.transform("max").astype("float32")
        out[f"leak_{col}_amt_ratio"] = (
            df["amount"] / grp.transform("mean").clip(lower=1e-6)
        ).astype("float32")
    if "is_fraud" in df.columns:
        for col in ("card_id", "merchant_id"):
            out[f"leak_{col}_fraud_rate"] = (
                df.groupby(col, observed=True)["is_fraud"].transform("mean").astype("float32")
            )
    return out
