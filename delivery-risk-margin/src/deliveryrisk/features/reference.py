"""An intentionally slow, intentionally obvious implementation of the same features.

The SQL in :mod:`deliveryrisk.features.sql` is fast because it is written as running window
aggregates over an event stream, and running window aggregates are exactly the construct whose
off-by-one errors are invisible. So the definition is also written out here the way one would
explain it at a whiteboard -- for each order, scan every other row and keep the ones that had
already happened -- and ``tests/test_pit_reference.py`` asserts the two agree to floating point.

    for order i:
        prior = {j : j's delivery landed strictly before i's approval}
        n_prior, late_rate, mean handling, mean transit  are statistics of `prior`
        queue depth  = entries that opened before i's approval and had not yet closed

This is O(n^2) and is only ever run on a few thousand rows. It is not an optimisation target; it
is the thing the optimisation is checked against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deliveryrisk.features.sql import DAY, STALE_QUEUE_DAYS, EntitySpec


def reference_entity_features(
    facts: pd.DataFrame,
    spec: EntitySpec,
    *,
    trailing_days: float = 30.0,
    prior_weight: float = 15.0,
    prior_rate: float = 0.08,
) -> pd.DataFrame:
    """Recompute one entity's history features the slow, obvious way."""
    f = facts.reset_index(drop=True)
    key = f[spec.key].to_numpy()
    approved = f.approved_ts.to_numpy(dtype=float)
    delivered = f.delivered_ts.to_numpy(dtype=float)
    pickup = f.pickup_ts.to_numpy(dtype=float)
    promised = f.estimated_delivery_ts.to_numpy(dtype=float)

    has_outcome = ~np.isnan(delivered) & ~np.isnan(pickup)
    is_late = np.where(has_outcome, delivered > promised, np.nan)
    handling = (pickup - approved) / DAY
    transit = (delivered - pickup) / DAY
    days_over = (delivered - promised) / DAY

    if spec.queue is not None:
        open_expr, close_expr = spec.queue
        open_ts = approved if open_expr == "approved_ts" else pickup
        if close_expr.startswith("COALESCE"):
            close_ts = np.where(np.isnan(pickup), approved + STALE_QUEUE_DAYS * DAY, pickup)
        else:
            close_ts = delivered if close_expr == "delivered_ts" else pickup
    else:
        open_ts = close_ts = None

    n = len(f)
    rows = []
    span = trailing_days * DAY
    for i in range(n):
        t = approved[i]
        same = key == key[i]
        prior = same & has_outcome & (delivered < t)
        recent = prior & (delivered >= t - span)
        n_prior = int(prior.sum())
        n_recent = int(recent.sum())
        n_late = float(np.nansum(is_late[prior]))
        n_late_recent = float(np.nansum(is_late[recent]))
        # `first_ts` is the earliest event of any kind, matching MIN(ts) in the window.
        ev_times = [approved[same & (approved < t)]]
        if has_outcome.any():
            ev_times.append(delivered[same & has_outcome & (delivered < t)])
        if open_ts is not None:
            ev_times.append(open_ts[same & ~np.isnan(open_ts) & (open_ts < t)])
            ev_times.append(close_ts[same & ~np.isnan(close_ts) & (close_ts < t)])
        pieces = [np.atleast_1d(a) for a in ev_times if len(np.atleast_1d(a))]
        all_ev = np.concatenate(pieces) if pieces else np.array([])
        # The order's own decision event is a peer, not a predecessor, but MIN over the frame
        # includes peers with an equal shifted timestamp: other decisions at the same instant.
        own = approved[same & (approved == t)]
        all_ev = np.concatenate([all_ev, own]) if len(own) else all_ev

        row = {
            "order_id": f.order_id.iloc[i],
            f"{spec.name}_n_prior": n_prior,
            f"{spec.name}_n_prior_recent": n_recent,
            f"{spec.name}_late_rate": (n_late + prior_weight * prior_rate) / (n_prior + prior_weight),
            f"{spec.name}_late_rate_recent": (n_late_recent + prior_weight * prior_rate)
            / (n_recent + prior_weight),
            f"{spec.name}_mean_handling": float(handling[prior].mean()) if n_prior else np.nan,
            f"{spec.name}_mean_transit": float(transit[prior].mean()) if n_prior else np.nan,
            f"{spec.name}_mean_days_over": float(days_over[prior].mean()) if n_prior else np.nan,
            f"{spec.name}_mean_transit_recent": float(transit[recent].mean()) if n_recent else np.nan,
            f"{spec.name}_days_since_outcome": (t - delivered[prior].max()) / DAY if n_prior else np.nan,
            f"{spec.name}_history_days": (t - all_ev.min()) / DAY if len(all_ev) else np.nan,
        }
        if spec.queue is not None:
            opened = same & ~np.isnan(open_ts) & (open_ts < t)
            closed = same & ~np.isnan(close_ts) & (close_ts < t)
            depth = int(opened.sum() - closed.sum())
            row[f"{spec.name}_queue_depth"] = depth
            still_open = opened & ~closed
            row[f"{spec.name}_queue_mean_age_days"] = (
                (t - float(open_ts[still_open].mean())) / DAY if depth > 0 and still_open.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
