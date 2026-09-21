"""What an order is worth, and what being late costs.

Everything the policy does rests on two per-order quantities:

``m_i``  contribution margin -- gross margin on the goods plus the thin margin on freight.
``L_i``  the cost of delivering this order late.

``L_i`` is the number people hand-wave. It is built here from three terms that can each be
argued with separately, which is the point:

1. **Service cost.** A late parcel generates contacts. Flat per order.
2. **Goodwill.** Some share of late orders are settled with a voucher or partial refund,
   proportional to order value.
3. **Retention.** Late delivery drives the review score down; a bad review is a proxy for the
   customer not coming back. This term is **estimated from the data**, not assumed:
   :func:`estimate_review_damage` measures the lift in bad-review probability caused by lateness
   over the *training window only*, and the cost model multiplies it by a stated churn-per-bad-
   review and a stated repeat horizon.

The reason it matters that ``L_i`` scales with order value while a nudge costs the same on every
order is that it makes the intervention threshold *per order*::

    intervene when  u_a * p_i * L_i > c_a(i)   <=>   p_i > c_a(i) / (u_a * L_i)

A single global probability cut-off is a special case of that rule, and it is the wrong one
whenever ``L_i`` varies -- which is always. ``docs/RESULTS.md`` §5 reports how far apart the two
end up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Gross margin rate by category. A fixed, documented table rather than a random draw, so the
#: same category has the same economics in every run and in every ablation.
def _margin_rates(categories: list[str]) -> dict[str, float]:
    return {c: 0.16 + 0.26 * ((i * 17) % 11) / 10.0 for i, c in enumerate(sorted(categories))}


@dataclass
class CostModel:
    """Per-order economics. Every constant here is an assumption with a name."""

    freight_margin_share: float = 0.10   # share of freight charged that is not shipping cost
    default_margin_rate: float = 0.28

    support_cost: float = 4.50           # contacts, re-delivery admin, per late order
    voucher_prob: float = 0.22           # share of late orders settled with goodwill
    voucher_share: float = 0.12          # ... worth this share of order value

    churn_per_bad_review: float = 0.35   # share of 1-2 star customers who do not return
    repeat_horizon_orders: float = 2.5   # future orders a retained customer is worth
    #: Filled by `estimate_review_damage`; the fallback is used only if reviews are unavailable.
    review_damage: float = 0.30

    margin_rates: dict[str, float] = field(default_factory=dict)

    def with_categories(self, categories) -> CostModel:
        out = CostModel(**{**self.__dict__})
        out.margin_rates = _margin_rates(sorted({str(c) for c in categories}))
        return out

    # ------------------------------------------------------------------ per order
    def margin_rate(self, category: pd.Series) -> np.ndarray:
        if not self.margin_rates:
            return np.full(len(category), self.default_margin_rate)
        return category.astype(str).map(self.margin_rates).fillna(self.default_margin_rate).to_numpy()

    def contribution_margin(self, df: pd.DataFrame) -> np.ndarray:
        """Margin the order earns when everything goes to plan."""
        return (
            df.total_price.to_numpy() * self.margin_rate(df.category)
            + self.freight_margin_share * df.total_freight.to_numpy()
        )

    def late_cost(self, df: pd.DataFrame) -> np.ndarray:
        """Expected cost of this order being delivered after its promise."""
        m = self.contribution_margin(df)
        goodwill = self.voucher_prob * self.voucher_share * df.total_price.to_numpy()
        retention = (
            self.review_damage * self.churn_per_bad_review * self.repeat_horizon_orders * m
        )
        return self.support_cost + goodwill + retention

    def describe(self) -> dict[str, float]:
        return {
            k: v for k, v in self.__dict__.items() if isinstance(v, (int, float))
        }


def estimate_review_damage(df: pd.DataFrame, reviews: pd.DataFrame, *, bad_star_max: int = 2) -> float:
    """P(bad review | late) - P(bad review | on time), on the rows given.

    Call it with the **training window only**. It is an input to the cost model, and a cost model
    fitted on the evaluation window is the same mistake as a feature fitted on it, just wearing a
    different hat.
    """
    joined = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="inner")
    joined = joined[joined.y_late.notna()]
    if joined.empty:
        log.warning("no reviews joined; keeping the default review_damage")
        return float("nan")
    bad = (joined.review_score <= bad_star_max).to_numpy(dtype=float)
    late = joined.y_late.to_numpy(dtype=bool)
    if late.sum() == 0 or (~late).sum() == 0:
        return float("nan")
    damage = float(bad[late].mean() - bad[~late].mean())
    log.info(
        "review damage: P(<=%d stars | late)=%.3f vs on time=%.3f -> %.3f",
        bad_star_max, bad[late].mean(), bad[~late].mean(), damage,
    )
    return damage


def order_economics(df: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    """Attach ``margin`` and ``late_cost`` to a feature frame."""
    out = df.copy()
    out["margin"] = cost.contribution_margin(df)
    out["late_cost"] = cost.late_cost(df)
    return out
