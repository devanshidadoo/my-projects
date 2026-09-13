"""The intervention catalogue: what each action costs, and what it is believed to do.

Two actions, and their combination, act on *different halves of the delivery clock*::

    nudge      an SLA escalation to the seller.        compresses handling. free-ish, weak.
    expedite   buy the express service on the lane.    compresses transit. dear, strong.

That asymmetry is the reason risk alone is not a sufficient statistic for the decision. An order
whose 40 % lateness risk is a seller sitting on a stalled pick is not helped by paying a carrier
more, and an order stuck behind a depot failure is not helped by emailing the seller. Assigning
by risk rank gives both of them the same treatment; assigning by *expected contribution* asks
which mechanism is failing before it spends anything.

Uplift, and where honesty lives in this project
-----------------------------------------------
``rho`` is the probability that the action actually delivers the compression it is supposed to.
It is **not** identified from a do-nothing log -- nothing in an observational extract says what
a nudge would have achieved. It is a stated assumption, and the policy is only as good as it.
So: the counterfactual generator knows the true response, ``configs/uplift_sensitivity.yaml``
sweeps the *believed* value against the true one, and ``docs/RESULTS.md`` §7 reports how wrong
the belief can be before the policy stops paying. A single assumed uplift with no sensitivity
analysis would be a number dressed up as a result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Disjoint cause classes among late orders. The names say which action can still save the order.
CAUSES = ("handling_only", "transit_only", "either", "neither")
ACTIONS = ("none", "nudge", "expedite", "both")


@dataclass
class ActionCosts:
    """Marginal cost of each action, per order."""

    nudge_cost: float = 1.20             # ops time for a seller escalation
    expedite_multiplier: float = 1.0     # scales the carrier's published surcharge
    expedite_weight_rate: float = 0.05   # surcharge uplift per kg

    def per_order(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        expedite = (
            df.expedite_surcharge.to_numpy()
            * self.expedite_multiplier
            * (1.0 + self.expedite_weight_rate * df.total_weight_g.to_numpy() / 1000.0)
        )
        nudge = np.full(n, self.nudge_cost)
        return {
            "none": np.zeros(n),
            "nudge": nudge,
            "expedite": expedite,
            "both": nudge + expedite,
        }


@dataclass
class UpliftBelief:
    """What the policy believes each action achieves, per unit of fixable risk."""

    #: Defaults are what a well-run uplift experiment on this population would report: the share
    #: of rescuable late orders each action actually rescues. They are *inputs*, and they are
    #: wrong by construction in `configs/uplift_sensitivity.yaml`, which is the point.
    rho_nudge: float = 0.46
    rho_expedite: float = 0.71

    def uplifts(self, p_cause: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Absolute reduction in P(late) for each action.

        ``p_cause`` holds the four disjoint probabilities ``P(late & cause)``. An order that only
        a nudge can save contributes to the nudge uplift and not to the expedite one; an order
        either could save contributes to both, and to their combination through the usual
        independent-failure product.
        """
        h = p_cause["handling_only"]
        t = p_cause["transit_only"]
        e = p_cause["either"]
        rn, re = self.rho_nudge, self.rho_expedite
        return {
            "none": np.zeros_like(h),
            "nudge": rn * (h + e),
            "expedite": re * (t + e),
            "both": rn * h + re * t + (1.0 - (1.0 - rn) * (1.0 - re)) * e,
        }

    def risk_only_uplifts(
        self, p_late: np.ndarray, mix: dict[str, float]
    ) -> dict[str, np.ndarray]:
        """The same beliefs, applied without a per-order cause split -- the ablation in §6.

        This is the *strongest* honest version of a risk-ranked policy: it still knows the
        population mix of causes among late orders (measured on the training window), it just
        cannot tell which mix applies to the order in front of it. Handicapping it further --
        letting it assume a nudge rescues any late order -- would win the comparison by
        misconfiguring the baseline, which proves nothing.
        """
        return self.uplifts(
            {c: float(mix.get(c, 0.0)) * p_late for c in CAUSES}
        )
