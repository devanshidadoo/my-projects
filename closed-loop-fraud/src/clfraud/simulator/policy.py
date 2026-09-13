r"""Decision policy: who gets declined, and who gets explored.

A production authorisation policy is a threshold on a risk score, chosen to spend a fixed
*decline budget* -- issuers do not pick a probability cut-off, they pick how much good traffic
they are willing to refuse. So the threshold here is a quantile of the previous cycle's score
distribution, which is also what makes the arms comparable: every policy declines the same
volume, so differences in fraud caught are differences in *ranking quality*, not in appetite.

Exploration modes
-----------------
All three approve a fraction of would-be declines; they differ in *which* ones, and therefore in
what they cost and what they identify.

``uniform``
    :math:`q(x) = \epsilon` for every decline. Unbiased and maximally informative, and the most
    expensive: exploration lands on the highest-risk transactions as often as the marginal ones.

``risk_tapered`` (default)
    :math:`q(x) \propto \exp\!\big(-(s(x) - \tau)/\sigma\big)`, renormalised so the expected
    exploration volume still equals :math:`\epsilon`. Explores densely just above the threshold,
    where the model is least sure and the label is worth most, and rarely deep in the tail,
    where a label costs real money and tells you little. Positivity survives because
    :math:`q(x) > 0` everywhere.

``amount_capped``
    Uniform exploration but only below an amount ceiling. Cheapest per label and the one an
    issuer will actually sign off on -- and the one that *breaks* identification: above the cap
    :math:`q(x) = 0`, so that region stays unrecoverable. Included because the honest comparison
    is between corrections that differ in which assumption they violate.

``cost_aware`` (what the headline result uses)
    Two observations drive it. First, exploring only near the threshold is a trap: the fraud a
    closed loop forgets fastest sits *deep* in the decline region, precisely where a taper
    almost never looks. Second, the cost of an exploration approval is its dollar amount, while
    its information value is roughly independent of it -- a $12 label teaches as much as a
    $1,200 one and costs a hundredth as much. So the budget is spread evenly across score
    deciles of the decline region (coverage everywhere, which the taper sacrifices) and, within
    each decile, tilted toward small tickets by :math:`u(x) \propto 1/(1 + a(x)/a_0)`
    (cheapness, which uniform sacrifices). A floor keeps :math:`q(x) > 0` everywhere, so
    positivity holds and IPW stays identified; the amount tilt is fully corrected by the same
    weights, because it is a known part of the logged propensity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExplorationConfig:
    """Exploration rule parameters."""

    #: Fraction of *would-be declines* approved for exploration. 0.02 = the headline setting.
    rate: float = 0.02
    #: {"none", "uniform", "risk_tapered", "amount_capped", "cost_aware"}
    mode: str = "cost_aware"
    #: Score-distance scale of the taper, in units of the decline-region score range.
    taper_scale: float = 0.25
    #: Amount ceiling for `amount_capped`, in currency units.
    amount_cap: float = 150.0
    #: Amount scale of the cost tilt in `cost_aware`: transactions at this amount get half the
    #: exploration probability of a $0 one.
    amount_scale: float = 60.0
    #: Score-decile strata used by `cost_aware` to force coverage across the decline region.
    n_strata: int = 10
    #: Hard floor on q. Positivity is not a nicety here: the whole IPW argument collapses
    #: without it, so the floor is on by default and only `uniform` (already floored at `rate`)
    #: can do without it.
    propensity_floor: float = 0.004

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("exploration rate must lie in [0, 1]")
        if self.mode not in {
            "none", "uniform", "risk_tapered", "amount_capped", "cost_aware"
        }:
            raise ValueError(f"unknown exploration mode: {self.mode}")


@dataclass
class DecisionOutcome:
    """Per-transaction result of one policy pass.

    Attributes
    ----------
    declined_by_policy:
        What the deterministic threshold rule wanted. Drives the propensity, not the label.
    explored:
        Whether the exploration rule overrode a decline.
    approved:
        Final action: ``~declined_by_policy | explored``. Only approved rows produce labels.
    explore_prob:
        :math:`q(x)`, the randomisation probability actually used. Logged, because a propensity
        you did not log is a propensity you will have to estimate.
    propensity:
        :math:`e(x) = P(\\text{label observed} \\mid x)`.
    threshold:
        The score cut-off applied this cycle.
    """

    declined_by_policy: np.ndarray
    explored: np.ndarray
    approved: np.ndarray
    explore_prob: np.ndarray
    propensity: np.ndarray
    threshold: float

    @property
    def decline_rate(self) -> float:
        return float(np.mean(~self.approved))

    def summary(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "policy_decline_rate": float(np.mean(self.declined_by_policy)),
            "realised_decline_rate": self.decline_rate,
            "explored_count": float(np.sum(self.explored)),
            "explored_share_of_declines": float(
                np.sum(self.explored) / max(np.sum(self.declined_by_policy), 1)
            ),
            "min_propensity": float(self.propensity.min()),
        }


class ThresholdPolicy:
    """Quantile-threshold decline rule with optional randomised exploration."""

    def __init__(
        self,
        decline_budget: float = 0.03,
        exploration: ExplorationConfig | None = None,
    ) -> None:
        if not 0.0 < decline_budget < 1.0:
            raise ValueError("decline_budget must lie in (0, 1)")
        self.decline_budget = decline_budget
        self.exploration = exploration or ExplorationConfig(mode="none", rate=0.0)
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ threshold
    def calibrate(self, scores: np.ndarray) -> float:
        """Set the threshold from a reference score distribution (the previous cycle's).

        Calibrating on the *previous* cycle rather than the one being decided is not a detail:
        an issuer cannot see the current cycle's scores before deciding on it. It also means
        score drift shows up as decline-rate drift, which is exactly how the failure presents
        itself in production dashboards.
        """
        scores = np.asarray(scores, dtype="float64")
        if scores.size == 0:
            self.threshold_ = 1.0
        else:
            self.threshold_ = float(np.quantile(scores, 1.0 - self.decline_budget))
        return self.threshold_

    # ------------------------------------------------------------------ exploration
    def _explore_prob(self, scores: np.ndarray, amounts: np.ndarray, declined: np.ndarray) -> np.ndarray:
        cfg = self.exploration
        q = np.zeros(scores.shape, dtype="float64")
        if cfg.mode == "none" or cfg.rate <= 0.0 or not declined.any():
            return q

        if cfg.mode == "uniform":
            q[declined] = cfg.rate

        elif cfg.mode == "risk_tapered":
            s = scores[declined]
            tau = self.threshold_ if self.threshold_ is not None else float(s.min())
            spread = max(float(s.max() - tau), 1e-9)
            raw = np.exp(-(s - tau) / (cfg.taper_scale * spread))
            # Renormalise so E[#explored] == rate * #declines regardless of the taper's shape.
            scale = cfg.rate * raw.size / max(raw.sum(), 1e-12)
            q[declined] = np.clip(raw * scale, cfg.propensity_floor, 1.0)

        elif cfg.mode == "cost_aware":
            idx = np.flatnonzero(declined)
            s = scores[idx]
            # Score-decile strata. Each gets the same expected share of the exploration budget,
            # so the deep tail is covered instead of being tapered into invisibility.
            ranks = np.argsort(np.argsort(s))
            strata = np.minimum((ranks * cfg.n_strata) // max(len(s), 1), cfg.n_strata - 1)
            u = 1.0 / (1.0 + amounts[idx] / max(cfg.amount_scale, 1e-9))
            q_local = np.zeros(len(idx))
            for k in range(cfg.n_strata):
                m = strata == k
                if not m.any():
                    continue
                q_local[m] = cfg.rate * u[m] / max(u[m].mean(), 1e-12)
            q_local = np.clip(q_local, cfg.propensity_floor, 1.0)
            # Renormalise once more: flooring and capping perturb the budget.
            scale = cfg.rate * len(idx) / max(q_local.sum(), 1e-12)
            q[idx] = np.clip(q_local * scale, cfg.propensity_floor, 1.0)

        elif cfg.mode == "amount_capped":
            eligible = declined & (amounts <= cfg.amount_cap)
            n_elig = int(eligible.sum())
            if n_elig:
                # Concentrate the same *budget* (rate x #declines) on eligible rows only.
                rate_eff = min(cfg.rate * int(declined.sum()) / n_elig, 1.0)
                q[eligible] = rate_eff
            if cfg.propensity_floor > 0:
                q[declined] = np.maximum(q[declined], cfg.propensity_floor)

        return q

    # ------------------------------------------------------------------ decide
    def decide(
        self,
        scores: np.ndarray,
        amounts: np.ndarray,
        rng: np.random.Generator,
    ) -> DecisionOutcome:
        """Apply the policy to a batch of scored transactions."""
        if self.threshold_ is None:
            raise RuntimeError("policy threshold is not calibrated; call calibrate() first")
        scores = np.asarray(scores, dtype="float64")
        amounts = np.asarray(amounts, dtype="float64")

        declined = scores >= self.threshold_
        q = self._explore_prob(scores, amounts, declined)
        explored = declined & (rng.uniform(size=scores.shape) < q)
        approved = (~declined) | explored
        propensity = np.where(declined, q, 1.0)
        return DecisionOutcome(
            declined_by_policy=declined,
            explored=explored,
            approved=approved,
            explore_prob=q,
            propensity=propensity,
            threshold=float(self.threshold_),
        )


class OracleAccessPolicy(ThresholdPolicy):
    """Control arm: acts like the threshold policy but every label comes back anyway.

    Physically impossible -- a declined transaction never charges back, so its outcome is never
    revealed. It is the upper bound the corrected arms are measured against: how good would the
    model be if selection bias simply did not exist, holding the decision policy fixed?
    """

    def decide(self, scores, amounts, rng) -> DecisionOutcome:
        out = super().decide(scores, amounts, rng)
        return DecisionOutcome(
            declined_by_policy=out.declined_by_policy,
            explored=out.explored,
            approved=out.approved,
            explore_prob=out.explore_prob,
            propensity=np.ones_like(out.propensity),  # labels observed unconditionally
            threshold=out.threshold,
        )
