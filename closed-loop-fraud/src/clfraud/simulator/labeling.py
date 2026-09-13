"""Label maturation: when an outcome actually becomes trainable.

Two facts about fraud labels that offline benchmarks quietly ignore:

1. **They arrive late.** A chargeback lands 30-90 days after the authorisation. A model
   retrained on the 1st of the month cannot use last week's transactions -- they have no
   outcome yet. So the freshest data is always the least usable, which is precisely backwards
   from what drift wants.
2. **They arrive incomplete.** Some fraud is never disputed (small tickets, unengaged
   cardholders) and some disputes are not fraud (friendly fraud, first-party misuse). Both show
   up as label noise in the positive class only, which biases the base rate downward.

Both are modelled here because both interact with the feedback loop: delay determines how much
of a cycle's evidence the next model gets, and the miss rate determines how many positives the
corrected arms have to work with after censoring has already thinned them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY = 86_400.0


@dataclass
class LabelPipelineConfig:
    """Parameters of the chargeback process."""

    #: Days from authorisation until the outcome is trainable.
    delay_days: float = 45.0
    #: Share of true fraud never reported -> label flips 1 to 0 (false negatives in the labels).
    unreported_fraud_rate: float = 0.0
    #: Share of legitimate transactions mislabelled as fraud (friendly fraud / first-party).
    friendly_fraud_rate: float = 0.0
    seed: int = 101

    def __post_init__(self) -> None:
        if self.delay_days < 0:
            raise ValueError("delay_days must be non-negative")
        for r in (self.unreported_fraud_rate, self.friendly_fraud_rate):
            if not 0.0 <= r <= 1.0:
                raise ValueError("label noise rates must lie in [0, 1]")


class LabelPipeline:
    """Turns true outcomes into the noisy, delayed labels a retraining job would actually see."""

    def __init__(self, config: LabelPipelineConfig | None = None) -> None:
        self.config = config or LabelPipelineConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def observed_labels(self, y_true: np.ndarray) -> np.ndarray:
        """Apply reporting noise. Identity when both noise rates are zero."""
        cfg = self.config
        y = np.asarray(y_true).astype(int).copy()
        if cfg.unreported_fraud_rate > 0:
            pos = np.flatnonzero(y == 1)
            drop = pos[self._rng.uniform(size=pos.size) < cfg.unreported_fraud_rate]
            y[drop] = 0
        if cfg.friendly_fraud_rate > 0:
            neg = np.flatnonzero(y == 0)
            flip = neg[self._rng.uniform(size=neg.size) < cfg.friendly_fraud_rate]
            y[flip] = 1
        return y

    def is_mature(self, ts: np.ndarray, as_of: float) -> np.ndarray:
        """Whether each transaction's outcome is known by wall-clock time ``as_of``."""
        return np.asarray(ts, dtype="float64") + self.config.delay_days * DAY <= as_of

    def maturity_lag_cycles(self, cycle_days: float) -> float:
        """How many retraining cycles of evidence the delay costs. Reported in the run header."""
        return self.config.delay_days / max(cycle_days, 1e-9)
