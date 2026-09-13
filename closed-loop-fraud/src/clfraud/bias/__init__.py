"""Selection-bias correction: propensities, weights, overlap diagnostics."""
from clfraud.bias.propensity import (
    PropensityModel,
    known_propensity,
    overlap_diagnostics,
)
from clfraud.bias.reweighting import WeightConfig, build_weights

__all__ = [
    "PropensityModel",
    "known_propensity",
    "overlap_diagnostics",
    "WeightConfig",
    "build_weights",
]
