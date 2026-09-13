"""Closed-loop credit card fraud detection under selection bias.

Sub-packages
------------
``clfraud.data``        canonical schema, IEEE-CIS adapter, calibrated synthetic generator
``clfraud.features``    strictly point-in-time velocity / aggregate features
``clfraud.models``      gradient-boosted scorers + probability calibration
``clfraud.simulator``   the closed loop: decisions censor labels, models retrain on what survives
``clfraud.bias``        propensity estimation and selection-bias corrections
``clfraud.evaluation``  oracle metrics, off-policy estimators, reporting
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
