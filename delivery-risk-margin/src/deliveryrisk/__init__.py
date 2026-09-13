"""Order-level delivery risk prediction and a margin-optimised intervention policy.

The package is organised around the decision it exists to make: *at order approval, before the
parcel is handed to a carrier, which orders should be intervened on and with which action?*

    data/        the nine-table operational schema, its DDL, a generator, the Olist adapter
    features/    point-in-time feature extraction in SQL, plus a brute-force reference
    models/      logistic regression / gradient boosting, and probability calibration
    policy/      the cost algebra, per-order thresholds, and budget-constrained assignment
    evaluation/  decile capture, calibration error, counterfactual policy value, the leak audit
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
