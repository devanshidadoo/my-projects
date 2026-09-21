"""End to end: nine tables in, a decision per order out.

    generate (or ingest)  ->  load into SQL  ->  point-in-time feature query
      ->  chronological split  ->  risk model + calibrator + cause model
      ->  cost model  ->  decision table  ->  policies  ->  realised contribution

Every stage is cached on the config that produced it, so re-running one experiment does not
regenerate the warehouse.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from deliveryrisk.config import ExperimentConfig
from deliveryrisk.data.db import Database
from deliveryrisk.data.synthetic import GeneratedData, dataset_summary, generate
from deliveryrisk.evaluation.decisions import align_truth, compare_policies, oracle_actions
from deliveryrisk.features.build import build_features, labelled, split_by_time
from deliveryrisk.models.calibration import Calibrator, calibration_metrics
from deliveryrisk.models.cause import CauseModel, population_mix
from deliveryrisk.models.linear import make_model
from deliveryrisk.policy import thresholds as th
from deliveryrisk.policy.actions import ACTIONS
from deliveryrisk.policy.allocation import allocate_under_budget, budget_frontier
from deliveryrisk.policy.economics import estimate_review_damage, order_economics

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    """The warehouse, the feature table, and the counterfactual truth beside it."""

    features: pd.DataFrame
    truth: pd.DataFrame
    reviews: pd.DataFrame
    summary: dict[str, float]
    db: Database
    counts: dict[str, int] = field(default_factory=dict)


def _fingerprint(cfg: ExperimentConfig) -> str:
    payload = json.dumps(
        {"data": cfg.to_dict()["data"], "features": asdict(cfg.features)}, sort_keys=True
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def prepare(cfg: ExperimentConfig, cache_dir: Path | None = None) -> Prepared:
    """Build (or load) the warehouse and the point-in-time feature table."""
    cache_dir = Path(cache_dir or "data/cache")
    key = _fingerprint(cfg)
    feat_path = cache_dir / f"features_{key}.parquet"
    truth_path = cache_dir / f"truth_{key}.parquet"
    reviews_path = cache_dir / f"reviews_{key}.parquet"
    summary_path = cache_dir / f"summary_{key}.json"

    db = Database(cfg.database_url)
    if feat_path.exists() and truth_path.exists():
        log.info("loading cached feature table %s", feat_path)
        return Prepared(
            features=pd.read_parquet(feat_path),
            truth=pd.read_parquet(truth_path),
            reviews=pd.read_parquet(reviews_path),
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {},
            db=db,
        )

    data = _materialise_source(cfg)
    t0 = time.time()
    db.create_schema()
    db.load_frames(data.frames)
    log.info("loaded the nine tables in %.1fs", time.time() - t0)
    features = build_features(db, cfg.features)
    summary = dataset_summary(data.frames)

    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feat_path, index=False)
    truth_out = data.truth.copy()
    truth_out.attrs = {}  # parquet cannot carry the promise-buffer note; RESULTS records it
    truth_out.to_parquet(truth_path, index=False)
    data.frames["order_reviews"].to_parquet(reviews_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    return Prepared(
        features=features, truth=data.truth, reviews=data.frames["order_reviews"],
        summary=summary, db=db, counts=db.table_counts(),
    )


def _materialise_source(cfg: ExperimentConfig) -> GeneratedData:
    if cfg.source == "synthetic":
        return generate(cfg.synthetic)
    from deliveryrisk.data.olist import load_olist

    if not cfg.raw_dir:
        raise ValueError("data.source: olist needs data.raw_dir pointing at the CSVs")
    return load_olist(Path(cfg.raw_dir))


# ------------------------------------------------------------------ modelling
@dataclass
class FittedModels:
    risk: object
    calibrator: Calibrator
    cause: CauseModel
    mix: dict[str, float]
    model_kind: str


def fit_models(cfg: ExperimentConfig, train: pd.DataFrame, valid: pd.DataFrame,
               kind: str | None = None) -> FittedModels:
    model_cfg = cfg.model if kind is None else type(cfg.model)(**{**asdict(cfg.model), "kind": kind})
    model = make_model(model_cfg)
    t0 = time.time()
    model.fit(train, train.y_late.to_numpy(dtype=float))
    cal = Calibrator(cfg.calibration_method).fit(
        model.predict_proba(valid), valid.y_late.to_numpy(dtype=float)
    )
    cause = CauseModel(model_cfg).fit(train)
    log.info("fitted %s + %s calibration + cause model in %.1fs", model_cfg.kind,
             cfg.calibration_method, time.time() - t0)
    return FittedModels(model, cal, cause, population_mix(train), model_cfg.kind)


def score(models: FittedModels, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw and calibrated ``P(late)``."""
    raw = models.risk.predict_proba(df)
    return raw, models.calibrator.transform(raw)


# ------------------------------------------------------------------ decisions
@dataclass
class PolicyInputs:
    df: pd.DataFrame
    truth: pd.DataFrame
    margin: np.ndarray
    late_cost: np.ndarray
    costs: dict[str, np.ndarray]
    p_late: np.ndarray
    table_cause: th.DecisionTable
    table_risk_only: th.DecisionTable


def build_policy_inputs(
    cfg: ExperimentConfig, models: FittedModels, test: pd.DataFrame, truth: pd.DataFrame,
    p_late: np.ndarray, cost_model=None,
) -> PolicyInputs:
    cost_model = cost_model or cfg.economics
    econ = order_economics(test, cost_model)
    costs = cfg.actions.per_order(test)
    joint = models.cause.predict_joint(test, p_late)
    table_cause = th.decision_table(cfg.uplift.uplifts(joint), econ.late_cost.to_numpy(), costs)
    table_risk = th.decision_table(
        cfg.uplift.risk_only_uplifts(p_late, models.mix), econ.late_cost.to_numpy(), costs
    )
    return PolicyInputs(
        df=test,
        truth=align_truth(test, truth),
        margin=econ.margin.to_numpy(),
        late_cost=econ.late_cost.to_numpy(),
        costs=costs,
        p_late=p_late,
        table_cause=table_cause,
        table_risk_only=table_risk,
    )


def build_policies(cfg: ExperimentConfig, inp: PolicyInputs, valid_scores: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Every decision rule under comparison, all reading the same expected-value table."""
    y_valid, p_valid = valid_scores
    n = len(inp.df)
    policies: dict[str, np.ndarray] = {"do_nothing": th.assign_none(n)}
    policies["nudge_everything"] = np.array(["nudge"] * n, dtype=object)
    for rate in cfg.fixed_rates:
        pct = int(round(rate * 100))
        policies[f"top_{pct}pct_nudge"] = th.assign_fixed_rate(inp.p_late, rate, "nudge")
        policies[f"top_{pct}pct_both"] = th.assign_fixed_rate(inp.p_late, rate, "both")
    tau_f1 = th.threshold_f1(y_valid, p_valid)
    # Both actions, because "optimise F1 then apply the standard intervention" is the shape of
    # the real mistake, and pairing the accuracy threshold only with the expensive action would
    # be beating a baseline nobody proposed.
    policies["f1_threshold_both"] = th.assign_threshold(inp.p_late, tau_f1, "both")
    policies["f1_threshold_nudge"] = th.assign_threshold(inp.p_late, tau_f1, "nudge")
    policies["youden_threshold_nudge"] = th.assign_threshold(
        inp.p_late, th.threshold_youden(y_valid, p_valid), "nudge"
    )
    tau = th.threshold_expected_contribution(inp.table_cause, inp.p_late)
    policies["margin_global_threshold"] = th.assign_margin_global(inp.table_cause, inp.p_late, tau)
    policies["margin_risk_only"] = th.assign_margin_per_order(inp.table_risk_only)
    policies["margin_cause_aware"] = th.assign_margin_per_order(inp.table_cause)
    unconstrained_spend = float(
        sum(inp.costs[a][policies["margin_cause_aware"] == a].sum() for a in ACTIONS)
    )
    budget = cfg.budget_share * unconstrained_spend
    policies[f"margin_budget_{int(cfg.budget_share * 100)}pct"] = allocate_under_budget(
        inp.table_cause, budget
    ).actions
    policies["oracle"] = oracle_actions(inp.truth, inp.late_cost, inp.costs)
    return policies


def run_experiment(cfg: ExperimentConfig, cache_dir: Path | None = None) -> dict[str, object]:
    """The whole comparison, returned as frames ready to be written out."""
    prep = prepare(cfg, cache_dir)
    lab = labelled(prep.features)
    train, valid, test = split_by_time(lab, cfg.train_share, cfg.valid_share)
    log.info("split: %d train / %d valid / %d test orders", len(train), len(valid), len(test))

    # The cost model's retention term is measured on the training window only.
    cost_model = cfg.economics.with_categories(prep.features.category.astype(str).unique())
    damage = estimate_review_damage(train, prep.reviews)
    if np.isfinite(damage):
        cost_model.review_damage = float(damage)

    out: dict[str, object] = {"dataset_summary": prep.summary}
    model_rows, decile_frames, policy_frames = [], [], []
    fitted: dict[str, FittedModels] = {}

    from deliveryrisk.evaluation.metrics import decile_table, ranking_metrics

    for kind in ("logistic", "gbdt"):
        models = fit_models(cfg, train, valid, kind=kind)
        fitted[kind] = models
        raw_test, cal_test = score(models, test)
        raw_valid, cal_valid = score(models, valid)
        y_test = test.y_late.to_numpy(dtype=float)
        for label, p in (("raw", raw_test), ("calibrated", cal_test)):
            row = {"model": kind, "scores": label, **ranking_metrics(y_test, p),
                   **calibration_metrics(y_test, p, cfg.calibration_bins)}
            model_rows.append(row)
        tab = decile_table(y_test, cal_test)
        tab.insert(0, "model", kind)
        decile_frames.append(tab)

        inp = build_policy_inputs(cfg, models, test, prep.truth, cal_test, cost_model)
        policies = build_policies(cfg, inp, (valid.y_late.to_numpy(dtype=float), cal_valid))
        cmp = compare_policies(policies, inp.truth, inp.margin, inp.late_cost, inp.costs,
                               seed=cfg.seed)
        cmp.insert(0, "model", kind)
        cmp.insert(1, "scores", "calibrated")
        policy_frames.append(cmp)

        if kind == "gbdt":
            out["inputs"] = inp
            out["policies"] = policies
            out["cost_model"] = cost_model
            # The same policy on uncalibrated scores: the §4 comparison.
            inp_raw = build_policy_inputs(cfg, models, test, prep.truth, raw_test, cost_model)
            pol_raw = build_policies(
                cfg, inp_raw, (valid.y_late.to_numpy(dtype=float), raw_valid)
            )
            cmp_raw = compare_policies(pol_raw, inp_raw.truth, inp_raw.margin, inp_raw.late_cost,
                                       inp_raw.costs, seed=cfg.seed)
            cmp_raw.insert(0, "model", kind)
            cmp_raw.insert(1, "scores", "raw")
            policy_frames.append(cmp_raw)
            out["budget_frontier"] = _budget_frame(cfg, inp)
            out["implied_thresholds"] = th.implied_thresholds(
                inp.table_cause, cfg.uplift.uplifts(models.cause.predict_joint(test, cal_test)),
                cal_test, "nudge"
            )

    out["model_metrics"] = pd.DataFrame(model_rows)
    out["deciles"] = pd.concat(decile_frames, ignore_index=True)
    out["policy_comparison"] = pd.concat(policy_frames, ignore_index=True)
    out["splits"] = {"train": len(train), "valid": len(valid), "test": len(test)}
    out["fitted"] = fitted
    out["test"] = test
    out["train"] = train
    out["valid"] = valid
    out["truth"] = prep.truth
    return out


def _budget_frame(cfg: ExperimentConfig, inp: PolicyInputs) -> pd.DataFrame:
    from deliveryrisk.evaluation.decisions import evaluate_policy

    unconstrained = inp.table_cause.best_action()
    full_spend = float(sum(inp.costs[a][unconstrained == a].sum() for a in ACTIONS))
    budgets = np.array(cfg.budget_grid) * full_spend
    rows = []
    for share, alloc in zip(cfg.budget_grid, budget_frontier(inp.table_cause, budgets)):
        res = evaluate_policy(f"budget_{share}", alloc.actions, inp.truth, inp.margin,
                              inp.late_cost, inp.costs, seed=cfg.seed)
        row = res.as_row()
        row["budget_share"] = share
        row["shadow_price"] = alloc.shadow_price
        rows.append(row)
    return pd.DataFrame(rows)
