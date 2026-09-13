"""The closed-loop retraining simulator.

One cycle
---------
1. The deployed model scores the cycle's transactions.
2. The policy declines the top ``decline_budget`` by score, and approves a small randomised
   slice of those declines if the arm explores.
3. Approved transactions eventually charge back or do not: **only they yield labels.**
   Declined-and-unexplored transactions leave the system with no outcome, ever.
4. Whatever labels have matured go into the training pool, weighted by the arm's rule.
5. A fresh model is fit on a trailing window of that pool and deployed for the next cycle.

Step 3 is the entire experiment. The training pool's positive class stops being "fraud" and
becomes "fraud the previous model failed to catch" -- the strongest, most obvious fraud
signatures are systematically the ones that never come back labelled. Retrain on that long
enough and the model unlearns the patterns it was best at, which shows up as decaying recall
against ground truth while every *observable* metric on approved traffic looks fine. That is
what makes this failure mode dangerous in production: the dashboards do not move.

Evaluation is always against oracle labels on the full cycle -- including transactions the
policy declined. No deployed system can compute this; the simulator can, which is the point.
"""
from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clfraud.bias.propensity import PropensityModel, overlap_diagnostics
from clfraud.bias.reweighting import WeightConfig, build_weights, effective_sample_size
from clfraud.evaluation.metrics import (
    BusinessCosts,
    business_outcome,
    ranking_metrics,
)
from clfraud.models.base import ModelConfig
from clfraud.models.calibration import brier_score, expected_calibration_error
from clfraud.models.gbdt import make_scorer
from clfraud.simulator.labeling import LabelPipeline, LabelPipelineConfig
from clfraud.simulator.policy import ExplorationConfig, OracleAccessPolicy, ThresholdPolicy

log = logging.getLogger(__name__)
DAY = 86_400.0


@dataclass
class LoopConfig:
    """Timing and capacity of the closed loop."""

    cycle_days: float = 30.0
    n_cycles: int = 18
    #: Cycles before any model is deployed. Traffic is approved wholesale, so labels are
    #: complete: this is the pre-deployment bootstrap every arm starts from, identically.
    warmup_cycles: int = 2
    #: Hard trailing window, in cycles, of matured labels used for each retrain.
    train_window_cycles: int = 12
    #: Exponential recency weighting half-life, in cycles. When set (the default), training
    #: rows are weighted ``0.5 ** (age_in_cycles / half_life)`` on top of any bias-correction
    #: weight, and ``train_window_cycles`` acts only as a memory bound.
    #:
    #: This matters more than it looks. A hard window makes the closed loop *oscillate*: the
    #: model blocks a fraud pattern, its labels vanish, the pattern falls out of the window in
    #: one step, the model forgets it wholesale, fraud floods back, labels return, it relearns.
    #: Real retraining pipelines decay old data smoothly instead, which turns that sawtooth into
    #: the steady erosion this project measures. Both are available; the ablation reports both.
    recency_half_life_cycles: float | None = 3.0
    #: Trailing cycles of *exploration* labels retained. Exploration data is scarce, expensive
    #: and the only unbiased view of the decline region, so it is kept longer than ordinary
    #: policy-selected data. ``None`` means "same window as everything else".
    explore_window_cycles: int | None = 12
    decline_budget: float = 0.03
    seed: int = 4242
    label: LabelPipelineConfig = field(default_factory=LabelPipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    costs: BusinessCosts = field(default_factory=BusinessCosts)
    #: Cap on training rows per retrain (uniformly subsampled). Keeps arms comparable in
    #: compute; set to None to use everything.
    max_train_rows: int | None = 250_000


@dataclass
class PolicyArm:
    """One experimental condition."""

    name: str
    exploration: ExplorationConfig = field(default_factory=lambda: ExplorationConfig(mode="none", rate=0.0))
    weights: WeightConfig = field(default_factory=lambda: WeightConfig(scheme="none"))
    #: Full-label oracle: labels return even for declines. Physically impossible; upper bound.
    oracle_labels: bool = False
    #: Freeze the warm-up model and never retrain. Isolates drift from the feedback loop.
    retrain: bool = True
    #: Estimate e(x) from features instead of reading the logged propensity.
    estimate_propensity: bool = False
    description: str = ""


@dataclass
class ArmResult:
    """Per-cycle records for one arm."""

    name: str
    cycles: pd.DataFrame
    importances: pd.DataFrame
    train_composition: pd.DataFrame
    runtime_s: float

    def recall_series(self) -> pd.Series:
        return self.cycles.set_index("cycle")["recall_at_budget"]


class ClosedLoopSimulator:
    """Runs one or more policy arms over a shared, pre-computed feature matrix."""

    def __init__(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
        config: LoopConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features:
            Point-in-time feature matrix, one row per transaction, already time-ordered.
        meta:
            Aligned frame with ``ts``, ``amount``, ``is_fraud`` and optionally ``archetype``.
            Feature construction is policy-independent -- an issuer computes features for
            declined transactions too -- so the matrix is built once and shared across arms.
        """
        if len(features) != len(meta):
            raise ValueError("features and meta must be aligned")
        self.cfg = config or LoopConfig()
        self.X = features.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True)
        self.feature_names = list(self.X.columns)

        t0 = float(self.meta["ts"].min())
        cycle = ((self.meta["ts"].to_numpy() - t0) // (self.cfg.cycle_days * DAY)).astype(int)
        self.cycle = np.clip(cycle, 0, self.cfg.n_cycles - 1)
        self.t0 = t0
        self.y = self.meta["is_fraud"].to_numpy().astype(int)
        self.amount = self.meta["amount"].to_numpy(dtype="float64")
        self.ts = self.meta["ts"].to_numpy(dtype="float64")
        self._cycle_idx = {c: np.flatnonzero(self.cycle == c) for c in range(self.cfg.n_cycles)}
        self.labeller = LabelPipeline(self.cfg.label)

    # ------------------------------------------------------------------ helpers
    def cycle_end_time(self, c: int) -> float:
        return self.t0 + (c + 1) * self.cfg.cycle_days * DAY

    def _fit(self, rows: np.ndarray, y_obs: np.ndarray, w: np.ndarray | None, seed: int):
        model = make_scorer(self.cfg.model)
        X = self.X.iloc[rows]
        model.fit(X, y_obs, sample_weight=w)
        return model

    def _subsample(self, rows, y_obs, w, rng):
        cap = self.cfg.max_train_rows
        if cap is None or len(rows) <= cap:
            return rows, y_obs, w
        # Keep every positive; subsample negatives. Positives are the scarce resource and the
        # thing censoring destroys, so throwing them away would confound the experiment.
        pos = np.flatnonzero(y_obs == 1)
        neg = np.flatnonzero(y_obs == 0)
        take = max(cap - pos.size, 1000)
        keep_neg = rng.choice(neg, size=min(take, neg.size), replace=False)
        sel = np.sort(np.concatenate([pos, keep_neg]))
        return rows[sel], y_obs[sel], (None if w is None else w[sel])

    # ------------------------------------------------------------------ main loop
    def run_arm(self, arm: PolicyArm) -> ArmResult:
        cfg = self.cfg
        started = time.time()
        # Each arm gets its own exploration stream, derived from the arm name so that adding or
        # reordering arms never perturbs another arm's draws. crc32 rather than hash(): Python
        # salts string hashing per process, so hash() would make results irreproducible across
        # runs -- and the in-process reproducibility test would never catch it.
        rng = np.random.default_rng(cfg.seed + zlib.crc32(arm.name.encode()) % 100_000)

        policy_cls = OracleAccessPolicy if arm.oracle_labels else ThresholdPolicy
        policy = policy_cls(decline_budget=cfg.decline_budget, exploration=arm.exploration)

        # ---- warm-up: traffic approved wholesale, so labels are complete and unbiased.
        warm_rows = np.concatenate([self._cycle_idx[c] for c in range(cfg.warmup_cycles)])
        y_warm = self.labeller.observed_labels(self.y[warm_rows])
        model = self._fit(warm_rows, y_warm, None, cfg.seed)
        log.info("[%s] warm-up on %d rows (%d positive)", arm.name, len(warm_rows), int(y_warm.sum()))

        # Pool of labelled evidence: row index, observed label, propensity, timestamp, cycle.
        pool_rows = list(warm_rows)
        pool_y = list(y_warm)
        pool_e = [1.0] * len(warm_rows)
        pool_cycle = list(self.cycle[warm_rows])
        # Approval decisions, kept separately from the label pool: the propensity model's target
        # is "was this approved", which is known for every transaction the moment it is decided.
        # Whether its *label* has matured yet is a different question entirely.
        approved_mask = np.zeros(len(self.X), dtype=bool)
        approved_mask[warm_rows] = True

        policy.calibrate(model.predict_proba(self.X.iloc[warm_rows]))

        records: list[dict] = []
        importances: dict[int, pd.Series] = {}
        composition: list[dict] = []

        for c in range(cfg.warmup_cycles, cfg.n_cycles):
            rows = self._cycle_idx[c]
            if rows.size == 0:
                continue
            Xc = self.X.iloc[rows]
            scores = model.predict_proba(Xc)
            y_true_c = self.y[rows]
            amt_c = self.amount[rows]

            decision = policy.decide(scores, amt_c, rng)

            # ---- evaluation against oracle labels on the *whole* cycle, declines included.
            rec = {"cycle": c, "arm": arm.name}
            rec |= ranking_metrics(y_true_c, scores, cfg.decline_budget, amounts=amt_c)
            rec |= business_outcome(
                y_true_c, amt_c, decision.approved, decision.explored, cfg.costs
            )
            rec |= decision.summary()
            rec["ece"] = expected_calibration_error(y_true_c, scores)
            rec["brier"] = brier_score(y_true_c, scores)

            # ---- labels: only approved transactions ever come back (unless oracle arm).
            approved_mask[rows] = decision.approved
            observed_mask = np.ones(rows.size, dtype=bool) if arm.oracle_labels else decision.approved
            obs_rows = rows[observed_mask]
            y_obs = self.labeller.observed_labels(self.y[obs_rows])
            e_obs = decision.propensity[observed_mask]
            pool_rows.extend(obs_rows.tolist())
            pool_y.extend(y_obs.tolist())
            pool_e.extend(e_obs.tolist())
            pool_cycle.extend([c] * obs_rows.size)

            rec["labels_gained"] = float(obs_rows.size)
            rec["label_censoring_rate"] = 1.0 - obs_rows.size / rows.size
            rec["fraud_labels_gained"] = float(y_obs.sum())
            rec["fraud_labels_censored"] = float(y_true_c.sum() - self.y[obs_rows].sum())

            # ---- retrain on matured labels inside the trailing window.
            if arm.retrain:
                as_of = self.cycle_end_time(c)
                pr = np.asarray(pool_rows)
                py = np.asarray(pool_y)
                pe = np.asarray(pool_e, dtype="float64")
                pc = np.asarray(pool_cycle)

                mature = self.labeller.is_mature(self.ts[pr], as_of)
                in_window = pc > c - cfg.train_window_cycles
                if cfg.explore_window_cycles is not None:
                    # Rows logged with propensity < 1 are exploration draws; give them the
                    # longer memory. Ordinary approvals keep the standard window.
                    is_explore = pe < 1.0
                    in_window = in_window | (
                        is_explore & (pc > c - cfg.explore_window_cycles)
                    )
                sel = mature & in_window
                tr_rows, tr_y, tr_e = pr[sel], py[sel], pe[sel]

                if tr_rows.size >= 500 and tr_y.sum() >= 20:
                    if arm.estimate_propensity:
                        tr_e = self._estimated_propensity(c, tr_rows, approved_mask)
                    w = (
                        None
                        if arm.weights.scheme == "none"
                        else build_weights(tr_e, tr_y, arm.weights)
                    )
                    if cfg.recency_half_life_cycles:
                        age = np.maximum(c - pc[sel], 0.0)
                        decay = 0.5 ** (age / cfg.recency_half_life_cycles)
                        # Recency weighting is a staleness control, not a bias correction; it
                        # applies to every arm, including the unweighted ones, so the arms
                        # differ only in whether they correct for *selection*.
                        w = decay if w is None else w * decay
                    tr_rows, tr_y, w = self._subsample(tr_rows, tr_y, w, rng)
                    model = self._fit(tr_rows, tr_y, w, cfg.seed + c)

                    rec["train_rows"] = float(tr_rows.size)
                    rec["train_positives"] = float(tr_y.sum())
                    rec["train_pos_rate"] = float(tr_y.mean())
                    rec["train_ess"] = (
                        effective_sample_size(w) if w is not None else float(tr_rows.size)
                    )
                    rec["train_ess_ratio"] = rec["train_ess"] / max(tr_rows.size, 1)
                    diag = overlap_diagnostics(tr_e, np.ones(tr_e.size, dtype=bool))
                    rec |= {f"ovl_{k}": v for k, v in diag.items()}
                    composition.append(
                        self._composition(arm.name, c, tr_rows, tr_y)
                    )
                    imp = model.feature_importance()
                    if not imp.empty:
                        importances[c] = imp
                else:
                    log.warning(
                        "[%s] cycle %d: insufficient labels (%d rows, %d pos); keeping model",
                        arm.name, c, tr_rows.size, int(tr_y.sum()),
                    )
                    rec["train_rows"] = float(tr_rows.size)
                    rec["train_positives"] = float(tr_y.sum())

            # Next cycle's threshold is calibrated on this cycle's scores: an issuer can only
            # set today's cut-off from yesterday's distribution.
            policy.calibrate(scores)
            records.append(rec)

        cycles = pd.DataFrame(records)
        imp_df = (
            pd.DataFrame(importances).T.rename_axis("cycle") if importances else pd.DataFrame()
        )
        comp_df = pd.DataFrame(composition)
        runtime = time.time() - started
        log.info("[%s] done in %.1fs", arm.name, runtime)
        return ArmResult(arm.name, cycles, imp_df, comp_df, runtime)

    # ------------------------------------------------------------------ diagnostics
    def _composition(self, arm: str, c: int, rows: np.ndarray, y_obs: np.ndarray) -> dict:
        """What the training set's positive class is *made of* this cycle.

        The clearest picture of the failure: as the loop tightens, the archetype mix of observed
        positives shifts away from whatever the model catches best. The model then retrains on a
        positive class that no longer represents fraud, only fraud-that-slipped-through.
        """
        out = {"arm": arm, "cycle": c, "n": float(rows.size), "n_pos": float(y_obs.sum())}
        if "archetype" in self.meta.columns:
            arche = self.meta["archetype"].to_numpy()
            pos_arche = arche[rows[y_obs == 1]]
            if pos_arche.size:
                vals, counts = np.unique(pos_arche, return_counts=True)
                for v, k in zip(vals, counts):
                    out[f"pos_{v}"] = float(k / pos_arche.size)
        out["mean_amount_pos"] = (
            float(self.amount[rows[y_obs == 1]].mean()) if (y_obs == 1).any() else float("nan")
        )
        return out

    def _estimated_propensity(
        self, c: int, tr_rows: np.ndarray, approved_mask: np.ndarray
    ) -> np.ndarray:
        """Fit e(x) = P(approved | x) on the recent full population, declines included.

        The training target here is the *approval decision*, which the issuer knows for every
        transaction it ever saw -- unlike the fraud label, which only comes back for approvals.
        That asymmetry is what makes the propensity estimable at all.

        An earlier version trained on "is this row in the training pool", which conflates the
        policy with label maturation: the most recent cycle's approvals have no chargeback
        outcome yet, so the model would have learned "recent means unobserved" and produced
        propensities that describe the calendar rather than the decision rule.
        """
        hist = np.concatenate([self._cycle_idx[k] for k in range(max(c - 2, 0), c + 1)])
        pm = PropensityModel().fit(self.X.iloc[hist], approved_mask[hist])
        return pm.predict(self.X.iloc[tr_rows])

    # ------------------------------------------------------------------ batch
    def run(self, arms: list[PolicyArm]) -> dict[str, ArmResult]:
        results: dict[str, ArmResult] = {}
        for arm in arms:
            log.info("running arm %-22s %s", arm.name, arm.description)
            results[arm.name] = self.run_arm(arm)
        return results
