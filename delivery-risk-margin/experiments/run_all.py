"""Run every experiment and regenerate ``docs/RESULTS.md`` and the figures.

    python experiments/run_all.py --config configs/headline.yaml

Stages
------
1. **dataset**      -- marginals of the generated warehouse against the published Olist
                       reference, so the calibration claim is checkable rather than asserted.
2. **leakage**      -- point-in-time features against a two-sided window, framed as train/serve
                       skew: the same fitted model scored once with features that can see the
                       future and once with the causal features production can compute. Measured
                       in ranking, in calibration, and in money.
3. **models**       -- logistic regression and gradient boosting, raw and calibrated: ranking,
                       decile capture, and reliability.
4. **policies**     -- every decision rule against the counterfactual truth, plus the spread of
                       per-order break-even thresholds that says why one global cut-off cannot
                       be right.
5. **budget**       -- contribution against spend, and the shadow price of the next unit.
6. **matched**      -- cause-aware against risk-ranked assignment at *equal spend*, which is the
                       only comparison of the two that means anything.
7. **prices**      -- the assignment at a grid of expedite prices. At the carrier's published
                       surcharge the express upgrade is worth buying for almost nobody; that is
                       a fact about the price list, not about the model, and the sweep shows
                       where the action mix turns over.
8. **sensitivity**  -- how wrong the believed uplift can be before the policy stops paying. The
                       one input no observational log identifies gets a sweep, not a footnote.

Everything is written to ``results/`` as CSV and rendered into ``docs/RESULTS.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deliveryrisk.config import ExperimentConfig  # noqa: E402
from deliveryrisk.evaluation import reporting as rep  # noqa: E402
from deliveryrisk.evaluation.decisions import evaluate_policy  # noqa: E402
from deliveryrisk.evaluation.leakage import run_leak_audit  # noqa: E402
from deliveryrisk.models.calibration import reliability_table  # noqa: E402
from deliveryrisk.models.cause import cause_labels  # noqa: E402
from deliveryrisk.pipeline import prepare, run_experiment  # noqa: E402
from deliveryrisk.policy.actions import ACTIONS, CAUSES  # noqa: E402
from deliveryrisk.policy.allocation import budget_frontier  # noqa: E402
from deliveryrisk.policy.thresholds import (  # noqa: E402
    decision_table,
    threshold_expected_contribution,
)

log = logging.getLogger("experiments")

STAGES = ("dataset", "leakage", "models", "policies", "budget", "matched", "prices",
          "sensitivity")


def _write(df: pd.DataFrame, out: Path, name: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    log.info("wrote %s (%d rows)", out / f"{name}.csv", len(df))
    return df


def _read(out: Path, name: str) -> pd.DataFrame | None:
    p = out / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------------- stages
def stage_dataset(cfg: ExperimentConfig, res: dict, out: Path) -> pd.DataFrame:
    summary = res["dataset_summary"]
    ref = cfg.calibration_reference
    rows = []
    for k in sorted(set(summary) | set(ref)):
        gen, r = summary.get(k), ref.get(k)
        rows.append(
            {
                "statistic": k,
                "reference": r if r is not None else float("nan"),
                "generated": gen if gen is not None else float("nan"),
                "ratio": (gen / r) if (r not in (None, 0) and gen is not None) else float("nan"),
            }
        )
    return _write(pd.DataFrame(rows), out, "dataset_calibration")


def stage_leakage(cfg: ExperimentConfig, cache: Path | None, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = prepare(cfg, cache)
    audit, breakdown = run_leak_audit(prep, cfg)
    return _write(audit, out, "leakage"), _write(breakdown, out, "leakage_by_history")


def stage_models(res: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics = _write(res["model_metrics"], out, "model_metrics")
    deciles = _write(res["deciles"], out, "deciles")
    test = res["test"]
    y = test.y_late.to_numpy(dtype=float)
    models = res["fitted"]["gbdt"]
    raw = models.risk.predict_proba(test)
    curves = {
        "raw": reliability_table(y, raw, 20),
        "calibrated": reliability_table(y, models.calibrator.transform(raw), 20),
    }
    for label, tab in curves.items():
        _write(tab, out, f"reliability_{label}")
    return metrics, deciles, curves


def stage_policies(cfg: ExperimentConfig, res: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = _write(res["policy_comparison"], out, "policy_comparison")
    inp = res["inputs"]
    tau = threshold_expected_contribution(inp.table_cause, inp.p_late)
    implied = res["implied_thresholds"]
    finite = implied[np.isfinite(implied) & (implied > 0) & (implied < 1)]
    spread = pd.DataFrame(
        {
            "quantile": [0.05, 0.25, 0.5, 0.75, 0.95],
            "break_even_p": [float(np.quantile(finite, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
        }
    )
    spread["best_single_threshold"] = tau
    spread["ratio_to_single"] = spread.break_even_p / tau
    _write(spread, out, "threshold_spread")

    # Cause mix by risk decile, and what each policy spends on.
    test = res["test"]
    lab = cause_labels(test)
    d = pd.DataFrame({"p": inp.p_late, "cause": lab.to_numpy()})
    d["decile"] = np.ceil(d.p.rank(method="first", ascending=False) / (len(d) / 10)).clip(1, 10)
    mix = (
        d.dropna(subset=["cause"])
        .pivot_table(index="decile", columns="cause", values="p", aggfunc="size")
        .fillna(0)
    )
    mix = mix.div(mix.sum(axis=1), axis=0).reset_index()
    for c in CAUSES:
        if c not in mix.columns:
            mix[c] = 0.0
    _write(mix, out, "cause_mix_by_decile")
    acts = pd.DataFrame(
        {
            "action": list(ACTIONS),
            "risk_only": [int((res["policies"]["margin_risk_only"] == a).sum()) for a in ACTIONS],
            "cause_aware": [int((res["policies"]["margin_cause_aware"] == a).sum()) for a in ACTIONS],
        }
    )
    _write(acts, out, "policy_action_counts")
    return comparison, mix


def stage_budget(res: dict, out: Path) -> pd.DataFrame:
    return _write(res["budget_frontier"], out, "budget_frontier")


def stage_matched(cfg: ExperimentConfig, res: dict, out: Path) -> pd.DataFrame:
    """Cause-aware against risk-ranked at equal spend.

    The unconstrained comparison is not clean: the two rules choose different treated volumes, so
    the winner could simply be the one that spends more. Sweeping both under the same budget
    removes that objection, and the gap that survives is the value of knowing *why* an order is
    at risk rather than only how much.
    """
    inp = res["inputs"]
    full = float(sum(inp.costs[a][inp.table_cause.best_action() == a].sum() for a in ACTIONS))
    grid = np.array(cfg.budget_grid)
    rows = []
    for label, table in (("cause_aware", inp.table_cause), ("risk_only", inp.table_risk_only)):
        for share, alloc in zip(grid, budget_frontier(table, grid * full)):
            res_i = evaluate_policy(label, alloc.actions, inp.truth, inp.margin, inp.late_cost,
                                    inp.costs, seed=cfg.seed)
            rows.append({"policy": label, "budget_share": float(share),
                         "spend_per_1k": res_i.spend_per_1k, "delta_per_1k": res_i.delta_per_1k,
                         "treated_share": res_i.treated_share, "roi": res_i.roi,
                         "late_rate": res_i.late_rate})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="budget_share", columns="policy", values="delta_per_1k")
    wide["uplift_pct"] = 100.0 * (wide.cause_aware / wide.risk_only - 1.0)
    wide = wide.reset_index()
    _write(wide, out, "matched_spend")
    return _write(df, out, "matched_spend_long")


def stage_prices(cfg: ExperimentConfig, res: dict, out: Path,
                 multipliers=(0.25, 0.4, 0.55, 0.7, 1.0, 1.4)) -> pd.DataFrame:
    """Re-run the assignment with the express upgrade repriced.

    The decision layer is a function of prices as much as of probabilities, and one of its
    prices -- what a carrier charges to upgrade a parcel on the day -- is a commercial term
    rather than a model output. Sweeping it answers the question the unconstrained result
    provokes: at what price does expediting start to be worth buying, and does knowing *why* an
    order is at risk matter more once there is a real choice of action to make?
    """
    from deliveryrisk.policy.actions import ActionCosts, UpliftBelief  # noqa: F401

    inp = res["inputs"]
    models = res["fitted"]["gbdt"]
    joint = models.cause.predict_joint(inp.df, inp.p_late)
    rows = []
    for mult in multipliers:
        costs = ActionCosts(
            nudge_cost=cfg.actions.nudge_cost,
            expedite_multiplier=mult,
            expedite_weight_rate=cfg.actions.expedite_weight_rate,
        ).per_order(inp.df)
        tables = {
            "margin_cause_aware": decision_table(cfg.uplift.uplifts(joint), inp.late_cost, costs),
            "margin_risk_only": decision_table(
                cfg.uplift.risk_only_uplifts(inp.p_late, models.mix), inp.late_cost, costs
            ),
        }
        for name, table in tables.items():
            actions = table.best_action()
            r = evaluate_policy(name, actions, inp.truth, inp.margin, inp.late_cost, costs,
                                seed=cfg.seed)
            row = r.as_row()
            row["expedite_multiplier"] = mult
            row["mean_expedite_cost"] = float(np.mean(costs["expedite"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["expedite_multiplier", "mean_expedite_cost", "policy", "treated_share",
            "spend_per_1k", "n_nudge", "n_expedite", "n_both", "delta_per_1k", "roi"]
    return _write(df[cols], out, "expedite_prices")


def stage_sensitivity(cfg: ExperimentConfig, res: dict, out: Path,
                      scales=(0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)) -> pd.DataFrame:
    """Sweep the believed uplift against the true one.

    Only the *belief* moves: the physics of the world, and therefore the realised outcome of
    every action, are held fixed. A policy that only works when its assumed uplift happens to be
    exactly right is not a policy, it is a coincidence.
    """
    from deliveryrisk.policy.actions import UpliftBelief

    inp = res["inputs"]
    models = res["fitted"]["gbdt"]
    joint = models.cause.predict_joint(inp.df, inp.p_late)
    rows = []
    for scale in scales:
        belief = UpliftBelief(rho_nudge=cfg.uplift.rho_nudge * scale,
                              rho_expedite=cfg.uplift.rho_expedite * scale)
        tables = {
            "margin_cause_aware": decision_table(belief.uplifts(joint), inp.late_cost, inp.costs),
            "margin_risk_only": decision_table(
                belief.risk_only_uplifts(inp.p_late, models.mix), inp.late_cost, inp.costs
            ),
        }
        for name, table in tables.items():
            actions = table.best_action()
            r = evaluate_policy(name, actions, inp.truth, inp.margin, inp.late_cost, inp.costs,
                                seed=cfg.seed)
            rows.append({"policy": name, "believed_rho_scale": scale,
                         "rho_nudge": belief.rho_nudge, "rho_expedite": belief.rho_expedite,
                         "treated_share": r.treated_share, "spend_per_1k": r.spend_per_1k,
                         "delta_per_1k": r.delta_per_1k, "roi": r.roi})
    return _write(pd.DataFrame(rows), out, "uplift_sensitivity")


# --------------------------------------------------------------------------- report
def render_report(cfg: ExperimentConfig, out: Path, docs: Path, figures: dict[str, Path]) -> Path:
    def tab(name, columns=None, **kw):
        """Render one result CSV. `columns` trims the wide frames to what a reader needs --
        the full frame is on disk in `results/` for anyone who wants the rest."""
        df = _read(out, name)
        if df is None:
            return "_(not run)_"
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return rep.markdown_table(df, **kw)

    policy = _read(out, "policy_comparison")
    headline = ""
    if policy is not None:
        g = policy[(policy.model == "gbdt") & (policy.scores == "calibrated")]
        g = g.set_index("policy").delta_per_1k
        headline = (
            f"Margin-optimised cause-aware assignment is worth **{g.get('margin_cause_aware', float('nan')):.0f}** "
            f"per 1,000 orders against doing nothing; the top-decile heuristic is worth "
            f"**{g.get('top_10pct_nudge', float('nan')):.0f}**; the F1-optimal threshold is worth "
            f"**{g.get('f1_threshold_both', float('nan')):.0f}**."
        )

    md = f"""# Results

> Generated by `python experiments/run_all.py`. Every table and figure below is written by that
> script from `results/*.csv` — nothing here is transcribed by hand, so a config change that
> moves a number moves it everywhere.

Configuration: `{cfg.name}` — {cfg.description.strip()}

{headline}

## 1. Is the generated warehouse the right shape?

Marginals of the generated nine tables against the published Olist reference. `span_days` and
`mean_promise_days` are the deliberate departures — see `docs/METHODOLOGY.md` §2.

{tab("dataset_calibration")}

## 2. What a two-sided aggregate manufactures

Every row scores the same held-out orders. Each "as served" row uses the **same fitted model** as
the "as reported offline" row above it, and differs only in whether the features it is fed can
see the future.

{tab("leakage", ["scenario", "trained_on", "scored_with", "pr_auc", "capture_top2",
                 "recall_at_10", "ece", "calibration_slope", "delta_per_1k", "treated_share"])}

![leakage](figures/leakage.png)

The leak is not spread evenly. It is largest exactly where an honest builder has nothing to work
with — a seller's first orders:

{tab("leakage_by_history", ["scenario", "bucket", "n", "roc_auc", "pr_auc", "capture_top2"])}

## 3. The risk models

{tab("model_metrics", ["model", "scores", "roc_auc", "pr_auc", "capture_top1", "capture_top2",
                       "capture_top3", "lift_top1", "brier", "ece", "mce",
                       "calibration_slope", "mean_predicted", "base_rate"])}

![decile capture](figures/decile_capture.png)

{tab("deciles", ["model", "decile", "n", "n_late", "late_rate", "lift", "capture",
                 "cum_capture"])}

## 4. Calibration, and why the threshold cares

![calibration](figures/calibration.png)

## 5. What each decision rule is worth

Realised contribution against the counterfactual truth, per 1,000 orders, with a 95 % bootstrap
interval over orders.

{tab("policy_comparison", ["model", "scores", "policy", "treated_share", "spend_per_1k",
                           "late_rate", "late_rate_change_pct", "contribution_per_1k",
                           "delta_per_1k", "delta_lo", "delta_hi", "roi", "n_nudge",
                           "n_expedite", "n_both"], floatfmt="{:.2f}")}

![policy value](figures/policy_value.png)

### The threshold is a property of the order

`c / (u · L)` — the break-even probability for a nudge — computed per order:

{tab("threshold_spread")}

![threshold spread](figures/threshold_spread.png)

## 6. Knowing *why* an order is at risk

{tab("cause_mix_by_decile")}

![cause mix](figures/cause_mix.png)

At equal spend, cause-aware assignment against risk-ranked assignment:

{tab("matched_spend", floatfmt="{:.2f}")}

{tab("policy_action_counts")}

### The express upgrade is a price, not a model output

{tab("expedite_prices", floatfmt="{:.2f}")}

## 7. Budget

{tab("budget_frontier", ["budget_share", "treated_share", "spend_per_1k", "late_rate",
                         "delta_per_1k", "delta_lo", "delta_hi", "roi", "shadow_price"],
     floatfmt="{:.3f}")}

![budget frontier](figures/budget_frontier.png)

## 8. How wrong can the uplift belief be?

The physics are fixed; only the belief moves.

{tab("uplift_sensitivity", ["policy", "believed_rho_scale", "rho_nudge", "rho_expedite",
                            "treated_share", "spend_per_1k", "delta_per_1k", "roi"],
     floatfmt="{:.3f}")}

![uplift sensitivity](figures/uplift_sensitivity.png)
"""
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / "RESULTS.md"
    path.write_text(md)
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/headline.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--only", default=None,
                    help=f"comma-separated subset of: {','.join(STAGES)}. "
                         "Pass an empty string to re-render docs/RESULTS.md from existing CSVs.")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)-7s %(name)-26s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = ExperimentConfig.load(args.config)
    out = Path(args.out_dir or cfg.output_dir)
    docs = Path(args.docs_dir)
    figures_dir = docs / "figures"
    cache = Path(args.cache_dir)
    # `--only ""` means "no stages, just re-render the report"; omitting it entirely means "all".
    stages = (
        STAGES if args.only is None
        else tuple(s.strip() for s in args.only.split(",") if s.strip())
    )

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

    figures: dict[str, Path] = {}
    cause_fig = None
    needs_run = {"dataset", "models", "policies", "budget", "matched", "sensitivity"} & set(stages)
    res = run_experiment(cfg, cache) if needs_run else None

    if "dataset" in stages:
        stage_dataset(cfg, res, out)
    if "leakage" in stages:
        audit, _ = stage_leakage(cfg, cache, out)
        figures["leakage"] = rep.plot_leakage(audit, figures_dir / "leakage.png")
    if "models" in stages:
        _, deciles, curves = stage_models(res, out)
        figures["deciles"] = rep.plot_decile_capture(deciles, figures_dir / "decile_capture.png")
        figures["calibration"] = rep.plot_calibration(curves, figures_dir / "calibration.png")
    if "policies" in stages:
        comparison, mix = stage_policies(cfg, res, out)
        figures["policy"] = rep.plot_policy_value(comparison, figures_dir / "policy_value.png")
        cause_fig = (mix, figures_dir / "cause_mix.png")
        tau = threshold_expected_contribution(res["inputs"].table_cause, res["inputs"].p_late)
        figures["threshold"] = rep.plot_threshold_spread(
            res["implied_thresholds"], tau, figures_dir / "threshold_spread.png"
        )
    if "budget" in stages:
        frontier = stage_budget(res, out)
        figures["budget"] = rep.plot_budget_frontier(frontier, figures_dir / "budget_frontier.png")
    if "matched" in stages:
        stage_matched(cfg, res, out)
    prices = stage_prices(cfg, res, out) if "prices" in stages else _read(out, "expedite_prices")
    if cause_fig is not None:
        figures["cause"] = rep.plot_cause_mix(cause_fig[0], prices, cause_fig[1])
    if "sensitivity" in stages:
        sens = stage_sensitivity(cfg, res, out)
        figures["sensitivity"] = rep.plot_uplift_sensitivity(
            sens, figures_dir / "uplift_sensitivity.png"
        )

    render_report(cfg, out, docs, figures)
    log.info("all stages finished in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
