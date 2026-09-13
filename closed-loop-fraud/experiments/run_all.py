"""Run every experiment and regenerate ``docs/RESULTS.md`` and the figures.

    python experiments/run_all.py --config configs/headline.yaml

Stages
------
1. **Dataset audit** -- marginals of the generated stream against IEEE-CIS's own, so the
   calibration claim is checkable rather than asserted.
2. **Leakage audit** -- point-in-time features vs. a two-sided velocity window, framed as
   train/serve skew: the same fitted model scored once with features that can see the future and
   once with the causal features a production system can actually compute. The gap is the part
   of the offline number that was never real.
3. **Closed loop** -- every arm in the config, evaluated against oracle labels each cycle.
4. **Exploration frontier** -- recall recovered vs. fraud dollars bought, across designs and
   rates. This is the trade-off the headline number is one point on.
5. **Off-policy evaluation** -- check IPS/SNIPS against the oracle truth they are estimating,
   which is only possible here because the simulator knows the counterfactual.
6. **Sensitivity** -- how the damage scales with retraining aggressiveness. A single number for
   "how bad is the feedback loop" would be a number about one configuration, not about the
   phenomenon.

Everything is written to ``results/`` as CSV and rendered into ``docs/RESULTS.md``; no figure in
the documentation is transcribed by hand.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clfraud.bias.reweighting import WeightConfig  # noqa: E402
from clfraud.config import ExperimentConfig  # noqa: E402
from clfraud.evaluation.counterfactual import policy_value_from_logs  # noqa: E402
from clfraud.evaluation.leakage import run_leak_audit  # noqa: E402
from clfraud.evaluation.reporting import (  # noqa: E402
    combine,
    markdown_table,
    plot_cost_frontier,
    plot_leakage_comparison,
    plot_positive_composition,
    plot_recall_trajectories,
    plot_training_positives,
    summarise_arms,
)
from clfraud.pipeline import dataset_summary, prepare  # noqa: E402
from clfraud.simulator.loop import ClosedLoopSimulator, PolicyArm  # noqa: E402
from clfraud.simulator.policy import ExplorationConfig  # noqa: E402

log = logging.getLogger("experiments")

# IEEE-CIS train_transaction.csv, for the calibration table.
IEEE_REFERENCE = {
    "n_transactions": 590_540.0,
    "fraud_rate": 0.03499,
    "n_cards": 13_553.0,
    "span_days": 181.8,
    "amount_median": 68.77,
    "amount_mean": 135.03,
    "amount_p99": 1_200.0,
}


def _cached(path: Path, force: bool) -> pd.DataFrame | None:
    """Reuse a stage's output unless asked not to.

    Stages have very different costs -- the leakage audit is four model fits, the sensitivity
    sweep is a dozen full closed loops -- so they are individually resumable. `--force` or
    deleting the CSV re-runs one.
    """
    if not force and path.exists():
        log.info("reusing %s", path)
        return pd.read_csv(path)
    return None


def stage_dataset(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    got = dataset_summary(df)
    rows = []
    for k, ref in IEEE_REFERENCE.items():
        mine = got.get(k, float("nan"))
        rows.append(
            {
                "statistic": k,
                "ieee_cis": ref,
                "generated": mine,
                "ratio": mine / ref if ref else float("nan"),
            }
        )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / "dataset_calibration.csv", index=False)
    return tbl


def stage_leakage(
    df: pd.DataFrame, X: pd.DataFrame, out_dir: Path, budget: float,
    docs_dir: Path, force: bool = False,
) -> pd.DataFrame:
    cached = _cached(out_dir / "leakage_audit.csv", force)
    if cached is not None:
        plot_leakage_comparison(cached, docs_dir / "figures/leakage.png")
        return cached
    audit = run_leak_audit(df, X, budget=budget)
    audit.to_csv(out_dir / "leakage_audit.csv", index=False)
    plot_leakage_comparison(audit, docs_dir / "figures/leakage.png")
    return audit


def stage_loop(cfg: ExperimentConfig, X, meta, out_dir: Path, docs_dir: Path, force: bool = False):
    cached = _cached(out_dir / "cycles.csv", force)
    if cached is not None:
        # A cached stage still regenerates its figures. Skipping them was how a throwaway smoke
        # run left three of the committed plots showing smoke data while the tables beside them
        # showed the real run -- the kind of inconsistency nobody notices by reading.
        summary = summarise_arms(cached, cfg.baseline_arm, cfg.oracle_arm)
        _plot_loop_figures(cached, out_dir, docs_dir, cfg.loop.decline_budget)
        return None, cached, summary
    sim = ClosedLoopSimulator(X, meta, cfg.loop)
    results = sim.run(cfg.arms)
    cycles = combine(results)
    cycles.to_csv(out_dir / "cycles.csv", index=False)

    summary = summarise_arms(cycles, cfg.baseline_arm, cfg.oracle_arm)
    summary.to_csv(out_dir / "summary.csv")

    comp = pd.concat(
        [r.train_composition for r in results.values() if not r.train_composition.empty],
        ignore_index=True,
    )
    if not comp.empty:
        comp.to_csv(out_dir / "train_composition.csv", index=False)

    imp = pd.concat(
        [r.importances.assign(arm=name) for name, r in results.items() if not r.importances.empty]
    )
    if not imp.empty:
        imp.to_csv(out_dir / "feature_importance.csv")

    _plot_loop_figures(cycles, out_dir, docs_dir, cfg.loop.decline_budget)
    return results, cycles, summary


def _plot_loop_figures(cycles: pd.DataFrame, out_dir: Path, docs_dir: Path, budget: float) -> None:
    """Every figure the closed-loop stage owns, rebuilt from the CSVs on disk."""
    plot_recall_trajectories(
        cycles, docs_dir / "figures/recall_trajectories.png", budget=budget
    )
    plot_training_positives(cycles, docs_dir / "figures/training_supply.png")
    comp_path = out_dir / "train_composition.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        plot_positive_composition(
            comp, docs_dir / "figures/positive_composition.png",
            arms=[a for a in ("oracle", "naive", "explore_ipw") if a in set(comp["arm"])],
        )


def stage_frontier(
    cfg: ExperimentConfig, X, meta, out_dir: Path, docs_dir: Path, force: bool = False
) -> pd.DataFrame:
    """Recall recovered vs. fraud dollars bought, for each exploration design and rate.

    Run on its own (smaller) configuration: the frontier needs a dozen full closed loops and the
    *ordering* of exploration designs is what matters here, which is stable across scale.
    """
    cached = _cached(out_dir / "exploration_frontier.csv", force)
    if cached is not None:
        plot_cost_frontier(cached, docs_dir / "figures/exploration_frontier.png")
        return cached

    sim = ClosedLoopSimulator(X, meta, cfg.loop)
    last = int(cfg.loop.n_cycles) - 1
    late = range(last - 2, last + 1)

    # The no-exploration reference has to come from *this* configuration, not the headline one.
    naive = sim.run_arm(PolicyArm("naive")).cycles.set_index("cycle")
    baseline_late = float(naive.loc[naive.index.isin(late), "recall_at_budget"].mean())
    oracle = sim.run_arm(PolicyArm("oracle", oracle_labels=True)).cycles.set_index("cycle")
    oracle_late = float(oracle.loc[oracle.index.isin(late), "recall_at_budget"].mean())
    log.info("frontier reference: naive %.4f, oracle %.4f", baseline_late, oracle_late)

    rows = []
    grid = [
        ("cost_aware", [0.005, 0.01, 0.02, 0.04]),
        ("uniform", [0.01, 0.02, 0.04]),
        ("risk_tapered", [0.02]),
        ("amount_capped", [0.02]),
    ]
    for mode, rates in grid:
        for rate in rates:
            name = f"{mode}_{rate}"
            arm = PolicyArm(
                name=name,
                exploration=ExplorationConfig(rate=rate, mode=mode),
                weights=WeightConfig(scheme="ipw"),
            )
            t = time.time()
            res = sim.run_arm(arm)
            g = res.cycles.set_index("cycle")
            rows.append(
                {
                    "mode": mode,
                    "rate": rate,
                    "recall_late": float(g.loc[g.index.isin(late), "recall_at_budget"].mean()),
                    "recovered_pts": float(
                        g.loc[g.index.isin(late), "recall_at_budget"].mean() - baseline_late
                    ),
                    "exploration_cost_share": float(g["exploration_fraud_cost_share"].mean()),
                    "explored_per_cycle": float(g["explored_count"].mean()),
                    "min_propensity": float(g["min_propensity"].min()),
                    "gap_closed": (
                        float(g.loc[g.index.isin(late), "recall_at_budget"].mean() - baseline_late)
                        / max(oracle_late - baseline_late, 1e-9)
                    ),
                    "ess_ratio": float(g["train_ess_ratio"].mean())
                    if "train_ess_ratio" in g else float("nan"),
                    "runtime_s": time.time() - t,
                }
            )
            log.info("frontier %-22s recovered %+.3f pts at %.3f%% cost",
                     name, rows[-1]["recovered_pts"], rows[-1]["exploration_cost_share"] * 100)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "exploration_frontier.csv", index=False)
    plot_cost_frontier(out, docs_dir / "figures/exploration_frontier.png")
    return out


def stage_ope(cfg: ExperimentConfig, X, meta, out_dir: Path, force: bool = False) -> pd.DataFrame:
    """Do the off-policy estimators recover a quantity we can independently verify?

    The candidate policy is "decline the top ``budget`` by score". Its true fraud-value-blocked
    share is computable here because the simulator holds every counterfactual label. IPS and
    SNIPS see only what an issuer's logs would contain. The error between them is the honest
    answer to "can we evaluate a policy change before shipping it?".
    """
    cached = _cached(out_dir / "off_policy_evaluation.csv", force)
    if cached is not None:
        return cached

    from clfraud.models.base import ModelConfig
    from clfraud.models.gbdt import make_scorer
    from clfraud.simulator.policy import ThresholdPolicy

    rng = np.random.default_rng(cfg.loop.seed)
    ts = meta["ts"].to_numpy()
    cycle = ((ts - ts.min()) // (cfg.loop.cycle_days * 86_400)).astype(int)
    y = meta["is_fraud"].to_numpy().astype(int)
    amt = meta["amount"].to_numpy(dtype="float64")

    rows = []
    for c in range(cfg.loop.warmup_cycles + 1, min(cfg.loop.n_cycles, 12)):
        tr = np.flatnonzero(cycle < c)
        te = np.flatnonzero(cycle == c)
        if te.size == 0 or y[tr].sum() < 50:
            continue
        model = make_scorer(ModelConfig(n_estimators=150, num_leaves=32))
        model.fit(X.iloc[tr], y[tr])
        scores = model.predict_proba(X.iloc[te])

        logger = ThresholdPolicy(
            cfg.loop.decline_budget, ExplorationConfig(rate=0.02, mode="cost_aware")
        )
        logger.calibrate(scores)
        dec = logger.decide(scores, amt[te], rng)

        # Candidate: a tighter policy that declines twice the volume.
        cand_thresh = float(np.quantile(scores, 1.0 - 2 * cfg.loop.decline_budget))
        candidate = scores >= cand_thresh

        truth = float(
            (amt[te][candidate & (y[te] == 1)]).sum() / max(amt[te][y[te] == 1].sum(), 1e-9)
        )
        # No clip: the exploration propensity floor is 0.004, so the correct weight on an
        # explored row is 250. Any tighter cap throws away most of the evidence about the
        # decline region -- which is the only region the candidate and the logger disagree on.
        est = policy_value_from_logs(
            y[te], amt[te], dec.approved, dec.propensity, candidate, clip=None
        )
        for name, r in est.items():
            rows.append(
                {
                    "cycle": c,
                    "estimator": name,
                    "estimate": r.estimate,
                    "std_error": r.std_error,
                    "truth": truth,
                    "abs_error": abs(r.estimate - truth),
                    "rel_error": abs(r.estimate - truth) / max(truth, 1e-9),
                    "covers_truth": float(r.ci95[0] <= truth <= r.ci95[1]),
                    "ess": r.ess,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "off_policy_evaluation.csv", index=False)
    return out


def stage_sensitivity(cfg: ExperimentConfig, X, meta, out_dir: Path, force: bool = False) -> pd.DataFrame:
    """How large is the closed-loop damage as a function of how the retrainer is configured?

    Reported because the headline decay is a property of a configuration, not a universal
    constant, and pretending otherwise would be the kind of claim this project exists to argue
    against.
    """
    cached = _cached(out_dir / "sensitivity.csv", force)
    if cached is not None:
        return cached

    grid = [
        {"train_window_cycles": 3, "decline_budget": 0.03},
        {"train_window_cycles": 3, "decline_budget": 0.06},
        {"train_window_cycles": 6, "decline_budget": 0.06},
        {"train_window_cycles": 12, "decline_budget": 0.06},
    ]
    last = cfg.loop.n_cycles - 1
    late = range(last - 2, last + 1)
    early = range(cfg.loop.warmup_cycles, cfg.loop.warmup_cycles + 3)

    rows = []
    for g in grid:
        loop_cfg = replace(cfg.loop, **g)
        sim = ClosedLoopSimulator(X, meta, loop_cfg)
        out = {}
        for arm in (
            PolicyArm("naive"),
            PolicyArm("oracle", oracle_labels=True),
            PolicyArm(
                "explore_ipw",
                exploration=ExplorationConfig(rate=0.02, mode="cost_aware"),
                weights=WeightConfig(scheme="ipw"),
            ),
        ):
            r = sim.run_arm(arm).cycles.set_index("cycle")
            out[arm.name] = r
        n_early = float(out["naive"].loc[out["naive"].index.isin(early), "recall_at_budget"].mean())
        n_late = float(out["naive"].loc[out["naive"].index.isin(late), "recall_at_budget"].mean())
        o_late = float(out["oracle"].loc[out["oracle"].index.isin(late), "recall_at_budget"].mean())
        e_late = float(
            out["explore_ipw"].loc[out["explore_ipw"].index.isin(late), "recall_at_budget"].mean()
        )
        rows.append(
            {
                **g,
                "naive_early": n_early,
                "naive_late": n_late,
                "decay_rel": (n_late - n_early) / n_early,
                "naive_worst_cycle": float(out["naive"]["recall_at_budget"].min()),
                "oracle_late": o_late,
                "explore_ipw_late": e_late,
                "gap_closed": (e_late - n_late) / max(o_late - n_late, 1e-9),
            }
        )
        log.info("sensitivity %s -> decay %.1f%%", g, rows[-1]["decay_rel"] * 100)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "sensitivity.csv", index=False)
    return out


def render_results_md(
    cfg: ExperimentConfig,
    data_tbl: pd.DataFrame,
    leak: pd.DataFrame,
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    frontier: pd.DataFrame | None,
    ope: pd.DataFrame | None,
    sens: pd.DataFrame | None,
    path: Path,
) -> None:
    base, orc = cfg.baseline_arm, cfg.oracle_arm
    m = "recall_at_budget"
    n_early = summary.loc[base, f"{m}_early"]
    n_late = summary.loc[base, f"{m}_late"]
    decay = summary.loc[base, "delta_rel"]
    worst = summary.loc[base, "min_cycle"]
    n_mean = summary.loc[base, "mean_over_run"]

    # The wide frame goes to summary.csv; the document shows the columns a reader needs.
    headline_cols = [
        f"{m}_early", f"{m}_late", "delta_rel", "mean_over_run", "p10_cycle", "min_cycle",
        "max_drawdown_rel", "recovered_mean_pts", "recovered_p10_pts", "gap_closed",
        "exploration_cost_share", "mean_ess_ratio", "total_cost",
    ]
    summary_view = summary[[c for c in headline_cols if c in summary.columns]]

    lines = [
        "# Results",
        "",
        "> Generated by `python experiments/run_all.py`. Every table and figure below is written "
        "by that script from `results/*.csv` -- nothing here is transcribed by hand, so a config "
        "change that moves a number moves it everywhere.",
        "",
        f"Configuration: `{cfg.name}` — {cfg.loop.n_cycles} cycles of "
        f"{cfg.loop.cycle_days:.0f} days, {cfg.loop.decline_budget:.0%} decline budget, "
        f"{cfg.loop.train_window_cycles}-cycle training window, "
        f"{cfg.loop.label.delay_days:.0f}-day label maturation.",
        "",
        "## 1. Dataset calibration",
        "",
        "The generated stream is calibrated against IEEE-CIS `train_transaction.csv`. Span is "
        "the one deliberate departure: the research question is about retraining cycles, so the "
        "stream is generated at 18 months natively rather than IEEE-CIS's ~182 days.",
        "",
        markdown_table(data_tbl.round(4)),
        "",
        "## 2. What a two-sided velocity window manufactures",
        "",
        "The leak that actually happens in fraud work is not target encoding, it is an unshifted "
        "`groupby().rolling()`: velocity aggregates over a window that includes transactions "
        "that had not happened yet. All three rows below score the same held-out future window; "
        "the second and third use the **same fitted model** and differ only in whether the "
        "features it is fed can see the future. The drop between them is the part of the offline "
        "number that was never real, and no amount of retraining recovers it.",
        "",
        markdown_table(leak.round(4)),
        "",
        "![leakage](figures/leakage.png)",
        "",
        "## 3. The closed loop",
        "",
        f"Recall at a {cfg.loop.decline_budget:.0%} decline budget, measured against oracle "
        "labels on the **whole** cycle -- declined transactions included. No deployed system can "
        "compute this column, which is the entire reason the failure goes unnoticed in "
        "production: every metric an issuer *can* compute is restricted to approved traffic.",
        "",
        markdown_table(summary_view.round(4)),
        "",
        "Recovery and gap-closed are measured on `mean_over_run` rather than on the final "
        "cycles. The closed loop sawtooths, so an end-of-run figure depends on where in the "
        "swing the run happens to stop; the run mean is also the business-relevant quantity, "
        "being proportional to fraud caught over the whole period.",
        "",
        "![recall trajectories](figures/recall_trajectories.png)",
        "",
        f"- `{base}` opens at **{n_early:.3f}** and closes at **{n_late:.3f}** "
        f"(**{decay:+.1%}** over {cfg.loop.n_cycles} cycles), averaging **{n_mean:.3f}** with a "
        f"worst cycle of **{worst:.3f}**.",
    ]
    if orc and orc in summary.index:
        gap = summary.loc[orc, "mean_over_run"] - n_mean
        lines.append(
            f"- `{orc}`, which declines identically but keeps every label, averages "
            f"**{summary.loc[orc, 'mean_over_run']:.3f}** — a **{gap:.3f}** gap attributable "
            "entirely to censoring, not to drift and not to the decision rule."
        )
    if "frozen" in summary.index:
        lines.append(
            f"- `frozen`, which is never retrained at all, averages "
            f"**{summary.loc['frozen', 'mean_over_run']:.3f}**, beating `{base}` by "
            f"**{summary.loc['frozen', 'mean_over_run'] - n_mean:+.3f}**. Under censored "
            "feedback, retraining is not a neutral act."
        )
    for arm in ("explore_ipw", "explore_ipw_cap50", "explore_only"):
        if arm in summary.index:
            r = summary.loc[arm]
            lines.append(
                f"- `{arm}` recovers **{r['recovered_mean_pts']:+.3f}** of mean recall "
                f"(**{r.get('gap_closed', float('nan')):.0%}** of the gap; "
                f"**{r['recovered_p10_pts']:+.3f}** in the bottom decile of cycles) for an added "
                f"fraud loss of **{r['exploration_cost_share']:.2%}** of fraud value."
            )
    lines += [
        "",
        "### Why it happens: the supply of labelled fraud",
        "",
        "![training supply](figures/training_supply.png)",
        "",
        "The loop does not merely shrink the training set, it *selects* it. Positives survive "
        "into training only when the previous model failed to stop them, so the positive class "
        "drifts from \"fraud\" toward \"fraud my predecessor missed\".",
        "",
        "![positive composition](figures/positive_composition.png)",
        "",
    ]
    if frontier is not None and not frontier.empty:
        lines += [
            "## 4. The exploration frontier",
            "",
            "Exploration is not free: every approved would-be decline is fraud risk bought on "
            "purpose. The question is what each design gets for the same money. Read the table "
            "horizontally: the designs recover similar amounts of recall and differ mostly in "
            "`exploration_cost_share`.",
            "",
            "Note that `recovered_pts` and `gap_closed` here are measured on the final three "
            "cycles, not on the run mean as in §3 -- these arms are compared against each other "
            "on one consistent basis, so the two sections' `gap_closed` columns are not "
            "interchangeable.",
            "",
            markdown_table(frontier.round(4)),
            "",
            "![frontier](figures/exploration_frontier.png)",
            "",
        ]
    if ope is not None and not ope.empty:
        agg = ope.groupby("estimator")[
            ["estimate", "truth", "abs_error", "rel_error", "covers_truth", "ess"]
        ].mean()
        lines += [
            "## 5. Off-policy evaluation against a knowable truth",
            "",
            "Can a candidate policy be scored *before* shipping it, from logs the incumbent "
            "produced? Here the answer is checkable, because the simulator holds the "
            "counterfactual labels the estimators are trying to work around.",
            "",
            markdown_table(agg.round(4)),
            "",
        ]
    if sens is not None and not sens.empty:
        lines += [
            "## 6. Sensitivity",
            "",
            "The headline decay is a property of a retraining configuration, not a constant. "
            "Shorter windows and larger decline budgets both deepen the damage, for the same "
            "reason: each removes more of the evidence the next model needs.",
            "",
            markdown_table(sens.round(4)),
            "",
        ]
    path.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/headline.yaml")
    p.add_argument(
        "--sweep-config", default="configs/exploration_frontier.yaml",
        help="smaller configuration used for the frontier and sensitivity sweeps",
    )
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir")
    p.add_argument(
        "--docs-dir", default="docs",
        help="where RESULTS.md and figures/ are written; point elsewhere for throwaway runs "
             "so a smoke run cannot overwrite the committed report",
    )
    p.add_argument("--skip", default="", help="comma-separated stages to skip")
    p.add_argument("--only", default="", help="comma-separated stages to run")
    p.add_argument("--force", action="store_true", help="re-run stages even if a CSV exists")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )

    all_stages = ["dataset", "leakage", "loop", "frontier", "ope", "sensitivity"]
    stages = (
        set(args.only.split(",")) if args.only
        else set(all_stages) - {s for s in args.skip.split(",") if s}
    )

    cfg = ExperimentConfig.load(args.config)
    out_dir = Path(args.out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(args.docs_dir)
    (docs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

    t0 = time.time()
    df, X, meta = prepare(cfg, Path(args.cache_dir))
    log.info("prepared %d transactions, %d features in %.1fs", len(df), X.shape[1], time.time() - t0)

    if "dataset" in stages:
        data_tbl = stage_dataset(df, out_dir)
    else:
        data_tbl = _cached(out_dir / "dataset_calibration.csv", False)
        data_tbl = pd.DataFrame() if data_tbl is None else data_tbl
    if "leakage" in stages:
        leak = stage_leakage(
            df, X, out_dir, cfg.loop.decline_budget, docs_dir, force=args.force
        )
    else:
        leak = _cached(out_dir / "leakage_audit.csv", False)
        leak = pd.DataFrame() if leak is None else leak

    _results, cycles, summary = stage_loop(cfg, X, meta, out_dir, docs_dir, force=args.force)

    frontier = ope = sens = None
    if {"frontier", "sensitivity"} & stages:
        # Sweeps run on their own, smaller configuration; conclusions are about ordering and
        # shape, both of which are stable across scale, and a dozen full loops at 590k rows is
        # an hour of compute for no extra insight.
        sweep_cfg = ExperimentConfig.load(args.sweep_config)
        _sdf, sX, smeta = prepare(sweep_cfg, Path(args.cache_dir))
        log.info("sweep dataset: %d transactions", len(sX))
        if "frontier" in stages:
            frontier = stage_frontier(
                sweep_cfg, sX, smeta, out_dir, docs_dir, force=args.force
            )
        if "sensitivity" in stages:
            sens = stage_sensitivity(sweep_cfg, sX, smeta, out_dir, force=args.force)
    if frontier is None:
        frontier = _cached(out_dir / "exploration_frontier.csv", False)
    if sens is None:
        sens = _cached(out_dir / "sensitivity.csv", False)
    if "ope" in stages:
        ope = stage_ope(cfg, X, meta, out_dir, force=args.force)
    else:
        ope = _cached(out_dir / "off_policy_evaluation.csv", False)

    render_results_md(
        cfg, data_tbl, leak, summary, cycles, frontier, ope, sens, docs_dir / "RESULTS.md"
    )
    log.info("all stages done in %.1f min", (time.time() - t0) / 60)
    print(markdown_table(summary.round(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
