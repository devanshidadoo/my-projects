"""Aggregation and figures.

Every number quoted in the README comes out of ``summarise_arms`` -- there is no hand-copied
figure anywhere in the docs, which is the only way the claims stay true after a config change.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Colour-blind-safe qualitative palette (Okabe-Ito), fixed per arm role so the same concept
#: keeps the same colour across every figure.
ARM_COLOURS = {
    # The two arms the eye should separate instantly get the strongest contrast available.
    "oracle": "#009E73",
    "naive": "#D55E00",
    "frozen": "#7F7F7F",
    "explore_only": "#E69F00",
    "explore_ipw": "#0072B2",
    "explore_ipw_estimated": "#56B4E9",
    "explore_ipw_cap50": "#CC79A7",
    "explore_ipw_unstable": "#8C6D31",
    "explore_uniform_ipw": "#6A3D9A",
    "explore_tapered_ipw": "#B15928",
}

#: Arms whose line should read as a reference rather than a result.
_DASHED = {"oracle", "frozen"}


def _colour(name: str, i: int) -> str:
    if name in ARM_COLOURS:
        return ARM_COLOURS[name]
    fallback = list(ARM_COLOURS.values())
    return fallback[i % len(fallback)]


def combine(results: dict) -> pd.DataFrame:
    """Stack every arm's per-cycle records into one long frame."""
    return pd.concat([r.cycles for r in results.values()], ignore_index=True)


def summarise_arms(
    cycles: pd.DataFrame,
    baseline_arm: str,
    oracle_arm: str | None = None,
    metric: str = "recall_at_budget",
    early_window: int = 3,
    late_window: int = 3,
) -> pd.DataFrame:
    """Headline table: where each arm started, where it ended, and what that is worth.

    ``early`` is the mean over the first ``early_window`` post-warm-up cycles. At that point the
    training window is still dominated by the fully-labelled bootstrap, so it is the closest
    thing to a "day one, complete labels" model -- the honest reference for decay.

    ``late`` is the mean over the final ``late_window`` cycles. Means rather than single cycles
    because the closed loop is noisy by construction: it is a feedback system, and any single
    cycle can catch it mid-swing.
    """
    rows = []
    first_cycle = int(cycles["cycle"].min())
    last_cycle = int(cycles["cycle"].max())
    early_cycles = range(first_cycle, first_cycle + early_window)
    late_cycles = range(last_cycle - late_window + 1, last_cycle + 1)

    for arm, g in cycles.groupby("arm"):
        g = g.set_index("cycle").sort_index()
        early = float(g.loc[g.index.isin(early_cycles), metric].mean())
        late = float(g.loc[g.index.isin(late_cycles), metric].mean())
        rows.append(
            {
                "arm": arm,
                f"{metric}_early": early,
                f"{metric}_late": late,
                "delta_pts": late - early,
                "delta_rel": (late - early) / early if early else np.nan,
                # A closed loop under a hard training window does not decay smoothly, it
                # sawtooths: block, starve, forget, flood, relearn. A single end-of-run number
                # therefore depends on where in the swing the run happens to stop, so the
                # distribution over cycles is reported alongside it. `mean_over_run` is the
                # business-relevant one -- it is proportional to fraud caught over 18 months --
                # and `p10_cycle` / `min_cycle` are the tail risk a fraud team actually fears.
                "mean_over_run": float(g[metric].mean()),
                "p10_cycle": float(g[metric].quantile(0.10)),
                "min_cycle": float(g[metric].min()),
                "max_drawdown_rel": float((g[metric].min() - early) / early) if early else np.nan,
                "value_recall_late": float(
                    g.loc[g.index.isin(late_cycles), "value_recall_at_budget"].mean()
                )
                if "value_recall_at_budget" in g
                else np.nan,
                "pr_auc_late": float(g.loc[g.index.isin(late_cycles), "pr_auc"].mean()),
                "fraud_loss_net": float(g["fraud_loss_net"].sum()),
                "false_positive_cost": float(g["false_positive_cost"].sum()),
                "total_cost": float(g["total_cost"].sum()),
                "exploration_cost_share": float(
                    g["exploration_fraud_cost_share"].mean()
                    if "exploration_fraud_cost_share" in g
                    else 0.0
                ),
                "mean_train_positives": float(g["train_positives"].mean())
                if "train_positives" in g
                else np.nan,
                "mean_ess_ratio": float(g["train_ess_ratio"].mean())
                if "train_ess_ratio" in g
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("arm")

    base_late = out.loc[baseline_arm, f"{metric}_late"]
    base_mean = out.loc[baseline_arm, "mean_over_run"]
    base_p10 = out.loc[baseline_arm, "p10_cycle"]
    out["recovered_pts"] = out[f"{metric}_late"] - base_late
    out["recovered_mean_pts"] = out["mean_over_run"] - base_mean
    out["recovered_p10_pts"] = out["p10_cycle"] - base_p10
    if oracle_arm and oracle_arm in out.index:
        gap = out.loc[oracle_arm, "mean_over_run"] - base_mean
        out["gap_to_oracle_pts"] = out.loc[oracle_arm, "mean_over_run"] - out["mean_over_run"]
        # Share of the closed-loop damage an arm undoes, measured on the run mean rather than on
        # the final cycles -- the final-cycle figure moves with the phase of the sawtooth.
        # 1.0 == indistinguishable from having had complete labels all along; above 1 is noise,
        # not a result.
        out["gap_closed"] = out["recovered_mean_pts"] / gap if abs(gap) > 1e-12 else np.nan
    return out.sort_values("mean_over_run", ascending=False)


# ---------------------------------------------------------------------- figures
def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_recall_trajectories(
    cycles: pd.DataFrame,
    out_path: Path,
    metric: str = "recall_at_budget",
    budget: float = 0.04,
    arms: list[str] | None = None,
) -> Path:
    """The headline figure: recall against oracle labels, cycle by cycle, per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.4))
    # Draw the headline arms last so they sit on top of the ablation spaghetti.
    names = arms or sorted(cycles["arm"].unique())
    order = sorted(names, key=lambda a: (a in ("naive", "oracle"), a))
    for i, arm in enumerate(order):
        g = cycles[cycles["arm"] == arm].sort_values("cycle")
        headline = arm in ("naive", "oracle")
        ax.plot(
            g["cycle"], g[metric],
            label=arm, color=_colour(arm, i),
            linewidth=2.6 if headline else 1.4,
            alpha=1.0 if headline else 0.75,
            marker="o", markersize=4.2 if headline else 3.0,
            linestyle="--" if arm in _DASHED else "-",
            zorder=3 if headline else 2,
        )
    _style(
        ax,
        f"Recall at a {budget:.0%} decline budget, measured against oracle labels",
        "retraining cycle (1 cycle = 30 days)",
        "recall",
    )
    ax.legend(
        frameon=False, fontsize=8, ncol=1,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_training_positives(cycles: pd.DataFrame, out_path: Path) -> Path:
    """What the loop does to the training set: positives available to each arm, per cycle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, arm in enumerate(sorted(cycles["arm"].unique())):
        g = cycles[cycles["arm"] == arm].sort_values("cycle")
        if "train_positives" in g:
            axes[0].plot(g["cycle"], g["train_positives"], label=arm, color=_colour(arm, i), linewidth=1.7)
        axes[1].plot(
            g["cycle"], g["label_censoring_rate"], label=arm, color=_colour(arm, i), linewidth=1.7
        )
    _style(axes[0], "Labelled fraud available at retrain", "cycle", "positive training rows")
    _style(axes[1], "Share of the cycle that returns no label", "cycle", "censoring rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_cost_frontier(frontier: pd.DataFrame, out_path: Path) -> Path:
    """Recall recovered against the fraud dollars exploration deliberately lets through.

    Read this figure horizontally. All four designs recover a similar amount of recall -- what
    separates them is the bill: at a 2 % rate, cost-aware exploration buys the same recovery as
    uniform exploration for roughly a quarter of the fraud losses.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_colours = {
        "cost_aware": "#0072B2",
        "uniform": "#D55E00",
        "risk_tapered": "#009E73",
        "amount_capped": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    # The oracle gap is the ceiling: recovering all of it means the censoring never happened.
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = float((frontier["recovered_pts"] / frontier["gap_closed"]).median())
    if np.isfinite(gap) and gap > 0:
        ax.axhline(gap * 100, color="#009E73", linestyle=":", linewidth=1.4, zorder=1)
        ax.annotate(
            f"full-label oracle  ({gap * 100:.1f} pts)",
            xy=(ax.get_xlim()[0], gap * 100), xytext=(4, 4),
            textcoords="offset points", fontsize=8, color="#009E73",
        )

    for mode, g in frontier.groupby("mode"):
        g = g.sort_values("rate")            # order by budget, not by the noisy cost axis
        colour = mode_colours.get(mode, "#666666")
        ax.plot(
            g["exploration_cost_share"] * 100, g["recovered_pts"] * 100,
            marker="o", markersize=6, linewidth=1.6 if len(g) > 1 else 0,
            color=colour, label=mode, zorder=3,
        )
        for _, r in g.iterrows():
            rate = r["rate"]
            label = f"{rate * 100:g}%"
            ax.annotate(
                label,
                (r["exploration_cost_share"] * 100, r["recovered_pts"] * 100),
                textcoords="offset points", xytext=(7, -3), fontsize=7.5, alpha=0.8,
            )
    _style(
        ax,
        "Exploration frontier: recall recovered vs. fraud losses bought",
        "added fraud loss (% of total fraud value)  —  lower is better",
        "recall recovered vs. naive (percentage points)",
    )
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, title="exploration design", title_fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_positive_composition(composition: pd.DataFrame, out_path: Path, arms: list[str]) -> Path:
    """The mechanism, made visible: what the observed positive class is *made of*, per cycle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted(c for c in composition.columns if c.startswith("pos_"))
    if not kinds:
        return out_path
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for ax, arm in zip(axes, arms):
        g = composition[composition["arm"] == arm].sort_values("cycle")
        if g.empty:
            continue
        ax.stackplot(
            g["cycle"],
            *[g[k].fillna(0.0) for k in kinds],
            labels=[k.replace("pos_", "") for k in kinds],
            colors=palette[: len(kinds)],
            alpha=0.9,
        )
        _style(ax, f"{arm}", "cycle", "share of labelled fraud")
        ax.set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(
        "Composition of the positive class the retrainer actually sees",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_leakage_comparison(rows: pd.DataFrame, out_path: Path) -> Path:
    """Train/serve skew: what a two-sided window reports, and what it delivers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("pr_auc", "PR-AUC"), ("recall_at_budget", "recall @ budget"),
               ("opening_recall_at_budget", "recall on episode openings")]
    metrics = [(k, lbl) for k, lbl in metrics if k in rows.columns]
    colours = {
        "honest (point-in-time)": "#0072B2",
        "leaky, as reported offline": "#E69F00",
        "leaky, as served": "#D55E00",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(metrics))
    scenarios = list(rows["scenario"])
    width = 0.8 / max(len(scenarios), 1)
    for i, scen in enumerate(scenarios):
        g = rows[rows["scenario"] == scen]
        vals = [float(g[k].iloc[0]) for k, _ in metrics]
        bars = ax.bar(
            x + (i - (len(scenarios) - 1) / 2) * width, vals, width,
            label=scen, color=colours.get(scen, _colour(scen, i)), alpha=0.92,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.05)
    _style(
        ax,
        "A two-sided velocity window: what it reports offline vs. what it delivers at serving",
        "", "score on a held-out future window",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal markdown renderer so the docs have no tabulate dependency."""
    d = df.copy()
    if d.index.name:
        d = d.reset_index()
    cols = list(d.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
