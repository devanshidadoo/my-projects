"""Tables and figures, written from the result frames rather than transcribed.

Nothing in ``docs/`` is typed by hand. If a config change moves a number, it moves in the
README, in ``docs/RESULTS.md`` and in every figure, because they are all rendered from the same
CSVs by ``experiments/run_all.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# One palette, used consistently: blue is honest/proposed, orange is the heuristic, red is the
# thing that loses money, grey is the unattainable reference.
C_GOOD = "#1f77b4"
C_ALT = "#ff7f0e"
C_BAD = "#d62728"
C_REF = "#7f7f7f"
C_FILL = "#c6dbef"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    return plt


def markdown_table(df: pd.DataFrame, *, floatfmt: str = "{:.4f}", max_rows: int | None = None) -> str:
    """GitHub-flavoured markdown table with numbers formatted once, consistently."""
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.clf()
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- figures
def plot_decile_capture(deciles: pd.DataFrame, path: Path) -> Path:
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    models = list(deciles.model.unique())
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        d = deciles[deciles.model == m].sort_values("decile")
        ax1.bar(d.decile + (i - (len(models) - 1) / 2) * width, d.late_rate, width,
                label=m, color=[C_GOOD, C_ALT][i % 2])
        ax2.plot(d.cum_share_of_orders, d.cum_capture, marker="o", ms=3,
                 color=[C_GOOD, C_ALT][i % 2], label=m)
    base = float(deciles.n_late.sum() / deciles.n.sum())
    ax1.axhline(base, color=C_REF, ls="--", lw=1, label="base rate")
    ax1.set_xlabel("risk decile (1 = riskiest)")
    ax1.set_ylabel("share delivered late")
    ax1.set_xticks(range(1, 11))
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("late rate by predicted risk decile")

    ax2.plot([0, 1], [0, 1], color=C_REF, ls="--", lw=1, label="no model")
    top2 = deciles[deciles.decile <= 2].groupby("model").capture.sum()
    for i, m in enumerate(models):
        ax2.annotate(f"top 2 deciles: {top2[m]:.1%}", (0.2, top2[m]),
                     textcoords="offset points", xytext=(6, -12 * (i + 1)), fontsize=8,
                     color=[C_GOOD, C_ALT][i % 2])
    ax2.axvline(0.2, color=C_REF, lw=0.8)
    ax2.set_xlabel("share of orders reviewed, riskiest first")
    ax2.set_ylabel("share of late orders caught")
    ax2.set_title("capture curve")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return save(fig, path)


def plot_calibration(curves: dict[str, pd.DataFrame], path: Path) -> Path:
    """Reliability curves, plus what the miscalibration does to the decision threshold."""
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    colours = {"raw": C_BAD, "calibrated": C_GOOD}
    lim = 0.0
    for label, tab in curves.items():
        ax1.plot(tab.predicted, tab.observed, marker="o", ms=3.5,
                 color=colours.get(label, C_ALT), label=label)
        lim = max(lim, float(tab.predicted.max()), float(tab.observed.max()))
    ax1.plot([0, lim], [0, lim], color=C_REF, ls="--", lw=1, label="perfect")
    ax1.set_xlabel("predicted P(late)")
    ax1.set_ylabel("observed share late")
    ax1.set_title("reliability (equal-count bins)")
    ax1.legend(frameon=False, fontsize=8)

    for label, tab in curves.items():
        ax2.plot(tab.predicted, tab.gap, marker="o", ms=3.5, color=colours.get(label, C_ALT),
                 label=label)
    ax2.axhline(0, color=C_REF, ls="--", lw=1)
    ax2.set_xlabel("predicted P(late)")
    ax2.set_ylabel("observed − predicted")
    ax2.set_title("calibration error, where the threshold lives")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def plot_policy_value(comparison: pd.DataFrame, path: Path, *, model: str = "gbdt") -> Path:
    plt = _mpl()
    d = comparison[(comparison.model == model) & (comparison.scores == "calibrated")]
    d = d.sort_values("delta_per_1k")
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(d) + 1.4))
    colours = []
    for name, v in zip(d.policy, d.delta_per_1k):
        if name == "oracle":
            colours.append(C_REF)
        elif name.startswith("margin"):
            colours.append(C_GOOD)
        elif v < 0:
            colours.append(C_BAD)
        else:
            colours.append(C_ALT)
    y = np.arange(len(d))
    ax.barh(y, d.delta_per_1k, color=colours)
    ax.errorbar(d.delta_per_1k, y, xerr=[d.delta_per_1k - d.delta_lo, d.delta_hi - d.delta_per_1k],
                fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.policy)
    ax.axvline(0, color="black", lw=0.9)
    # Symlog: the losing rules lose an order of magnitude more than the winning ones win, and on
    # a linear axis that flattens every rule worth telling apart into the same sliver.
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("contribution vs doing nothing, per 1,000 orders (95 % bootstrap, symlog)")
    ax.set_title("what each decision rule is worth")
    fig.tight_layout()
    return save(fig, path)


def plot_leakage(audit: pd.DataFrame, path: Path) -> Path:
    plt = _mpl()
    metrics = [("pr_auc", "PR-AUC"), ("capture_top2", "capture, top 2 deciles"),
               ("delta_per_1k", "contribution / 1k orders")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    labels = [str(s).replace(", ", "\n") for s in audit.scenario]
    colours = [C_GOOD if "honest" in s else (C_BAD if "reported" in s else "#f4a582")
               for s in audit.scenario]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(range(len(audit)), audit[col], color=colours)
        ax.set_xticks(range(len(audit)))
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_title(title)
        for i, v in enumerate(audit[col]):
            ax.annotate(f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}", (i, v),
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle("what a two-sided aggregate buys you offline, and gives back at serving time",
                 fontsize=10)
    fig.tight_layout()
    return save(fig, path)


def plot_budget_frontier(frontier: pd.DataFrame, path: Path) -> Path:
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax1.plot(frontier.spend_per_1k, frontier.delta_per_1k, marker="o", ms=4, color=C_GOOD)
    ax1.set_xlabel("spend per 1,000 orders")
    ax1.set_ylabel("contribution gained per 1,000 orders")
    ax1.set_title("budget frontier")
    for _, r in frontier.iterrows():
        ax1.annotate(f"{r.budget_share:.0%}", (r.spend_per_1k, r.delta_per_1k),
                     textcoords="offset points", xytext=(4, -9), fontsize=7, color=C_REF)
    ax2.plot(frontier.spend_per_1k, frontier.shadow_price, marker="o", ms=4, color=C_ALT)
    ax2.axhline(0, color=C_REF, ls="--", lw=1)
    ax2.set_xlabel("spend per 1,000 orders")
    ax2.set_ylabel("shadow price (return on the next unit)")
    ax2.set_title("what the next unit of budget earns")
    fig.tight_layout()
    return save(fig, path)


def plot_threshold_spread(implied: np.ndarray, global_tau: float, path: Path) -> Path:
    plt = _mpl()
    imp = np.asarray(implied, dtype=float)
    imp = imp[np.isfinite(imp) & (imp > 0) & (imp < 1)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(imp, bins=60, color=C_FILL, edgecolor=C_GOOD, linewidth=0.5)
    ax.axvline(global_tau, color=C_BAD, lw=1.6,
               label=f"best single threshold = {global_tau:.3f}")
    for q, ls in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
        ax.axvline(np.quantile(imp, q), color=C_REF, ls=ls, lw=1,
                   label=f"p{int(q * 100)} = {np.quantile(imp, q):.3f}")
    ax.set_xlabel("break-even P(late) for a nudge on this order:  c / (u · L)")
    ax.set_ylabel("orders")
    ax.set_title("the threshold is a property of the order, not of the model")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def plot_uplift_sensitivity(sens: pd.DataFrame, path: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for policy, grp in sens.groupby("policy"):
        grp = grp.sort_values("believed_rho_scale")
        colour = C_GOOD if "cause" in str(policy) else C_ALT
        ax.plot(grp.believed_rho_scale, grp.delta_per_1k, marker="o", ms=4, label=policy,
                color=colour)
    ax.axvline(1.0, color=C_REF, ls="--", lw=1, label="belief = truth")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("believed uplift ÷ true uplift")
    ax.set_ylabel("realised contribution per 1,000 orders")
    ax.set_title("how wrong the uplift assumption can be before the policy stops paying")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def plot_cause_mix(mix: pd.DataFrame, prices: pd.DataFrame | None, path: Path) -> Path:
    """Where the two policies disagree: risk rank against cause, and what it buys."""
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    order = ["handling_only", "transit_only", "either", "neither"]
    present = [c for c in order if c in mix.columns]
    bottom = np.zeros(len(mix))
    colours = {"handling_only": C_GOOD, "transit_only": C_ALT, "either": "#9ecae1",
               "neither": C_BAD}
    for c in present:
        ax1.bar(mix.decile, mix[c], bottom=bottom, color=colours[c], label=c)
        bottom += mix[c].to_numpy()
    ax1.set_xlabel("risk decile (1 = riskiest)")
    ax1.set_ylabel("share of late orders")
    ax1.set_title("why the order is late, by risk decile")
    ax1.set_xticks(range(1, 11))
    ax1.legend(frameon=False, fontsize=7, ncol=2)

    if prices is not None and len(prices):
        for policy, grp in prices.groupby("policy"):
            grp = grp.sort_values("mean_expedite_cost")
            colour = C_GOOD if "cause" in str(policy) else C_ALT
            label = "cause-aware" if "cause" in str(policy) else "risk-ranked"
            ax2.plot(grp.mean_expedite_cost, grp.n_expedite + grp.n_both, marker="o", ms=4,
                     color=colour, label=label)
        ax2.set_xlabel("mean cost of an express upgrade")
        ax2.set_ylabel("orders bought an upgrade")
        ax2.set_title("who buys express, and at what price")
        ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, path)
