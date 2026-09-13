"""``clfraud`` command line.

    clfraud describe    configs/headline.yaml      # resolved config + dataset marginals
    clfraud generate    configs/headline.yaml      # materialise + cache the stream
    clfraud features    configs/headline.yaml      # build + cache the feature matrix
    clfraud simulate    configs/headline.yaml      # run every arm, write results/
    clfraud ingest-ieee --raw-dir data/raw --out data/processed/ieee.parquet
    clfraud leak-audit  configs/headline.yaml      # point-in-time vs. leaky aggregates
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from clfraud import __version__

log = logging.getLogger("clfraud")


def _setup_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> tuple:
    from clfraud.config import ExperimentConfig
    from clfraud.pipeline import prepare

    cfg = ExperimentConfig.load(args.config)
    cache = Path(args.cache_dir) if args.cache_dir else None
    return cfg, prepare(cfg, cache)


# ---------------------------------------------------------------------- commands
def cmd_describe(args) -> int:
    from clfraud.config import ExperimentConfig
    from clfraud.pipeline import dataset_summary, load_stream

    cfg = ExperimentConfig.load(args.config)
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    if args.with_data:
        df = load_stream(cfg, Path(args.cache_dir) if args.cache_dir else None)
        print("\ndataset marginals:")
        print(json.dumps(dataset_summary(df), indent=2))
    return 0


def cmd_generate(args) -> int:
    from clfraud.config import ExperimentConfig
    from clfraud.pipeline import dataset_summary, load_stream

    cfg = ExperimentConfig.load(args.config)
    df = load_stream(cfg, Path(args.cache_dir) if args.cache_dir else None)
    print(json.dumps(dataset_summary(df), indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_features(args) -> int:
    _cfg, (df, X, _meta) = _load(args)
    print(f"{len(X):,} rows x {X.shape[1]} features")
    print(f"point-in-time features: {list(X.columns)[:12]} ...")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_simulate(args) -> int:
    from clfraud.evaluation.reporting import combine, markdown_table, summarise_arms
    from clfraud.simulator.loop import ClosedLoopSimulator

    cfg, (df, X, meta) = _load(args)
    out_dir = Path(args.out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = cfg.arms
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = [a for a in arms if a.name in wanted]
        if not arms:
            print(f"no arms match {sorted(wanted)}", file=sys.stderr)
            return 2

    sim = ClosedLoopSimulator(X, meta, cfg.loop)
    results = sim.run(arms)

    cycles = combine(results)
    cycles.to_csv(out_dir / "cycles.csv", index=False)
    comp = pd.concat([r.train_composition for r in results.values()], ignore_index=True)
    if not comp.empty:
        comp.to_csv(out_dir / "train_composition.csv", index=False)

    summary = summarise_arms(cycles, cfg.baseline_arm, cfg.oracle_arm)
    summary.to_csv(out_dir / "summary.csv")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    print(markdown_table(summary.round(4)))
    print(f"\nwrote {out_dir}/cycles.csv, summary.csv, config.json")
    return 0


def cmd_ingest_ieee(args) -> int:
    from clfraud.data.ieee_cis import load_ieee_cis, summarise

    df = load_ieee_cis(args.raw_dir, nrows=args.nrows)
    print(json.dumps(summarise(df), indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_leak_audit(args) -> int:
    from clfraud.evaluation.leakage import run_leak_audit

    cfg, (df, X, meta) = _load(args)
    rows = run_leak_audit(df, X, holdout_frac=args.holdout_frac, budget=cfg.loop.decline_budget)
    print(rows.round(4).to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.out, index=False)
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clfraud",
        description="Closed-loop fraud detection under selection bias.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"clfraud {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=1, help="repeat for debug logs")
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp):
        sp.add_argument("config", help="path to an experiment YAML")
        sp.add_argument("--cache-dir", default="data/cache", help="parquet cache directory")
        return sp

    d = with_config(sub.add_parser("describe", help="print the resolved configuration"))
    d.add_argument("--with-data", action="store_true", help="also materialise and summarise data")
    d.set_defaults(func=cmd_describe)

    g = with_config(sub.add_parser("generate", help="materialise the transaction stream"))
    g.add_argument("--out", help="optional parquet output path")
    g.set_defaults(func=cmd_generate)

    f = with_config(sub.add_parser("features", help="build the point-in-time feature matrix"))
    f.add_argument("--out")
    f.set_defaults(func=cmd_features)

    s = with_config(sub.add_parser("simulate", help="run the closed-loop experiment"))
    s.add_argument("--out-dir")
    s.add_argument("--arms", help="comma-separated subset of arm names")
    s.set_defaults(func=cmd_simulate)

    la = with_config(sub.add_parser("leak-audit", help="quantify what leaky aggregates buy"))
    la.add_argument("--holdout-frac", type=float, default=0.25)
    la.add_argument("--out")
    la.set_defaults(func=cmd_leak_audit)

    i = sub.add_parser("ingest-ieee", help="canonicalise the Kaggle IEEE-CIS files")
    i.add_argument("--raw-dir", required=True)
    i.add_argument("--out")
    i.add_argument("--nrows", type=int)
    i.set_defaults(func=cmd_ingest_ieee)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s: %s", type(exc).__name__, exc)
        if args.verbose > 1:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
