"""``drisk`` command line.

    drisk schema --dialect mysql            # the nine-table DDL, for a data engineer
    drisk describe  configs/headline.yaml   # resolved config (+ --with-data for marginals)
    drisk build     configs/headline.yaml   # generate, load, extract features, cache
    drisk features  configs/headline.yaml   # the feature query, or --show-sql to read it
    drisk train     configs/headline.yaml   # fit, calibrate, report ranking + calibration
    drisk policy    configs/headline.yaml   # every decision rule against the truth
    drisk leak-audit configs/headline.yaml  # point-in-time vs two-sided aggregates
    drisk ingest-olist --raw-dir data/raw   # load the real files into the same schema
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from deliveryrisk import __version__

log = logging.getLogger("deliveryrisk")


def _setup_logging(verbosity: int) -> None:
    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG),
        format="%(asctime)s %(levelname)-7s %(name)-26s %(message)s",
        datefmt="%H:%M:%S",
    )


def _show(df: pd.DataFrame, n: int | None = None) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df if n is None else df.head(n))


# ---------------------------------------------------------------------- commands
def cmd_schema(args) -> int:
    from deliveryrisk.data.schema import TABLES, drop_ddl, schema_ddl

    sql = (drop_ddl(args.dialect) + "\n" if args.drop else "") + schema_ddl(args.dialect)
    if args.out:
        Path(args.out).write_text(sql)
        print(f"wrote {args.out} ({len(TABLES)} tables, {args.dialect})")
    else:
        print(sql)
    return 0


def cmd_describe(args) -> int:
    from deliveryrisk.config import ExperimentConfig

    cfg = ExperimentConfig.load(args.config)
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    if args.with_data:
        from deliveryrisk.pipeline import prepare

        prep = prepare(cfg, Path(args.cache_dir))
        print("\ndataset marginals:")
        print(json.dumps(prep.summary, indent=2))
    return 0


def cmd_build(args) -> int:
    from deliveryrisk.config import ExperimentConfig
    from deliveryrisk.pipeline import prepare

    cfg = ExperimentConfig.load(args.config)
    prep = prepare(cfg, Path(args.cache_dir))
    print(json.dumps(prep.summary, indent=2))
    print(f"\nfeature table: {prep.features.shape[0]} rows x {prep.features.shape[1]} columns")
    if prep.counts:
        print(json.dumps(prep.counts, indent=2))
    return 0


def cmd_features(args) -> int:
    from deliveryrisk.config import ExperimentConfig
    from deliveryrisk.features.sql import (
        ENTITIES,
        ORDER_FACTS_SQL,
        build_entity_sql,
        build_events_sql,
        build_features_sql,
    )

    cfg = ExperimentConfig.load(args.config)
    if args.show_sql:
        print(ORDER_FACTS_SQL)
        print(build_events_sql(trailing_days=cfg.features.trailing_days))
        for spec in ENTITIES:
            print(build_entity_sql(spec, mode=cfg.features.mode,
                                   trailing_days=cfg.features.trailing_days,
                                   prior_weight=cfg.features.prior_weight,
                                   prior_rate=cfg.features.prior_rate or 0.08))
        print(build_features_sql(mode=cfg.features.mode))
        return 0
    from deliveryrisk.pipeline import prepare

    prep = prepare(cfg, Path(args.cache_dir))
    _show(prep.features.describe().T, 40)
    if args.out:
        prep.features.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_train(args) -> int:
    from deliveryrisk.config import ExperimentConfig
    from deliveryrisk.evaluation.metrics import decile_table, ranking_metrics
    from deliveryrisk.features.build import labelled, split_by_time
    from deliveryrisk.models.calibration import calibration_metrics
    from deliveryrisk.pipeline import fit_models, prepare, score

    cfg = ExperimentConfig.load(args.config)
    prep = prepare(cfg, Path(args.cache_dir))
    train, valid, test = split_by_time(labelled(prep.features), cfg.train_share, cfg.valid_share)
    models = fit_models(cfg, train, valid, kind=args.kind)
    raw, cal = score(models, test)
    y = test.y_late.to_numpy(dtype=float)
    rows = [
        {"scores": label, **ranking_metrics(y, p), **calibration_metrics(y, p)}
        for label, p in (("raw", raw), ("calibrated", cal))
    ]
    _show(pd.DataFrame(rows).round(4))
    print()
    _show(decile_table(y, cal).round(4))
    return 0


def cmd_policy(args) -> int:
    from deliveryrisk.config import ExperimentConfig
    from deliveryrisk.pipeline import run_experiment

    cfg = ExperimentConfig.load(args.config)
    res = run_experiment(cfg, Path(args.cache_dir))
    cols = ["model", "scores", "policy", "treated_share", "spend_per_1k", "late_rate",
            "delta_per_1k", "delta_lo", "delta_hi", "roi"]
    _show(res["policy_comparison"][cols].round(2))
    return 0


def cmd_leak_audit(args) -> int:
    from deliveryrisk.config import ExperimentConfig
    from deliveryrisk.evaluation.leakage import run_leak_audit
    from deliveryrisk.pipeline import prepare

    cfg = ExperimentConfig.load(args.config)
    audit, breakdown = run_leak_audit(prepare(cfg, Path(args.cache_dir)), cfg)
    _show(audit[["scenario", "pr_auc", "capture_top2", "ece", "delta_per_1k"]].round(4))
    print()
    _show(breakdown.round(4))
    return 0


def cmd_ingest_olist(args) -> int:
    from deliveryrisk.data.db import Database
    from deliveryrisk.data.olist import load_olist

    data = load_olist(Path(args.raw_dir))
    db = Database(args.database_url)
    db.create_schema()
    db.load_frames(data.frames)
    print(json.dumps(db.table_counts(), indent=2))
    print(
        "\nNote: the realised-contribution comparison cannot run on this data -- a real extract\n"
        "has no counterfactual outcome for an action that was not taken. Features, the leak\n"
        "audit, the models and the calibration diagnostics all run unchanged."
    )
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="drisk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"deliveryrisk {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    ap.add_argument("--cache-dir", default="data/cache")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schema", help="print the nine-table DDL")
    p.add_argument("--dialect", default="sqlite", choices=("sqlite", "mysql"))
    p.add_argument("--drop", action="store_true", help="prepend DROP TABLE statements")
    p.add_argument("--out")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("describe", help="print the resolved config")
    p.add_argument("config")
    p.add_argument("--with-data", action="store_true")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("build", help="generate, load and extract features")
    p.add_argument("config")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("features", help="the feature table, or the SQL that makes it")
    p.add_argument("config")
    p.add_argument("--show-sql", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="fit, calibrate and report")
    p.add_argument("config")
    p.add_argument("--kind", default="gbdt", choices=("gbdt", "logistic"))
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("policy", help="compare every decision rule")
    p.add_argument("config")
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser("leak-audit", help="point-in-time vs two-sided aggregates")
    p.add_argument("config")
    p.set_defaults(func=cmd_leak_audit)

    p = sub.add_parser("ingest-olist", help="load the real Olist CSVs into the same schema")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--database-url", default="sqlite:///data/olist.db")
    p.set_defaults(func=cmd_ingest_olist)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - top-level UX
        log.error("%s: %s", type(exc).__name__, exc)
        if args.verbose > 1:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
