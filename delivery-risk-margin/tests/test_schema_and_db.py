"""The nine tables, their DDL, and the loader that refuses bad data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deliveryrisk.data.db import Database, DatabaseError, to_datetime
from deliveryrisk.data.schema import (
    POST_DECISION_COLUMNS,
    TABLES,
    drop_ddl,
    schema_ddl,
    validate_tables,
)


def test_there_are_nine_tables_and_they_load_parents_first():
    assert len(TABLES) == 9
    seen: set[str] = set()
    for t in TABLES:
        for _, ref_table, _ in t.foreign_keys:
            assert ref_table in seen or ref_table == t.name, (
                f"{t.name} references {ref_table} before it is declared; the loader inserts in "
                "declaration order, so this would fail the foreign key"
            )
        seen.add(t.name)


@pytest.mark.parametrize("dialect", ["sqlite", "mysql"])
def test_ddl_covers_every_table_and_column(dialect):
    sql = schema_ddl(dialect)
    for t in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {t.name}" in sql
        for c in t.columns:
            assert c.name in sql
    assert drop_ddl(dialect).count("DROP TABLE") == len(TABLES)


def test_mysql_ddl_is_innodb_and_sqlite_is_not():
    assert "ENGINE=InnoDB" in schema_ddl("mysql")
    assert "ENGINE=InnoDB" not in schema_ddl("sqlite")


def test_unknown_dialect_is_rejected():
    with pytest.raises(ValueError):
        TABLES[0].ddl("postgres")


def test_validate_catches_what_the_driver_would_only_mumble_about(tiny_data):
    frames = {k: v.copy() for k, v in tiny_data.frames.items()}
    assert validate_tables(frames) == []

    frames["orders"] = frames["orders"].drop(columns=["approved_ts"])
    problems = validate_tables(frames)
    assert any(p.table == "orders" and "missing columns" in p.problem for p in problems)

    frames = {k: v.copy() for k, v in tiny_data.frames.items()}
    frames["sellers"] = pd.concat([frames["sellers"], frames["sellers"].head(1)])
    assert any(p.problem == "duplicate primary key" for p in validate_tables(frames))

    frames = {k: v.copy() for k, v in tiny_data.frames.items()}
    frames["orders"].loc[frames["orders"].index[0], "purchase_ts"] = np.nan
    assert any("NOT NULL" in p.problem for p in validate_tables(frames))


def test_loader_refuses_frames_that_do_not_match(tiny_data):
    db = Database("sqlite://:memory:")
    db.create_schema()
    broken = {k: v.copy() for k, v in tiny_data.frames.items()}
    broken["products"] = broken["products"].drop(columns=["category"])
    with pytest.raises(DatabaseError):
        db.load_frames(broken)


def test_counts_match_what_went_in(tiny_db, tiny_data):
    counts = tiny_db.table_counts()
    for name, frame in tiny_data.frames.items():
        assert counts[name] == len(frame), name


def test_unsupported_url():
    with pytest.raises(DatabaseError):
        Database("postgres://localhost/x")


def test_epoch_conversion_round_trips():
    ts = to_datetime(pd.Series([0.0, 86_400.0]))
    assert str(ts.iloc[0].date()) == "2017-01-01"
    assert str(ts.iloc[1].date()) == "2017-01-02"


def test_post_decision_columns_are_named_in_one_place():
    """A tripwire: the outcome columns are declared centrally so a feature review has a list."""
    assert "orders.delivered_ts" in POST_DECISION_COLUMNS
    assert "orders.pickup_ts" in POST_DECISION_COLUMNS
