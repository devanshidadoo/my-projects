"""Database handle over the operational schema.

Two engines, one set of SQL. SQLite is the default because a 100 K-order warehouse fits in a
file and CI should not need a service container; MySQL 8 is supported because that is what the
schema is shaped for and because the point-in-time query leans on window functions that MySQL
only grew in 8.0.

Everything downstream of this module speaks SQL, not pandas. That is deliberate: the feature
definitions are the artefact worth having, and a feature definition that only exists inside a
``groupby`` cannot be handed to a data engineer to schedule.
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from deliveryrisk.data.schema import TABLES, TABLES_BY_NAME, drop_ddl, schema_ddl, validate_tables

log = logging.getLogger(__name__)

EPOCH = _dt.datetime(2017, 1, 1, tzinfo=_dt.timezone.utc)


def to_datetime(seconds: pd.Series) -> pd.Series:
    """Epoch-seconds column to a pandas timestamp column, for humans and for plots."""
    return pd.to_datetime(seconds, unit="s", origin=pd.Timestamp(EPOCH).tz_localize(None)).dt.tz_localize("UTC")


class DatabaseError(RuntimeError):
    """Raised for a connection or load failure that is worth a readable message."""


@dataclass
class Database:
    """A connection target plus the operations the pipeline needs from it.

    Parameters
    ----------
    url:
        ``sqlite:///relative/or/absolute.db``, ``sqlite://:memory:``, or any SQLAlchemy MySQL URL
        (``mysql+pymysql://user:pass@host:3306/deliveryrisk``).
    """

    url: str = "sqlite:///data/deliveryrisk.db"

    def __post_init__(self) -> None:
        if self.url.startswith("sqlite"):
            self.dialect = "sqlite"
            self.path = self.url.split("sqlite://", 1)[1].lstrip("/")
            if self.url in ("sqlite://:memory:", "sqlite:///:memory:"):
                self.path = ":memory:"
            self._engine = None
            self._memory_conn: sqlite3.Connection | None = None
        elif self.url.startswith("mysql"):
            self.dialect = "mysql"
            self.path = None
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise DatabaseError(
                    "MySQL support needs the `mysql` extra: pip install -e '.[mysql]'"
                ) from exc
            self._engine = create_engine(self.url, pool_pre_ping=True)
        else:
            raise DatabaseError(f"unsupported database url {self.url!r}")

    # ------------------------------------------------------------------ connections
    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.dialect == "sqlite":
            if self.path == ":memory:":
                # An in-memory database dies with its connection, so one is held open for the
                # lifetime of the handle. Tests rely on this; file-backed runs do not.
                if self._memory_conn is None:
                    self._memory_conn = sqlite3.connect(":memory:")
                yield self._memory_conn
                return
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=OFF")  # a rebuildable analytics database
                conn.execute("PRAGMA foreign_keys=ON")
                yield conn
                conn.commit()
            finally:
                conn.close()
        else:
            with self._engine.begin() as conn:  # type: ignore[union-attr]
                yield conn

    # ------------------------------------------------------------------ schema + load
    def create_schema(self, *, drop: bool = True, foreign_keys: bool = True) -> None:
        with self.connect() as conn:
            script = ""
            if drop:
                script += drop_ddl(self.dialect)
            script += schema_ddl(self.dialect, foreign_keys=foreign_keys)
            self._exec_script(conn, script)
        log.info("schema created (%s, %d tables)", self.dialect, len(TABLES))

    def load_frames(self, frames: dict[str, pd.DataFrame], *, chunksize: int = 20_000) -> None:
        """Insert the nine frames, parents first, after validating them against the schema."""
        violations = validate_tables(frames)
        if violations:
            detail = "; ".join(f"{v.table}: {v.problem} {v.detail}".strip() for v in violations)
            raise DatabaseError(f"frames do not satisfy the schema -- {detail}")
        for table in TABLES:  # declaration order == foreign-key-safe load order
            df = frames[table.name][list(table.column_names)]
            self._insert(table.name, df, chunksize=chunksize)
            log.info("loaded %s: %d rows", table.name, len(df))

    def _insert(self, name: str, df: pd.DataFrame, *, chunksize: int) -> None:
        with self.connect() as conn:
            if self.dialect == "sqlite":
                cols = ", ".join(df.columns)
                marks = ", ".join("?" * len(df.columns))
                sql = f"INSERT INTO {name} ({cols}) VALUES ({marks})"
                rows = list(df.itertuples(index=False, name=None))
                conn.executemany(sql, rows)
            else:
                df.to_sql(name, conn, if_exists="append", index=False, chunksize=chunksize)

    # ------------------------------------------------------------------ reads
    def read_sql(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a query and return a DataFrame.

        Named parameters use ``:name`` in both dialects; the SQLite path binds them as a dict,
        which ``sqlite3`` supports natively, and the MySQL path hands them to SQLAlchemy.
        """
        with self.connect() as conn:
            if self.dialect == "sqlite":
                return pd.read_sql_query(sql, conn, params=params or {})
            from sqlalchemy import text

            return pd.read_sql_query(text(sql), conn, params=params or {})

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            if self.dialect == "sqlite":
                conn.execute(sql, params or {})
            else:
                from sqlalchemy import text

                conn.execute(text(sql), params or {})

    def executescript(self, script: str) -> None:
        with self.connect() as conn:
            self._exec_script(conn, script)

    def _exec_script(self, conn: Any, script: str) -> None:
        if self.dialect == "sqlite":
            conn.executescript(script)
            return
        from sqlalchemy import text

        for stmt in _split_statements(script):
            conn.execute(text(stmt))

    def table_counts(self) -> dict[str, int]:
        return {
            name: int(self.read_sql(f"SELECT COUNT(*) AS n FROM {name}")["n"].iloc[0])
            for name in TABLES_BY_NAME
        }

    def drop_temp(self, name: str) -> None:
        self.execute(f"DROP TABLE IF EXISTS {name}")


def _split_statements(script: str) -> list[str]:
    """Split a DDL script on semicolons, dropping comments and blanks.

    Good enough for the DDL this project emits -- no procedures, no string literals containing
    semicolons -- and the alternative is a SQL parser dependency for nine CREATE TABLEs.
    """
    out = []
    for raw in script.split(";"):
        stmt = "\n".join(
            line for line in raw.splitlines() if line.strip() and not line.strip().startswith("--")
        ).strip()
        if stmt:
            out.append(stmt)
    return out


def open_database(url: str, *, create: bool = False) -> Database:
    db = Database(url)
    if create:
        db.create_schema()
    return db
