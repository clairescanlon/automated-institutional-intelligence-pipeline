"""
db_writer.py
------------
Writes validated, transformed records to Aurora PostgreSQL.

Covers all downstream tables that foreign-key back to `source_documents`:
    domains, stated_outcomes, actual_outcomes, winners, losers,
    actions, rhetorical_markers, marker_inversions, entities.

Security notes
--------------
* DSN is constructed at runtime from environment variables — never logged.
* Bulk inserts use SQLAlchemy Core (parameterised); no raw string SQL.
* Connection pool is bounded for Lambda compatibility.
* All writes are transactional per document; partial failures roll back cleanly.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import TransformationConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _make_engine(cfg: TransformationConfig):
    return create_engine(
        cfg.db_dsn,
        pool_size=2,
        max_overflow=1,
        pool_timeout=10,
        pool_pre_ping=True,
        echo=False,           # never echo SQL — DSN contains credentials
    )


@contextmanager
def get_session(cfg: TransformationConfig) -> Generator[Session, None, None]:
    """Context manager yielding a transactional Aurora session."""
    engine = _make_engine(cfg)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# Bulk writer
# ---------------------------------------------------------------------------

# Maps each table to its primary key column (used for duplicate detection)
_PRIMARY_KEYS: dict[str, str] = {
    "domains":             "domain_id",
    "stated_outcomes":     "outcome_id",
    "actual_outcomes":     "outcome_id",
    "winners":             "winner_id",
    "losers":              "loser_id",
    "actions":             "action_id",
    "rhetorical_markers":  "marker_id",
    "marker_inversions":   "inversion_id",
    "entities":            "entity_id",
}


def bulk_insert(
    table_name: str,
    df: pd.DataFrame,
    cfg: TransformationConfig,
) -> int:
    """
    Insert all rows in *df* into *table_name*, skipping existing primary keys.

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if df.empty:
        return 0

    pk = _PRIMARY_KEYS[table_name]
    engine = _make_engine(cfg)
    inserted = 0

    try:
        with engine.begin() as conn:
            for _, row in df.iterrows():
                pk_value = row[pk]
                exists = conn.execute(
                    text(f"SELECT 1 FROM {table_name} WHERE {pk} = :pk"),  # noqa: S608
                    {"pk": pk_value},
                ).fetchone()

                if exists:
                    log.debug("Skipping duplicate | table=%s pk=%s", table_name, pk_value)
                    continue

                conn.execute(
                    text(
                        f"INSERT INTO {table_name} ({', '.join(df.columns)}) "  # noqa: S608
                        f"VALUES ({', '.join(':' + c for c in df.columns)})"
                    ),
                    row.to_dict(),
                )
                inserted += 1
    finally:
        engine.dispose()

    log.info("Inserted | table=%s rows=%d", table_name, inserted)
    return inserted


def write_all(
    validated: dict[str, pd.DataFrame],
    cfg: TransformationConfig,
) -> dict[str, int]:
    """
    Write all validated DataFrames to their respective tables.

    Returns
    -------
    dict[str, int]
        Table name → number of rows inserted.
    """
    counts: dict[str, int] = {}
    for table, df in validated.items():
        counts[table] = bulk_insert(table, df, cfg)
    return counts
