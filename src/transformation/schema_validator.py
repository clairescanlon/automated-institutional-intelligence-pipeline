"""
schema_validator.py
-------------------
Pandera validation gate that enforces schema constraints on every batch
of records before they reach Aurora PostgreSQL.

Validation failures are raised as SchemaValidationError, which the
transformer catches per-document — bad data never silently reaches
production tables.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera as pa

from .models import (
    ActionSchema,
    ActualOutcomeSchema,
    DomainSchema,
    EntitySchema,
    LoserSchema,
    MarkerInversionSchema,
    RhetoricalMarkerSchema,
    StatedOutcomeSchema,
    WinnerSchema,
)

log = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when a record batch fails Pandera schema validation."""


# Map table name → Pandera schema
_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "domains":             DomainSchema,
    "stated_outcomes":     StatedOutcomeSchema,
    "actual_outcomes":     ActualOutcomeSchema,
    "winners":             WinnerSchema,
    "losers":              LoserSchema,
    "actions":             ActionSchema,
    "rhetorical_markers":  RhetoricalMarkerSchema,
    "marker_inversions":   MarkerInversionSchema,
    "entities":            EntitySchema,
}


def validate(table_name: str, records: list[dict]) -> pd.DataFrame:
    """
    Validate *records* against the schema for *table_name*.

    Parameters
    ----------
    table_name:
        One of the downstream table names (e.g. "domains", "entities").
    records:
        List of record dicts to validate.

    Returns
    -------
    pd.DataFrame
        Validated, coerced DataFrame ready for bulk insert.

    Raises
    ------
    SchemaValidationError
        If any row fails schema constraints.
    KeyError
        If *table_name* has no registered schema.
    """
    if not records:
        return pd.DataFrame()

    schema = _SCHEMAS.get(table_name)
    if schema is None:
        raise KeyError(f"No schema registered for table '{table_name}'.")

    df = pd.DataFrame(records)
    try:
        validated = schema.validate(df, lazy=True)
        log.debug("Validated | table=%s rows=%d", table_name, len(validated))
        return validated
    except pa.errors.SchemaErrors as exc:
        raise SchemaValidationError(
            f"Schema validation failed for '{table_name}': {exc.failure_cases.to_dict()}"
        ) from exc


def validate_all(result) -> dict[str, pd.DataFrame]:
    """
    Validate every table in a DocumentTransformResult.

    Returns
    -------
    dict[str, pd.DataFrame]
        Table name → validated DataFrame. Empty tables are omitted.
    """
    table_map = {
        "domains":             result.domains,
        "stated_outcomes":     result.stated_outcomes,
        "actual_outcomes":     result.actual_outcomes,
        "winners":             result.winners,
        "losers":              result.losers,
        "actions":             result.actions,
        "rhetorical_markers":  result.rhetorical_markers,
        "marker_inversions":   result.marker_inversions,
        "entities":            result.entities,
    }

    validated: dict[str, pd.DataFrame] = {}
    for table, records in table_map.items():
        if records:
            validated[table] = validate(table, records)

    return validated
