"""
transformer.py
--------------
Top-level orchestrator for the transformation layer.

Flow per document
-----------------
1. Fetch unprocessed `source_documents` records from Aurora.
2. Run rhetoric neutralization (markers, SVO, stated outcomes, actions).
3. Run entity mapping (NER + resolution).
4. Run domain classification; link actions to domains.
5. Validate all output records with Pandera before any write.
6. Bulk-insert validated records into downstream tables.
7. Mark the source document as transformed.
8. Log outcomes; errors per document never crash the batch.

Entry point
-----------
    from src.transformation import run_transformation
    run_transformation()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text

from .config import TransformationConfig
from .db_writer import write_all
from .domain_classifier import classify_domains, link_actions_to_domains
from .entity_mapper import extract_entities
from .models import DocumentTransformResult
from .rhetoric_neutralizer import (
    build_marker_inversions,
    extract_actions,
    extract_rhetorical_markers,
    extract_stated_outcomes,
    extract_svo_triplets,
)
from .schema_validator import SchemaValidationError, validate_all

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch result summary  (mirrors ingestion.BatchResult)
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    total:    int = 0
    success:  int = 0
    failed:   int = 0
    errors:   list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source document fetching
# ---------------------------------------------------------------------------

def _fetch_pending(cfg: TransformationConfig) -> list[dict]:
    """
    Return source documents that have not yet been transformed.
    Uses a `transformed_at` column (NULL = pending) on `source_documents`.
    """
    engine = create_engine(cfg.db_dsn, echo=False)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT document_id, title, raw_text "
                    "FROM source_documents "
                    "WHERE transformed_at IS NULL "
                    "ORDER BY ingestion_timestamp ASC "
                    "LIMIT :limit"
                ),
                {"limit": cfg.batch_size},
            ).mappings().all()
        return [dict(r) for r in rows]
    finally:
        engine.dispose()


def _mark_transformed(document_id: str, cfg: TransformationConfig) -> None:
    """Stamp `transformed_at` on the source document record."""
    engine = create_engine(cfg.db_dsn, echo=False)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE source_documents "
                    "SET transformed_at = NOW() "
                    "WHERE document_id = :doc_id"
                ),
                {"doc_id": document_id},
            )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Per-document transformation
# ---------------------------------------------------------------------------

def _transform_document(doc: dict, cfg: TransformationConfig) -> DocumentTransformResult:
    """
    Run the full transformation pipeline for a single source document.
    All sub-steps are isolated; failures surface as exceptions to the caller.
    """
    document_id = doc["document_id"]
    text        = doc["raw_text"]

    result = DocumentTransformResult(document_id=document_id)

    # 1. Rhetorical markers
    result.rhetorical_markers = extract_rhetorical_markers(
        document_id, text, threshold=cfg.rhetoric_threshold
    )

    # 2. SVO triplets (used by inversion builder)
    svo_triplets = extract_svo_triplets(text, model_name=cfg.spacy_model)

    # 3. Stated outcomes
    result.stated_outcomes = extract_stated_outcomes(document_id, text)

    # 4. Actions
    result.actions = extract_actions(document_id, text)

    # 5. Marker inversions (rhetoric-to-reality gap)
    result.marker_inversions = build_marker_inversions(
        document_id, result.rhetorical_markers, svo_triplets
    )

    # 6. Entity mapping
    result.entities = extract_entities(
        document_id, text,
        model_name=cfg.spacy_model,
        min_confidence=cfg.entity_confidence,
    )

    # 7. Domain classification
    result.domains = classify_domains(document_id, text)

    # 8. Link actions → domains
    result.actions = link_actions_to_domains(result.actions, result.domains)

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_transformation(cfg: TransformationConfig | None = None) -> BatchResult:
    """
    Execute a full transformation batch.

    Parameters
    ----------
    cfg:
        Optional pre-built TransformationConfig. If omitted, one is
        constructed from environment variables.

    Returns
    -------
    BatchResult
        Summary counts and per-document error messages.
    """
    cfg = cfg or TransformationConfig()
    result = BatchResult()

    log.info("Transformation batch starting.")

    pending = _fetch_pending(cfg)
    log.info("Pending documents | count=%d", len(pending))

    for doc in pending:
        document_id = doc["document_id"]
        title       = doc.get("title", document_id)
        result.total += 1

        try:
            # Transform
            transform_result = _transform_document(doc, cfg)

            # Validate all output tables with Pandera
            validated = validate_all(transform_result)

            # Write to Aurora
            counts = write_all(validated, cfg)
            log.info("Written | document_id=%s tables=%s", document_id, counts)

            # Mark complete
            _mark_transformed(document_id, cfg)
            result.success += 1

        except SchemaValidationError as exc:
            result.failed += 1
            msg = f"[{title}] Validation error: {exc}"
            result.errors.append(msg)
            log.error(msg)

        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            msg = f"[{title}] Unexpected error: {exc}"
            result.errors.append(msg)
            log.exception(msg)

    log.info(
        "Transformation batch complete | total=%d success=%d failed=%d",
        result.total, result.success, result.failed,
    )
    return result
