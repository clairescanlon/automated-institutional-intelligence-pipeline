"""
ingestor.py
-----------
Top-level orchestrator for the ingestion layer.

Flow per asset
--------------
1. Retrieve asset URL list from the configured REST API.
2. Download each binary asset to ephemeral local storage.
3. Validate (path safety, magic bytes, file size, filename).
4. Extract text via the hybrid extraction suite (pdfminer → PyPDF2 → OCR).
5. Construct a validated SourceDocument Pydantic model.
6. Upload raw binary to S3 (raw bucket, AES-256 SSE).
7. Upsert the SourceDocument record into Aurora PostgreSQL.
8. Log outcomes; errors on individual assets never crash the batch.

Entry point
-----------
    from src.ingestion import run_ingestion
    run_ingestion()
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import IngestionConfig
from .db_writer import upsert_source_document
from .models import SourceDocument
from .pdf_extractor import extract_text
from .s3_client import S3Error, upload_asset
from .validator import ValidationError, validate_asset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch result summary
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    total:    int = 0
    inserted: int = 0
    skipped:  int = 0
    failed:   int = 0
    errors:   list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP session with retry logic
# ---------------------------------------------------------------------------

def _build_http_session(cfg: IngestionConfig) -> requests.Session:
    """
    Create a requests Session with:
    * Retry on transient server errors (503, 429) with exponential back-off.
    * Auth header injected from config (never hard-coded).
    * No credentials stored on the session object beyond the lifetime of the call.
    """
    retry = Retry(
        total=cfg.max_retries,
        backoff_factor=1.5,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {cfg.source_api_key}",
        "Accept": "application/json",
        "User-Agent": "AIIP-Ingestion/1.0",
    })
    return session


# ---------------------------------------------------------------------------
# Asset retrieval
# ---------------------------------------------------------------------------

def _fetch_asset_list(session: requests.Session, cfg: IngestionConfig) -> list[dict]:
    """
    Call the source REST API and return a list of asset metadata dicts.

    Expected response shape
    -----------------------
    {
        "assets": [
            {"title": "...", "url": "https://...", "filename": "..."},
            ...
        ]
    }
    """
    url = f"{cfg.source_api_base_url.rstrip('/')}/assets"
    resp = session.get(url, timeout=cfg.request_timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    assets = data.get("assets", [])
    log.info("API returned %d asset(s).", len(assets))
    return assets


def _download_asset(url: str, dest: Path, session: requests.Session, cfg: IngestionConfig) -> None:
    """Stream-download an asset to *dest* without loading it fully into memory."""
    with session.get(url, stream=True, timeout=cfg.request_timeout_seconds) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Per-asset processing
# ---------------------------------------------------------------------------

def _process_asset(
    asset: dict,
    tmp_dir: Path,
    session: requests.Session,
    cfg: IngestionConfig,
    result: BatchResult,
) -> None:
    """
    Handle the full lifecycle of a single asset.
    All exceptions are caught so one failure never blocks the batch.
    """
    title    = asset.get("title", "untitled")
    url      = asset.get("url", "")
    filename = asset.get("filename", Path(url).name)

    result.total += 1
    local_path = tmp_dir / filename

    try:
        # 1. Download
        log.debug("Downloading | title=%s url=%s", title, url)
        _download_asset(url, local_path, session, cfg)

        # 2. Validate
        file_hash = validate_asset(local_path, tmp_dir, cfg.max_file_size_bytes)

        # 3. Extract text
        extraction = extract_text(local_path, language=cfg.ocr_language, dpi=cfg.ocr_dpi)

        # 4. Upload raw asset to S3
        s3_key = upload_asset(local_path, cfg)

        # 5. Build validated model
        doc = SourceDocument(
            title             = title,
            source_url        = url,
            s3_key            = s3_key,
            file_hash         = file_hash,
            file_size_bytes   = local_path.stat().st_size,
            page_count        = extraction.pages,
            raw_text          = extraction.text,
            extraction_method = extraction.method,
            warnings          = extraction.warnings,
        )

        # 6. Write to Aurora PostgreSQL
        inserted = upsert_source_document(doc, cfg)
        if inserted:
            result.inserted += 1
        else:
            result.skipped += 1

    except (ValidationError, S3Error, requests.RequestException) as exc:
        result.failed += 1
        msg = f"[{title}] {type(exc).__name__}: {exc}"
        result.errors.append(msg)
        log.error(msg)
    except Exception as exc:  # noqa: BLE001
        result.failed += 1
        msg = f"[{title}] Unexpected error: {exc}"
        result.errors.append(msg)
        log.exception(msg)
    finally:
        # Always clean up the local download regardless of outcome
        if local_path.exists():
            local_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ingestion(cfg: IngestionConfig | None = None) -> BatchResult:
    """
    Execute a full ingestion batch.

    Parameters
    ----------
    cfg:
        Optional pre-built IngestionConfig. If omitted, one is constructed
        from environment variables.

    Returns
    -------
    BatchResult
        Summary counts and any per-asset error messages.
    """
    cfg = cfg or IngestionConfig()
    result = BatchResult()

    log.info("Ingestion batch starting.")

    http = _build_http_session(cfg)

    try:
        assets = _fetch_asset_list(http, cfg)
    except requests.RequestException as exc:
        log.error("Failed to retrieve asset list: %s", exc)
        result.errors.append(str(exc))
        return result

    # Use a single temp directory for the whole batch (auto-cleaned on exit)
    with tempfile.TemporaryDirectory(prefix="aiip_batch_") as tmp:
        tmp_dir = Path(tmp)
        for asset in assets:
            _process_asset(asset, tmp_dir, http, cfg, result)

    log.info(
        "Ingestion batch complete | total=%d inserted=%d skipped=%d failed=%d",
        result.total, result.inserted, result.skipped, result.failed,
    )
    return result
