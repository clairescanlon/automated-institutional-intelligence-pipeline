"""
pdf_extractor.py
----------------
Hybrid text-recovery suite for the ingestion layer.

Extraction strategy (in order)
-------------------------------
1. pdfminer.six   – high-fidelity layout-aware parsing for digital-native assets.
2. PyPDF2         – fallback for files that pdfminer cannot parse cleanly.
3. Tesseract OCR  – final fallback for image-based / legacy scanned assets.

Security notes
--------------
* No subprocess(shell=True) calls anywhere in this module.
* Temporary files are written to an isolated, automatically-cleaned directory.
* All compiled regexes are module-level constants (thread-safe after import).
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

from .models import ExtractionMethod

log = logging.getLogger(__name__)

# Minimum substantive character count before we consider direct extraction sufficient
_MIN_DIRECT_CHARS = 100

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ExtractionResult(NamedTuple):
    text:      str
    pages:     int
    method:    ExtractionMethod
    warnings:  list[str]


# ---------------------------------------------------------------------------
# Direct extraction helpers
# ---------------------------------------------------------------------------

def _try_pdfminer(path: Path) -> tuple[str, int] | None:
    """Attempt extraction with pdfminer.six. Returns (text, page_count) or None."""
    try:
        from pdfminer.high_level import extract_pages, extract_text
        from pdfminer.layout import LAParams

        params = LAParams(line_margin=0.5, char_margin=2.0)
        text = extract_text(str(path), laparams=params)
        pages = sum(1 for _ in extract_pages(str(path)))
        return text, pages
    except Exception as exc:  # noqa: BLE001
        log.warning("pdfminer failed | file=%s error=%s", path.name, exc)
        return None


def _try_pypdf2(path: Path) -> tuple[str, int] | None:
    """Attempt extraction with PyPDF2. Returns (text, page_count) or None."""
    try:
        import PyPDF2

        parts: list[str] = []
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            pages = len(reader.pages)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts), pages
    except Exception as exc:  # noqa: BLE001
        log.warning("PyPDF2 failed | file=%s error=%s", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------

def _extract_ocr(path: Path, language: str, dpi: int) -> tuple[str, int]:
    """
    Render each page to an image and run Tesseract OCR.
    Uses an isolated temporary directory; no shell calls.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies missing. Install with: pip install pytesseract pdf2image"
        ) from exc

    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="aiip_ocr_") as tmp:
        images = convert_from_path(
            str(path),
            dpi=dpi,
            output_folder=tmp,
            fmt="png",
            thread_count=1,      # deterministic; avoids race conditions in Lambda
        )
        for img in images:
            parts.append(pytesseract.image_to_string(img, lang=language))

    return "\n".join(parts), len(images)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZ_SPACE_RE    = re.compile(r"[ \t]+")
_EXCESS_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean(raw: str) -> str:
    """Strip control characters, normalise whitespace, collapse blank lines."""
    text = _CONTROL_CHAR_RE.sub("", raw)
    text = _HORIZ_SPACE_RE.sub(" ", text)
    text = _EXCESS_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_text(path: Path, language: str = "eng", dpi: int = 300) -> ExtractionResult:
    """
    Extract text from *path* using the best available method.

    Parameters
    ----------
    path:
        Validated, resolved path to the asset on disk.
    language:
        Tesseract language code (used only if OCR fallback is triggered).
    dpi:
        Page-render resolution for OCR (default 300 DPI).

    Returns
    -------
    ExtractionResult
        Named tuple: text, page count, method used, and any warnings.
    """
    warnings: list[str] = []

    # --- Pass 1: pdfminer ---
    result = _try_pdfminer(path)
    if result and len(result[0].strip()) >= _MIN_DIRECT_CHARS:
        text, pages = result
        return ExtractionResult(_clean(text), pages, ExtractionMethod.DIRECT_PDFMINER, warnings)

    if result:
        warnings.append("pdfminer returned minimal text; trying PyPDF2.")

    # --- Pass 2: PyPDF2 ---
    result = _try_pypdf2(path)
    if result and len(result[0].strip()) >= _MIN_DIRECT_CHARS:
        text, pages = result
        return ExtractionResult(_clean(text), pages, ExtractionMethod.DIRECT_PYPDF2, warnings)

    warnings.append("Direct extraction insufficient; engaging OCR fallback.")
    log.info("OCR fallback triggered | file=%s", path.name)

    # --- Pass 3: Tesseract OCR ---
    text, pages = _extract_ocr(path, language=language, dpi=dpi)
    return ExtractionResult(_clean(text), pages, ExtractionMethod.OCR, warnings)
