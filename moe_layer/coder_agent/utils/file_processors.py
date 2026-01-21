#  Utility helpers for extracting text snippets from uploaded assets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Extract text from a PDF file when the pypdf dependency is available.
def extract_text_from_pdf(path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:  # pragma: no cover
        logger.warning("pypdf not available; skipping PDF extraction for %s", path)
        return None

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pragma: no cover
                continue
        text = "\n".join(pages).strip()
        return text or None
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read PDF %s: %s", path, exc)
        return None


# Perform OCR on an image when Pillow and pytesseract are installed.
def extract_text_from_image(path: Path) -> Optional[str]:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:  # pragma: no cover
        logger.warning("pytesseract/Pillow not available; skipping OCR for %s", path)
        return None

    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
            return text.strip() or None
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to OCR image %s: %s", path, exc)
        return None


def extract_text(path: Path, mime_type: str) -> Optional[str]:
    # Route to the appropriate extractor based on the file MIME type.
    if mime_type == "application/pdf":
        return extract_text_from_pdf(path)
    if mime_type.startswith("image/"):
        return extract_text_from_image(path)
    return None


def summarize_text(text: Optional[str], max_chars: int = 1500) -> Optional[str]:
    # Trim extracted text to a safe prompt length while preserving content.
    if not text:
        return None
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


__all__ = [
    "extract_text",
    "extract_text_from_pdf",
    "extract_text_from_image",
    "summarize_text",
]
