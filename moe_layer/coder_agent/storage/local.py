# Local storage helpers for user uploads and generated artifacts.

from __future__ import annotations

import asyncio
import json
import mimetypes
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from fastapi import UploadFile

from ..utils.file_processors import extract_text

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = ".source_registry.json"
_REGISTRY_LOCK = threading.Lock()
_TEXT_CACHE_DIRNAME = ".source_text_cache"
_SECTION_LABEL_PATTERN = r"(?:chapter|chap(?:ter)?|ch|section|sec(?:tion)?|topic)"
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "about",
    "into",
    "chapter",
    "topic",
    "please",
    "show",
    "explain",
    "what",
    "when",
    "where",
    "which",
    "me",
}
_WORD_TO_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_WORD_PATTERN = "|".join(sorted(_WORD_TO_NUMBER.keys(), key=len, reverse=True))
_INT_TO_WORD = {value: key for key, value in _WORD_TO_NUMBER.items()}


@dataclass
# Capture persistent asset info
class StoredAsset:
    path: Path
    url: str
    mime_type: str
    original_filename: str
    extracted_text: Optional[str] = None

    # convert the metadata into the structure downstream agents expect
    def as_descriptor(self) -> dict:
        return {
            "type": "file",
            "url": self.url,
            "mime_type": self.mime_type,
            "description": f"Uploaded asset: {self.original_filename}",
        }

# provide a fallback mime type guessing function
def _guess_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(upload_dir: Path) -> Path:
    return upload_dir / _REGISTRY_FILENAME


def _text_cache_dir(upload_dir: Path) -> Path:
    return upload_dir / _TEXT_CACHE_DIRNAME


def _text_cache_path(upload_dir: Path, stored_name: str) -> Path:
    return _text_cache_dir(upload_dir) / f"{stored_name}.txt"


def _load_registry(upload_dir: Path) -> list[dict]:
    path = _registry_path(upload_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _save_registry(upload_dir: Path, records: list[dict]) -> None:
    path = _registry_path(upload_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")


def _upsert_registry_record(upload_dir: Path, record: dict) -> None:
    with _REGISTRY_LOCK:
        records = _load_registry(upload_dir)
        url = str(record.get("url", "")).strip()
        updated = False
        for index, existing in enumerate(records):
            if existing.get("url") == url:
                merged = dict(existing)
                merged.update(record)
                records[index] = merged
                updated = True
                break
        if not updated:
            records.append(record)
        _save_registry(upload_dir, records)


def _update_registry_record(upload_dir: Path, url: str, updates: dict) -> None:
    with _REGISTRY_LOCK:
        records = _load_registry(upload_dir)
        for index, existing in enumerate(records):
            if existing.get("url") == url:
                merged = dict(existing)
                merged.update(updates)
                records[index] = merged
                _save_registry(upload_dir, records)
                return
        fallback = {"url": url}
        fallback.update(updates)
        records.append(fallback)
        _save_registry(upload_dir, records)


def _resolve_source_ref(ref: str) -> tuple[str, str]:
    normalized = (ref or "").strip()
    basename = ""
    if normalized:
        basename = Path(normalized).name
    return normalized, basename


def _persist_text_cache(upload_dir: Path, stored_name: str, text: str) -> Optional[str]:
    if not text:
        return None
    cache_path = _text_cache_path(upload_dir, stored_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8", errors="replace")
    return str(cache_path)


def _extract_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z0-9_]+", (query or "").lower())
    return [term for term in terms if len(term) >= 3 and term not in _STOPWORDS]


def _slice_window(text: str, center: int, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    start = max(0, center - half)
    end = min(len(text), start + max_chars)
    return text[start:end]


def _roman_to_int(token: str) -> Optional[int]:
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    cleaned = (token or "").strip().upper()
    if not cleaned or any(ch not in roman_values for ch in cleaned):
        return None

    total = 0
    previous = 0
    for ch in reversed(cleaned):
        value = roman_values[ch]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if total > 0 else None


def _int_to_roman(value: int) -> Optional[str]:
    if value <= 0 or value > 3999:
        return None
    mapping = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    remaining = value
    output = []
    for base, symbol in mapping:
        while remaining >= base:
            output.append(symbol)
            remaining -= base
    return "".join(output) or None


def _extract_requested_section_number(query: str) -> Optional[int]:
    if not query:
        return None
    match = re.search(
        rf"\b{_SECTION_LABEL_PATTERN}\.?\s*([0-9]{{1,3}}|[ivxlcdm]+|{_NUMBER_WORD_PATTERN})\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw_token = (match.group(1) or "").strip().lower()
    if raw_token.isdigit():
        return int(raw_token)
    if raw_token in _WORD_TO_NUMBER:
        return _WORD_TO_NUMBER[raw_token]
    return _roman_to_int(raw_token)


def _section_tokens(section_num: int) -> list[str]:
    tokens: list[str] = [str(section_num)]
    word = _INT_TO_WORD.get(section_num)
    if word:
        tokens.append(word)
    roman = _int_to_roman(section_num)
    if roman:
        tokens.append(roman)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(tokens))


def _find_section_heading(text: str, section_num: int) -> Optional[re.Match[str]]:
    for token in _section_tokens(section_num):
        match = re.search(
            rf"\b{_SECTION_LABEL_PATTERN}\.?\s*{re.escape(token)}\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match
    return None


def _find_chapter_section(text: str, query: str, max_chars: int) -> Optional[str]:
    section_num = _extract_requested_section_number(query)
    if section_num is None:
        return None

    start_match = _find_section_heading(text, section_num)
    if not start_match:
        return None

    start = start_match.start()

    next_match = _find_section_heading(text[start + 20 :], section_num + 1)
    if next_match:
        end = start + 20 + next_match.start()
        chunk = text[start:end].strip()
        if chunk:
            return chunk[:max_chars]
    return _slice_window(text, start, max_chars)


def _build_query_snippet(text: str, query: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    chapter_chunk = _find_chapter_section(text, query, max_chars=max_chars)
    if chapter_chunk:
        return chapter_chunk

    lowered = text.lower()
    for term in sorted(_extract_query_terms(query), key=len, reverse=True):
        idx = lowered.find(term)
        if idx >= 0:
            return _slice_window(text, idx, max_chars)

    return text[:max_chars]


def get_uploaded_source_record(upload_dir: Path, ref: str) -> Optional[dict]:
    normalized, basename = _resolve_source_ref(ref)
    with _REGISTRY_LOCK:
        records = _load_registry(upload_dir)
    for item in records:
        if normalized and item.get("url") == normalized:
            return dict(item)
        if basename and item.get("stored_name") == basename:
            return dict(item)
        if basename and item.get("display_name") == basename:
            return dict(item)
        if normalized and item.get("path") == normalized:
            return dict(item)
    return None


def list_uploaded_sources(upload_dir: Path) -> list[dict]:
    with _REGISTRY_LOCK:
        records = _load_registry(upload_dir)
    sorted_records = sorted(
        records,
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )
    compact = []
    for item in sorted_records:
        status = item.get("status", "unknown")
        error = item.get("error")
        index_mode = item.get("index_mode", "unknown")
        # Backward compatibility: old records marked failed only because chromadb was missing.
        if status == "failed" and isinstance(error, str) and "chromadb" in error.lower():
            status = "ready"
            error = None
            index_mode = "text_only"

        compact.append(
            {
                "path": item.get("url"),
                "display_name": item.get("display_name") or item.get("stored_name"),
                "stored_name": item.get("stored_name"),
                "status": status,
                "error": error,
                "updated_at": item.get("updated_at") or item.get("created_at"),
                "index_mode": index_mode,
            }
        )
    return compact


def get_uploaded_source_context(
    upload_dir: Path,
    ref: str,
    *,
    query: Optional[str] = None,
    max_chars: int = 12000,
) -> Optional[str]:
    record = get_uploaded_source_record(upload_dir, ref)
    if not record:
        return None

    text = ""
    cache_path = record.get("text_cache_path")
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                text = cache_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""

    if not text:
        text = record.get("preview_text") or ""

    if not text:
        record_path = record.get("path")
        mime_type = record.get("mime_type", "application/octet-stream")
        stored_name = record.get("stored_name") or Path(record_path or "").name
        if record_path:
            source_path = Path(record_path)
            if source_path.exists():
                try:
                    recovered_text = extract_text(source_path, mime_type) or ""
                except Exception:
                    recovered_text = ""
                if recovered_text:
                    text = recovered_text
                    new_cache_path = _persist_text_cache(upload_dir, stored_name, recovered_text)
                    _update_registry_record(
                        upload_dir,
                        str(record.get("url") or ref),
                        {
                            "status": "ready",
                            "error": None,
                            "updated_at": _now_iso(),
                            "text_chars": len(recovered_text),
                            "preview_text": recovered_text[:8000],
                            "text_cache_path": new_cache_path,
                            "index_mode": record.get("index_mode") or "text_only",
                        },
                    )

    if not text:
        return None

    return _build_query_snippet(text, query or "", max_chars=max_chars)


def _ingest_uploaded_source(
    *,
    target_path: Path,
    mime_type: str,
    url: str,
    display_name: str,
    stored_name: str,
    upload_dir: Path,
) -> None:
    extracted_text: Optional[str] = None
    indexed_chunks = 0
    status = "ready"
    error = None
    index_mode = "text_only"
    index_warning = None
    text_cache_path = None

    try:
        extracted_text = extract_text(target_path, mime_type)
        if extracted_text:
            text_cache_path = _persist_text_cache(upload_dir, stored_name, extracted_text)
        if extracted_text and len(extracted_text) > 50:
            try:
                from ai_pedia_mcp_server.mcp_tools.rag_search import RAGEngine

                engine = RAGEngine.get_instance()
                metadata = {
                    "source": url,
                    "filename": stored_name,
                    "display_name": display_name,
                    "mime_type": mime_type,
                }
                indexed_chunks = int(engine.ingest_document(extracted_text, metadata))
                index_mode = "vector"
            except Exception as exc:
                index_mode = "text_only"
                index_warning = str(exc)
                logger.warning(
                    "Vector indexing unavailable for %s, fallback to text-only context: %s",
                    target_path,
                    exc,
                )
    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.warning("Background ingestion failed for %s: %s", target_path, exc)

    preview_text = (extracted_text or "")[:8000]
    _update_registry_record(
        upload_dir,
        url,
        {
            "status": status,
            "error": error,
            "updated_at": _now_iso(),
            "indexed_chunks": indexed_chunks,
            "text_chars": len(extracted_text or ""),
            "preview_text": preview_text,
            "text_cache_path": text_cache_path,
            "index_mode": index_mode,
            "index_warning": index_warning,
        },
    )


# Persist an uploaded file and return its descriptor
async def persist_upload(
    upload: UploadFile,
    upload_dir: Path,
    base_url: str,
    ingest_async: bool = True,
) -> StoredAsset:
   # Store an uploaded file locally and build its pipeline descriptor.  
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename using timestamp + original name to prevent "hallucinations"
    # Agent sees "ML_example.pdf", file on disk is "ML_example.pdf" (or similar)
    import time
    timestamp = int(time.time() * 1000)
    
    # Safe regex for filename
    import re
    SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
    
    original_stem = Path(upload.filename or "upload").stem
    suffix = Path(upload.filename or "").suffix
    
    # Clean stem
    safe_stem = SAFE_NAME_RE.sub("_", original_stem).strip("_")
    
    # Format: 1736281234000_a1b2c3d4_ML_Example.pdf
    unique_name = f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_stem}{suffix}"
    target_path = upload_dir / unique_name
    display_name = Path(upload.filename or unique_name).name

    try:
        upload.file.seek(0)
    except Exception:
        pass

    with target_path.open("wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    mime_type = upload.content_type or _guess_mime_type(upload.filename or unique_name)
    url = urljoin(base_url.rstrip("/") + "/", unique_name)
    created_at = _now_iso()
    _upsert_registry_record(
        upload_dir,
        {
            "source_id": uuid.uuid4().hex,
            "url": url,
            "path": str(target_path),
            "display_name": display_name,
            "stored_name": unique_name,
            "mime_type": mime_type,
            "status": "processing",
            "error": None,
            "created_at": created_at,
            "updated_at": created_at,
            "indexed_chunks": 0,
            "text_chars": 0,
            "preview_text": "",
        },
    )

    if ingest_async:
        worker = threading.Thread(
            target=_ingest_uploaded_source,
            kwargs={
                "target_path": target_path,
                "mime_type": mime_type,
                "url": url,
                "display_name": display_name,
                "stored_name": unique_name,
                "upload_dir": upload_dir,
            },
            daemon=True,
        )
        worker.start()
        extracted_text = None
    else:
        await asyncio.to_thread(
            _ingest_uploaded_source,
            target_path=target_path,
            mime_type=mime_type,
            url=url,
            display_name=display_name,
            stored_name=unique_name,
            upload_dir=upload_dir,
        )
        latest = get_uploaded_source_record(upload_dir, url) or {}
        extracted_text = latest.get("preview_text") or None

    return StoredAsset(
        path=target_path,
        url=url,
        mime_type=mime_type,
        original_filename=unique_name, # IMPORTANT: Agent must know the ACTUAL filename on disk
        extracted_text=extracted_text,
    )


__all__ = [
    "StoredAsset",
    "persist_upload",
    "list_uploaded_sources",
    "get_uploaded_source_record",
    "get_uploaded_source_context",
]
