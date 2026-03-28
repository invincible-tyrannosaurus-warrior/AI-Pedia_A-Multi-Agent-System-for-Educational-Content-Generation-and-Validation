# Local storage helpers for user uploads and generated artifacts.

from __future__ import annotations

import mimetypes
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from fastapi import UploadFile

from ..utils.file_processors import extract_text

logger = logging.getLogger(__name__)


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

# Persist an uploaded file and return its descriptor
async def persist_upload(
    upload: UploadFile,
    upload_dir: Path,
    base_url: str,
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

    try:
        upload.file.seek(0)
    except Exception:
        pass

    with target_path.open("wb") as buffer:
        buffer.write(upload.file.read())

    mime_type = upload.content_type or _guess_mime_type(upload.filename or unique_name)
    url = urljoin(base_url.rstrip("/") + "/", unique_name)

    extracted_text = extract_text(target_path, mime_type)

    # RAG Ingestion (New)
    try:
        if extracted_text and len(extracted_text) > 50:
            import asyncio
            from ai_pedia_mcp_server.mcp_tools.rag_search import RAGEngine
            
            # Run ingestion in a separate thread to avoid blocking the main event loop
            # This is critical because embedding calculation is CPU intensive
            engine = RAGEngine.get_instance()
            
            metadata = {
                "source": url,
                "filename": unique_name,
                "mime_type": mime_type
            }
            # We await the thread execution so we don't return until it's "accepted" 
            # (or we could fire-and-forget if we want faster UI response, but awaiting is safer for now)
            await asyncio.to_thread(engine.ingest_document, extracted_text, metadata)
    except Exception as e:
        # Don't fail the upload just because RAG failed
        logger.warning("RAG ingestion failed for %s: %s", target_path, e)

    return StoredAsset(
        path=target_path,
        url=url,
        mime_type=mime_type,
        original_filename=unique_name, # IMPORTANT: Agent must know the ACTUAL filename on disk
        extracted_text=extracted_text,
    )


__all__ = ["StoredAsset", "persist_upload"]
