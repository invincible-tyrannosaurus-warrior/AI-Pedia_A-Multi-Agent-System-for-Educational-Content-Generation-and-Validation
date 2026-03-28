
from __future__ import annotations

import logging
import os
import subprocess
import shutil
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
    FITZ_IMPORT_ERROR = None
except ImportError as exc:
    fitz = None
    FITZ_AVAILABLE = False
    FITZ_IMPORT_ERROR = exc
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

class PPTConverter:
    """
    Converts PowerPoint slides to images using a two-step process:
    1. PPTX -> PDF (via LibreOffice headless)
    2. PDF -> Images (via PyMuPDF)
    
    Why this approach?
    - LibreOffice headless is cross-platform (Linux, macOS, Windows).
    - PDF export preserves vector fonts and exact layout.
    - PyMuPDF is faster and easier to install than pdf2image+poppler.
    """
    
    def __init__(self):
        if not FITZ_AVAILABLE:
            raise RuntimeError(
                "PyMuPDF is not installed. Install PyMuPDF before running the video agent. "
                f"Original import error: {FITZ_IMPORT_ERROR}"
            )
        if not shutil.which("libreoffice"):
            raise RuntimeError(
                "PPTConverter requires LibreOffice. "
                "Install via: apt-get install -y libreoffice (Linux) "
                "or download from https://www.libreoffice.org/ (Windows/macOS)"
            )

    def convert_to_images(
        self,
        pptx_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[Path]:
        """
        Convert the presentation at `pptx_path` to a list of image paths in `output_dir`.
        Returns sorted list of image paths.
        """
        pptx_path = pptx_path.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / f"{pptx_path.stem}.pdf"
        
        # Step 1: PPTX -> PDF
        self._pptx_to_pdf(pptx_path, pdf_path)
        if progress_callback is not None:
            progress_callback(
                stage_progress=0.25,
                message="Converted PPTX to PDF.",
            )
        
        # Step 2: PDF -> Images
        image_paths = self._pdf_to_images(pdf_path, output_dir, progress_callback=progress_callback)
        
        return image_paths

    def _pptx_to_pdf(self, pptx_path: Path, pdf_path: Path):
        """Use LibreOffice headless to export PPTX as PDF."""
        logger.info(f"Converting PPTX to PDF: {pptx_path} -> {pdf_path}")
        
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--norestore",
                "--convert-to", "pdf",
                "--outdir", str(pdf_path.parent),
                str(pptx_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed (exit {result.returncode}): {result.stderr}"
            )
        
        # LibreOffice outputs to <stem>.pdf in outdir
        # Rename if our desired pdf_path differs from what LibreOffice produces
        actual = pdf_path.parent / f"{pptx_path.stem}.pdf"
        if actual != pdf_path and actual.exists():
            actual.rename(pdf_path)
        
        if not pdf_path.exists():
            raise RuntimeError(
                f"PDF not found after conversion. Expected at: {pdf_path}"
            )

    def _pdf_to_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[Path]:
        """Use PyMuPDF to render PDF pages as high-res images."""
        logger.info(f"Rendering PDF to Images: {pdf_path}")
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for i, page in enumerate(doc):
                # 2.0 = 2x zoom (High DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                
                image_name = f"slide_{i+1:03d}.png"
                image_path = output_dir / image_name
                pix.save(str(image_path))
                
                image_paths.append(image_path)
                if progress_callback is not None:
                    progress = 0.25 + (0.75 * ((i + 1) / max(total_pages, 1)))
                    progress_callback(
                        stage_progress=progress,
                        current=i + 1,
                        total=total_pages,
                        message=f"Rendered slide {i + 1} of {total_pages}.",
                    )
                
            doc.close()
        except Exception as e:
            logger.error(f"Failed during PDF->Image rendering: {e}")
            raise
            
        return sorted(image_paths)

if __name__ == "__main__":
    # Internal test
    import sys
    converter = PPTConverter()
    # converter.convert_to_images(Path("test.pptx"), Path("output"))
