
from __future__ import annotations

import logging
import os
import comtypes.client
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class PPTConverter:
    """
    Converts PowerPoint slides to images using a two-step process:
    1. PPTX -> PDF (via Windows COM / Microsoft PowerPoint)
    2. PDF -> Images (via PyMuPDF)
    
    Why this approach?
    - Direct export to images via COM often yields poor quality or artifacts.
    - PDF export preserves vector fonts and exact layout.
    - PyMuPDF is faster and easier to install than pdf2image+poppler.
    """
    
    def __init__(self):
        # Ensure we are on Windows for COM
        if os.name != 'nt':
            raise RuntimeError("PPTConverter requires Windows with Microsoft PowerPoint installed.")

    def convert_to_images(self, pptx_path: Path, output_dir: Path) -> List[Path]:
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
        
        # Step 2: PDF -> Images
        image_paths = self._pdf_to_images(pdf_path, output_dir)
        
        return image_paths

    def _pptx_to_pdf(self, pptx_path: Path, pdf_path: Path):
        """Use PowerPoint COM interface to export as PDF."""
        logger.info(f"Converting PPTX to PDF: {pptx_path} -> {pdf_path}")
        powerpoint = None
        presentation = None
        
        try:
            # Initialize PowerPoint
            powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
            # powerpoint.Visible = True # Debugging
            
            # Open presentation (ReadOnly=True, Untitled=False, WithWindow=False)
            presentation = powerpoint.Presentations.Open(str(pptx_path), True, False, False)
            
            # Save as PDF
            # 32 = ppSaveAsPDF
            presentation.SaveAs(str(pdf_path), 32)
            
        except Exception as e:
            logger.error(f"Failed during PPTX->PDF conversion: {e}")
            raise
        finally:
            if presentation:
                presentation.Close()
            if powerpoint:
                # We don't Quit because user might have other files open
                # But for headless server mode, maybe we should?
                # Let's just release logic
                pass

    def _pdf_to_images(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Use PyMuPDF to render PDF pages as high-res images."""
        logger.info(f"Rendering PDF to Images: {pdf_path}")
        image_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                # 2.0 = 2x zoom (High DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                
                image_name = f"slide_{i+1:03d}.png"
                image_path = output_dir / image_name
                pix.save(str(image_path))
                
                image_paths.append(image_path)
                
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
