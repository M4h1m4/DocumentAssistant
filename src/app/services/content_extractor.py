from __future__ import annotations

from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

from .hashing import decode_text

# Import PDF processing functions
from .pdf_extractor import extract_pdf_content, PDFExtractionError, PDFContent

# Import image processing for combining text + images
from .image_processor import combine_text_images

log = logging.getLogger("precisbox.services.content_extractor")

@dataclass
class ExtractedContent:
    text: str
    mime_type: str
    metadata: dict

class ContentExtractionError(RuntimeError):
    """Raised when content extraction fails."""
    pass

def extract_content(
    file_bytes: bytes, 
    mime_type: str, 
    extract_images: bool = True, 
    use_ocr: bool = False
) -> ExtractedContent:
    try:
        if mime_type in ("text/plain", "text/markdown"):
            # WHY: Text files are simple - just decode UTF-8
            return _extract_text_content(file_bytes, mime_type)
        elif mime_type == "application/pdf":
            return _extract_pdf_content_wrapper(file_bytes, extract_images, use_ocr)
        else:
            raise ContentExtractionError(f"Unsupported MIME type: {mime_type}")
    except ContentExtractionError:
        raise
    except Exception as e:
        raise ContentExtractionError(f"Failed to extract content: {e}") from e

def _extract_text_content(
    file_bytes: bytes, 
    mime_type: str, 
) -> ExtractedContent:
    try:
        text = decode_text(file_bytes)
        return ExtractedContent(
            text = text, 
            mime_type=mime_type,
            metadata={}
        )
    except UnicodeDecodeError as e:
        raise ContentExtractionError(f"File must be UTF-8 encoded text: {e}") from e


def _extract_pdf_content_wrapper(
    file_bytes: bytes,
    extract_images: bool,
    use_ocr: bool
) -> ExtractedContent:
    try:
        pdf_content=extract_pdf_content(file_bytes, extract_images=extract_images)
        combined_text = combine_text_images(
            pdf_content.text,
            pdf_content.images,
            use_ocr=use_ocr,
        )
        metadata = {
            "total_pages": pdf_content.total_pages,  
            "image_count": len(pdf_content.images),  
            "pdf_metadata": pdf_content.metadata,
        }

        return ExtractedContent(
            text=combined_text,
            mime_type="application/pdf",
            metadata=metadata,
        )
    except PDFExtractionError as e:
        raise ContentExtractionError(f"PDF extraction failed: {e}") from e
