from __future__ import annotations

import base64
from typing import List, Optional, Dict, Tuple, Any
import logging 
from io import BytesIO

try:
    from PIL import Image as PILImage
    import pytesseract  # OCR library (requires Tesseract binary installed)
except ImportError:
    PILImage = None 
    pytesseract = None

from .pdf_extractor import ExtractedImage  # Our dataclass from previous step

log = logging.getLogger("precisbox.services.image_processor")


class ImageProcessingError(RuntimeError):
    """Raised when image processing fails."""
    pass

def extract_image_text(image_data: bytes, use_ocr:bool = False) -> Optional[str]:
    if not use_ocr: 
        return None 
    if PILImage is None or pytesseract is None:
        log.warning("OCR libraries not available. Install: pip install pillow pytesseract")
        return None

    try:
        img = PILImage.open(BytesIO(image_data))
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else None 

    except Exception as e:
        log.warning("OCR failed: %s", e)
        return None

# - We need to convert image information to text so LLM can understand it
# - This description gets included in the document content for summarization
def describe_image(image: ExtractedImage, use_ocr:bool = False) -> str:
    description_parts: List[str] = []
    description_parts.append(
        f"[Image {image.image_index + 1} on page {image.page_number}]"
    )
    description_parts.append(f"Dimensions: {image.width}x{image.height} pixels")
    description_parts.append(f"Format: {image.format.upper()}")  # PNG, JPEG, etc.
    description_parts.append(f"Size: {image.size_bytes} bytes")
    
    # WHY: Add OCR text if available
    # - If image contains text, extract it and include in description
    # - This way LLM can summarize the text that was in the image
    if use_ocr:
        ocr_text = extract_image_text(image.data, use_ocr=True)
        if ocr_text:
            description_parts.append(f"Extracted text: {ocr_text}")
        else:
            description_parts.append("(No text detected in image)")
    else:
        description_parts.append("(Image content - no OCR performed)")
    
    return "\n".join(description_parts)



def combine_text_images(
    text: str, 
    images: List[ExtractedImage],
    use_ocr: bool = False
) -> str:
    parts : List[str] =[]
    if text.strip():
        parts.append("=== PDF Text Content ===")
        parts.append(text)
    if images:
        parts.append("\n=== PDF Images ===")
        for img in images:
            img_desc = describe_image(img, use_ocr=use_ocr)  # Convert image to text
            parts.append(f"\n{img_desc}\n")
    return "\n".join(parts)
