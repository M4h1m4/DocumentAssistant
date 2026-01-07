from __future__ import annotations

import io 
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging 

try: 
    import fitz
except:
    fitz=None 

try:
    from PIL import Image
except ImportError:
    Image = None 

from ..config import Defaults


log = logging.getLogger("precisbox.services.pdf_extractor")

@dataclass
class ExtractedImage: 
    page_number: int 
    image_index: int 
    data: bytes
    width: int 
    height: int 
    format: str
    size_bytes: int 

@dataclass 
class PDFContent:
    text: str
    images: List[ExtractedImage]
    total_pages: int 
    metadata: Dict[str, Any]

class PDFExtractionError(RuntimeError):
    pass 


#takes the raw pdf file and structures the content (text+image+metadata)
def extract_pdf_content(pdf_bytes: bytes, extract_images: bool=True) -> PDFContent:
    if fitz is None:
        raise ImportError(
            "PyMuPDF (pymupdf) is required for PDF support. "
            "Install it with: pip install pymupdf"
        )
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        #that is recieving it from FastAPI upload
        total_pages = len(pdf_doc)

        text_parts: List[str] = []
        images: List[ExtractedImage]=[]

        for page_num in range(total_pages):
            page = pdf_doc[page_num] # get page object 0-indexed
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
            
            if extract_images:
                image_list = page.get_images(full=True)  # Note: get_images() not get_image()
                for img_index, img in enumerate(image_list):
                    try:
                        extracted_img = _extract_image_from_page(
                            pdf_doc, page_num, img_index, img
                        )
                        if extracted_img:
                            images.append(extracted_img)
                    except Exception as e:
                        log.warning(
                            "Failed to extract image page=%d index=%d: %s",
                            page_num, img_index, e
                        )
        pdf_doc.close()

        full_text = "\n".join(text_parts)

        metadata = _extract_pdf_metadata(pdf_doc)

        return PDFContent(
            text=full_text,
            images=images,
            total_pages=total_pages,
            metadata=metadata,
        )
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract PDF content: {e}") from e
                    
def _extract_image_from_page(
    pdf_doc: fitz.document, 
    page_num: int, 
    img_index: int,
    img_info: Tuple[int, int, int, int, str, str, str, int, int], # we get this information from the function get_image()
) -> Optional[ExtractedImage]:
    """
    PDFs use XREF table to locate objects
    First element of the tuple is the XREF number which has the actual image data
    """

    try:
        xref = img_info[0]
        #extract_image is used to extract the data from the image
        base_image = pdf_doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_format = base_image["ext"]

        width = base_image.get("width","0")
        height = base_image.get("height", "0")

        if Image is not None:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

        return ExtractedImage(
            page_number = page_num+1,
            image_index= img_index, 
            data = image_bytes, 
            width = width, 
            height = height, 
            format = image_foramt, 
            size_bytes= len(image_bytes), 
        )
    except Exception as e:
        log.warning("Failed to extract image xref=%d: %s", xref, e)
        return None 

def _extract_pdf_metadata(pdf_doc: fitz.document) -> Dict[str, Any]:
    metadata = pdf_doc.metadata  
    return {
        "title": metadata.get("title", ""), 
        "author": metadata.get("author", ""), 
        "subject": metadata.get("subject", ""),  
        "creator": metadata.get("creator", ""),  
        "producer": metadata.get("producer", ""),  
        "creation_date": metadata.get("creationDate", ""),  
        "modification_date": metadata.get("modDate", ""),  
    }





