git from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import io
from io import BytesIO


import fitz  # PyMuPDF
from PIL import Image as PILImage
import pytesseract  # OCR library (requires Tesseract binary installed)

from ..config import Defaults
from ..utils import decode_text  # From app/utils.py (after moving hashing functions)
from ..logging_config import get_logger

log = get_logger("precisbox.services.extractors")



# Data Classes

@dataclass
class ExtractedContent:
    """Represents extracted content from any supported file type."""
    text: str
    mime_type: str
    metadata: Dict[str, Any]


@dataclass
class ExtractedImage:
    """Represents an extracted image from PDF."""
    page_number: int
    image_index: int
    data: bytes
    width: int
    height: int
    format: str  # e.g., "png", "jpeg"
    size_bytes: int


@dataclass
class PDFContent:
    """Internal representation of extracted PDF content."""
    text: str
    images: List[ExtractedImage]
    total_pages: int
    metadata: Dict[str, Any]



# Exception Classes

class ContentExtractionError(RuntimeError):
    """Raised when content extraction fails."""
    pass


class PDFExtractionError(RuntimeError):
    """Raised when PDF extraction fails."""
    pass



# Base Content Extractor


class BaseContentExtractor(ABC):
    """Abstract base class for all content extractors."""
    
    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """Return list of MIME types this extractor supports."""
        pass
    
    @abstractmethod
    def extract(self, file_bytes: bytes, **kwargs) -> ExtractedContent:
        """
        Extract content from file bytes.
        
        Args:
            file_bytes: Raw file content
            **kwargs: Extractor-specific options (e.g., extract_images, use_ocr)
            
        Returns:
            ExtractedContent object
        """
        pass
    
    def can_extract(self, mime_type: str) -> bool:
        """Check if this extractor supports the given MIME type."""
        return mime_type in self.supported_mime_types


# Text Extractor
class TextExtractor(BaseContentExtractor):
    """Extractor for plain text and markdown files."""
    
    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/plain", "text/markdown"]
    
    def extract(self, file_bytes: bytes, **kwargs) -> ExtractedContent:
        """Extract text by decoding UTF-8."""
        try:
            text = decode_text(file_bytes)
            # Detect MIME type from kwargs if provided, otherwise default to text/plain
            mime_type = kwargs.get("mime_type", "text/plain")
            return ExtractedContent(
                text=text,
                mime_type=mime_type,
                metadata={}
            )
        except UnicodeDecodeError as e:
            raise ContentExtractionError(f"File must be UTF-8 encoded: {e}") from e


# PDF Extractor (Consolidates PDF + Image Processing)
class PDFExtractor(BaseContentExtractor):
    """Extractor for PDF files with text, images, and optional OCR."""
    
    def __init__(
        self,
        extract_images: bool = True,
        use_ocr: bool = False,
    ):
        """
        Initialize PDF extractor.
        
        Args:
            extract_images: Whether to extract images from PDF
            use_ocr: Whether to perform OCR on images
        """
        self.extract_images = extract_images
        self.use_ocr = use_ocr
    
    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]
    
    def extract(self, file_bytes: bytes, **kwargs) -> ExtractedContent:
        """
        Extract content from PDF.
        
        Consolidates ALL PDF processing:
        - Text extraction (from text layer)
        - Image extraction (if enabled)
        - OCR (if enabled)
        - Metadata extraction
        - Combining text + image descriptions
        """
        # Extract PDF structure (text + images)
        pdf_content = self._extract_pdf_structure(file_bytes)
        
        # Process images (OCR, descriptions) and combine with text
        combined_text = self._combine_text_and_images(
            pdf_content.text,
            pdf_content.images,
        )
        
        # Build metadata
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
    
    def _extract_pdf_structure(self, pdf_bytes: bytes) -> PDFContent:
        """Extract text and images from PDF structure."""
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_doc)
            
            text_parts: List[str] = []
            images: List[ExtractedImage] = []
            
            for page_num in range(total_pages):
                page = pdf_doc[page_num]  # Get page object (0-indexed)
                
                # Extract text from page
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
                
                # Extract images from page (if enabled)
                if self.extract_images:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        try:
                            extracted_img = self._extract_image_from_page(
                                pdf_doc, page_num, img_index, img_info
                            )
                            if extracted_img:
                                images.append(extracted_img)
                        except Exception as e:
                            log.warning(
                                "Failed to extract image page=%d index=%d: %s",
                                page_num, img_index, e
                            )
            
            # Extract metadata before closing
            metadata = self._extract_pdf_metadata(pdf_doc)
            
            pdf_doc.close()
            
            # Combine text from all pages
            full_text = "\n".join(text_parts)
            
            return PDFContent(
                text=full_text,
                images=images,
                total_pages=total_pages,
                metadata=metadata,
            )
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract PDF content: {e}") from e
    
    def _extract_image_from_page(
        self,
        pdf_doc: fitz.Document,
        page_num: int,
        img_index: int,
        img_info: Tuple[int, int, int, int, str, str, str, int, int],
    ) -> Optional[ExtractedImage]:
        """
        Extract a single image from a PDF page.
        
        PDFs use XREF table to locate objects.
        First element of the tuple is the XREF number which has the actual image data.
        """
        try:
            xref = img_info[0]
            
            # Extract image bytes using XREF
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_format = base_image["ext"]  # "png", "jpeg", etc.
            
            # Get dimensions (try from base_image first, then use PIL)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            
            # Verify and get accurate dimensions using PIL
            img = PILImage.open(BytesIO(image_bytes))
            width, height = img.size
            
            return ExtractedImage(
                page_number=page_num + 1,  # 1-indexed for display
                image_index=img_index,
                data=image_bytes,
                width=width,
                height=height,
                format=image_format,
                size_bytes=len(image_bytes),
            )
        except Exception as e:
            log.warning("Failed to extract image xref=%d: %s", img_info[0] if img_info else None, e)
            return None
    
    def _extract_pdf_metadata(self, pdf_doc: fitz.Document) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
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
    
    def _perform_ocr(self, image_data: bytes) -> Optional[str]:
        """Perform OCR on image data."""
        if not self.use_ocr:
            return None
        
        img = PILImage.open(BytesIO(image_data))
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else None
    
    def _describe_image(self, image: ExtractedImage) -> str:
        """Generate a textual description of an image for LLM processing."""
        description_parts: List[str] = []
        
        description_parts.append(
            f"[Image {image.image_index + 1} on page {image.page_number}]"
        )
        description_parts.append(f"Dimensions: {image.width}x{image.height} pixels")
        description_parts.append(f"Format: {image.format.upper()}")  # PNG, JPEG, etc.
        description_parts.append(f"Size: {image.size_bytes} bytes")
        
        # Add OCR text if available
        if self.use_ocr:
            ocr_text = self._perform_ocr(image.data)
            if ocr_text:
                description_parts.append(f"Extracted text: {ocr_text}")
            else:
                description_parts.append("(No text detected in image)")
        else:
            description_parts.append("(Image content - no OCR performed)")
        
        return "\n".join(description_parts)
    
    def _combine_text_and_images(
        self,
        text: str,
        images: List[ExtractedImage],
    ) -> str:
        """Combine extracted PDF text with textual descriptions of its images."""
        parts: List[str] = []
        
        # Add main PDF text content if present
        if text.strip():
            parts.append("=== PDF Text Content ===")
            parts.append(text)
        
        # Add descriptions for all extracted images
        if images:
            parts.append("\n=== PDF Images ===")
            for img in images:
                img_desc = self._describe_image(img)
                parts.append(f"\n{img_desc}\n")
        
        return "\n".join(parts)



# Content Extractor Factory

class ContentExtractorFactory:
    """Factory for creating appropriate content extractor based on MIME type."""
    
    _extractors: List[BaseContentExtractor] = []
    
    @classmethod
    def register(cls, extractor: BaseContentExtractor) -> None:
        """Register an extractor (for dependency injection)."""
        cls._extractors.append(extractor)
    
    @classmethod
    def create(
        cls,
        mime_type: str,
        extract_images: bool = True,
        use_ocr: bool = False,
    ) -> BaseContentExtractor:
        """
        Create appropriate extractor for MIME type.
        
        Args:
            mime_type: MIME type of the file
            extract_images: Whether to extract images (for PDF)
            use_ocr: Whether to use OCR (for PDF)
            
        Returns:
            Appropriate extractor instance
            
        Raises:
            ContentExtractionError: If no extractor supports the MIME type
        """
        # Check registered extractors first
        for extractor in cls._extractors:
            if extractor.can_extract(mime_type):
                # For PDF, create new instance with config
                if isinstance(extractor, PDFExtractor):
                    return PDFExtractor(
                        extract_images=extract_images,
                        use_ocr=use_ocr,
                    )
                return extractor
        
        # Fallback to default extractors
        if mime_type in ["text/plain", "text/markdown"]:
            return TextExtractor()
        elif mime_type == "application/pdf":
            return PDFExtractor(
                extract_images=extract_images,
                use_ocr=use_ocr,
            )
        
        raise ContentExtractionError(f"Unsupported MIME type: {mime_type}")
    
    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize with default extractors."""
        cls.register(TextExtractor())
        # PDFExtractor created on-demand with config


# Initialize default extractors
ContentExtractorFactory.initialize_defaults()

