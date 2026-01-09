# Services Folder Refactoring Plan

## Executive Summary

This document outlines industry-standard approaches for organizing the `services/` folder and refactoring the PDF extraction code to use Object-Oriented Programming (OOP) principles. The current structure has too many small files with functional programming, making it harder to maintain and extend.

---

## 1. Industry Standards for Services Folder Organization

### 1.1 Current Issues

**Current Structure:**
```
services/
├── chunking.py          # RAG: Document chunking
├── content_extractor.py # Router/dispatcher for extraction
├── embeddings.py        # RAG: Vector embeddings
├── hashing.py          # Utility functions
├── image_processor.py   # PDF image processing
├── pdf_extractor.py    # PDF text extraction
└── summarize.py        # LLM summarization
```

**Problems:**
1. **Too many small files**: Each file has 1-3 functions, creating fragmentation
2. **Tight coupling**: PDF extraction logic is split across 3 files (`pdf_extractor.py`, `image_processor.py`, `content_extractor.py`)
3. **No clear domain boundaries**: Related functionality is scattered
4. **Functional programming only**: No inheritance/polymorphism, making extension harder
5. **Mixed concerns**: Utilities (`hashing.py`) mixed with business logic

### 1.2 Industry Standards

#### **Principle 1: Domain-Driven Organization**
Group services by **business domain/feature**, not by technical layer:

```
services/
├── document/           # Document processing domain
│   ├── extractors.py  # All content extractors (OOP classes)
│   └── summarizer.py  # LLM summarization
├── embeddings/        # RAG/embeddings domain
│   ├── embeddings.py
│   └── chunking.py
└── utils/             # Shared utilities (if not in app/utils.py)
    └── hashing.py
```

**OR** (simpler, for smaller projects):

```
services/
├── extractors.py      # All content extractors (OOP)
├── summarizer.py      # LLM summarization
├── embeddings.py      # Vector embeddings
├── chunking.py        # Document chunking
└── (move hashing.py to app/utils.py)
```

#### **Principle 2: Single Responsibility per Service**
Each service class should handle **one domain concern**:
- `PDFExtractor`: Handles ALL PDF-related extraction (text, images, metadata, OCR)
- `TextExtractor`: Handles plain text extraction
- `Summarizer`: Handles LLM summarization only

#### **Principle 3: OOP with Inheritance**
Use **base classes** for common patterns:
- `BaseContentExtractor`: Abstract base class with common interface
- `TextExtractor(BaseContentExtractor)`: Text-specific implementation
- `PDFExtractor(BaseContentExtractor)`: PDF-specific implementation

#### **Principle 4: Dependency Injection**
Services should be **injectable and testable**:
- Accept dependencies via constructor (not global imports)
- Use interfaces/abstract classes for external dependencies
- Enable easy mocking in tests

#### **Principle 5: Factory Pattern**
Use a **factory** to create extractors based on MIME type:
- `ContentExtractorFactory.create(mime_type)` returns appropriate extractor
- Centralizes creation logic
- Easy to add new extractor types

---

## 2. Recommended Refactoring: OOP-Based Structure

### 2.1 Proposed Structure

```
services/
├── extractors.py          # All content extractors (OOP classes)
│   ├── BaseContentExtractor (ABC)
│   ├── TextExtractor
│   ├── MarkdownExtractor
│   ├── PDFExtractor
│   └── ContentExtractorFactory
├── summarizer.py          # LLM summarization (can stay functional or become class)
├── embeddings.py          # Vector embeddings (RAG)
└── chunking.py            # Document chunking (RAG)
```

**Move to `app/utils.py`:**
- `hashing.py` functions (already have `utils.py`, consolidate there)

### 2.2 OOP Design: Base Class + Child Classes

#### **Base Class: `BaseContentExtractor`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExtractedContent:
    text: str
    mime_type: str
    metadata: Dict[str, Any]

class BaseContentExtractor(ABC):
    """Abstract base class for all content extractors."""
    
    @property
    @abstractmethod
    def supported_mime_types(self) -> list[str]:
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
```

#### **Child Class: `TextExtractor`**

```python
class TextExtractor(BaseContentExtractor):
    """Extractor for plain text and markdown files."""
    
    @property
    def supported_mime_types(self) -> list[str]:
        return ["text/plain", "text/markdown"]
    
    def extract(self, file_bytes: bytes, **kwargs) -> ExtractedContent:
        """Extract text by decoding UTF-8."""
        from ..utils import decode_text  # Or from app.utils
        
        try:
            text = decode_text(file_bytes)
            return ExtractedContent(
                text=text,
                mime_type="text/plain",  # Or detect from kwargs
                metadata={}
            )
        except UnicodeDecodeError as e:
            raise ContentExtractionError(f"File must be UTF-8 encoded: {e}") from e
```

#### **Child Class: `PDFExtractor`**

```python
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
    def supported_mime_types(self) -> list[str]:
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
        
        # Process images (OCR, descriptions)
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
        """Extract text and images from PDF (from current pdf_extractor.py)."""
        # Move logic from pdf_extractor.py here
        pass
    
    def _extract_image_from_page(self, ...) -> Optional[ExtractedImage]:
        """Extract single image from PDF page (from current pdf_extractor.py)."""
        # Move logic from pdf_extractor.py here
        pass
    
    def _extract_pdf_metadata(self, pdf_doc) -> Dict[str, Any]:
        """Extract PDF metadata (from current pdf_extractor.py)."""
        # Move logic from pdf_extractor.py here
        pass
    
    def _perform_ocr(self, image_data: bytes) -> Optional[str]:
        """Perform OCR on image (from current image_processor.py)."""
        # Move logic from image_processor.py here
        pass
    
    def _describe_image(self, image: ExtractedImage) -> str:
        """Generate text description of image (from current image_processor.py)."""
        # Move logic from image_processor.py here
        pass
    
    def _combine_text_and_images(
        self,
        text: str,
        images: List[ExtractedImage],
    ) -> str:
        """Combine PDF text with image descriptions (from current image_processor.py)."""
        # Move logic from image_processor.py here
        pass
```

#### **Factory: `ContentExtractorFactory`**

```python
class ContentExtractorFactory:
    """Factory for creating appropriate content extractor based on MIME type."""
    
    _extractors: list[BaseContentExtractor] = []
    
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
        for extractor in cls._extractors:
            if extractor.can_extract(mime_type):
                # For PDF, pass configuration
                if isinstance(extractor, PDFExtractor):
                    return PDFExtractor(
                        extract_images=extract_images,
                        use_ocr=use_ocr,
                    )
                return extractor
        
        raise ContentExtractionError(f"Unsupported MIME type: {mime_type}")
    
    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize with default extractors."""
        cls.register(TextExtractor())
        cls.register(MarkdownExtractor())  # Or combine with TextExtractor
        # PDFExtractor registered on-demand with config
```

### 2.3 Usage in API Layer

**Before (functional):**
```python
from .services.content_extractor import extract_content

extracted = extract_content(
    file_bytes=raw,
    mime_type=mime,
    extract_images=settings.pdf_extract_images,
    use_ocr=settings.pdf_use_ocr,
)
```

**After (OOP):**
```python
from .services.extractors import ContentExtractorFactory

extractor = ContentExtractorFactory.create(
    mime_type=mime,
    extract_images=settings.pdf_extract_images,
    use_ocr=settings.pdf_use_ocr,
)
extracted = extractor.extract(file_bytes=raw)
```

---

## 3. File Consolidation Plan

### 3.1 Files to Merge

**Merge into `services/extractors.py`:**
1. `pdf_extractor.py` → `PDFExtractor` class methods
2. `image_processor.py` → `PDFExtractor` class methods (private methods)
3. `content_extractor.py` → `BaseContentExtractor`, `TextExtractor`, `PDFExtractor`, `ContentExtractorFactory`

**Keep separate:**
- `summarizer.py` (different domain: LLM, not extraction)
- `embeddings.py` (different domain: RAG)
- `chunking.py` (different domain: RAG)

**Move to `app/utils.py`:**
- `hashing.py` functions (already have `utils.py`)

### 3.2 Resulting Structure

```
services/
├── extractors.py      # ~400-500 lines (all extractors in one file)
├── summarizer.py      # ~50 lines (unchanged or minor refactor)
├── embeddings.py      # RAG (unchanged)
└── chunking.py        # RAG (unchanged)

app/
└── utils.py           # Add hashing functions here
```

**Benefits:**
- **Reduced file count**: 7 files → 4 files in services/
- **Better cohesion**: All extraction logic in one place
- **Easier navigation**: Related code is together
- **OOP benefits**: Inheritance, polymorphism, easier testing

---

## 4. Implementation Steps

### Step 1: Create Base Class
1. Create `services/extractors.py`
2. Define `BaseContentExtractor` abstract class
3. Define `ExtractedContent` dataclass (move from `content_extractor.py`)

### Step 2: Implement Text Extractor
1. Create `TextExtractor(BaseContentExtractor)`
2. Move logic from `content_extractor.py._extract_text_content()`
3. Test with text files

### Step 3: Implement PDF Extractor (Consolidate)
1. Create `PDFExtractor(BaseContentExtractor)`
2. Move ALL PDF logic from `pdf_extractor.py`:
   - `extract_pdf_content()` → `_extract_pdf_structure()`
   - `_extract_image_from_page()` → `_extract_image_from_page()`
   - `_extract_pdf_metadata()` → `_extract_pdf_metadata()`
3. Move ALL image logic from `image_processor.py`:
   - `extract_image_text()` → `_perform_ocr()`
   - `describe_image()` → `_describe_image()`
   - `combine_text_images()` → `_combine_text_and_images()`
4. Implement `extract()` method that orchestrates all steps
5. Test with PDF files

### Step 4: Create Factory
1. Implement `ContentExtractorFactory`
2. Register default extractors
3. Update factory to handle PDF configuration

### Step 5: Update API Layer
1. Update `api.py` to use factory pattern
2. Remove old imports
3. Test all endpoints

### Step 6: Cleanup
1. Delete `pdf_extractor.py`
2. Delete `image_processor.py`
3. Delete `content_extractor.py`
4. Move `hashing.py` functions to `app/utils.py`
5. Delete `hashing.py`

### Step 7: Update Tests
1. Update unit tests to use new OOP structure
2. Test factory pattern
3. Test inheritance/polymorphism

---

## 5. Benefits of OOP Approach

### 5.1 Maintainability
- **Single file for extraction**: All extraction logic in `extractors.py`
- **Clear inheritance hierarchy**: Easy to understand relationships
- **Consolidated PDF logic**: All PDF processing in `PDFExtractor` class

### 5.2 Extensibility
- **Easy to add new extractors**: Just create new class inheriting from `BaseContentExtractor`
- **No need to modify existing code**: Open/Closed Principle
- **Example**: Add `DOCXExtractor`, `ImageExtractor`, etc.

### 5.3 Testability
- **Mock base class**: Easy to mock `BaseContentExtractor` in tests
- **Test each extractor independently**: Isolated unit tests
- **Dependency injection**: Pass dependencies via constructor

### 5.4 Code Organization
- **Reduced file count**: 7 files → 4 files
- **Better cohesion**: Related functionality grouped together
- **Clearer boundaries**: Each class has single responsibility

---

## 6. Alternative: Simpler Functional Approach (If OOP is Overkill)

If the team prefers functional programming, consider:

### Option A: Consolidate Files Only
- Merge `pdf_extractor.py` + `image_processor.py` → `pdf_extractor.py` (larger file, ~300 lines)
- Keep `content_extractor.py` as router
- **Result**: 5 files instead of 7

### Option B: Group by Domain (Subdirectories)
```
services/
├── extraction/
│   ├── __init__.py
│   ├── text.py
│   ├── pdf.py          # Merge pdf_extractor + image_processor
│   └── factory.py
├── summarization/
│   └── summarizer.py
└── embeddings/
    ├── embeddings.py
    └── chunking.py
```

**Recommendation**: Use OOP approach (Section 2) for better long-term maintainability.

---

## 7. Industry Examples

### 7.1 Django Services Pattern
Django projects often use:
- Service classes (not just functions)
- Factory patterns for creation
- Dependency injection via constructors

### 7.2 FastAPI Best Practices
FastAPI documentation recommends:
- Service classes for business logic
- Dependency injection for testability
- Clear separation of concerns

### 7.3 Enterprise Patterns
Large codebases use:
- **Strategy Pattern**: Different extractors as strategies
- **Factory Pattern**: Centralized creation
- **Template Method**: Base class defines algorithm, subclasses implement steps

---

## 8. Migration Checklist

- [ ] Create `services/extractors.py` with base class
- [ ] Implement `TextExtractor` class
- [ ] Implement `PDFExtractor` class (consolidate PDF + image logic)
- [ ] Implement `ContentExtractorFactory`
- [ ] Update `api.py` to use factory
- [ ] Test text file upload
- [ ] Test PDF file upload
- [ ] Test PDF with images
- [ ] Test OCR (if enabled)
- [ ] Move `hashing.py` to `app/utils.py`
- [ ] Delete old files (`pdf_extractor.py`, `image_processor.py`, `content_extractor.py`, `hashing.py`)
- [ ] Update imports in other files
- [ ] Update tests
- [ ] Update documentation

---

## 9. Code Size Estimates

**Current:**
- `pdf_extractor.py`: ~148 lines
- `image_processor.py`: ~71 lines
- `content_extractor.py`: ~86 lines
- **Total**: ~305 lines across 3 files

**After Refactoring:**
- `extractors.py`: ~400-500 lines (all extractors + factory)
- **Reduction**: 3 files → 1 file
- **Lines**: Similar total, but better organized

---

## 10. Conclusion

**Recommended Approach:**
1. **Use OOP with base class + child classes** for extractors
2. **Consolidate PDF logic** into single `PDFExtractor` class
3. **Use factory pattern** for creating extractors
4. **Reduce file count** from 7 to 4 in services/
5. **Move utilities** to `app/utils.py`

**Benefits:**
- ✅ Better code organization
- ✅ Easier to extend (add new extractors)
- ✅ Better testability
- ✅ Industry-standard patterns
- ✅ Reduced file fragmentation

**Trade-offs:**
- ⚠️ Slightly more complex initially (OOP vs functional)
- ⚠️ One larger file instead of multiple small files (but better cohesion)

**Recommendation**: Proceed with OOP refactoring for long-term maintainability.

