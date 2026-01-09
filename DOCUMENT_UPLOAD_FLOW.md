# Document Upload and Processing Flow

This document explains the complete flow of what happens when a document is uploaded to PrecisBox, detailing which functions are called and what each one does.

## Overview

```
User Uploads File
  ↓
API Layer (api.py)
  ↓
Content Extraction (content_extractor.py)
  ↓
[For Text/MD] → Text Decoder (hashing.py)
[For PDF] → PDF Extractor (pdf_extractor.py) → Image Processor (image_processor.py)
  ↓
Content Stored in MongoDB
  ↓
Job Enqueued in Redis
  ↓
Background Worker (redis_queue_worker.py)
  ↓
Summarization (summarize.py) → OpenAI API
  ↓
Summary Stored in MongoDB
```

---

## Step-by-Step Flow

### 1. Upload Request Received

**File:** `src/app/api.py`  
**Function:** `upload_doc()`

**What happens:**
1. FastAPI receives POST request to `/docs` endpoint
2. File is read as bytes: `raw: bytes = await file.read()`
3. File size is validated against `settings.max_upload_bytes` (default: 2MB)
4. MIME type is extracted from `file.content_type`
5. MIME type is validated against `SUPPORTED_MIME` (`text/plain`, `text/markdown`, `application/pdf`)
6. Document ID is generated: `doc_id = uuid4().hex`

---

### 2. Content Extraction

**File:** `src/app/api.py`  
**Function:** `upload_doc()` → calls `extract_content()`

**File:** `src/app/services/content_extractor.py`  
**Function:** `extract_content(file_bytes, mime_type, extract_images, use_ocr)`

**What happens:**
- This is the **routing function** that determines how to process the file based on MIME type
- Runs in thread pool (CPU-bound operation) using `anyio.to_thread.run_sync()`

**Routing Logic:**
```python
if mime_type in ("text/plain", "text/markdown"):
    return _extract_text_content(file_bytes, mime_type)
elif mime_type == "application/pdf":
    return _extract_pdf_content_wrapper(file_bytes, extract_images, use_ocr)
```

**Returns:** `ExtractedContent` dataclass with:
- `text`: Combined text content (ready for storage/summarization)
- `mime_type`: Original MIME type
- `metadata`: Additional info (page count, image count for PDFs)

---

### 3a. Text/Markdown File Processing

**File:** `src/app/services/content_extractor.py`  
**Function:** `_extract_text_content(file_bytes, mime_type)`

**What happens:**
1. Calls `decode_text(file_bytes)` from `hashing.py`
2. Converts bytes to UTF-8 string
3. Returns `ExtractedContent` with the decoded text

**File:** `src/app/services/hashing.py`  
**Function:** `decode_text(raw: bytes) -> str`

**What happens:**
- Simply decodes bytes using UTF-8: `return raw.decode("utf-8")`
- Raises `UnicodeDecodeError` if file is not valid UTF-8

---

### 3b. PDF File Processing

**File:** `src/app/services/content_extractor.py`  
**Function:** `_extract_pdf_content_wrapper(file_bytes, extract_images, use_ocr)`

**What happens:**
1. Calls `extract_pdf_content()` to extract text and images from PDF
2. Calls `combine_text_images()` to merge text + image descriptions
3. Collects metadata (page count, image count, PDF properties)
4. Returns `ExtractedContent` with combined text

---

### 4. PDF Content Extraction

**File:** `src/app/services/pdf_extractor.py`  
**Function:** `extract_pdf_content(pdf_bytes, extract_images=True)`

**What happens:**
1. **Opens PDF from bytes:**
   - `pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")`
   - Uses PyMuPDF (fitz) to parse PDF structure

2. **Iterates through each page:**
   ```python
   for page_num in range(total_pages):
       page = pdf_doc[page_num]
   ```

3. **Extracts text from each page:**
   - `page_text = page.get_text()`
   - Adds page marker: `"--- Page {page_num + 1} ---\n{page_text}\n"`
   - Stores in `text_parts` list

4. **Extracts images (if `extract_images=True`):**
   - `image_list = page.get_images(full=True)`
   - For each image, calls `_extract_image_from_page()`
   - Stores extracted images in `images` list

5. **Extracts PDF metadata:**
   - Calls `_extract_pdf_metadata(pdf_doc)`
   - Gets title, author, creation date, etc.

6. **Returns:** `PDFContent` dataclass with:
   - `text`: All page text combined
   - `images`: List of `ExtractedImage` objects
   - `total_pages`: Page count
   - `metadata`: PDF properties

---

### 5. Image Extraction (PDF only)

**File:** `src/app/services/pdf_extractor.py`  
**Function:** `_extract_image_from_page(pdf_doc, page_num, img_index, img_info)`

**What happens:**
1. Gets image XREF (cross-reference number) from `img_info`
2. Extracts image bytes: `base_image = pdf_doc.extract_image(xref)`
3. Gets image metadata (format, dimensions, size)
4. Validates image using PIL if available
5. Returns `ExtractedImage` dataclass with:
   - `page_number`: Which page the image is on
   - `image_index`: Index of image on that page
   - `data`: Raw image bytes (JPEG/PNG)
   - `width`, `height`: Dimensions in pixels
   - `format`: Image format ("png", "jpeg", etc.)
   - `size_bytes`: Size of image data

---

### 6. Combining Text and Images (PDF only)

**File:** `src/app/services/image_processor.py`  
**Function:** `combine_text_images(text, images, use_ocr=False)`

**What happens:**
1. Creates list of content parts
2. Adds PDF text with header: `"=== PDF Text Content ==="`
3. For each image:
   - Calls `describe_image(image, use_ocr)` to convert image to text description
   - Adds image descriptions with header: `"=== PDF Images ==="`
4. Joins all parts with newlines
5. Returns single combined text string

**File:** `src/app/services/image_processor.py`  
**Function:** `describe_image(image, use_ocr=False) -> str`

**What happens:**
1. Creates text description of image metadata:
   - Image location: `"[Image {index} on page {page}]"`
   - Dimensions: `"{width}x{height} pixels"`
   - Format: `"{format.upper()}"`
   - Size: `"{size_bytes} bytes"`

2. **If OCR enabled (`use_ocr=True`):**
   - Calls `extract_image_text(image.data, use_ocr=True)`
   - Uses pytesseract to extract text from image
   - Adds OCR text to description

3. Returns formatted text description

**File:** `src/app/services/image_processor.py`  
**Function:** `extract_image_text(image_data, use_ocr=False) -> Optional[str]`

**What happens (if OCR enabled):**
1. Opens image from bytes using PIL
2. Runs OCR: `pytesseract.image_to_string(img)`
3. Returns extracted text or None if OCR fails

---

### 7. Store in MongoDB

**File:** `src/app/api.py`  
**Function:** `upload_doc()`

**What happens:**
1. Calls `write_raw_doc(settings.mongo_uri, settings.mongo_db, doc_id, text)`
   - Runs in thread pool: `anyio.to_thread.run_sync()`
   - Stores the combined text string in MongoDB

**File:** `src/app/database/mongo.py`  
**Function:** `write_raw_doc(mongo_uri, mongo_db, doc_id, text)`

**What happens:**
1. Gets MongoDB client (from connection pool)
2. Gets database: `db = client[mongo_db]`
3. Stores document: `db.docs.update_one({"_id": doc_id}, {"$set": {"text": text}}, upsert=True)`

---

### 8. Store Metadata in SQLite

**File:** `src/app/api.py`  
**Function:** `upload_doc()`

**What happens:**
1. Calls `insert_document()` with document metadata
   - Runs in thread pool: `anyio.to_thread.run_sync()`
   - Stores metadata in SQLite database

**File:** `src/app/database/sqlite.py`  
**Function:** `insert_document(sqlite_path, doc_id, filename, size, mime, sha256, status, model)`

**What happens:**
1. Connects to SQLite database
2. Inserts row into `documents` table with:
   - `id`: Document ID
   - `filename`: Original filename
   - `size`: File size in bytes
   - `mime`: MIME type
   - `sha256`: SHA-256 hash
   - `status`: "pending" (DocumentStatus.PENDING)
   - `model`: OpenAI model name
   - `created_at`, `updated_at`: Timestamps

---

### 9. Enqueue Job in Redis

**File:** `src/app/api.py`  
**Function:** `upload_doc()`

**What happens:**
1. Gets Redis client: `r = get_redis()`
2. Enqueues job: `enqueue_job(r, doc_id)`

**File:** `src/app/queue/redis_queue.py`  
**Function:** `enqueue_job(redis_client, doc_id)`

**What happens:**
1. Pushes document ID to Redis list: `redis_client.lpush("precisbox:jobs", doc_id)`
2. Job is now in queue for background processing

**Returns:** `DocCreateResponse` with:
- `id`: Document ID
- `status`: "pending"
- `display_name`: `"{filename}_{short_id}"`

---

### 10. Background Worker Processing

**File:** `src/app/queue/redis_queue_worker.py`  
**Function:** `_process_one(cfg, doc_id)`

**What happens:**
1. **Updates status to "processing":**
   - `set_status(cfg.sqlite_path, doc_id, DocumentStatus.PROCESSING)`

2. **Reads text from MongoDB:**
   - `text = get_raw_doc(db, doc_id)`
   - This is the combined text (text + image descriptions for PDFs)

3. **Calls OpenAI API:**
   - `summary, prompt_tokens, completion_tokens = summarize_text(...)`

**File:** `src/app/services/summarize.py`  
**Function:** `summarize_text(api_key, model, text, timeout)`

**What happens:**
1. Creates OpenAI client
2. Calls `client.chat.completions.create()` with:
   - Model: `gpt-4o-mini` (or configured model)
   - System message: "You are a helpful assistant that summarizes documents clearly."
   - User message: "summarize the following document into 6-10 bullet points:\n\n{text}"
   - Temperature: 0.2 (low, for factual summaries)
3. Returns: `(summary_text, prompt_tokens, completion_tokens)`

4. **Stores summary in MongoDB:**
   - `put_summary(db, doc_id, summary, prompt_tokens, completion_tokens)`

5. **Updates status to "done":**
   - `set_status(cfg.sqlite_path, doc_id, DocumentStatus.DONE, ...)`

---

## Complete Function Call Chain

### For Text/Markdown Files:
```
upload_doc()
  → extract_content()
    → _extract_text_content()
      → decode_text() [from hashing.py]
        → raw.decode("utf-8")
  → write_raw_doc() [stores text in MongoDB]
  → insert_document() [stores metadata in SQLite]
  → enqueue_job() [adds to Redis queue]
```

### For PDF Files:
```
upload_doc()
  → extract_content()
    → _extract_pdf_content_wrapper()
      → extract_pdf_content() [from pdf_extractor.py]
        → fitz.open() [opens PDF]
        → For each page:
          → page.get_text() [extracts text]
          → page.get_images() [gets image list]
          → For each image:
            → _extract_image_from_page() [extracts image bytes]
        → _extract_pdf_metadata() [gets PDF properties]
      → combine_text_images() [from image_processor.py]
        → For each image:
          → describe_image() [converts image to text description]
            → extract_image_text() [optional OCR]
        → Joins text + image descriptions
  → write_raw_doc() [stores combined text in MongoDB]
  → insert_document() [stores metadata in SQLite]
  → enqueue_job() [adds to Redis queue]
```

### Background Worker (same for all file types):
```
_worker_loop() [polls Redis queue]
  → _process_one()
    → set_status("processing")
    → get_raw_doc() [reads text from MongoDB]
    → summarize_text() [calls OpenAI API]
      → OpenAI API processes text
    → put_summary() [stores summary in MongoDB]
    → set_status("done")
```

---

## Data Structures

### ExtractedContent
```python
@dataclass
class ExtractedContent:
    text: str              # Combined text (ready for LLM)
    mime_type: str         # Original file type
    metadata: dict         # Additional info (pages, images, etc.)
```

### PDFContent
```python
@dataclass
class PDFContent:
    text: str              # All page text combined
    images: List[ExtractedImage]  # Extracted images
    total_pages: int       # Page count
    metadata: dict         # PDF properties
```

### ExtractedImage
```python
@dataclass
class ExtractedImage:
    page_number: int       # Page number (1-indexed)
    image_index: int       # Index on page
    data: bytes           # Raw image bytes
    width: int            # Width in pixels
    height: int           # Height in pixels
    format: str           # Image format ("png", "jpeg")
    size_bytes: int       # Size in bytes
```

---

## Key Points

1. **Text files** are processed directly - just decode UTF-8
2. **PDFs** require multi-step processing:
   - Parse PDF structure
   - Extract text from each page
   - Extract images as binary data
   - Convert images to text descriptions (with optional OCR)
   - Combine everything into single text string
3. **All files** (text or PDF) end up as a text string in MongoDB
4. **Worker** doesn't know the source format - it just reads text and sends to OpenAI
5. **OpenAI** receives a text string (same format regardless of source file type)

---

## Example Flow for PDF with Images

```
1. User uploads "report.pdf" (has 3 pages, 2 images)

2. extract_content() routes to PDF processing

3. extract_pdf_content():
   - Opens PDF
   - Page 1: Extracts text "Q4 Report", extracts Image 1 (chart)
   - Page 2: Extracts text "Revenue increased...", extracts Image 2 (screenshot)
   - Page 3: Extracts text "Conclusion..."
   - Returns PDFContent with text + 2 ExtractedImage objects

4. combine_text_images():
   - Takes text from all pages
   - Converts Image 1 to description: "[Image 1 on page 1]\nDimensions: 800x600..."
   - Converts Image 2 to description: "[Image 2 on page 2]\nDimensions: 1200x900..."
   - Combines: "=== PDF Text Content ===\nQ4 Report\n...\n=== PDF Images ===\n[Image 1...]\n[Image 2...]"

5. Stores combined text in MongoDB

6. Worker reads combined text, sends to OpenAI

7. OpenAI receives single text string and generates summary
```

---

This flow ensures that regardless of file type, the system processes everything into a unified text format that can be sent to OpenAI for summarization.

