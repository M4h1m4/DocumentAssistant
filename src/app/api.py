from __future__ import annotations

from uuid import uuid4
from typing import Tuple, Dict, List, Any, Optional 
from datetime import datetime 
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
import anyio
from .config import settings, Defaults
from .schemas import (
    ApiError,
    ApiErrorCode,
    DocCreateResponse,
    DocMetaResponse,
    DocSummary,
    DocListResponse,
    DocContentResponse,
    DocumentStatus,
    RAGQueryRequest,
)
from .utils import sha256_bytes
from .database.sqlite import (
    insert_document,
    fetch_document,
    list_documents,
    set_status,
    delete_document as delete_document_sqlite,
)
from .database.mongo import (
    write_raw_doc,
    read_summary,
    delete_document as delete_document_mongo,
    read_raw_doc,
)
from .queue.redis_queue import get_redis, enqueue_job
from .utils import safe_int

from .services.document.extractors import (
    ContentExtractorFactory,
    ExtractedContent,
    ContentExtractionError,
)

from .services.rag.rag import RAGService
from .services.rag.vector_store import VectorStore 
from .services.embeddings.embeddings import EmbeddingService

from .logging_config import get_logger


log = get_logger("precisbox.api")

router = APIRouter(prefix="/docs")

SUPPORTED_MIME = Defaults.SUPPORTED_MIME_TYPES

rag_service: Optional[RAGService] = None

def init_rag_service() -> None:
    """Initialize RAG service during startup."""
    global rag_service
    if settings.enable_rag:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        vector_store = VectorStore(
            collection_name="documents",
            persist_directory=settings.vector_store_path,
        )
        rag_service = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )

def _err(code: ApiErrorCode, msg: str) -> ApiError:
    return ApiError(code=code, message=msg)

def build_doc_meta_response(row: Dict[str, Any]) -> DocMetaResponse:
    doc_id = row["id"]
    filename = row["filename"]
    created_at = datetime.fromisoformat(row["created_at"])
    updated_at = datetime.fromisoformat(row["updated_at"])

    # Defensive
    size = safe_int(row.get("size"), 0, "size", doc_id)
    attempts = safe_int(row.get("attempts"), 0, "attempts", doc_id)

    mime = row.get("mime") or "application/octet-stream"
    sha256 = row.get("sha256") or ""
    status = row.get("status") or "unknown"

    model = row.get("model")
    prompt_tokens = row.get("prompt_tokens")
    completion_tokens = row.get("completion_tokens")
    last_error = row.get("last_error")

    # Optional: log missing non-guaranteed fields
    for k in ("size", "mime", "sha256", "status"):
        if k not in row:
            log.warning("Row missing '%s' for doc_id=%s; using default. keys=%s", k, doc_id, list(row.keys()))

    return DocMetaResponse(
        id=str(doc_id),
        filename=str(filename),
        size=size,
        mime=str(mime),
        sha256=str(sha256),
        status=status,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        attempts=attempts,
        last_error=last_error,
        created_at=created_at,
        updated_at=updated_at,
    )
    
@router.post("", response_model=DocCreateResponse, status_code=201)
async def upload_doc(file: UploadFile = File(...)) -> DocCreateResponse:
    if not settings.is_summarizer_enabled:
        raise HTTPException(status_code=503, detail="Summarizer disabled (OPENAI_API_KEY missing)")

    raw: bytes = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large(max {settings.max_upload_bytes}bytes)")
    mime: str = file.content_type or Defaults.DEFAULT_MIME_TYPE
    if mime not in SUPPORTED_MIME:
        raise HTTPException(status_code=400, detail=f"unsupported mime type{mime}. Use text/plain for now")
    
    doc_id: str = uuid4().hex  # Generate doc_id early for error logging
    
    try:
        # Create appropriate extractor based on MIME type
        extractor = ContentExtractorFactory.create(
            mime_type=mime,
            extract_images=settings.pdf_extract_images,
            use_ocr=settings.pdf_use_ocr,
        )
        # Extract content using the extractor
        extracted_content: ExtractedContent = await anyio.to_thread.run_sync(
            lambda: extractor.extract(file_bytes=raw, mime_type=mime)
        )
        text = extracted_content.text

    except ContentExtractionError as e:
        log.exception("Content extraction failed for doc_id=%s: %s", doc_id, e)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract content: {str(e)}"
        )
    filename: str = file.filename or Defaults.DEFAULT_FILENAME
    original: str = file.filename or Defaults.DEFAULT_FILENAME
    short = doc_id[:8]
    display_name = f"{original}_{short}"
    size: int = int(len(raw))
    sha256: str = sha256_bytes(raw)

    try:
        await anyio.to_thread.run_sync(
            lambda: insert_document(
                settings.sqlite_path,
                doc_id, 
                filename,
                size, 
                mime, 
                sha256,
                DocumentStatus.PENDING,
                settings.openai_model,
            )
        )
    except Exception as e:
        log.exception("Failed to write metadata for doc_id=%s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    # def mongo_write() -> None: #Write the raw document into mongodb
    #     client = get_mongo_client(MONGO_URI)
    #     db = client[MONGO_DB]
    #     put_raw_doc(db, doc_id, text)

    try:
        await anyio.to_thread.run_sync(lambda: write_raw_doc(settings.mongo_uri, settings.mongo_db, doc_id, text)) 
        if settings.enable_rag and rag_service:
            try:
                await anyio.to_thread.run_sync(
                    lambda: rag_service.index_document(
                        doc_id=doc_id,
                        text=text,
                        metadata={
                            "filename": filename,
                            "mime_type": mime,
                            "size": size,
                        }
                    )
                )
            except Exception as e:
                log.warning("Failed to index document for RAG doc_id=%s: %s", doc_id, e)
    except Exception as e:
        log.exception("Failed to write document to MongoDB for doc_id=%s: %s", doc_id, e)
        await anyio.to_thread.run_sync(
            lambda: set_status(
                settings.sqlite_path,
                doc_id,
                "failed",
                model=settings.openai_model,
                last_error=ApiErrorCode.MONGO_WRITE_FAILED.value,
            )
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
    """
    run_sync runs the function in background, so it does not block the async event loop that is this is pushed into the thread_loop
    In mean the server can handle other requests 
    anyio.to_thread --> Schedules the function in threadpool worker. 
    await tells the event loop : I am waiting for the threadppol to complete/finish meantime feel free to run other requests/tasks 
    """
    # Enqueue job in Redis queue for background processing
    r = get_redis()
    enqueue_job(r, doc_id)
    return DocCreateResponse(id=doc_id, status=DocumentStatus.PENDING, display_name=display_name)

@router.get("/{doc_id}", response_model=DocMetaResponse)
async def get_doc(doc_id: str) -> DocMetaResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    return build_doc_meta_response(raw)

@router.get("/{doc_id}/summary", response_model=DocSummary)
async def get_doc_summary(doc_id: str) -> DocSummary:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    status = DocumentStatus(raw["status"])
    if status == DocumentStatus.FAILED:
        body = DocSummary(
            id=doc_id, 
            status=DocumentStatus.FAILED,
            summary=None, 
            err=_err(ApiErrorCode.SUMMARY_NOT_READY, "Summary not ready yet"),
        )
        return JSONResponse(status_code=409, content=body.model_dump())

    if status != DocumentStatus.DONE:
        body = DocSummary(
            id=doc_id,
            status=status,
            summary=None,
            err=_err(ApiErrorCode.SUMMARY_NOT_READY, "Summary not ready yet"),
        )
        return JSONResponse(status_code=202, content=body.model_dump())
    
    doc: Optional[Dict[str, Any]] = await anyio.to_thread.run_sync(lambda: read_summary(settings.mongo_uri, settings.mongo_db, doc_id))
    summary = None if not doc else doc.get("summary")
    if summary is None:
        log.warning("Summary is None for doc_id=%s despite status=done. doc=%s", doc_id, doc)
    return DocSummary(id=doc_id, status=DocumentStatus.DONE, summary=summary, err=None)

@router.get("", response_model=DocListResponse)
async def list_docs( #GET /docs?page=2&size=10&status=done
    page: int = Query(1, ge=1), # GET /docs?page=2 (default is greater than 1)
    size: int = Query(10, ge=1, le=100), # GET /docs?size=20 (default ge than 1 and less than 100)
    status: Optional[DocumentStatus] = Query(None) #GET /docs?status=done
) -> DocListResponse:
    status_str = status.value if status else None
    items, total = await anyio.to_thread.run_sync(lambda: list_documents(settings.sqlite_path, page, size, status_str))
    return DocListResponse(
        items=[build_doc_meta_response(r) for r in items],
        total=total, 
        page=page,
        size=size, 
    )

@router.post("/search", status_code=200)
async def search_documents(request: RAGQueryRequest) -> Dict[str, Any]:
    """
    Search/query documents using RAG (Retrieval-Augmented Generation).
    
    Accepts natural language queries in the request body - no URL encoding needed!
    Works like a chatbot - just write your question naturally.
    
    Example:
    {
        "query": "What are the key points discussed in this document?",
        "doc_id": "abc123...",  # optional: filter to specific document
        "top_k": 5              # optional: number of chunks to retrieve (default: 5)
    }
    """
    if not settings.enable_rag:
        raise HTTPException(status_code=503, detail="RAG is not enabled")
    
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Query using RAG
    response, retrieved_chunks = await anyio.to_thread.run_sync(
        lambda: rag_service.query(
            query_text=request.query,
            top_k=request.top_k,
            doc_filter=request.doc_id,
        )
    )
    
    return {
        "query": request.query,
        "answer": response,
        "sources": [
            {
                "doc_id": chunk.metadata.get("doc_id"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "score": chunk.score,
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            }
            for chunk in retrieved_chunks
        ],
    }

@router.get("/{doc_id}/content", response_model=DocContentResponse)
async def get_doc_content(doc_id: str) -> DocContentResponse:
    """
    Get raw document content.
    
    Returns the original text content extracted from the document.
    """
    # Check if document exists
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get content from MongoDB
    content = await anyio.to_thread.run_sync(
        lambda: read_raw_doc(settings.mongo_uri, settings.mongo_db, doc_id)
    )
    
    if content is None:
        raise HTTPException(status_code=404, detail="Document content not found")
    
    mime_type = raw.get("mime") or "text/plain"
    return DocContentResponse(
        id=doc_id,
        content=content,
        mime_type=mime_type,
    )

@router.delete("/{doc_id}", status_code=204)
async def delete_doc(doc_id: str) -> Response:
    """
    Delete a document.
    
    Deletes document from:
    - SQLite (metadata)
    - MongoDB (content and summary)
    - Vector store (embeddings, if RAG enabled)
    
    Returns 204 No Content on success.
    """
    # Check if document exists
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from SQLite
    deleted_sqlite = await anyio.to_thread.run_sync(
        lambda: delete_document_sqlite(settings.sqlite_path, doc_id)
    )
    if not deleted_sqlite:
        log.warning("Document not found in SQLite for deletion doc_id=%s", doc_id)
    
    # Delete from MongoDB
    try:
        deleted_mongo = await anyio.to_thread.run_sync(
            lambda: delete_document_mongo(settings.mongo_uri, settings.mongo_db, doc_id)
        )
        if not deleted_mongo:
            log.warning("Document not found in MongoDB for deletion doc_id=%s", doc_id)
    except Exception as e:
        log.exception("Failed to delete document from MongoDB doc_id=%s: %s", doc_id, e)
        # Continue with deletion even if MongoDB fails
    
    # Delete from vector store if RAG is enabled
    if settings.enable_rag and rag_service:
        try:
            await anyio.to_thread.run_sync(
                lambda: rag_service.vector_store.delete_document(doc_id)
            )
            log.info("Deleted document from vector store doc_id=%s", doc_id)
        except Exception as e:
            log.warning("Failed to delete document from vector store doc_id=%s: %s", doc_id, e)
            # Continue even if vector store deletion fails
    
    # Return 204 No Content (standard for DELETE)
    return Response(status_code=204)
    