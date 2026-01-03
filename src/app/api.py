from __future__ import annotations

from uuid import uuid4
from typing import Tuple, Dict, List, Any, Optional 
from datetime import datetime 
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import anyio
import logging
from .config import settings
from .schemas import (
    ApiError,
    ApiErrorCode,
    DocCreateResponse,
    DocMetaResponse,
    DocSummaryResponse,
    DocListResponse,
    DocumentStatus,
)
from .services.hashing import sha256_bytes, decode_text
from .database.sqlite import (
    insert_document,
    fetch_document,
    list_documents,
    set_status,
)
from .database.mongo import (
    write_raw_doc,
    read_summary
)
from .queue.redis_queue import get_redis, enqueue_job

log = logging.getLogger("precisbox.api")
router = APIRouter(prefix="/docs")

SUPPORTED_MIME = {"text/plain", "text/markdown"}  

def _err(code: ApiErrorCode, msg: str) -> ApiError:
    return ApiError(code=code, message=msg)


"""
#row["x"] is used for columns that are guaranteed to exist
#Incase if they don't exists and we still want to continue the application without any error like keyerrors we use row.get("x") cause .get returns null if the key does not exist. 
Hence we should also define these columns optional in the schemas.py
"""

def _safe_int(val: Any, default: int, field: str, doc_id: str) -> int: 
    try:
        if val is None:
            return default 
        return int(val)
    except Exception:
        log.exception("Invalid %s=%r for doc_id=%s; defaulting to %d", field, val, doc_id, default)
        return default

def _to_meta(row: Dict[str, Any]) -> DocMetaResponse:
    doc_id = row["id"]
    filename = row["filename"]
    created_at = datetime.fromisoformat(row["created_at"])
    updated_at = datetime.fromisoformat(row["updated_at"])

    # Defensive
    size = _safe_int(row.get("size"), 0, "size", doc_id)
    attempts = _safe_int(row.get("attempts"), 0, "attempts", doc_id)

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
async def upload_doc(request: Request, file: UploadFile = File(...)) -> DocCreateResponse:
    # if not OPENAI_API_KEY:
    #     raise HTTPException(status_code=500, detail="OPEN_API_KEY not set")
    summarizer_enabled = bool(getattr(request.app.state, "SUMMARIZER_ENABLED", False))
    if not summarizer_enabled:
        raise HTTPException(status_code=503, detail="Summarizer disabled (OPENAI_API_KEY missing)")

    raw: bytes = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large(max {settings.max_upload_bytes}bytes)")
    mime: str = file.content_type or "application/octet-stream"
    if mime not in ("text/plain", "text/markdown"):
        raise HTTPException(status_code=400, detail=f"unsupported mime type{mime}. Use text/plain for now")
    try:
        text: str = decode_text(raw)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    doc_id: str = uuid4().hex
    filename: str = file.filename or "upload.txt"
    original: str = file.filename or "upload.txt"
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
                "pending",
                settings.openai_model,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write metadata")

    # def mongo_write() -> None: #Write the raw document into mongodb
    #     client = get_mongo_client(MONGO_URI)
    #     db = client[MONGO_DB]
    #     put_raw_doc(db, doc_id, text)

    try:
        await anyio.to_thread.run_sync(lambda: write_raw_doc(settings.mongo_uri, settings.mongo_db, doc_id, text)) 
    except Exception as e:
        await anyio.to_thread.run_sync(
            lambda: set_status(
                settings.sqlite_path,
                doc_id,
                "failed",
                model=settings.openai_model,
                last_error=f"Mongo write failed {e}",
            )
        )
        raise HTTPException(status_code=500, detail=f"Mongo write failed {e}")
    """
    run_sync runs the function in background, so it does not block the async event loop that is this is pushed into the thread_loop
    In mean the server can handle other requests 
    anyio.to_thread --> Schedules the function in threadpool worker. 
    await tells the event loop : I am waiting for the threadppol to complete/finish meantime feel free to run other requests/tasks 
    """
    # Enqueue job in Redis queue for background processing
    r = get_redis()
    enqueue_job(r, doc_id)
    return DocCreateResponse(id=doc_id, status="pending", display_name=display_name)

@router.get("/{doc_id}", response_model=DocMetaResponse)
async def get_doc(doc_id: str) -> DocMetaResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    return _to_meta(raw)

@router.get("/{doc_id}/summary", response_model=DocSummaryResponse)
async def get_doc_summary(doc_id: str) -> DocSummaryResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    status: str = raw["status"]
    if status == DocumentStatus.failed.value:
        body = DocSummaryResponse(
            id=doc_id, 
            status=DocumentStatus.failed,
            summary=None, 
            err=_err(ApiErrorCode.SUMMARY_NOT_READY, "Summary not ready yet"),
        )
        return JSONResponse(status_code=409, content=body.model_dump())

    if status != DocumentStatus.done.value:
        body = DocSummaryResponse(
            id=doc_id,
            status=DocumentStatus(status),
            summary=None,
            err=_err(ApiErrorCode.SUMMARY_NOT_READY, "Summary not ready yet"),
        )
        return JSONResponse(status_code=202, content=body.model_dump())
    
    doc = await anyio.to_thread.run_sync(lambda: read_summary(settings.mongo_uri, settings.mongo_db, doc_id))
    summary = None if not doc else doc.get("summary")
    return DocSummaryResponse(id=doc_id, status=DocumentStatus.done, summary=summary, err=None)

@router.get("", response_model=DocListResponse)
async def list_docs( #GET /docs?page=2&size=10&status=done
    page: int = Query(1, ge=1), # GET /docs?page=2 (default is greater than 1)
    size: int = Query(10, ge=1, le=100), # GET /docs?size=20 (default ge than 1 and less than 100)
    status: Optional[str] = Query(None) #GET /docs?status=done
) -> DocListResponse:
    items, total = await anyio.to_thread.run_sync(lambda: list_documents(settings.sqlite_path, page, size, status))
    return DocListResponse(
        items=[_to_meta(r) for r in items],
        total=total, 
        page=page,
        size=size, 
    )
    