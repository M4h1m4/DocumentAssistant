from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class DocCreateResponse(BaseModel):
    id: str
    status: DocumentStatus
    display_name: str

class DocMetaResponse(BaseModel):
    id: str
    filename: str
    size: int
    mime: str
    sha256: str
    status: DocumentStatus
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None 
    completion_tokens: Optional[int] = None
    last_error: Optional[str] = None
    created_at: datetime 
    updated_at: datetime 

class DocSummary(BaseModel):
    id: str
    status: DocumentStatus
    summary: Optional[str] = None 
    err: Optional[ApiError] = None

class DocListResponse(BaseModel):
    items: List[DocMetaResponse]
    total: int
    page: int 
    size: int 

class DocumentStatus(str, Enum):
    PENDING ="pending"
    PROCESSING ="processing"
    DONE = "done"
    FAILED = "failed"

class ApiErrorCode(str, Enum):
    NOT_FOUND: str = "NOT_FOUND"
    UNSUPPORTED_MIME: str= "UNSUPPORTED_MIME"
    FILE_TOO_LARGE :str= "FILE_TOO_LARGE"
    MONGO_WRITE_FAILED : str= "MONGO_WRITE_FAILED"
    SQLITE_WRITE_FAILED : str= "SQLITE_WRITE_FAILED"
    SUMMARY_NOT_READY :str = "SUMMARY_NOT_READY"
    SUMMARY_FAILED :str = "SUMMARY_FAILED"
    SERVER_MISCONFIG :str = "SERVER_MISCONFIG"
    RETRY_LIMIT_EXCEEDED = "RETRY_LIMIT_EXCEEDED"

class ApiError(BaseModel):
    code: ApiErrorCode
    message: str

class RAGQueryRequest(BaseModel):
    """Request model for RAG query endpoint - allows natural language queries without URL encoding."""
    query: str = Field(..., description="Natural language query - write your question normally, no URL encoding needed!")
    doc_id: Optional[str] = Field(None, description="Filter query to a specific document ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant chunks to retrieve (1-20)")
