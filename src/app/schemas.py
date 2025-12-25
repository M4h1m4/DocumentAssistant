from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from enum import Enum

class DocCreateResponse(BaseModel):
    id: str
    status: DocumentStatus

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

class DocSummaryResponse(BaseModel):
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
    pending: str ="pending"
    processing: str="processing"
    done: str="done"
    failed:str="failed"

class ApiErrorCode(BaseModel):
    NOT_FOUND: str = "NOT_FOUND"
    UNSUPPORTED_MIME: str= "UNSUPPORTED_MIME"
    FILE_TOO_LARGE :str= "FILE_TOO_LARGE"
    MONGO_WRITE_FAILED : str= "MONGO_WRITE_FAILED"
    SQLITE_WRITE_FAILED : str= "SQLITE_WRITE_FAILED"
    SUMMARY_NOT_READY :str = "SUMMARY_NOT_READY"
    SUMMARY_FAILED :str = "SUMMARY_FAILED"
    SERVER_MISCONFIG :str = "SERVER_MISCONFIG"

class ApiError(BaseModel):
    code: ApiErrorCode
    message: str
