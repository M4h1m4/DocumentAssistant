from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocCreateResponse(BaseModel):
    id: str
    status: str

class DocMetaResponse(BaseModel):
    id: str
    filename: str
    size: int
    mime: str
    sha256: str
    status: str
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None 
    completion_tokens: Optional[int] = None
    last_error: Optional[str] = None
    created_at: datetime 
    updated_at: datetime 

class DocSummaryResponse(BaseModel):
    id: str
    status: str
    summary: Optional[str] = None 

class DocListResponse(BaseModel):
    items: List[DocMetaResponse]
    total: int
    page: int 
    size: int 