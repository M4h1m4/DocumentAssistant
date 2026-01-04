# FastAPI Dependency Injection for Database Clients

This guide explains how to implement FastAPI dependency injection patterns for MongoDB and SQLite database clients. Using dependencies makes the code more testable, maintainable, and follows FastAPI best practices.

## Why Use FastAPI Dependencies?

### Current Issues (Lines 145-146 in `api.py`)

**Current approach:**
- Directly calling database functions with connection parameters
- Hard to mock in unit tests
- Tight coupling between routes and database implementation
- Repetitive connection setup code

### Benefits of Dependency Injection

1. **Easier Testing**: Mock dependencies in unit tests
2. **Standard Pattern**: FastAPI's recommended approach
3. **Reusability**: Share dependencies across routes
4. **Lifecycle Management**: Control when resources are created/closed
5. **Configuration**: Centralized dependency configuration
6. **Type Safety**: Better IDE support and type checking

## Understanding FastAPI Dependencies

### What are Dependencies?

Dependencies are functions that FastAPI automatically calls when needed. They can:
- Provide shared resources (database clients, HTTP clients)
- Perform authentication/authorization
- Validate request data
- Extract common parameters

### Dependency Injection Pattern

```python
from fastapi import Depends

# Define a dependency function
def get_db():
    db = create_database_connection()
    try:
        yield db  # Provide the resource
    finally:
        db.close()  # Cleanup after use

# Use in route
@app.get("/items")
def get_items(db = Depends(get_db)):
    return db.query(...)
```

**Key points:**
- Use `yield` for resources that need cleanup (connection closing)
- FastAPI handles the lifecycle automatically
- Dependencies can depend on other dependencies

## Implementation Steps

### Step 1: Create Dependency Functions Module

**Create a new file: `src/app/dependencies.py`**

This file will contain all dependency functions for database clients.

```python
from __future__ import annotations

from typing import Generator
from fastapi import Depends
from pymongo import MongoClient
from pymongo.database import Database
import sqlite3
from sqlite3 import Connection

from .config import settings
from .database.mongo import get_mongo_client
from .database.sqlite import get_conn


def get_mongo_db() -> Generator[Database, None, None]:
    """
    FastAPI dependency to get MongoDB database instance.
    
    Yields:
        Database: MongoDB database instance
        
    Usage:
        @router.get("/docs")
        def get_docs(db: Database = Depends(get_mongo_db)):
            return db.docs.find_one(...)
    """
    client: MongoClient = get_mongo_client(settings.mongo_uri)
    db: Database = client[settings.mongo_db]
    yield db
    # Note: We don't close the client here as it's pooled and shared
    # The client will be closed during application shutdown


def get_sqlite_db() -> Generator[Connection, None, None]:
    """
    FastAPI dependency to get SQLite database connection.
    
    Yields:
        Connection: SQLite connection instance
        
    Usage:
        @router.get("/docs")
        def get_docs(conn: Connection = Depends(get_sqlite_db)):
            cursor = conn.execute("SELECT * FROM documents")
            return cursor.fetchall()
    """
    conn: Connection = get_conn(settings.sqlite_path)
    try:
        yield conn
    finally:
        conn.close()


# Alternative: If you want to use the existing database functions
# You can create dependencies that return the connection/client directly

def get_mongo_client_dep() -> Generator[MongoClient, None, None]:
    """
    FastAPI dependency to get MongoDB client instance.
    
    Yields:
        MongoClient: MongoDB client instance
        
    Usage:
        @router.get("/docs")
        def get_docs(client: MongoClient = Depends(get_mongo_client_dep)):
            db = client[settings.mongo_db]
            return db.docs.find_one(...)
    """
    client: MongoClient = get_mongo_client(settings.mongo_uri)
    yield client
    # Client is pooled, don't close here
```

### Step 2: Update API Routes to Use Dependencies

**File: `src/app/api.py`**

**Current approach (lines 145-146 and similar):**
```python
async def get_doc_summary(doc_id: str) -> DocSummaryResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document(settings.sqlite_path, doc_id))
    # ...
    doc = await anyio.to_thread.run_sync(lambda: read_summary(settings.mongo_uri, settings.mongo_db, doc_id))
```

**Changes needed:**

1. **Add imports at the top:**
```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import Optional
from pymongo.database import Database
from sqlite3 import Connection

from .dependencies import get_mongo_db, get_sqlite_db
```

2. **Update route functions to accept dependencies:**

**For MongoDB operations (e.g., `get_doc_summary`):**
```python
@router.get("/{doc_id}/summary", response_model=DocSummaryResponse)
async def get_doc_summary(
    doc_id: str,
    sqlite_db: Connection = Depends(get_sqlite_db),
    mongo_db: Database = Depends(get_mongo_db),
) -> DocSummaryResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document_from_conn(sqlite_db, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    status = DocumentStatus(raw["status"])
    if status == DocumentStatus.failed:
        # ... existing code ...

    if status != DocumentStatus.done:
        # ... existing code ...
    
    # Use mongo_db dependency instead of calling read_summary with URI
    doc: Optional[Dict[str, Any]] = await anyio.to_thread.run_sync(
        lambda: get_summary(mongo_db, doc_id)
    )
    summary = None if not doc else doc.get("summary")
    if summary is None:
        log.warning("Summary is None for doc_id=%s despite status=done. doc=%s", doc_id, doc)
    return DocSummary(id=doc_id, status=DocumentStatus.done, summary=summary, err=None)
```

**For SQLite operations (e.g., `get_doc`):**
```python
@router.get("/{doc_id}", response_model=DocMetaResponse)
async def get_doc(
    doc_id: str,
    sqlite_db: Connection = Depends(get_sqlite_db),
) -> DocMetaResponse:
    raw = await anyio.to_thread.run_sync(lambda: fetch_document_from_conn(sqlite_db, doc_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found")

    return build_doc_meta_response(raw)
```

**For list operations:**
```python
@router.get("", response_model=DocListResponse)
async def list_docs(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    status: Optional[DocumentStatus] = Query(None),
    sqlite_db: Connection = Depends(get_sqlite_db),
) -> DocListResponse:
    status_str = status.value if status else None
    items, total = await anyio.to_thread.run_sync(
        lambda: list_documents_from_conn(sqlite_db, page, size, status_str)
    )
    return DocListResponse(
        items=[build_doc_meta_response(r) for r in items],
        total=total,
        page=page,
        size=size,
    )
```

**For upload operations:**
```python
@router.post("", response_model=DocCreateResponse, status_code=201)
async def upload_doc(
    file: UploadFile = File(...),
    sqlite_db: Connection = Depends(get_sqlite_db),
    mongo_db: Database = Depends(get_mongo_db),
) -> DocCreateResponse:
    # ... existing validation code ...

    try:
        await anyio.to_thread.run_sync(
            lambda: insert_document_to_conn(
                sqlite_db,
                doc_id,
                filename,
                size,
                mime,
                sha256,
                DocumentStatus.pending.value,
                settings.openai_model,
            )
        )
    except Exception as e:
        log.exception("Failed to write metadata for doc_id=%s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    try:
        await anyio.to_thread.run_sync(
            lambda: put_raw_doc(mongo_db, doc_id, text)
        )
    except Exception as e:
        log.exception("Failed to write document to MongoDB for doc_id=%s: %s", doc_id, e)
        await anyio.to_thread.run_sync(
            lambda: set_status_to_conn(
                sqlite_db,
                doc_id,
                "failed",
                model=settings.openai_model,
                last_error=ApiErrorCode.MONGO_WRITE_FAILED.value,
            )
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")

    # ... rest of code ...
```

### Step 3: Create Helper Functions in Database Modules

**File: `src/app/database/sqlite.py`**

**Add functions that accept Connection instead of sqlite_path:**

```python
def fetch_document_from_conn(conn: Connection, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a document from SQLite using an existing connection.
    
    Args:
        conn: SQLite connection
        doc_id: Document ID
        
    Returns:
        Dictionary with document data or None if not found
    """
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (doc_id,),
    ).fetchone()
    return None if row is None else dict(row)


def list_documents_from_conn(
    conn: Connection,
    page: int,
    size: int,
    status: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    List documents from SQLite using an existing connection.
    
    Args:
        conn: SQLite connection
        page: Page number (1-indexed)
        size: Page size
        status: Optional status filter
        
    Returns:
        Tuple of (items list, total count)
    """
    offset = (page - 1) * size
    if status:
        total_row = conn.execute(
            "SELECT COUNT (*) AS c FROM documents WHERE status = ?",
            (status,),
        ).fetchone()
        total = int(total_row["c"]) if total_row else 0

        rows = conn.execute(
            """
            SELECT * FROM documents
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
            (status, size, offset)
        ).fetchall()
    else:
        total_row = conn.execute("SELECT COUNT (*) AS c FROM documents").fetchone()
        total = int(total_row["c"]) if total_row else 0
        rows = conn.execute(
            """
            SELECT * from documents
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
            (size, offset),
        ).fetchall()
    return [dict(r) for r in rows], total


def insert_document_to_conn(
    conn: Connection,
    doc_id: str,
    filename: str,
    size: int,
    mime: str,
    sha256: str,
    status: str,
    model: Optional[str],
) -> None:
    """
    Insert a document into SQLite using an existing connection.
    
    Args:
        conn: SQLite connection
        doc_id: Document ID
        filename: Filename
        size: File size in bytes
        mime: MIME type
        sha256: SHA256 hash
        status: Document status
        model: OpenAI model name
    """
    now = utcnow_iso()
    conn.execute(
        """
        INSERT INTO documents (id, filename, size, mime, sha256, status, model, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, filename, size, mime, sha256, status, model, now, now),
    )
    conn.commit()


def set_status_to_conn(
    conn: Connection,
    doc_id: str,
    status: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[str] = None,
    completion_tokens: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    """
    Update document status in SQLite using an existing connection.
    
    Args:
        conn: SQLite connection
        doc_id: Document ID
        status: New status
        model: Model name
        prompt_tokens: Prompt tokens
        completion_tokens: Completion tokens
        last_error: Error message
    """
    now = utcnow_iso()
    conn.execute(
        """
        UPDATE documents
        SET status = ?,
            model = COALESCE(?, model),
            prompt_tokens = COALESCE(?, prompt_tokens),
            completion_tokens = COALESCE(?, completion_tokens),
            last_error = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (status, model, prompt_tokens, completion_tokens, last_error, now, doc_id),
    )
    conn.commit()
```

**File: `src/app/database/mongo.py`**

**Add functions that accept Database instead of mongo_uri/mongo_db:**

```python
def get_summary_from_db(db: Database, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get summary from MongoDB using an existing database instance.
    
    Args:
        db: MongoDB database instance
        doc_id: Document ID
        
    Returns:
        Dictionary with summary data or None if not found
    """
    return db.docs.find_one(
        {"_id": doc_id},
        {"summary": 1, "summary_meta": 1}
    )


# Note: put_raw_doc, put_summary, get_raw_doc already accept Database/DB,
# so they can be used directly with the dependency
```

### Step 4: Update Imports in api.py

**File: `src/app/api.py`**

**Update imports section:**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import Tuple, Dict, List, Any, Optional
from datetime import datetime
from fastapi.responses import JSONResponse
from pymongo.database import Database
from sqlite3 import Connection
import anyio
import logging

from .config import settings, Defaults
from .dependencies import get_mongo_db, get_sqlite_db
from .schemas import (
    ApiError,
    ApiErrorCode,
    DocCreateResponse,
    DocMetaResponse,
    DocSummary,
    DocListResponse,
    DocumentStatus,
)
from .services.hashing import sha256_bytes, decode_text
from .database.sqlite import (
    fetch_document_from_conn,
    list_documents_from_conn,
    insert_document_to_conn,
    set_status_to_conn,
)
from .database.mongo import (
    put_raw_doc,
    get_summary_from_db,
)
from .queue.redis_queue import get_redis, enqueue_job
from .utils import safe_int
```

## Testing with Mocked Dependencies

### Example Unit Test

**File: `tests/test_api.py`**

```python
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies import get_mongo_db, get_sqlite_db


def test_get_doc_summary_success():
    # Create mock dependencies
    mock_mongo_db = MagicMock()
    mock_sqlite_db = MagicMock()
    
    # Mock MongoDB response
    mock_mongo_db.docs.find_one.return_value = {
        "summary": "This is a test summary",
        "summary_meta": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    
    # Mock SQLite response
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda self, key: {
        "id": "test-doc-id",
        "status": "done",
        "filename": "test.txt",
        # ... other fields
    }.get(key)
    mock_sqlite_db.execute.return_value.fetchone.return_value = mock_row
    
    # Override dependencies
    app.dependency_overrides[get_mongo_db] = lambda: mock_mongo_db
    app.dependency_overrides[get_sqlite_db] = lambda: mock_sqlite_db
    
    # Test
    client = TestClient(app)
    response = client.get("/docs/test-doc-id/summary")
    
    assert response.status_code == 200
    assert response.json()["summary"] == "This is a test summary"
    
    # Cleanup
    app.dependency_overrides.clear()
```

### Using pytest fixtures

**File: `tests/conftest.py`**

```python
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.dependencies import get_mongo_db, get_sqlite_db


@pytest.fixture
def mock_mongo_db():
    """Fixture providing a mock MongoDB database."""
    return MagicMock()


@pytest.fixture
def mock_sqlite_db():
    """Fixture providing a mock SQLite connection."""
    return MagicMock()


@pytest.fixture
def client(mock_mongo_db, mock_sqlite_db):
    """Fixture providing a test client with mocked dependencies."""
    app.dependency_overrides[get_mongo_db] = lambda: mock_mongo_db
    app.dependency_overrides[get_sqlite_db] = lambda: mock_sqlite_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# Usage in tests
def test_get_doc(client, mock_sqlite_db):
    # Setup mock
    mock_sqlite_db.execute.return_value.fetchone.return_value = {
        "id": "test-id",
        "status": "pending",
        # ...
    }
    
    # Test
    response = client.get("/docs/test-id")
    assert response.status_code == 200
```

## Advanced: Sub-dependencies and Configuration

### Creating Configurable Dependencies

**File: `src/app/dependencies.py`**

```python
from fastapi import Depends
from .config import settings, Settings


def get_settings() -> Settings:
    """Dependency to get application settings."""
    return settings


def get_mongo_db(
    settings: Settings = Depends(get_settings)
) -> Generator[Database, None, None]:
    """
    MongoDB database dependency with configurable settings.
    """
    client = get_mongo_client(settings.mongo_uri)
    db = client[settings.mongo_db]
    yield db


def get_sqlite_db(
    settings: Settings = Depends(get_settings)
) -> Generator[Connection, None, None]:
    """
    SQLite database dependency with configurable settings.
    """
    conn = get_conn(settings.sqlite_path)
    try:
        yield conn
    finally:
        conn.close()
```

## Migration Checklist

- [ ] Create `src/app/dependencies.py` file
- [ ] Add `get_mongo_db()` dependency function
- [ ] Add `get_sqlite_db()` dependency function
- [ ] Add helper functions to `sqlite.py` (accepting Connection)
- [ ] Add helper functions to `mongo.py` (accepting Database) - if needed
- [ ] Update imports in `api.py`
- [ ] Update route functions to use `Depends()`:
  - [ ] `upload_doc()`
  - [ ] `get_doc()`
  - [ ] `get_doc_summary()`
  - [ ] `list_docs()`
- [ ] Replace direct database calls with dependency-based calls
- [ ] Test all routes still work
- [ ] Write unit tests with mocked dependencies
- [ ] Remove unused imports (if any)
- [ ] Update documentation if needed

## Benefits Summary

✅ **Testability**: Easy to mock dependencies in unit tests  
✅ **Standard Pattern**: Follows FastAPI best practices  
✅ **Maintainability**: Centralized dependency configuration  
✅ **Type Safety**: Better IDE support and type checking  
✅ **Lifecycle Management**: Automatic resource cleanup  
✅ **Reusability**: Share dependencies across multiple routes  
✅ **Flexibility**: Easy to swap implementations (e.g., different DB for testing)  

## Common Patterns

### Pattern 1: Simple Resource Dependency
```python
def get_db():
    db = create_connection()
    yield db
    db.close()
```

### Pattern 2: Dependency with Configuration
```python
def get_db(settings: Settings = Depends(get_settings)):
    db = create_connection(settings.db_url)
    yield db
    db.close()
```

### Pattern 3: Dependency on Another Dependency
```python
def get_user(db: Connection = Depends(get_db)):
    # Use db to get user
    yield user
```

## References

- [FastAPI Dependencies Documentation](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python Dependency Injection Patterns](https://docs.python.org/3/library/unittest.mock.html)

