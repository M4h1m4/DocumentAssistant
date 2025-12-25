from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any 

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def get_conn(sqlite_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    #In this a new connection is created per call. 
    #With this we are not sharing a connections across threads. Hence check_same_thread is assigned False
    conn.row_factory = sqlite3.Row 
    #Helps to acess the row with attributes instead only the index numbers
    conn.execute("PRAGMA journal_mode=WAL;")
    #Write-Ahead Logging: Multiple Reads and Single Write
    conn.execute("PRAGMA busy_timeout=3000;")
    #wait till 3 seconds before failing the job when the db is locked. 
    return conn 

def init_db(sqlite_path: str) -> None:
    with get_conn(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
            id          TEXT    PRIMARY KEY,
            filename    TEXT    NOT NULL, 
            size        INTEGER NOT NULL,
            mime        TEXT    NOT NULL,
            sha256      TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            model       TEXT, 
            prompt_tokens   INTEGER,
            completion_tokens   INTEGER,
            attempts        INTEGER NOT NULL DEFAULT 0,
            last_error      TEXT,
            created_at      TEXT NOT NULL, 
            updated_at      TEXT NOT NULL
            );
            """
        )
        #mime is to store the extenstion/type of file.
        #sha256 to store the fingerprint of the file. Helps in deduplication of files 
        #prompt_tokens to store the no of tokens used by the input file 
        #completion_tokens to store the no of tokens used by the summary generated. 
        #total tokens = prompt_tokens + completion_tokens
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "attempts" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_sha256 ON documents(sha256);")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_created_at ON documents(created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_updated_at ON documents(updated_at);")

def insert_document(
    sqlite_path: str, 
    doc_id: str,
    filename: str, 
    size: str,
    mime: str, 
    sha256: str, 
    status: str, 
    model: Optional[str],
) -> None:
    now = utcnow_iso()
    with get_conn(sqlite_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, size, mime, sha256, status, model, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, size, mime, sha256, status, model, now, now), 
        )

def set_status(
    sqlite_path: str, 
    doc_id: str, 
    status: str, 
    model: Optional[str] = None, 
    prompt_tokens: Optional[str] = None, 
    completion_tokens: Optional[str] = None, 
    last_error: Optional[str] = None,
) -> None:
    now = utcnow_iso()
    with get_conn(sqlite_path) as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = ?,
                model = COALESCE(?, model),
                prompt_tokens = COALESCE(?, prompt_tokens),
                completion_tokens = COALESCE(?, completion_tokens),
                last_error = ?,
                updated_at =?
            WHERE id = ?
            """,
            (status, model, prompt_tokens, completion_tokens, last_error, now, doc_id),
        )

def fetch_documents(sqlite_path: str, doc_id: str) -> Optional[Dict[str, Any]]:
    with get_conn(sqlite_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return None if row is None else dict(row)
    
def list_documents(
    sqlite_path: str, 
    page: int, 
    size: int, 
    status: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    #To paginate documents. Optional filter by status 
    offset = (page-1)*size
    with get_conn(sqlite_path) as conn:
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
                    LIMIT ? OFFSET = ?
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
        return [dict[r] for r in rows], total

"""
list_documents returns the a page of rows
total (rows) count so that the API can show pagination info 
gets the size rows and skips the offset rows. 
"""

def get_attempts(sqlite_path: str, doc_id: str) -> int:
    with get_conn(sqlite_path) as conn:
        row = conn.execute(
            "SELECT attempts FROM documents WHERE id = ?", 
            (doc_id,)
        ).fetchone()
        if row is None:
            return 0 
        return int(row["attempts"] or 0)

def record_failure(sqlite_path: str, doc_id: str, err: str, status: str) -> int:
    # This function is to increment attempts and update stautus and error 
    now: str = utcnow_iso()
    with get_conn(sqlite_path) as conn:
        conn.execute(
            """
            UPDATE documents 
                SET attempts = COALESCE(attempts, 0) + 1,
                    last_error = ?,
                    status = ?,
                    updated_at = ?
                WHERE id = ?
            """,
            (err[:200], status, now, doc_id),
        )
        row = conn.execute("SELECT attempts FROM documents WHERE id =?", (doc_id,)).fetchone()
        return 0 if row is None else int(row["attempts"] or 0)

def get_doc_meta(sqlite_path: str, doc_id: str) -> Optional[Dict[str, Any]]:
    with get_conn(sqlite_path) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,),
            ).fetchone()
        if row is None:
            return None 
        return dict(row)