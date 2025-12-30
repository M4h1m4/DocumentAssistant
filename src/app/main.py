import os
from fastapi import FastAPI, Request, status 
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from .rate_limit import TokenBucketLimiter
import math
from dotenv import load_dotenv
import logging 

from .api import router 
from .db_sql import init_db
from .queue_worker import start_workers, WorkerConfig, _worker_loop
from .rate_limit import TokenBucketLimiter
from .rate_limit_middleware import RateLimitMiddleware

load_dotenv() 
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("precisbox.main")

SQLITE_PATH: str= os.getenv("SQLITE_PATH", "./meta.db")
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "precisbox")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

WORKERS: int = int(os.getenv("WORKERS", "1"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_SEC: float = float(os.getenv("RETRY_BACKOFF", "0.5"))
CRASH_P: float = float(os.getenv("CRASH_P", "0.0"))
OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "120.0"))

# Rate limit knobs
UPLOAD_PER_MIN = int(os.getenv("RL_UPLOAD_PER_MIN", "1"))
SUMMARY_PER_MIN = int(os.getenv("RL_SUMMARY_PER_MIN", "2"))

app = FastAPI(title="PrecisBox")
limiter = TokenBucketLimiter()
app.add_middleware(
    RateLimitMiddleware,
    limiter=limiter,
    upload_limit_per_min=UPLOAD_PER_MIN,
    summary_limit_per_min=SUMMARY_PER_MIN,
)
app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    # Create/migrate SQLite schema
    init_db(SQLITE_PATH)
    log.info("init_db ok (%s)", SQLITE_PATH)

    summarizer_enabled = bool(OPENAI_API_KEY)
    app.state.SUMMARIZER_ENABLED = summarizer_enabled
    if summarizer_enabled:
        log.info("summarizer enabled (model=%s)", OPENAI_MODEL)
    else:
        # App can still run (uploads/metadata), but summarization should be treated as unavailable.
        log.warning("OPENAI_API_KEY missing -> summarizer disabled")

    # Start queue worker threads
    cfg = WorkerConfig(
        sqlite_path=SQLITE_PATH,
        mongo_uri=MONGO_URI,
        mongo_db=MONGO_DB,
        openai_api_key=OPENAI_API_KEY,
        openai_model=OPENAI_MODEL,
        workers=WORKERS,
        max_retries=MAX_RETRIES,
        retry_backoff=RETRY_BACKOFF_SEC,
        crash_p=CRASH_P,
        openai_timeout=OPENAI_TIMEOUT,
    )
    start_workers(cfg)
    log.info("workers started (n=%d)", WORKERS)

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service": "precisbox"}

@app.get("/ready")
def ready() -> JSONResponse:
    if not app.state.SUMMARIZER_ENABLED:
        return JSONResponse(
            status_code=503, 
            content={
                "ready": False,
                "reason": "summarizer_disabled",
                "detail": "OPENAI_API_KEY missing or SUMMARIZER_ENABLED=false",
            },
        )
    return {"ready": True}