from fastapi import FastAPI
from starlette.responses import JSONResponse
import redis

from .api import router 
from .database.sqlite import init_db
from .database.mongo import init_mongo_client
from .queue.redis_queue_worker import start_worker, workerconfig
from .middleware.rate_limit_redis import RedisTokenBucketLimiter
from .middleware.rate_limit_middleware import RateLimitMiddleware

from .logging_config import setup_json_logging, get_logger
from .config import settings

setup_json_logging(log_level="INFO")
log = get_logger("precisbox.main")

app = FastAPI(title="PrecisBox")
app.include_router(router)

r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
limiter = RedisTokenBucketLimiter(
    redis_client=r, 
    enforce_known_user=False,
)

app.add_middleware(
    RateLimitMiddleware,
    limiter=limiter,
    upload_limit_per_min=settings.upload_per_min,
    summary_limit_per_min=settings.summary_per_min,
)


@app.on_event("startup")
def _startup() -> None:
    # Create/migrate SQLite schema
    init_db(settings.sqlite_path)
    log.info("init_db ok (%s)", settings.sqlite_path)

    init_mongo_client(
        settings.mongo_uri,
        max_pool_size=settings.mongo_max_pool_size,
        min_pool_size=settings.mongo_min_pool_size,
    )
    log.info(
        "MongoDB client initialized with connection pooling (uri=%s, max_pool=%d, min_pool=%d)",
        settings.mongo_uri,
        settings.mongo_max_pool_size,
        settings.mongo_min_pool_size,
    )

    app.state.SUMMARIZER_ENABLED = settings.is_summarizer_enabled
    if settings.is_summarizer_enabled:
        log.info("summarizer enabled (model=%s)", settings.openai_model)
    else:
        # App can still run (uploads/metadata), but summarization should be treated as unavailable.
        log.warning("OPENAI_API_KEY missing -> summarizer disabled")

    # Start queue worker threads
    cfg = workerconfig(
        sqlite_path=settings.sqlite_path,
        mongo_uri=settings.mongo_uri,
        mongo_db=settings.mongo_db,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        workers=settings.workers,
        max_retries=settings.max_retries,
        retry_backoff=settings.retry_backoff,
        openai_timeout=settings.openai_timeout,
    )
    start_worker(cfg)
    log.info("workers started (n=%d)", settings.workers)

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service": "precisbox"}

@app.get("/ready")
def ready() -> JSONResponse:
    if not settings.is_summarizer_enabled:
        return JSONResponse(
            status_code=503, 
            content={
                "ready": False,
                "reason": "summarizer_disabled",
                "detail": "OPENAI_API_KEY missing or SUMMARIZER_ENABLED=false",
            },
        )
    return {"ready": True}