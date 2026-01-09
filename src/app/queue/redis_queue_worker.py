from __future__ import annotations 

import threading
import time 

from dataclasses import dataclass
from typing import Optional 
import redis 


from .redis_queue import get_redis, ensure_group, read_job, ack_job, enqueue_job
from ..database.sqlite import record_failure, set_status, get_doc_meta
from ..database.mongo import get_mongo_client, get_raw_doc, put_summary
from ..services.document.summarizer import summarize_text
from ..schemas import DocumentStatus

from ..logging_config import get_logger
log = get_logger("precisbox.redis_worker")


@dataclass(frozen=True)
class workerconfig:
    sqlite_path: str
    mongo_uri: str
    mongo_db: str
    openai_api_key: str
    openai_model: str
    openai_timeout: float = 120.0
    workers: int = 1
    max_retries: int = 2
    retry_backoff: float = 0.5 

    consumer_prefix: str = "worker"

_threads : list[threading.Thread] = []
_stop_event = threading.Event()

def start_worker(cfg: workerconfig) -> None:
    r = get_redis()
    ensure_group(r) 

    _stop_event.clear()

    for i in range(cfg.workers):
        consumer = f"{cfg.consumer_prefix}-{i+1}"
        t = threading.Thread(
            target= _worker_loop, 
            args=(cfg, consumer),
            daemon=True, 
            name=f"precisbox-redis-worker-{i+1}",
        )
        t.start()
        _threads.append(t)
    log.info("Redis workers started (n=%d)", cfg.workers)

def stop_workers() -> None:
    _stop_event.set()

def _worker_loop(cfg: workerconfig, consumer: str) -> None:
    r = get_redis()
    ensure_group(r)

    log.info("Worker started consumer=%s", consumer)

    while not _stop_event.is_set():
        job = read_job(r, consumer=consumer, block_ms=5000, count=1)
        if job is None:
            continue 
        msg_id, doc_id = job 
        try:
            _process_one(cfg, doc_id)
            ack_job(r, msg_id)

        except Exception as e:
            log.exception("Worker error doc_id=%s msg_id=%s", doc_id, msg_id)
            new_attempts = record_failure(
                cfg.sqlite_path,
                doc_id, 
                err=str(e),
                status=DocumentStatus.PENDING if hasattr(DocumentStatus, "pending") else "pending",
            )
            if new_attempts >= cfg.max_retries: 
                set_status(
                    cfg.sqlite_path,
                    doc_id, 
                      DocumentStatus.FAILED if hasattr(DocumentStatus, "failed") else "failed",
                      last_error = str(e),
                )
                ack_job(r, msg_id)
                continue 
            time.sleep(cfg.retry_backoff* new_attempts)
            enqueue_job(r, doc_id)
            ack_job(r, msg_id)

def _process_one(cfg: workerconfig, doc_id: str) -> None:
    meta = get_doc_meta(cfg.sqlite_path, doc_id)
    if meta is None:
        raise RuntimeError(f"doc_id is not found in SQLite: {doc_id}")

    # failures = int(meta.get("attempts") or 0) 
    # if failures >= cfg.max_retries:
    #     log.warning("Document %s exceeded max retries (%d)", doc_id, cfg.max_retries)
    #     set_status(
    #         cfg.sqlite_path,
    #         doc_id,
    #         status=DocumentStatus.failed.value,
    #         last_error=ApiErrorCode.RETRY_LIMIT_EXCEEDED.value,  # optional if you store codes
    #     )
    #     return
    # log.info("Processing document %s (attempt %d)", doc_id, failures + 1)
    set_status(cfg.sqlite_path, doc_id, status=DocumentStatus.PROCESSING, model=cfg.openai_model)
    client = get_mongo_client(cfg.mongo_uri)
    db = client[cfg.mongo_db]
    text: Optional[str] = get_raw_doc(db, doc_id)
    if not text:
        raise RuntimeError("raw document is missing from mongo")

    # summarize
    log.debug("Calling OpenAI API for document %s", doc_id)
    summary, prompt_tokens, completion_tokens = summarize_text(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        text=text,
        timeout=cfg.openai_timeout,
    )
    if not summary or not summary.strip():
        raise RuntimeError(f"OpenAI returned empty summary for doc_id={doc_id}")
    log.info("OpenAI API call succeeded for document %s", doc_id)
    put_summary(db, doc_id, summary, prompt_tokens or 0, completion_tokens or 0)
    set_status(
        cfg.sqlite_path,
        doc_id,
        status=DocumentStatus.DONE,
        model=cfg.openai_model,
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        last_error=None,
    )







