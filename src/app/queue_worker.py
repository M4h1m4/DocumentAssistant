from __future__ import annotations 

import queue 
import threading 
import time 
import random 
import logging
from typing import Optional
from dataclasses import dataclass

from .db_sql import set_status, record_failure, get_attempts, get_doc_meta
from .db_mongo import get_mongo_client, put_summary, get_raw_doc
from .services.summarize import summarize_text

log = logging.getLogger("precisbox")



"""

Worker loop = infinite job processor

Worker thread = background executor

Queue = buffer between HTTP requests and slow work

This gives:

concurrency control

failure handling

retry capability

better scalability

"""

job_queue: "queue.Queue[str]" = queue.Queue()


@dataclass(frozen=True) #frozen=true means that the class is immutable. The configuraitons cannot be changed at runtime.
class WorkerConfig:
    sqlite_path: str
    mongo_uri: str
    mongo_db: str
    openai_api_key: str
    openai_model : str
    workers: int = 1
    max_retries: int = 2
    retry_backoff: float = 0.5 
    crash_p: float = 0.0
    openai_timeout: float = 120.0 

_threads : list[threading.Thread] = [] # helps to track worker threads.
_stop_event= threading.Event() # a boolean shared across threads.

def enqueue_job(doc_id: str) -> None:
    job_queue.put(doc_id)

def start_workers(cfg: WorkerConfig) -> None:
    _stop_event.clear() # to ensure stop flag is off before starting the threads. 
    for i in range(cfg.workers):
        t = threading.Thread(
            target= _worker_loop, # this means to say start the workerloop when the threads starts
            args=(i, cfg), # these args are passed into the worker loop
            daemon=True,  #If only daemon threads are left, Python will exit and kill them automatically.
            name=f"precisbox-worker{i}",
        )
        t.start()
        _threads.append(t)

def stop_workers() -> None:
    _stop_event.set()

def _worker_loop(index: int, cfg: WorkerConfig) -> None:
    label = f"w-{index+1}"
    while not _stop_event.is_set():
        try:
            doc_id: str = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue 
        try:
            if cfg.crash_p > 0.0 and random.random() < cfg.crash_p:
                raise RuntimeError(f"Simulated crash in {label}") 
            _process_one(doc_id, cfg)
        except Exception as e:
            set_status(cfg.sqlite_path, doc_id, status="failed", last_error=f"Workerloop crashed{e}")
            # record_failure(cfg.sqlite_path, doc_id, str(e), status="failed")
        finally:
            job_queue.task_done()

def _process_one(doc_id: str, cfg: WorkerConfig) -> None:
    meta = get_doc_meta(cfg.sqlite_path, doc_id)
    if meta is None:
        log.warning("Document %s not found in database", doc_id)
        return 
    last_error: Optional[str] = None
    while True:
        failures: int = get_attempts(cfg.sqlite_path, doc_id)
        if failures >= cfg.max_retries:
            log.warning("Document %s exceeded max retries (%d)", doc_id, cfg.max_retries)
            set_status(cfg.sqlite_path, doc_id, status="failed", last_error="retry limit exceeded")
            return
        try:
            log.info("Processing document %s (attempt %d)", doc_id, failures + 1)
            set_status(cfg.sqlite_path, doc_id, "processing", model=cfg.openai_model)
            client = get_mongo_client(cfg.mongo_uri)
            db = client[cfg.mongo_db]
            text: Optional[str] = get_raw_doc(db, doc_id)
            if not text:
                raise RuntimeError("raw document is missing from mongo")
            log.debug("Calling OpenAI API for document %s", doc_id)
            summary, prompt_tokens, completion_tokens = summarize_text(
                api_key = cfg.openai_api_key,
                model=cfg.openai_model,
                text=text,
                timeout=cfg.openai_timeout
            )
            log.info("OpenAI API call succeeded for document %s", doc_id)
            put_summary(db, doc_id, summary, prompt_tokens, completion_tokens)
            set_status(
                cfg.sqlite_path,
                doc_id,
                "done",
                model=cfg.openai_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, 
                last_error=None,
            )
            return 
        except Exception as e:
            log.error("Error processing document %s: %s", doc_id, str(e), exc_info=True)
            new_failures: int = record_failure(
                cfg.sqlite_path,
                doc_id,
                err=str(e),
                status="pending",
            )
            if new_failures >= cfg.max_retries:
                set_status(
                    cfg.sqlite_path,
                    doc_id,
                    "failed",
                    last_error=None
                )
                return                 
            time.sleep(cfg.retry_backoff*new_failures)

    