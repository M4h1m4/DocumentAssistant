import asyncio
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Any, Dict, List, Tuple, Optional 
import httpx 


@dataclass
class JobResult:
    file: str
    doc_id: Optional[str]
    upload_s: float 
    summary_s: float 
    final_status: str
    http_status: str 
    error: Optional[str]

async def upload_doc(client: httpx.AsyncClient, base_url: str, path: Path) -> Tuple[Optional[str], float, Optional[str]]:
    t0= time.perf_counter()
    try: 
        files = {"file": (path.name, path.read_bytes(), "text/plain")}
        #The key "file" must match what your FastAPI endpoint expects: file: UploadFile = File(...)
        #curl -F "file=@doc1.txt" http://127.0.0.1:8000/docs
        r = await client.post(f"{base_url}/docs", files=files, timeout=30.0)
        #Sends the request to http://.../docs
        dt = time.perf_counter() - t0 # time taken from start to response returned
        r.raise_for_status() #
        data: Dict[str, Any] = r.json()
        return str(data["id"]), dt, None 
    except:
        dt = time.perf_counter() - t0
        return None, dt, str(e)

async def poll_summary(
    client: httpx.AsyncClient,
    base_url: str,
    doc_id: str,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.25,
) -> Tuple[int, float, str, Dict[str, Any]]:
    """
    Poll GET /docs/{id}/summary until it is ready (200), failed (409), or timeout.

    Returns:
      (http_status, seconds_waited, final_state, payload_json)

      final_state is one of: "done" | "failed" | "timeout"
      payload_json is the response body if JSON; otherwise {}.
    """
    t0: float = time.perf_counter()
    deadline: float = t0 + timeout_s  

    last_status: int = 0
    last_payload: Dict[str, Any] = {}

    while time.perf_counter() <= deadline:  
        r: httpx.Response = await client.get(
            f"{base_url}/docs/{doc_id}/summary",
            timeout=30.0,
        )
        last_status = r.status_code

        # Try to parse JSON (even for 202/409)
        try:
            last_payload = r.json()
        except Exception:
            last_payload = {}

        if r.status_code == 200:
            return 200, time.perf_counter() - t0, "done", last_payload

        if r.status_code == 409:
            # Your API uses 409 when status=failed
            return 409, time.perf_counter() - t0, "failed", last_payload

        if r.status_code != 202:
            # Unexpected response
            return r.status_code, time.perf_counter() - t0, "failed", last_payload

        # 202 -> not ready yet
        await asyncio.sleep(poll_interval_s)

    # Timeout
    return last_status or 202, time.perf_counter() - t0, "timeout", last_payload


async def run_once(base_url: str, paths: List[Path]) -> List[JobResult]:
    async with httpx.AsyncClient() as client:
        upload_tasks = [upload_doc(client, base_url, p) for p in paths]
        upload_results = await asyncio.gather(*upload_tasks)

        poll_coros: List[Optional[asyncio.Future]] = []
        for (doc_id, _upload_s, _err) in upload_results:
            if doc_id is not None:
                # poll_summary returns: (final_status, http_status, seconds_waited, error)
                poll_coros.append(poll_summary(client, base_url, doc_id))
            else:
                poll_coros.append(None)

        # 3) Gather poll results (only non-None)
        to_gather = [c for c in poll_coros if c is not None]
        polled: List[Tuple[str, int, float, Optional[str]]] = await asyncio.gather(*to_gather) if to_gather else []
        
        final: List[JobResult] = []
        polled_i = 0
        for i, p in enumerate(paths):
            doc_id, upload_s, up_err = upload_results[i]
            if doc_id is None:
                final.append(
                    JobResult(
                        file=p.name,
                        doc_id=None,
                        upload_s=upload_s,
                        summary_s=0.0,
                        final_status="upload_failed",
                        http_status=0,
                        error=up_err,
                    )
                )
            else:
                status, http_status, summary_s, err = polled[polled_i]
                polled_i += 1
                final.append(
                    JobResult(
                        file=p.name,
                        doc_id=doc_id,
                        upload_s=upload_s,
                        summary_s=summary_s,
                        final_status=status,
                        http_status=http_status,
                        error=err,
                    )
                )
        return final 

def _as_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def print_report(results: List[JobResult]) -> None:
    print("\n=== Mini Load Test Report (5 parallel docs) ===")
    for r in results:
        upload_s = _as_float(getattr(r, "upload_s", None))
        summary_s = _as_float(getattr(r, "summary_s", None))

        print(
            f"- {r.file:10s} id={r.doc_id} upload={upload_s:.3f}s "
            f"summary_wait={summary_s:.3f}s http={r.http_status} status={r.final_status} "
            f"{'' if not r.error else 'error=' + str(r.error)}"
        )

    ok = [x for x in results if x.http_status == 200]
    if ok:
        avg_upload = sum(x.upload_s for x in ok) / len(ok)
        avg_summary = sum(x.summary_s for x in ok) / len(ok)
        print(f"\nDone count: {len(ok)}/{len(results)}")
        print(f"Avg upload time: {avg_upload:.3f}s")
        print(f"Avg summary wait: {avg_summary:.3f}s")
    else:
        print("\nNo successful summaries (200) to compute averages.")

if __name__ == "__main__":
    BASE_URL = "http://127.0.0.1:8000"

    paths = [Path("doc1.txt"), Path("doc2.txt"), Path("doc3.md"), Path("doc4.txt"), Path("doc5.txt")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}. Create them first.")

    results = asyncio.run(run_once(BASE_URL, paths))
    print_report(results)

