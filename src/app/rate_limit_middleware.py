from __future__ import annotations

import math
from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request
# from starlette.responses import JSONResponse, Response
from fastapi.responses import JSONResponse

from .rate_limit import TokenBucketLimiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        limiter: TokenBucketLimiter,
        upload_limit_per_min: int,
        summary_limit_per_min: int,
        header_user_id: str = "X-User-Id",
    ) -> None:
        super().__init__(app)
        self.limiter = limiter
        self.upload_limit_per_min = upload_limit_per_min
        self.summary_limit_per_min = summary_limit_per_min
        self.header_user_id = header_user_id

    async def dispatch(self, request: Request, call_next) -> Response:
        # Allow health endpoints etc.
        if request.url.path in ("/healthz", "/readyz", "/openapi.json") or request.url.path.startswith("/docs"):
            pass  # continue; we'll still rate-limit below for /docs routes if you want
        # Decide endpoint key + limits
        path = request.url.path
        method = request.method.upper()

        # Only rate limit these:
        is_upload = (method == "POST" and path == "/docs")
        is_summary = (method == "GET" and path.endswith("/summary"))

        if not (is_upload or is_summary):
            return await call_next(request)   # IMPORTANT RETURN

        user_id = request.headers.get(self.header_user_id, "").strip()
        if not user_id:
            # No auth: either treat as anonymous bucket or reject.
            user_id = "anonymous"

        endpoint_key = "upload" if is_upload else "summary"
        refill = self.upload_limit_per_min if is_upload else self.summary_limit_per_min

        allowed, retry_after = self.limiter.check(
            user_id=user_id,
            endpoint=endpoint_key,
            capacity=refill,            # bucket size == per-minute allowance
            refill_per_min=refill,
            amount=1.0,
        )

        if not allowed:
            secs = max(1, int(math.ceil(retry_after)))
            return JSONResponse(          # IMPORTANT RETURN
                status_code=429,
                content={
                    "detail": "rate_limited",
                    "user_id": user_id,
                    "endpoint": endpoint_key,
                    "retry_after_seconds": secs,
                },
                headers={"Retry-After": str(secs)},
            )

        return await call_next(request)    # IMPORTANT RETURN
