from __future__ import annotations

import time 
import threading
from dataclasses import dataclass 
from typing import Any, Dict, Tuple, List, Optional 
import math
import redis


@dataclass
class TokenBucket:
    capacity: float # the bucket capacity
    refill_per_sec: float #refill rate
    tokens: float #tokens available at point in time. 
    last_refill: float #time stamp when we last updated tokens 
    def refill(self, now: float) -> None:
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return 
        self.tokens = min(self.capacity, self.tokens+ elapsed * self.refill_per_sec)
        self.last_refill = now
    
    def try_consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        self.refill(now)
        if self.tokens >= amount:
            self.tokens -= amount 
            return True, 0.0 
        needed = amount - self.tokens 
        if self.refill_per_sec <= 0.0:
            return False, 60.0 
        retry_after = needed/self.refill_per_sec
        return False, retry_after

class TokenBucketLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str], TokenBucket] = {}
    
    def check(
        self,
        user_id: str,
        endpoint: str,
        capacity: int,
        refill_per_min: int,
        amount: float = 1.0,
    ) -> tuple[bool, float]:
        now = time.monotonic()
        refill_per_sec = refill_per_min / 60.0
        k = (user_id, endpoint)

        with self._lock:
            b = self._buckets.get(k)
            if b is None:
                b = TokenBucket(
                    capacity=float(capacity),
                    refill_per_sec=float(refill_per_sec),
                    tokens=float(capacity),     # start full
                    last_refill=now,
                )
                self._buckets[k] = b

            allowed, retry_after = b.try_consume(amount)  # refill should happen inside
            return allowed, retry_after

