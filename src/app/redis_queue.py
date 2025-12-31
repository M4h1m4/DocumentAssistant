# REDIS_URL=redis://localhost:6379/0  -> 0 is the Database number. Redis has logical number from 0-15
# REDIS_STREAM=precisbox:jobs --> is the redis stream key that holds the job. Basically it is the queue 
# REDIS_GROUP=precisbox-workers --> consumer groups. let's one consumer per group take up the job. Tracks pending jobs using PEL - Penidng Entries list
# REDIS_CONSUMER=worker-1 --> Name or identity of the worker inside the group

from __future__ import annotations
import os 
from typing import Optional, List, Tuple

import redis 

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM = os.getenv("REDIS_STREAM", "precisbox:jobs")
GROUP = os.getenv("REDIS_GROUP", "precisbox-workers")

def get_redis() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)

