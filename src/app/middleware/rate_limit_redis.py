from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math 
import time 

import redis 


#Lua scripting is used to ensures rate limting is atomic in Lua.

#Lua Script: refill + consume automatically 
#Redis key holds a hash: Every bucket is stored as a Redis hash
#   tokens: current tokens(float)
#   ts: last_refill timestamp(float)

#Args:
#ARGV[1]= now 
#ARGV[2]= capacity
#ARGV[3] refill_per_sec
#ARGV[4]= amount 

#returns array: [allowed(0/1), retry_after_seconds]

#HMGET reads the fields from redis hash

TOKEN_BUCKET_LUA = r"""
local key = KEYS[1]
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local refill_per_sec = tonumber(ARGV[3])
local amount = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil or ts == nil then 
    tokens = capacity 
    ts = now 
end 

local elapsed = now - ts
if elapsed > 0 then 
    local added = elapsed * refill_per_sec
    tokens = math.min(capacity, tokens+added)
    ts = now 
end 

if tokens >= amount then 
    tokens = tokens - amount 
    redis.call("HMSET", key, "tokens", tokens, "ts", ts)
    -- keep the key around a bit so older buckets disappear
    redis.call("EXPIRE", key, 3600)
    return {1, 0}
else
    redis.call("HMSET", key, "tokens", tokens, "ts", ts)
    redis.call("EXPIRE", key, 3600)
    if refill_per_sec <=0 then 
        return {0, 60}
    end 
    local needed = amount - tokens 
    local retry_after = needed/refill_per_sec 
    return {0, retry_after}
end 
"""

# So in the lua script we are writing the logic for allowing the tokens to be used if there are sufficicent amount of tokens. 
#If not we are also write the retry logic. 
class RedisTokenBucketLimiter:
    def __init__(
        self, 
        redis_client: redis.Redis, 
        rate_key_prefix: str = "precisbox:rl:",
        user_cache_prefix: str = "precisbox:user:",
        enforce_known_user: bool = False, 
    ) -> None:
        self.r = redis_client 
        self.rate_key_prefix= rate_key_prefix 
        self.user_cache_prefix = user_cache_prefix
        self.enforce_known_user = enforce_known_user
        self._lua = self.r.register_script(TOKEN_BUCKET_LUA)

    def resolve_user_id(self, header_value: str) -> Optional[str]:
        """
        Look up for the user in cache 
        if not found:
            if enforce_known = True --> return None the user should be rejected
            else return header value (use raw header as user_id)
        """
        hv = (header_value or "").strip()
        if not hv:
            return None 
        cached = self.r.get(f"{self.user_cache_prefix}{hv}")
        if cached:
            return str(cached)
        if self.enforce_known_user:
            return None 
        return hv 

    def check( # takes the user_id, endpoints and the rate limit settings
        self, 
        user_id: str, 
        end_point: str, 
        capacity: int, 
        refill_per_min: int, 
        amount: float = 1.0,
    ) -> Tuple[bool, float]:
        #Returns (allowed, retry_after_seconds)
        now = time.monotonic()
        refill_per_sec = float(refill_per_min)/60.0
        key = f"{self.rate_key_prefix}{user_id}:{end_point}" #build the redis bucket key here
        res = self._lua(
            keys=[key],
            args=[now, float(capacity), float(refill_per_sec), float(amount)],
        )
        allowed = bool(int(res[0]))
        retry_after = float(res[1])
        return allowed, retry_after