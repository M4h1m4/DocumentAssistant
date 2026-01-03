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
    #decode=True returns str instead of bytes. 

def ensure_group(r: redis.Redis) -> None: 
    # to create consumer group if it does not exist 
    try:
        r.xgroup_create(name=STREAM, groupname=GROUP, id="0", mkstream=True) # this creates a consumer group over the stream
        #mkstream creates a key if it does not exist
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
            #BUSYGROUP means the group already exists

def enqueue_job(r: redis.Redis, doc_id: str) -> str: #this is the produces and put the doc_id on stream and returns the stream message_id
    msg_id = r.xadd(STREAM, {"doc_id": doc_id})
    return msg_id

def read_job( 
    r:redis.Redis,
    consumer: str, #Worker name eg: worker_1, wokrer_2
    block_ms: int = 5000, #how long to wait for new_messages before returning
    count: int =1, #Number of messages to fetch
    ) -> Optional[Tuple[str, str]]:
        resp = r.xreadgroup(
            groupname=GROUP, 
            consumername=consumer, 
            streams={STREAM: ">"}, # this says that give me messages that have not been delivered to any consumer in this group
            count=count, 
            block=block_ms,
        )
        if not resp:
            return None
        
        #resp format: [(stream_name, [(msg_id, {field: val})])]
        _stream, messages = resp[0]
        msg_id, fields = messages[0]
        doc_id = fields.get("doc_id")
        if not doc_id:
            r.xack(STREAM, GROUP, msg_id)
            return None 
        return msg_id, doc_id

def ack_job(r: redis.Redis, msg_id: str)-> None:
    r.xack(STREAM, GROUP, msg_id) 




