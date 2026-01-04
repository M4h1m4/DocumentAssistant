from __future__ import annotations
from pymongo import MongoClient
from typing import Optional, Dict, Any
from pymongo.database import Database


_client: Optional[MongoClient] = None
_default_uri: Optional[str] = None
_max_pool_size: int = 10
_min_pool_size: int = 1

def get_mongo_client(
    mongo_uri: Optional[str]=None,
    max_pool_size: Optional[int] = None, 
    min_pool_size: Optional[int] = None,
) -> MongoClient:
    global _client, _default_uri, _max_pool_size, _min_pool_size
    
    # If no URI provided and we have a client, return existing client
    if mongo_uri is None and _client is not None:
        return _client
    
    # If URI changed or client doesn't exist, create new client
    if _client is None or (mongo_uri is not None and mongo_uri != _default_uri):
        if mongo_uri is None:
            raise ValueError("mongo_uri must be provided on first call")
        _default_uri = mongo_uri
        if max_pool_size is not None:
            _max_pool_size = max_pool_size
        if min_pool_size is not None:
            _min_pool_size = min_pool_size
        _client = MongoClient(
            mongo_uri,
            maxPoolSize=_max_pool_size,
            minPoolSize=_min_pool_size,
        )
    
    return _client

def init_mongo_client(mongo_uri: str, max_pool_size: int = 10, min_pool_size: int = 1) -> None:
    get_mongo_client(mongo_uri, max_pool_size=max_pool_size, min_pool_size=min_pool_size)

def put_raw_doc(db, doc_id: str, text: str) -> None:
    """
    To store the raw content into mongo
    upsert is set to true so that the reuploads can overwrite the content of the doc
    """
    db.docs.update_one(
        {"_id": doc_id},
        {"$set": {"text": text}},
        upsert=True,
    )

def put_summary(db, doc_id: str, summary: str, prompt_tokens: int, completion_tokens: int) -> None:
    db.docs.update_one(
        {"_id": doc_id},
        {"$set": {"summary": summary, "summary_meta":{"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}},
        upsert=False, 
    )

def get_summary(db, doc_id: str) -> Optional[Dict[str, Any]]:
    return db.docs.find_one(
        {"_id":doc_id}, 
        {"summary": 1, "summary_meta": 1}
        )

def get_raw_doc(db, doc_id: str) -> Optional[str]:
    raw = db.docs.find_one({"_id": doc_id}, {"text":1})
    return None if raw is None else raw.get("text")

def write_raw_doc(mongo_uri: str, mongo_db: str, doc_id: str, text: str) -> None:
    client = get_mongo_client(mongo_uri)
    db = client[mongo_db]
    put_raw_doc(db, doc_id, text)

def read_raw_doc(mongo_uri: str, mongo_db:str, doc_id: str) -> Optional[str]:
    client = get_mongo_client(mongo_uri)
    db = client[mongo_db]
    return get_raw_doc(db, doc_id)

def write_summary(mongo_uri: str, mongo_db: str, doc_id: str, summary:str, prompt_tokens: int, completion_tokens:str) -> None:
    client = get_mongo_client(mongo_uri)
    db = client[mongo_db]
    put_summary(db, doc_id, summary, prompt_tokens, completion_tokens)


def read_summary(mongo_uri: str, mongo_db: str, doc_id: str) -> Optional[Dict[str, Any]]:
    client = get_mongo_client(mongo_uri)
    db = client[mongo_db]
    return get_summary(db, doc_id)
