from __future__ import annotations
from pymongo import MongoClient
from typing import Optional, Dict, Any

def get_mongo_client(mongo_uri: str) -> MongoClient:
    return MongoClient(mongo_uri)

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

def put_summary(db, doc_id: str, summary: str, summary_meta: Dict[str, Any]) -> None:
    db.docs.update_one(
        {"_id": doc_id},
        {"$set": {"summary": summary, "summary_meta": summary_meta}},
        upsert=False, 
    )

def get_summary(db, doc_id: str) -> Optional[Dict[str, Any]]:
    return db.docs.find_one(
        {"_id":doc_id}, 
        {"summary": 1, "summary_meta": 1})