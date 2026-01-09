from __future__ import annotations 

from typing import Dict, List, Any, Optional, Tuple 
from dataclasses import dataclass 

import chromadb

from ...logging_config import get_logger 

log = get_logger("precisbox.services.rag.vector_store")

@dataclass 
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str

class VectorStore: 
    def __init__(
        self, 
        collection_name : str = "documents", 
        persist_directory: Optional[str] = None,
    ):
        if persist_directory:
            #persistent storage survives restarts
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            #In-memory, useful for testing
            self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(collection_name)
            log.info("Loaded existing collection: %s", collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document Embeddings for RAG"}
            )
            log.info("Created new collection: %s", collection_name)

    def add_chunks(
        self, 
        chunks:List[tuple[str, List[float], Dict[str, Any]]]
    ) -> None: 
        """ chunk is a list (text, embeddings, metadata)"""
        if not chunks:
            return 
        texts = [chunk[0] for chunk in chunks]
        embeddings = [chunk[1] for chunk in chunks]
        metadatas = [chunk[2] for chunk in chunks ]

        ids = [
            f"{metadata.get('doc_id')}_{metadata.get('chunk_index', i)}"
            for i, metadata in enumerate(metadatas)
        ]
        self.collection.add(
            embeddings=embeddings, 
            documents= texts, 
            metadatas=metadatas, 
            ids=ids
        )
        log.info("Added chunks to vector store %s", len(chunks))

    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5, 
        filter_metadata : Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k, 
            where=filter_metadata #to filter by metadata if provided
        )
        search_results: List[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for i, (text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                score = 1.0/(1.0+distance) 
                search_results.append(SearchResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                    chunk_id=results["ids"][0][i] if results["ids"] else f"chunk_{i}",
                ))
        return search_results

    def delete_document(self, doc_id: str) -> None: 
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                log.info("Deleted %d chunks for doc_id=%s", len(results["ids"]), doc_id)
        except Exception as e:
                log.warning("Failed to delete chunks for doc_id=%s: %s", doc_id, e)
