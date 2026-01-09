from __future__ import annotations 

from typing import List, Tuple, Dict, Any, Optional 

from ..embeddings.embeddings import EmbeddingService
from ..embeddings.chunking import TextChunk, FixedSizeChunker
from .vector_store import VectorStore, SearchResult 

from ..document.summarizer import summarize_text

from ...config import settings 
from ...logging_config import get_logger 

log = get_logger("precisbox.services.rag.rag")

class RAGService:
    def __init__(
        self, 
        embedding_service: EmbeddingService, 
        vector_store: VectorStore,
        openai_api_key: str, 
        openai_model: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embedding_service = embedding_service 
        self.vector_store = vector_store 
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def index_document( #chunk, embed and store in vectorDB
        self, 
        doc_id: str, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None: 
        doc_metadata = {"doc_id": doc_id}
        if metadata:
            doc_metadata.update(metadata)

        chunks = self.chunker.chunk(text, doc_metadata)
        log.info("Chunked document doc_id=%s into %d chunks", doc_id, len(chunks))

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(chunk_texts)

        chunks_with_embeddings = [
            (
                chunk.text,
                embedding,
                {
                    **chunk.metadata,
                    "chunk_index": chunk.chunk_index, 
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                } 
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.vector_store.add_chunks(chunks_with_embeddings)
        log.info("Indexed document doc_id=%s with %d chunks", doc_id, len(chunks))


    def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        doc_filter: Optional[str] = None, 
    ) -> Tuple[str, List[SearchResult]]:
        query_embedding = self.embedding_service.embed_text(query_text)
        filter_metadata = {"doc_id": doc_filter} if doc_filter else None 
        search_results = self.vector_store.search(
            query_embedding=query_embedding, 
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        if not search_results:
            return "No search results found.", []

        #Build context from chunks
        context_parts = []
        for i, result in enumerate(search_results, 1):
            doc_id = result.metadata.get("doc_id", "unknown")
            context_parts.append(
                f"[Context {i} from document {doc_id}]:\n{result.text}\n"
            )
        
        context = "\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {query_text}

Answer the question based on the context above. 
If the context doesn't contain enough information to answer the question, say so. 
Cite which context sources you used (e.g., "According to Context 1...")."""

        summary, prompt_tokens, completion_tokens = summarize_text(
            api_key=self.openai_api_key,
            model=self.openai_model,
            text=prompt,
            timeout=120.0,
        )
        
        return summary, search_results
        