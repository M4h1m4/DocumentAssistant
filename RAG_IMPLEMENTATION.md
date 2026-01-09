# RAG Implementation Guide for PrecisBox

This guide explains how to extend PrecisBox with RAG (Retrieval-Augmented Generation) to enable document querying and chatbot functionality.

## Overview

### What is RAG?
RAG (Retrieval-Augmented Generation) combines:
1. **Retrieval**: Finding relevant document chunks from your knowledge base using semantic search
2. **Augmentation**: Injecting retrieved context into the LLM prompt
3. **Generation**: LLM generates response using both its training and retrieved context

### What We're Building
- **Document Indexing**: Convert uploaded documents into searchable vector embeddings
- **Semantic Search**: Query documents by meaning, not just keywords
- **Chatbot Interface**: Natural language queries that return answers synthesized from your documents
- **Multi-Document Support**: Query across all uploaded documents simultaneously

---

## Architecture

```
User Query → Generate Query Embedding → Vector Search → Retrieve Relevant Chunks
                                                              ↓
                        Combine Chunks + Query → LLM → Response + Sources
```

### Current State vs RAG State

**Current:**
- Documents uploaded → Stored → Summarized
- No querying capability

**With RAG:**
- Documents uploaded → Indexed (chunked + embedded) → Stored in vector DB
- User queries → Vector search → Retrieve chunks → LLM generates answer from chunks

---

## Step 1: Add Dependencies

**File: `pyproject.toml`**

Add to dependencies:
```toml
dependencies = [
    # ... existing dependencies ...
    
    # Vector Database
    "chromadb>=0.4.0",  # Simple, Python-native vector DB (good for MVP)
    
    # Alternative options:
    # "qdrant-client>=1.7.0",  # Fast, efficient (if you prefer Qdrant)
    # "pinecone-client>=3.0.0",  # Managed service (if you prefer Pinecone)
]
```

**Install:**
```bash
uv sync
```

**Why ChromaDB?**
- Simple to use, Python-native
- Good for MVP and development
- Can scale to production with some configuration
- No external service required (runs locally)

**Alternative Vector DBs:**
- **Qdrant**: Fast, self-hosted option
- **Pinecone**: Managed service (good for production)
- **Weaviate**: Open source with GraphQL
- **pgvector**: If you want to use PostgreSQL

---

## Step 2: Create Embeddings Service

**File: `src/app/services/embeddings.py`**

### What are Embeddings?

**Embeddings** are numerical representations of text that capture semantic meaning. Think of them as:
- **Text → Numbers**: Convert words/sentences into arrays of numbers (vectors)
- **Semantic Meaning**: Similar meaning = similar numbers
- **High-Dimensional**: Usually 1536 numbers (dimensions) for OpenAI embeddings

**Example:**
```
"Hello world" → [0.123, -0.456, 0.789, ...] (1536 numbers)
"Hi there"    → [0.120, -0.451, 0.785, ...] (similar numbers because similar meaning)
"Quantum physics" → [0.987, 0.234, -0.123, ...] (very different numbers - different topic)
```

**Why we need embeddings:**
- Enable **semantic search**: Find documents by meaning, not just keywords
- Compare text mathematically: Calculate similarity between texts
- Search in vector space: Find similar documents instantly

```python
from __future__ import annotations

from typing import List, Optional
from openai import OpenAI

from ..config import settings
from ..logging_config import get_logger

log = get_logger("precisbox.services.embeddings")


class EmbeddingService:
    """
    Service for generating document embeddings using OpenAI.
    
    WHY: Embeddings convert text into numerical vectors that capture semantic meaning.
    Similar text will have similar vectors, enabling semantic search.
    
    WHAT ARE EMBEDDINGS?
    - Numerical representation of text (array of numbers)
    - Captures semantic meaning (similar meaning = similar numbers)
    - High-dimensional vectors (e.g., 1536 dimensions for OpenAI)
    - Enables semantic search and similarity calculations
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",  # or "text-embedding-3-large"
        timeout: float = 30.0,
    ):
        """
        Initialize embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name (text-embedding-3-small is cheaper, -large is better)
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (max ~8000 tokens for most models)
            
        Returns:
            List of floats representing the embedding vector (1536 dimensions for 3-small)
        """
        try:
            # WHY: Call OpenAI Embeddings API
            # - model: Which embedding model to use (e.g., "text-embedding-3-small")
            # - input: The text string to embed (single text, not a list)
            # - OpenAI's neural network converts text → numbers
            response = self.client.embeddings.create(
                model=self.model,
                input=text,  # Single string, not a list
            )
            
            # WHY: Extract embedding from response
            # - response.data is a list (even though we sent one text)
            # - response.data[0] is the first (and only) embedding
            # - .embedding is the actual vector (list of floats)
            # - Example: [0.123, -0.456, 0.789, ..., 0.234] (1536 numbers)
            return response.data[0].embedding
        except Exception as e:
            log.exception("Failed to generate embedding: %s", e)
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for MULTIPLE texts in batches (more efficient).
        
        WHAT IT DOES:
        - Takes a list of text strings (e.g., ["Chunk 1 text...", "Chunk 2 text...", ...])
        - Sends multiple texts to OpenAI in a single API call (batch)
        - Gets back embeddings for all texts at once
        - Processes large lists by splitting into batches
        
        WHY BATCH PROCESSING:
        - **Much faster**: One API call for 100 texts vs 100 separate calls
        - **More cost-effective**: Batch requests are cheaper per text
        - **Efficient**: Processes many document chunks at once during indexing
        
        EXAMPLE:
        Input:  ["Machine learning is...", "Deep learning involves...", "Neural networks..."]
        Output: [
            [0.123, -0.456, ...],  # Embedding for text 1
            [0.234, -0.567, ...],  # Embedding for text 2
            [0.345, -0.678, ...],  # Embedding for text 3
        ]
        
        PROCESS:
        1. Split texts into batches (e.g., 100 texts per batch)
        2. For each batch:
           - Send all texts in one API call: client.embeddings.create(model=..., input=[text1, text2, ...])
           - OpenAI processes all texts in parallel
           - Returns embeddings for all texts in the batch
        3. Combine all batch results into one list
        
        WHY BATCH SIZE:
        - OpenAI allows up to 2048 texts per request
        - Using 100-200 is a good balance (not too large, efficient enough)
        - Smaller batches = faster individual responses
        - Larger batches = fewer API calls but slower per batch
        
        WHEN TO USE:
        - **embed_text()**: Single queries from users (one question)
        - **embed_batch()**: Indexing documents (many chunks at once)
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch (OpenAI allows up to 2048)
            
        Returns:
            List of embedding vectors (one per input text)
            Example: [[0.123, ...], [0.234, ...], [0.345, ...]]
            - Same order as input texts
            - Each inner list is an embedding vector (1536 numbers)
        """
        # WHY: Initialize empty list to collect all embeddings
        embeddings = []
        
        # WHY: Process texts in batches (e.g., 100 at a time)
        # - Loop through texts, taking batch_size at a time
        # - Example: If we have 250 texts and batch_size=100:
        #   - Batch 1: texts[0:100]   (0-99)
        #   - Batch 2: texts[100:200] (100-199)
        #   - Batch 3: texts[200:250] (200-249)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]  # Get next batch of texts
            
            try:
                # WHY: Send entire batch in ONE API call
                # - OpenAI processes all texts in the batch together
                # - Much faster than calling embed_text() for each text individually
                # - More cost-effective (batch pricing)
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,  # List of texts, not single text
                )
                
                # WHY: Extract embeddings from response
                # - response.data is a list of embedding objects
                # - Each object has an .embedding attribute (the vector)
                # - Extract all embeddings from this batch
                batch_embeddings = [item.embedding for item in response.data]
                
                # WHY: Add batch embeddings to our collection
                # - We're building up the full list as we process batches
                # - Maintains order (batch 1, batch 2, batch 3, etc.)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                log.exception("Failed to generate batch embeddings: %s", e)
                raise
        
        # WHY: Return all embeddings
        # - Same order as input texts
        # - Each embedding is a list of floats (vector)
        return embeddings
```

---

## Step 3: Create Chunking Service

**File: `src/app/services/chunking.py`**

```python
from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

from ..logging_config import get_logger

log = get_logger("precisbox.services.chunking")


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata.
    
    WHY: Documents are too large to embed whole. We chunk them into smaller pieces
    (e.g., 1000 characters) that can be embedded and searched individually.
    """
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]  # doc_id, page, filename, etc.


class FixedSizeChunker:
    """
    Chunks text into fixed-size pieces with overlap.
    
    WHY: Fixed-size chunks are simple and effective. Overlap ensures we don't lose
    context at chunk boundaries (e.g., if a sentence spans two chunks).
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Text separator to use for splitting (paragraphs work well)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to all chunks
            
        Returns:
            List of TextChunk objects
        """
        chunks: List[TextChunk] = []
        
        # Split by paragraphs first (respects natural boundaries)
        paragraphs = text.split(self.separator)
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = self.separator.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata.copy(),
                ))
                
                # Start new chunk with overlap (keep last N characters)
                overlap_text = chunk_text[-self.chunk_overlap:]
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_size = len(self.separator.join(current_chunk))
                start_char = end_char - self.chunk_overlap
                chunk_index += 1
            else:
                current_chunk.append(para)
                current_size += para_size + len(self.separator)
        
        # Add final chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata.copy(),
            ))
        
        return chunks
```

---

## Step 4: Create Vector Store Service

**File: `src/app/services/vector_store.py`**

```python
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from ..logging_config import get_logger

log = get_logger("precisbox.services.vector_store")


@dataclass
class SearchResult:
    """
    Represents a search result with similarity score.
    
    WHY: We need to know not just which chunks match, but how well they match.
    Higher scores mean more relevant content.
    """
    text: str
    score: float  # Similarity score (0-1, higher is better)
    metadata: Dict[str, Any]
    chunk_id: str


class VectorStore:
    """
    Vector store for storing and retrieving document embeddings.
    
    WHY: Vector stores enable fast similarity search. We can find chunks similar
    to a query in milliseconds, even with millions of chunks.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection (like a database table)
            persist_directory: Directory to persist data (None = in-memory)
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        # Initialize ChromaDB client
        if persist_directory:
            # Persistent storage (survives restarts)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # In-memory (for testing)
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            log.info("Loaded existing collection: %s", collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )
            log.info("Created new collection: %s", collection_name)
    
    def add_chunks(
        self,
        chunks: List[tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        """
        Add text chunks with embeddings to vector store.
        
        Args:
            chunks: List of (text, embedding, metadata) tuples
        """
        if not chunks:
            return
        
        texts = [chunk[0] for chunk in chunks]
        embeddings = [chunk[1] for chunk in chunks]
        metadatas = [chunk[2] for chunk in chunks]
        ids = [
            f"{metadata.get('doc_id')}_{metadata.get('chunk_index', i)}"
            for i, metadata in enumerate(metadatas)
        ]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        log.info("Added %d chunks to vector store", len(chunks))
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"doc_id": "abc123"})
            
        Returns:
            List of SearchResult objects sorted by relevance (best first)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,  # Filter by metadata if provided
        )
        
        search_results: List[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for i, (text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score (lower distance = higher similarity)
                # ChromaDB returns distances, not similarities
                score = 1.0 / (1.0 + distance)
                search_results.append(SearchResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                    chunk_id=results["ids"][0][i] if results["ids"] else f"chunk_{i}",
                ))
        
        return search_results
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document ID to delete
        """
        # ChromaDB doesn't have a direct delete by filter, so we need to:
        # 1. Query to get all chunk IDs for this doc
        # 2. Delete those IDs
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
```

---

## Step 5: Create RAG Service

**File: `src/app/services/rag.py`**

```python
from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple

from .embeddings import EmbeddingService
from .vector_store import VectorStore, SearchResult
from .chunking import FixedSizeChunker, TextChunk
from .summarize import summarize_text

from ..config import settings
from ..logging_config import get_logger

log = get_logger("precisbox.services.rag")


class RAGService:
    """
    Service for Retrieval-Augmented Generation.
    
    WHY: This service coordinates the RAG pipeline:
    1. Index documents (chunk + embed + store)
    2. Query documents (search + retrieve + generate)
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        openai_api_key: str,
        openai_model: str,
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for storing/retrieving chunks
            openai_api_key: OpenAI API key for LLM
            openai_model: OpenAI model for generation (e.g., "gpt-4o-mini")
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
    
    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a document by chunking, embedding, and storing in vector DB.
        
        Args:
            doc_id: Document ID
            text: Document text content
            metadata: Additional metadata (filename, mime_type, etc.)
        """
        # Prepare metadata
        doc_metadata = {"doc_id": doc_id}
        if metadata:
            doc_metadata.update(metadata)
        
        # Chunk the document
        chunks = self.chunker.chunk(text, doc_metadata)
        log.info("Chunked document doc_id=%s into %d chunks", doc_id, len(chunks))
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(chunk_texts)
        
        # Prepare chunks for storage
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
        
        # Store in vector database
        self.vector_store.add_chunks(chunks_with_embeddings)
        log.info("Indexed document doc_id=%s with %d chunks", doc_id, len(chunks))
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> Tuple[str, List[SearchResult]]:
        """
        Query the knowledge base and generate a response using RAG.
        
        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            doc_filter: Optional document ID filter (search only this document)
            
        Returns:
            Tuple of (generated response, retrieved chunks)
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query_text)
        
        # Search for relevant chunks
        filter_metadata = {"doc_id": doc_filter} if doc_filter else None
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        if not search_results:
            return "No relevant documents found.", []
        
        # Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(search_results, 1):
            doc_id = result.metadata.get("doc_id", "unknown")
            context_parts.append(
                f"[Context {i} from document {doc_id}]:\n{result.text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Generate response using RAG prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {query_text}

Answer the question based on the context above. If the context doesn't contain enough information to answer the question, say so. Cite which context sources you used (e.g., "According to Context 1...")."""

        summary, prompt_tokens, completion_tokens = summarize_text(
            api_key=self.openai_api_key,
            model=self.openai_model,
            text=prompt,
            timeout=120.0,
        )
        
        return summary, search_results
```

---

## Step 6: Update Configuration

**File: `src/app/config.py`**

Add RAG settings:

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # RAG Configuration
    enable_rag: bool = Field(default=False, description="Enable RAG capabilities")
    vector_store_path: str = Field(default="./vector_store", description="Vector store directory")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    chunk_size: int = Field(default=1000, ge=100, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    rag_top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve for RAG")
    
    # ... rest of fields ...
```

---

## Step 7: Update Document Upload to Index Documents

**File: `src/app/api.py`**

Add RAG indexing after document storage:

```python
from .services.rag import RAGService
from .services.embeddings import EmbeddingService
from .services.vector_store import VectorStore

# Initialize RAG service (do this at module level or in startup)
rag_service: Optional[RAGService] = None

def init_rag_service() -> None:
    """Initialize RAG service during startup."""
    global rag_service
    if settings.enable_rag:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        vector_store = VectorStore(
            collection_name="documents",
            persist_directory=settings.vector_store_path,
        )
        rag_service = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
```

**In `upload_doc()` function, after storing document:**

```python
# Store in MongoDB (existing code)
await anyio.to_thread.run_sync(
    lambda: write_raw_doc(settings.mongo_uri, settings.mongo_db, doc_id, text)
)

# Index document for RAG (NEW)
if settings.enable_rag and rag_service:
    try:
        await anyio.to_thread.run_sync(
            lambda: rag_service.index_document(
                doc_id=doc_id,
                text=text,
                metadata={
                    "filename": filename,
                    "mime_type": mime,
                    "size": size,
                }
            )
        )
    except Exception as e:
        log.warning("Failed to index document for RAG doc_id=%s: %s", doc_id, e)
        # Don't fail the upload if indexing fails
```

**In `main.py` startup:**

```python
@app.on_event("startup")
def _startup() -> None:
    # ... existing startup code ...
    
    # Initialize RAG service
    if settings.enable_rag:
        init_rag_service()
        log.info("RAG service initialized")
```

---

## Step 8: Add Query Endpoint

**File: `src/app/api.py`**

Add new query endpoint:

```python
@router.post("/query", status_code=200)
async def query_documents(
    query: str = Query(..., description="Query string"),
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    top_k: int = Query(5, ge=1, le=20, description="Number of chunks to retrieve"),
) -> Dict[str, Any]:
    """
    Query the knowledge base using RAG.
    
    WHY: This endpoint enables users to ask questions about their documents
    and get answers synthesized from relevant document chunks.
    """
    if not settings.enable_rag:
        raise HTTPException(status_code=503, detail="RAG is not enabled")
    
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Query using RAG
    response, retrieved_chunks = await anyio.to_thread.run_sync(
        lambda: rag_service.query(
            query_text=query,
            top_k=top_k,
            doc_filter=doc_id,
        )
    )
    
    return {
        "query": query,
        "answer": response,
        "sources": [
            {
                "doc_id": chunk.metadata.get("doc_id"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "score": chunk.score,
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            }
            for chunk in retrieved_chunks
        ],
    }
```

---

## Step 9: Update Main Startup

**File: `src/app/main.py`**

```python
from .api import init_rag_service

@app.on_event("startup")
def _startup() -> None:
    # ... existing startup code ...
    
    # Initialize RAG service
    if settings.enable_rag:
        init_rag_service()
        log.info("RAG service initialized (vector_store=%s)", settings.vector_store_path)
```

---

## Step 10: Testing

### 1. Upload a document:
```bash
curl -H "X-User-Id: user1" -F "file=@document.txt" http://localhost:8000/docs
```

### 2. Wait for indexing (happens automatically after upload)

### 3. Query documents:
```bash
curl -H "X-User-Id: user1" \
  "http://localhost:8000/docs/query?query=What%20is%20the%20main%20topic?&top_k=5"
```

### 4. Query specific document:
```bash
curl -H "X-User-Id: user1" \
  "http://localhost:8000/docs/query?query=What%20is%20discussed?&doc_id=abc123&top_k=3"
```

---

## Configuration in .env

Add to `.env`:
```env
# RAG Configuration
ENABLE_RAG=true
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_TOP_K=5
```

---

## How RAG Works in PrecisBox

### Indexing Flow:
```
1. Document uploaded
2. Text extracted (existing flow)
3. Document chunked into ~1000 char pieces
4. Each chunk embedded (converted to vector)
5. Chunks + embeddings stored in vector DB
```

### Query Flow:
```
1. User asks question
2. Question embedded (same embedding model)
3. Vector search finds similar chunks
4. Top K chunks retrieved
5. Chunks + question sent to LLM
6. LLM generates answer from chunks
7. Response + sources returned to user
```

### Example:
```
User Query: "What were the revenue numbers in Q4?"

1. Query embedded: [0.123, -0.456, ...] (1536 dimensions)
2. Vector search finds chunks with similar vectors
3. Retrieves:
   - "Q4 revenue was $2M" (score: 0.92)
   - "Revenue increased 15% in Q4" (score: 0.88)
   - "Q4 planning session notes" (score: 0.75)
4. LLM receives chunks + question
5. LLM generates: "According to the documents, Q4 revenue was $2M, representing a 15% increase..."
6. Returns answer + source citations
```

---

## Next Steps

Once RAG is working:
1. Add conversation history (multi-turn conversations)
2. Add document deletion (remove from vector store)
3. Add hybrid search (combine semantic + keyword search)
4. Add filtering (filter by date, author, etc.)
5. Add analytics (track query patterns)

See `GEN_AI_EXTENSION_IDEAS.md` for more advanced features like agents and MCP.

