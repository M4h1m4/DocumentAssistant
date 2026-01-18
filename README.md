# PrecisBox

A FastAPI-based document summarization and RAG (Retrieval-Augmented Generation) service that processes text documents, generates summaries, and enables semantic search using OpenAI and vector embeddings.

## Features

- Document upload and storage (SQLite for metadata, MongoDB for content)
- Background job processing with Redis queue
- Rate limiting with Redis token bucket algorithm
- Document summarization using OpenAI API
- **RAG (Retrieval-Augmented Generation)** - Semantic search and Q&A over documents
  - Automatic text chunking and embedding generation
  - Vector storage with ChromaDB
  - Natural language querying with LLM-generated responses
  - Support for multi-document search and single-document filtering
- RESTful API for document management
- Support for text, markdown, and PDF documents

## Requirements

- Python 3.11+
- Redis
- MongoDB
- OpenAI API key
- ChromaDB (optional, required for RAG functionality)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PrecisBox
```

2. Install dependencies:
```bash
uv sync
# or
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# IMPORTANT: Replace 'your_openai_api_key_here' with your actual OpenAI API key
```

4. Ensure Redis and MongoDB are running:
```bash
# Redis
redis-server

# MongoDB
mongod
```

## Configuration

Copy `.env.example` to `.env` and configure the following variables:

### Core Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MONGO_URI`: MongoDB connection string
- `REDIS_URL`: Redis connection URL
- `WORKERS`: Number of worker threads

### RAG Configuration (Optional)
- `ENABLE_RAG`: Enable RAG functionality (default: `false`)
- `VECTOR_STORE_PATH`: Directory for vector store (default: `./vector_store`)
- `EMBEDDING_MODEL`: OpenAI embedding model (default: `text-embedding-3-small`)
- `CHUNK_SIZE`: Text chunk size for RAG (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `200`)
- `RAG_TOP_K`: Number of chunks to retrieve for queries (default: `5`)

Other configuration options (see `.env.example`)

## Running the Application

### Using uv (Recommended)

```bash
# Install/sync dependencies
uv sync

# Run the application
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using standard Python

```bash
uvicorn src.app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

### Quick Test

1. **Health Check:**
```bash
curl http://localhost:8000/healthz
```

2. **Upload a Document:**
```bash
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: test-user" \
  -F "file=@doc1.txt"
```

3. **Get Document Summary:**
```bash
curl http://localhost:8000/docs/<doc_id>/summary \
  -H "X-User-Id: test-user"
```

4. **Search Documents with RAG** (requires `ENABLE_RAG=true`):
```bash
curl -X POST http://localhost:8000/docs/search \
  -H "Content-Type: application/json" \
  -H "X-User-Id: test-user" \
  -d '{
    "query": "What are the key points discussed in this document?",
    "doc_id": "abc123...",
    "top_k": 5
  }'
```

5. **Get Document Content:**
```bash
curl http://localhost:8000/docs/<doc_id>/content \
  -H "X-User-Id: test-user"
```

6. **Delete Document:**
```bash
curl -X DELETE http://localhost:8000/docs/<doc_id> \
  -H "X-User-Id: test-user"
```

**Note**: Rate limiting requires the `X-User-Id` header. Default limits:
- Upload: 1 per minute
- Summary: 2 per minute

## Project Structure

```
src/app/
├── config.py              # Centralized configuration
├── main.py                # FastAPI application entry point
├── api.py                 # API endpoints
├── schemas.py             # Pydantic models
├── utils.py               # Utility functions
├── database/              # Database operations
│   ├── sqlite.py         # SQLite metadata storage
│   └── mongo.py          # MongoDB document storage
├── queue/                 # Queue and workers
│   ├── redis_queue.py    # Redis queue implementation
│   └── redis_queue_worker.py  # Background workers
├── middleware/            # HTTP middleware
│   ├── rate_limit_redis.py
│   └── rate_limit_middleware.py
└── services/              # Business logic
    ├── document/          # Document processing
    │   └── extractors.py  # Content extractors (Text, PDF)
    ├── embeddings/        # Embedding generation
    │   ├── embeddings.py  # OpenAI embeddings
    │   └── chunking.py    # Text chunking
    └── rag/               # RAG pipeline
        ├── rag.py         # RAG orchestration
        └── vector_store.py # ChromaDB integration
```

## Development

The project uses:
- **FastAPI** for the web framework
- **Pydantic** for data validation
- **SQLite** for metadata storage
- **MongoDB** for document content and summaries
- **Redis** for job queue and rate limiting
- **OpenAI API** for summarization and embeddings
- **ChromaDB** for vector storage (RAG)
- **PyMuPDF** for PDF processing
- **Pillow** and **PyTesseract** for image/OCR processing

### RAG Workflow

1. **Document Upload**: Documents are uploaded via `POST /docs`
2. **Content Extraction**: Text is extracted based on MIME type (text, markdown, PDF)
3. **Chunking**: Text is split into overlapping chunks (configurable size and overlap)
4. **Embedding**: Chunks are converted to embeddings using OpenAI's embedding model
5. **Indexing**: Embeddings are stored in ChromaDB vector store with metadata
6. **Querying**: Natural language queries are converted to embeddings and matched against stored chunks
7. **Response Generation**: Top-k relevant chunks are retrieved and used as context for LLM-generated responses

