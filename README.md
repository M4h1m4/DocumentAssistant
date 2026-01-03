# PrecisBox

A FastAPI-based document summarization service that processes text documents and generates summaries using OpenAI.

## Features

- Document upload and storage (SQLite for metadata, MongoDB for content)
- Background job processing with Redis queue
- Rate limiting with Redis token bucket algorithm
- Document summarization using OpenAI API
- RESTful API for document management

## Requirements

- Python 3.11+
- Redis
- MongoDB
- OpenAI API key

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

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MONGO_URI`: MongoDB connection string
- `REDIS_URL`: Redis connection URL
- `WORKERS`: Number of worker threads
- Other configuration options (see `.env.example`)

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

**Note**: Rate limiting requires the `X-User-Id` header. Default limits:
- Upload: 1 per minute
- Summary: 2 per minute

## Project Structure

```
src/app/
├── config.py              # Centralized configuration
├── main.py                # FastAPI application entry point
├── schemas.py             # Pydantic models
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
    ├── hashing.py
    └── summarize.py
```

## Development

The project uses:
- FastAPI for the web framework
- Pydantic for data validation
- SQLite for metadata storage
- MongoDB for document content
- Redis for job queue and rate limiting
- OpenAI API for summarization

## License

[Add your license here]
