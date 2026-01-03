# Testing Guide - PrecisBox

Complete guide to set up, run, and test the PrecisBox application using `uv`.

## Prerequisites

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS with Homebrew:
brew install uv
```

2. **Start Redis**:
```bash
redis-server
# Or with Homebrew services:
brew services start redis
```

3. **Start MongoDB**:
```bash
mongod
# Or with Homebrew services:
brew services start mongodb-community
```

## Setup Steps

### 1. Install Dependencies with uv

```bash
# Navigate to project directory
cd PrecisBox

# Sync dependencies (creates virtual environment and installs packages)
uv sync
```

### 2. Create .env File

```bash
# Create .env file from template (if .env.example exists)
cp .env.example .env

# Or create manually
cat > .env << EOF
OPENAI_API_KEY=sk-your-actual-api-key-here
MONGO_URI=mongodb://localhost:27017
MONGO_DB=precisbox
REDIS_URL=redis://localhost:6379/0
SQLITE_PATH=./meta.db
WORKERS=1
MAX_RETRIES=2
RETRY_BACKOFF=0.5
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=120.0
UPLOAD_PER_MIN=1
SUMMARY_PER_MIN=2
MAX_UPLOAD_BYTES=2000000
EOF
```

**Important**: Replace `sk-your-actual-api-key-here` with your real OpenAI API key!

### 3. Run the Application

```bash
# Using uv to run uvicorn
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will start at `http://localhost:8000`

## Testing the Application

### 1. Health Checks

#### Check Health Endpoint
```bash
curl http://localhost:8000/healthz
```
**Expected**: `{"ok":true,"service":"precisbox"}`

#### Check Ready Endpoint
```bash
curl http://localhost:8000/ready
```
**Expected**: `{"ready":true}` (if OpenAI API key is set)

### 2. View API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test Document Upload (with Rate Limiting)

#### Create a test document
```bash
echo "This is a test document. It contains some content that will be summarized by the OpenAI API. The document discusses various topics and provides information for testing purposes." > test_doc.txt
```

#### Upload Document (with User ID header for rate limiting)
```bash
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: user-123" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_doc.txt"

# Expected response:
# {"id":"<doc_id>","status":"pending","display_name":"test_doc.txt_<short_id>"}
```

**Save the `doc_id` from the response** - you'll need it for next steps!

### 4. Test Rate Limiting

#### Test Upload Rate Limit (default: 1 per minute)
```bash
# First upload (should succeed)
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: user-123" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_doc.txt"

# Second upload immediately (should be rate limited - 429 status)
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: user-123" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_doc.txt"

# Expected: {"detail":"rate_limited","user_id":"user-123","endpoint":"upload","retry_after_seconds":X}
# Status: 429 Too Many Requests
```

#### Test Summary Rate Limit (default: 2 per minute)
```bash
# Replace <doc_id> with actual document ID from upload
DOC_ID="<your-doc-id>"

# First summary request (should succeed)
curl -X GET http://localhost:8000/docs/${DOC_ID}/summary \
  -H "X-User-Id: user-123"

# Second summary request (should succeed)
curl -X GET http://localhost:8000/docs/${DOC_ID}/summary \
  -H "X-User-Id: user-123"

# Third summary request (should be rate limited - 429)
curl -X GET http://localhost:8000/docs/${DOC_ID}/summary \
  -H "X-User-Id: user-123"

# Expected: {"detail":"rate_limited","user_id":"user-123","endpoint":"summary","retry_after_seconds":X}
# Status: 429 Too Many Requests
```

#### Test Missing User ID (401 Unauthorized)
```bash
# Upload without X-User-Id header
curl -X POST http://localhost:8000/docs \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_doc.txt"

# Expected: {"detail":"missing or unknown user"}
# Status: 401 Unauthorized
```

### 5. Test Document Retrieval

#### Get Document Metadata
```bash
# Replace <doc_id> with actual document ID
curl http://localhost:8000/docs/<doc_id> \
  -H "X-User-Id: user-123"
```

#### Get Document Summary
```bash
# Replace <doc_id> with actual document ID
# Wait a few seconds after upload for processing
curl http://localhost:8000/docs/<doc_id>/summary \
  -H "X-User-Id: user-123"

# Expected responses:
# - 202 Accepted: {"id":"...","status":"pending","summary":null,"error":{...}}
# - 202 Accepted: {"id":"...","status":"processing","summary":null,"error":{...}}
# - 200 OK: {"id":"...","status":"done","summary":"<summary text>","error":null}
```

#### List All Documents
```bash
curl "http://localhost:8000/docs?page=1&size=10" \
  -H "X-User-Id: user-123"

# With status filter
curl "http://localhost:8000/docs?page=1&size=10&status=done" \
  -H "X-User-Id: user-123"
```

### 6. Complete End-to-End Test

```bash
# Set your user ID
USER_ID="test-user-$(date +%s)"

# 1. Upload a document
echo "This is a comprehensive test document for PrecisBox. It contains multiple sentences and paragraphs to ensure the summarization service works correctly. The content should be processed and summarized by OpenAI's API." > test_doc.txt

UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8000/docs \
  -H "X-User-Id: $USER_ID" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_doc.txt")

echo "Upload response: $UPLOAD_RESPONSE"

# Extract doc_id (using jq if available, or manual extraction)
DOC_ID=$(echo $UPLOAD_RESPONSE | grep -o '"id":"[^"]*' | cut -d'"' -f4)
echo "Document ID: $DOC_ID"

# 2. Check document status
curl http://localhost:8000/docs/$DOC_ID -H "X-User-Id: $USER_ID"

# 3. Poll for summary (wait up to 60 seconds)
for i in {1..12}; do
  echo "Polling attempt $i..."
  SUMMARY_RESPONSE=$(curl -s http://localhost:8000/docs/$DOC_ID/summary -H "X-User-Id: $USER_ID")
  STATUS=$(echo $SUMMARY_RESPONSE | grep -o '"status":"[^"]*' | cut -d'"' -f4)
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "done" ]; then
    echo "Summary ready!"
    echo $SUMMARY_RESPONSE | grep -o '"summary":"[^"]*'
    break
  fi
  
  sleep 5
done

# 4. List all documents
curl "http://localhost:8000/docs?page=1&size=10" -H "X-User-Id: $USER_ID"
```

## Testing with Different Users

Rate limiting is per-user (based on `X-User-Id` header):

```bash
# User 1 uploads
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: alice" \
  -F "file=@test_doc.txt"

# User 2 can still upload (different user)
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: bob" \
  -F "file=@test_doc.txt"

# User 1's second upload is rate limited
curl -X POST http://localhost:8000/docs \
  -H "X-User-Id: alice" \
  -F "file=@test_doc.txt"
# Expected: 429 Rate Limited
```

## Quick Test Script

Create a test script `test_api.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"
USER_ID="test-user-123"

echo "1. Health check..."
curl -s $BASE_URL/healthz | jq .

echo -e "\n2. Ready check..."
curl -s $BASE_URL/ready | jq .

echo -e "\n3. Upload document..."
UPLOAD=$(curl -s -X POST $BASE_URL/docs \
  -H "X-User-Id: $USER_ID" \
  -F "file=@test_doc.txt")
echo $UPLOAD | jq .

DOC_ID=$(echo $UPLOAD | jq -r '.id')
echo -e "\n4. Get document metadata..."
curl -s $BASE_URL/docs/$DOC_ID -H "X-User-Id: $USER_ID" | jq .

echo -e "\n5. Test rate limiting (should fail)..."
curl -s -X POST $BASE_URL/docs \
  -H "X-User-Id: $USER_ID" \
  -F "file=@test_doc.txt" | jq .

echo -e "\n6. Wait for summary (polling)..."
sleep 10
curl -s $BASE_URL/docs/$DOC_ID/summary -H "X-User-Id: $USER_ID" | jq .
```

Make it executable and run:
```bash
chmod +x test_api.sh
./test_api.sh
```

## Troubleshooting

### Application won't start
- Check Redis is running: `redis-cli ping` (should return "PONG")
- Check MongoDB is running: `mongosh --eval "db.runCommand('ping')"`
- Check `.env` file exists and has valid `OPENAI_API_KEY`

### Rate limiting not working
- Verify Redis is running and accessible
- Check `REDIS_URL` in `.env` matches your Redis instance
- Verify `X-User-Id` header is being sent

### Summaries not appearing
- Check worker logs in the application console
- Verify OpenAI API key is valid
- Check document status: `curl http://localhost:8000/docs/<doc_id>`
- Wait longer (summarization takes time)

### 401 Unauthorized errors
- Ensure `X-User-Id` header is included in requests
- Header name must be exactly `X-User-Id` (case-sensitive)

