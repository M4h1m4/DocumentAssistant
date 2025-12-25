# PrecisBox Architecture Notes

## Components
- FastAPI service exposing endpoints
- MongoDB stores the raw document + summary
- SQLite stores metadata, status, and error fields

## Flow
1. POST /docs receives a file
2. Store raw doc in Mongo
3. Store metadata row in SQLite (status=pending)
4. Background task calls OpenAI summarizer
5. Store summary in Mongo and mark SQLite status=done

## Failure Handling
- If OpenAI fails, mark SQLite row failed and store last_error.
- If Mongo write fails, fail the request and do not insert metadata.

## Future Enhancements
- Move background work to a real queue (Redis/Celery)
- Add pagination to GET /docs
- Add PDF extraction pipeline
