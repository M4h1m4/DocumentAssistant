from pydantic_settings import BaseSettings
from pydantic import Field


class Defaults:
    """Default values used throughout the application."""
    
    # File defaults
    DEFAULT_FILENAME: str = "upload.txt"
    
    # MIME type defaults
    DEFAULT_MIME_TYPE: str = "application/octet-stream"
    
    SUPPORTED_MIME_TYPES: set[str] = {
        "text/plain",
        "text/markdown",
        "application/pdf",
    }
    # Status defaults
    DEFAULT_STATUS: str = "unknown"


class Settings(BaseSettings):
    # Database
    sqlite_path: str = Field(default="./meta.db")
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db: str = Field(default="precisbox")

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_timeout: float = Field(default=120.0)

    # Workers
    workers: int = Field(default=1, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff: float = Field(default=0.5, ge=0)
    
    # Rate Limiting
    redis_url: str = Field(default="redis://localhost:6379/0")
    upload_per_min: int = Field(default=1, ge=1)
    summary_per_min: int = Field(default=2, ge=1)
    
    # Upload
    max_upload_bytes: int = Field(default=2000000, ge=1)
    mongo_max_pool_size: int = Field(default=10, ge=1)
    mongo_min_pool_size: int = Field(default=1, ge=0)

    pdf_extract_images: bool = Field(default=True, description="Extract images from PDFs")
    pdf_use_ocr: bool = Field(default=False, description="Use OCR to extract text from images")
    pdf_max_image_size_mb: int = Field(default=10, ge=1, description="Max image size in MB")

    enable_rag: bool = Field(default=False, description="Enable RAG capabilities")
    vector_store_path: str = Field(default="./vector_store", description="Vector store directory")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    chunk_size: int = Field(default=1000, ge=100, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    rag_top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve for RAG")
    
    @property
    def is_summarizer_enabled(self) -> bool:
        """Check if summarizer is enabled based on OpenAI API key."""
        return bool(self.openai_api_key)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()