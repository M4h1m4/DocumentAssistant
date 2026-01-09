"""Embedding utilities - Core components for text embeddings and chunking."""
from .embeddings import EmbeddingService
from .chunking import FixedSizeChunker, TextChunk

__all__ = [
    "EmbeddingService",
    "FixedSizeChunker",
    "TextChunk",
]

