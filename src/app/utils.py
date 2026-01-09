from __future__ import annotations

import hashlib
from typing import Any
from .logging_config import get_logger

log = get_logger("precisbox.utils")


def safe_int(val: Any, default: int, field: str, doc_id: str) -> int:
    """
    Safely convert a value to an integer.
    
    Args:
        val: Value to convert
        default: Default value to return if conversion fails
        field: Field name (for logging purposes)
        doc_id: Document ID (for logging purposes)
        
    Returns:
        Integer value, or default if conversion fails
    """
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        log.exception("Invalid %s=%r for doc_id=%s; defaulting to %d", field, val, doc_id, default)
        return default


def sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.
    
    Returns SHA-256 hex digest for deduplication
    (same file uploaded again gets the same hash).
    
    Args:
        data: Raw bytes to hash
        
    Returns:
        Hexadecimal string representation of SHA-256 hash
    """
    return hashlib.sha256(data).hexdigest()


def decode_text(raw: bytes) -> str:
    """
    Decode bytes as UTF-8 text.
    
    Args:
        raw: Raw bytes to decode
        
    Returns:
        Decoded UTF-8 string
        
    Raises:
        UnicodeDecodeError: If bytes are not valid UTF-8
    """
    return raw.decode("utf-8")

