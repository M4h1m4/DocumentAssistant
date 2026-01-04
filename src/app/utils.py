from __future__ import annotations

from typing import Any
import ..logging_config import get_logger

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

