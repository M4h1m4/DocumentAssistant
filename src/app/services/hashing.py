from __future__ import annotations
import hashlib
from typing import Tuple, Dict, List


def sha256_bytes(data: bytes) -> str:
    #returns sha 256 jex digest to dedeup (same file uploaded again gets the same hash)
    return hashlib.sha256(data).hexdigest()

def decode_text(raw: bytes) -> str:
    return raw.decode("utf-8") # decode text to UTF-8 else raise UniCodeDecode Error if it is not valid UTF-8