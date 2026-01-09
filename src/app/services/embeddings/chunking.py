from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

from dataclasses import dataclass 

from ...logging_config import get_logger

log = get_logger("precisbox.services.chunking")


"""
Documents are too large to chunk them as whole hence we chunk them into smaller pieces 
that can be embedded and searched individually.
"""
@dataclass 
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int 
    metadata: Dict[str, Any] #doc_id, pagenumber, filename etc

class FixedSizeChunker:
    #This has overap between chunks to make sure we don't loose information at chunk boundaries 
    def __init__(
        self,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200, 
        seperator: str = "\n\n",
    ):
        #Initialize chunker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap 
        self.seperator = seperator 

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        #split text into overlapping chunks
        #returns TextChunk objects 
        chunks: List[TextChunk] = []
        #split by paragraphs
        paragraphs = text.split(self.seperator) 
        current_chunk = []
        curr_size = 0
        chunk_index = 0
        start_char = 0
        for para in paragraphs:
            para_size = len(para)
            #If adding this para exceeds the chunk size finalize this chunk
            if curr_size + para_size > self.chunk_size and current_chunk:
                chunk_text = self.seperator.join(current_chunk)
                end_char = start_char + len(chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text, 
                    chunk_index = chunk_index,
                    start_char = start_char,
                    end_char=end_char,
                    metadata= metadata.copy()
                ))
                overlap_text = chunk_text[-self.chunk_overlap:]
                if overlap_text:
                    current_chunk = [overlap_text, para]
                else:
                    current_chunk = [para]
                start_char = end_char - self.chunk_overlap
                chunk_index += 1
                curr_size = len(self.seperator.join(current_chunk))
            else:
                current_chunk.append(para)
                curr_size += para_size + len(self.seperator)

        #Adding final chunk
        if current_chunk:
            chunk_text = self.seperator.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata.copy(),
            ))
        
        return chunks
