from __future__ import annotations 

from typing import List, Tuple, Dict, Any 
from openai import OpenAI
from ..logging_config import get_logger 

log = get_logger("precisbox.services.embeddings")

class EmbeddingService:
    # this is to generate embeddings using OpenAI 

    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ): #to initialize the service
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model 

    def embed_text(self, text: str) -> List[float]:
        #creating embedding for single text
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            log.exception("Failed to generate embedding %s", e)
            raise 

    def embed_batch(self, text: List[str], batch_size: int = 100) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                log.exception("Failed to generate batch embeddings %s", e)
                raise
        return embeddings