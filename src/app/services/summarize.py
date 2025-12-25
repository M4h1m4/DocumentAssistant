from __future__ import annotations
from typing import Tuple, Optional
from openai import OpenAI 

def summarize_text(api_key: str, model: str, text: str, timeout: float = 120.0) -> Tuple[str, Optional[int], Optional[int]]:
    client = OpenAI(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes documents clearly."},
            {"role": "user", "content": f"summarize the following document into 6-10 bullet points:\n\n{text}"},
        ],
        temperature = 0.2 #how random/creative the model is. Here the temperature is low that is sticking to the facts. 
        #Higher temperature may lead to hallucinations. 
    )
    #The system message is the highest-priority instruction
    #User message is the actual request. 

    """
    Why do we need both? (role: system, user)

    Because they serve different purposes:

    System: sets permanent behavior across the conversation/request.

    User: gives the specific task + data for this request.
    """

    summary = resp.choices[0].message.content or ""
    usage = resp.usage 
    prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
    completion_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens else 0
    return summary, prompt_tokens, completion_tokens

