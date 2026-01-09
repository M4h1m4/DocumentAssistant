from __future__ import annotations
from typing import Tuple, Optional
from openai import OpenAI 

class SummarizeError(RuntimeError):
    pass 

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
    if not getattr(resp, "choices", None):
        raise SummarizeError("OpenAI returned no choices")

    msg = resp.choices[0].message
    content = (msg.content or "").strip() if msg else ""
    if not content:
        # could be refusal, empty output, or other non-server failure
        raise SummarizeError("OpenAI returned empty summary (possible refusal or truncated output)")
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens",0)) if usage else 0
    completion_tokens = int(getattr(usage, "completion_tokens",0)) if usage else 0
    return content, prompt_tokens, completion_tokens

