"""API client for OpenAI/OpenRouter models."""

import os
from openai import OpenAI
from typing import List, Tuple


def create_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Create OpenAI client with optional custom base URL."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    
    if base_url is None:
        base_url = os.getenv("API_BASE_URL")
    
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def chat(
    client: OpenAI,
    model: str,
    user_message: str,
    history: List[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    system_message: str = None
) -> Tuple[str, List[dict]]:
    """Send message and get response. Returns (reply, updated_history)."""
    if history is None:
        history = []
    
    messages = history.copy()
    messages.append({"role": "user", "content": user_message})
    
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    reply = response.choices[0].message.content
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]
    
    return reply, updated_history
