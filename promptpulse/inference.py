from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from huggingface_hub import InferenceClient

from .config import DEFAULT_MODEL, DEFAULT_PROVIDER, SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationConfig:
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    temperature: float = 0.2
    max_tokens: int = 220


def get_hf_token() -> str | None:
    """Read a token without ever logging it."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def build_messages(user_query: str, reference_context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "APPROVED CONTEXT:\n"
                f"{reference_context}\n\n"
                "USER QUESTION:\n"
                f"{user_query}"
            ),
        },
    ]


def generate_response(
    user_query: str,
    reference_context: str,
    config: GenerationConfig,
) -> str:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    client = InferenceClient(api_key=token, provider=config.provider)
    completion = client.chat.completions.create(
        model=config.model,
        messages=build_messages(user_query, reference_context),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return completion.choices[0].message.content.strip()


def stream_response(
    user_query: str,
    reference_context: str,
    config: GenerationConfig,
) -> Iterator[str]:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    client = InferenceClient(api_key=token, provider=config.provider)
    stream = client.chat.completions.create(
        model=config.model,
        messages=build_messages(user_query, reference_context),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
