from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from huggingface_hub import InferenceClient

from .config import DEFAULT_MODEL, DEFAULT_PROVIDER, SYSTEM_PROMPT

# Keep the provider fallback registry in the inference module that consumes it.
# This avoids deployment-time import mismatches when Streamlit Cloud rebuilds from
# closely spaced commits and keeps model-routing concerns out of static app config.
FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-4B-Thinking-2507",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct-1M",
)


@dataclass(frozen=True)
class GenerationConfig:
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    temperature: float = 0.2
    max_tokens: int = 220


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    attempted_models: tuple[str, ...]


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


def _candidate_models(requested_model: str) -> tuple[str, ...]:
    ordered = [requested_model, *FALLBACK_MODELS]
    return tuple(dict.fromkeys(model for model in ordered if model))


def _is_provider_compatibility_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    markers = (
        "model_not_supported",
        "not supported by any provider",
        "no provider available",
        "provider is not available",
        "model is not supported",
    )
    return any(marker in message for marker in markers)


def generate_response_with_fallback(
    user_query: str,
    reference_context: str,
    config: GenerationConfig,
) -> GenerationResult:
    """Generate with HF Inference Providers and recover from unsupported models.

    The requested model is attempted first. If the Hugging Face router reports that
    the model is unavailable for the user's enabled providers, PromptPulse tries a
    small ordered set of known chat-completion models. Authentication, quota, and
    other non-compatibility errors are surfaced immediately instead of being hidden.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    attempted: list[str] = []
    compatibility_errors: list[str] = []

    for model in _candidate_models(config.model):
        attempted.append(model)
        try:
            client = InferenceClient(api_key=token, provider=config.provider)
            completion = client.chat.completions.create(
                model=model,
                messages=build_messages(user_query, reference_context),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            text = completion.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError(f"Model {model!r} returned an empty response.")
            return GenerationResult(
                text=text.strip(),
                model=model,
                provider=config.provider,
                attempted_models=tuple(attempted),
            )
        except Exception as exc:
            if not _is_provider_compatibility_error(exc):
                raise
            compatibility_errors.append(f"{model}: {exc}")

    summary = " | ".join(compatibility_errors)
    raise RuntimeError(
        "No configured Hugging Face fallback model is available through the enabled "
        f"Inference Providers. Attempts: {summary}"
    )


def generate_response(
    user_query: str,
    reference_context: str,
    config: GenerationConfig,
) -> str:
    """Backward-compatible text-only generation helper."""
    return generate_response_with_fallback(user_query, reference_context, config).text


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
