from __future__ import annotations

import os

DEFAULT_MODEL = os.getenv("PROMPTPULSE_MODEL", "openai/gpt-oss-120b")
DEFAULT_PROVIDER = os.getenv("PROMPTPULSE_PROVIDER", "auto")

# Ordered fallbacks for Hugging Face Inference Providers. The runtime de-duplicates
# this list against the user-selected model and only advances when the router reports
# that a model/provider combination is unavailable.
FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-4B-Thinking-2507",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct-1M",
)

SYSTEM_PROMPT = """You are a concise customer-support assistant.
Answer only from the APPROVED CONTEXT supplied with the request.
If the context does not contain the answer, say that the approved context does not specify it.
Never invent policies, prices, deadlines, legal claims, or account-specific facts."""

PULSE_PASS_THRESHOLD = 0.70
