from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

from promptpulse.config import DEFAULT_MODEL, DEFAULT_PROVIDER, PULSE_PASS_THRESHOLD, SYSTEM_PROMPT
from promptpulse.data import load_dataset
from promptpulse.evaluation import evaluate_response

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
    attempted_models: tuple[str, ...]


def get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
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


def _is_provider_compatibility_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "model_not_supported",
            "not supported by any provider",
            "no provider available",
            "provider is not available",
            "model is not supported",
        )
    )


def generate_response_with_fallback(
    user_query: str,
    reference_context: str,
    config: GenerationConfig,
) -> GenerationResult:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    candidates = tuple(
        dict.fromkeys(
            model_name
            for model_name in (config.model, *FALLBACK_MODELS)
            if model_name
        )
    )
    attempted: list[str] = []
    compatibility_errors: list[str] = []

    client = InferenceClient(api_key=token, provider=config.provider)

    for model_name in candidates:
        attempted.append(model_name)
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=build_messages(user_query, reference_context),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            text = completion.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError(f"Model {model_name!r} returned an empty response.")
            return GenerationResult(
                text=text.strip(),
                model=model_name,
                attempted_models=tuple(attempted),
            )
        except Exception as exc:
            if not _is_provider_compatibility_error(exc):
                raise
            compatibility_errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "No configured Hugging Face model is available through the enabled "
        "Inference Providers. " + " | ".join(compatibility_errors)
    )


st.set_page_config(
    page_title="PromptPulse · LLM Quality Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 16px;
        padding: 12px 16px;
        background: rgba(15,23,42,.35);
    }
    .pulse-hero {
        border: 1px solid rgba(139,92,246,.30);
        background: linear-gradient(135deg, rgba(76,29,149,.28), rgba(15,23,42,.18));
        border-radius: 24px;
        padding: 24px 28px;
        margin-bottom: 22px;
    }
    .pulse-kicker {color:#A78BFA; font-weight:700; letter-spacing:.12em; font-size:.78rem;}
    .pulse-title {font-size:2.6rem; font-weight:800; line-height:1.05; margin:.35rem 0 .6rem;}
    .pulse-sub {color:#CBD5E1; max-width:760px; font-size:1.02rem;}
    .status-pass {color:#34D399; font-weight:800;}
    .status-fail {color:#FB7185; font-weight:800;}
    </style>
    """,
    unsafe_allow_html=True,
)

dataset = load_dataset()
scenario_by_name = {row["scenario"]: row for row in dataset}

with st.sidebar:
    st.markdown("## ⚡ Control panel")
    scenario_name = st.selectbox("Evaluation scenario", list(scenario_by_name))
    row = scenario_by_name[scenario_name]

    model = st.text_input("Preferred Hugging Face model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.05)
    max_tokens = st.slider("Max output tokens", 64, 512, 220, 16)

    live_available = bool(get_hf_token())
    st.divider()
    if live_available:
        st.success("Live HF inference enabled")
        st.caption(
            "If the preferred model is unavailable for your enabled Inference Providers, "
            "PromptPulse automatically tries compatible fallback models."
        )
        with st.expander("Fallback order"):
            for candidate in FALLBACK_MODELS:
                st.code(candidate, language=None)
    else:
        st.warning("Demo mode: HF_TOKEN not configured")
    st.caption("Demo mode uses the curated expected answer so visitors can explore the evaluation UI without secrets.")

st.markdown(
    """
    <div class="pulse-hero">
      <div class="pulse-kicker">CONTINUOUS LLM EVALUATION</div>
      <div class="pulse-title">PromptPulse ⚡</div>
      <div class="pulse-sub">
        Run a chatbot scenario, inspect the generated answer, and see whether it clears
        relevance, grounding, coverage, and policy gates before release.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### Test case")
    st.text_area("User query", value=row["user_query"], height=90, disabled=True)
    with st.expander("Approved reference context", expanded=True):
        st.write(row["reference_context"])
    with st.expander("Quality contract"):
        st.markdown(f"**Expected answer**  \n{row['expected_answer']}")
        st.markdown(f"**Required terms:** {', '.join(row['required_terms']) or 'None'}")
        st.markdown(f"**Forbidden terms:** {', '.join(row['forbidden_terms']) or 'None'}")

with right:
    st.markdown("### Release gate")
    st.metric("Required Pulse", f"{PULSE_PASS_THRESHOLD:.0%}")
    st.write("A response must meet the overall threshold **and** have perfect explicit policy compliance.")
    st.info("Deterministic metrics are transparent regression signals. Optional DeepEval tests in CI provide a semantic LLM-as-a-judge layer.")

run = st.button("⚡ Run Live Evaluation", type="primary", use_container_width=True)

if run:
    config = GenerationConfig(
        model=model.strip() or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    started = time.perf_counter()
    mode = "Live Hugging Face inference"
    error = None
    actual_model = config.model
    attempted_models: tuple[str, ...] = (config.model,)

    if live_available:
        try:
            with st.spinner("Calling Hugging Face Inference Providers…"):
                generation = generate_response_with_fallback(
                    row["user_query"],
                    row["reference_context"],
                    config,
                )
            response = generation.text
            actual_model = generation.model
            attempted_models = generation.attempted_models
            if actual_model != config.model:
                mode = "Live Hugging Face inference · automatic model fallback"
        except Exception as exc:
            error = str(exc)
            response = row["expected_answer"]
            mode = "Fallback demo response"
    else:
        response = row["expected_answer"]
        mode = "Curated demo response"

    latency_ms = int((time.perf_counter() - started) * 1000)

    result = evaluate_response(
        user_query=row["user_query"],
        reference_context=row["reference_context"],
        response=response,
        required_terms=row["required_terms"],
        forbidden_terms=row["forbidden_terms"],
    )

    st.divider()
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown("## Evaluation result")
        st.caption(f"{mode} · {latency_ms} ms · model: {actual_model}")
    with header_right:
        status = "PASS" if result.passed else "FAIL"
        klass = "status-pass" if result.passed else "status-fail"
        st.markdown(f"<div class='{klass}' style='font-size:1.7rem;text-align:right'>{status}</div>", unsafe_allow_html=True)

    if error:
        st.warning(f"Live inference failed, so the demo used the curated response. Provider message: {error}")
    elif actual_model != config.model:
        st.info(
            f"The preferred model `{config.model}` was unavailable through your enabled providers. "
            f"PromptPulse recovered automatically with `{actual_model}`."
        )

    st.markdown("### Model response")
    st.write(response)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pulse", f"{result.pulse_score:.0%}")
    c2.metric("Relevance", f"{result.answer_relevance:.0%}")
    c3.metric("Grounded", f"{result.groundedness:.0%}")
    c4.metric("Coverage", f"{result.reference_coverage:.0%}")
    c5.metric("Policy", f"{result.policy_compliance:.0%}")

    st.progress(min(max(result.pulse_score, 0.0), 1.0))

    score_df = pd.DataFrame(
        {
            "Metric": ["Answer relevance", "Groundedness", "Reference coverage", "Policy compliance"],
            "Score": [
                result.answer_relevance,
                result.groundedness,
                result.reference_coverage,
                result.policy_compliance,
            ],
            "Threshold / intent": [
                "Query alignment",
                "Supported by approved context",
                "Reference facts represented",
                "Required/forbidden rule checks",
            ],
        }
    )
    st.dataframe(
        score_df,
        hide_index=True,
        use_container_width=True,
        column_config={"Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.0%%")},
    )

    st.markdown("### Gate reasons")
    for reason in result.reasons:
        st.write(f"• {reason}")

    with st.expander("Machine-readable result"):
        payload = {
            "scenario_id": row["id"],
            "requested_model": config.model,
            "actual_model": actual_model,
            "attempted_models": list(attempted_models),
            "mode": mode,
            "latency_ms": latency_ms,
            "response": response,
            "evaluation": result.to_dict(),
        }
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")

st.divider()
st.caption("PromptPulse · GitHub Actions quality gates · Hugging Face Inference Providers · Streamlit")
st.markdown(
    """
<div style="text-align:center;color:#64748b;font-size:0.88rem;padding:0.5rem 0 1rem;">
  <strong>Hendarmawan, PhD Eng.</strong> &nbsp;·&nbsp;
  <a href="https://github.com/h00w/" target="_blank">GitHub</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/hender/" target="_blank">LinkedIn</a> &nbsp;·&nbsp;
  <a href="https://hendarmawan.se" target="_blank">hendarmawan.se</a>
</div>
""",
    unsafe_allow_html=True,
)
