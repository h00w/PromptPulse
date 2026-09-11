from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .config import PULSE_PASS_THRESHOLD

_WORD = re.compile(r"[A-Za-z0-9À-ÖØ-öø-ÿ'-]+")
_STOP = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "can", "could", "should",
    "would", "do", "does", "did", "i", "you", "we", "they", "it", "this", "that",
    "what", "how", "when", "where", "why", "my", "your", "our", "their", "with",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD.findall(text) if token.lower() not in _STOP]


def _set(text: str) -> set[str]:
    return set(_tokens(text))


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    return default if den == 0 else num / den


def query_relevance(user_query: str, response: str) -> float:
    q, r = _set(user_query), _set(response)
    return round(_safe_ratio(len(q & r), len(q), 0.0), 3)


def reference_coverage(reference_context: str, response: str) -> float:
    c, r = _set(reference_context), _set(response)
    return round(_safe_ratio(len(c & r), len(c), 0.0), 3)


def groundedness(reference_context: str, response: str) -> float:
    c, r = _set(reference_context), _set(response)
    if not r:
        return 0.0
    supported = len(c & r)
    return round(min(1.0, _safe_ratio(supported, len(r), 0.0) * 1.35), 3)


def policy_compliance(
    response: str,
    required_terms: Iterable[str] = (),
    forbidden_terms: Iterable[str] = (),
) -> tuple[float, list[str]]:
    normalized = response.casefold()
    failures: list[str] = []
    required = list(required_terms)
    forbidden = list(forbidden_terms)
    checks = 0
    passed = 0

    for term in required:
        checks += 1
        if term.casefold() in normalized:
            passed += 1
        else:
            failures.append(f"Missing required term: {term}")

    for term in forbidden:
        checks += 1
        if term.casefold() not in normalized:
            passed += 1
        else:
            failures.append(f"Forbidden term present: {term}")

    score = _safe_ratio(passed, checks, 1.0)
    return round(score, 3), failures


@dataclass(frozen=True)
class EvaluationResult:
    answer_relevance: float
    groundedness: float
    reference_coverage: float
    policy_compliance: float
    pulse_score: float
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_response(
    *,
    user_query: str,
    reference_context: str,
    response: str,
    required_terms: Iterable[str] = (),
    forbidden_terms: Iterable[str] = (),
    pass_threshold: float = PULSE_PASS_THRESHOLD,
) -> EvaluationResult:
    relevance = query_relevance(user_query, response)
    ground = groundedness(reference_context, response)
    coverage = reference_coverage(reference_context, response)
    policy, failures = policy_compliance(response, required_terms, forbidden_terms)

    pulse = round(
        (0.25 * relevance)
        + (0.30 * ground)
        + (0.20 * coverage)
        + (0.25 * policy),
        3,
    )

    reasons = list(failures)
    if relevance < 0.45:
        reasons.append("Low lexical relevance to the user question.")
    if ground < 0.55:
        reasons.append("Response contains substantial content not supported by approved context.")
    if coverage < 0.45:
        reasons.append("Response covers too little of the approved reference context.")
    if not reasons:
        reasons.append("No deterministic quality-gate violations detected.")

    return EvaluationResult(
        answer_relevance=relevance,
        groundedness=ground,
        reference_coverage=coverage,
        policy_compliance=policy,
        pulse_score=pulse,
        passed=pulse >= pass_threshold and policy >= 0.999,
        reasons=reasons,
    )
