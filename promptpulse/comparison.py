"""Paired, offline baseline-versus-candidate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping

from .evaluation import evaluate_response


@dataclass(frozen=True)
class ComparisonReport:
    decision: str
    baseline_mean: float
    candidate_mean: float
    regressed_cases: tuple[str, ...]
    failed_candidate_cases: tuple[str, ...]
    missing_cases: tuple[str, ...]
    unexpected_cases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_runs(
    dataset: list[dict[str, Any]],
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    mean_tolerance: float = 0.02,
) -> ComparisonReport:
    if not dataset or not isinstance(mean_tolerance, (int, float)) or not isfinite(mean_tolerance) or mean_tolerance < 0:
        raise ValueError("A non-empty dataset and finite non-negative tolerance are required.")
    cases = {row["id"]: row for row in dataset}
    if len(cases) != len(dataset):
        raise ValueError("Dataset case IDs must be unique.")
    expected = set(cases)
    missing = tuple(sorted((expected - baseline.keys()) | (expected - candidate.keys())))
    unexpected = tuple(sorted((baseline.keys() | candidate.keys()) - expected))
    aligned = sorted(expected & baseline.keys() & candidate.keys())
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    regressions: list[str] = []
    failures: list[str] = []

    for case_id in aligned:
        row = cases[case_id]
        answers = (baseline[case_id], candidate[case_id])
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError(f"Case {case_id}: both responses must be non-empty strings.")
        results = [
            evaluate_response(
                user_query=row["user_query"],
                reference_context=row["reference_context"],
                response=answer,
                required_terms=row.get("required_terms", []),
                forbidden_terms=row.get("forbidden_terms", []),
            )
            for answer in answers
        ]
        baseline_scores.append(results[0].pulse_score)
        candidate_scores.append(results[1].pulse_score)
        if results[0].passed and not results[1].passed:
            regressions.append(case_id)
        if not results[1].passed:
            failures.append(case_id)

    base_mean = sum(baseline_scores) / len(aligned) if aligned else 0.0
    cand_mean = sum(candidate_scores) / len(aligned) if aligned else 0.0
    decision = "HOLD" if missing or unexpected or failures else (
        "INVESTIGATE" if cand_mean < base_mean - mean_tolerance else "PASS"
    )
    return ComparisonReport(
        decision, round(base_mean, 3), round(cand_mean, 3), tuple(regressions), tuple(failures), missing, unexpected
    )
