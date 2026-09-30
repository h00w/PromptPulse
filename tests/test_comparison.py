from promptpulse.comparison import compare_runs
from promptpulse.data import load_dataset
import pytest
from types import SimpleNamespace


def _answers():
    rows = load_dataset()
    return rows, {row["id"]: row["expected_answer"] for row in rows}


def test_same_cases_and_answers_pass():
    rows, answers = _answers()
    report = compare_runs(rows, answers, answers)
    assert report.decision == "PASS"
    assert report.regressed_cases == ()


def test_missing_case_holds_even_if_remaining_answers_are_good():
    rows, answers = _answers()
    candidate = dict(answers)
    candidate.pop(rows[0]["id"])
    report = compare_runs(rows, answers, candidate)
    assert report.decision == "HOLD"
    assert report.missing_cases == (rows[0]["id"],)


def test_policy_regression_holds_despite_other_good_cases():
    rows, answers = _answers()
    candidate = dict(answers)
    candidate["refund-policy"] = "Unused products can be refunded within 60 days."
    report = compare_runs(rows, answers, candidate)
    assert report.decision == "HOLD"
    assert "refund-policy" in report.regressed_cases


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -0.01])
def test_invalid_tolerance_cannot_turn_regression_into_pass(tolerance):
    rows, answers = _answers()
    with pytest.raises(ValueError, match="finite non-negative"):
        compare_runs(rows, answers, answers, mean_tolerance=tolerance)


def test_decision_uses_unrounded_means_at_tolerance_boundary(monkeypatch):
    scores = iter((0.9004, 0.8996))
    monkeypatch.setattr("promptpulse.comparison.evaluate_response", lambda **kwargs: SimpleNamespace(pulse_score=next(scores), passed=True))
    row = {"id": "precision", "user_query": "query", "reference_context": "context"}
    report = compare_runs([row], {"precision": "baseline"}, {"precision": "candidate"}, mean_tolerance=0.0005)
    assert report.decision == "INVESTIGATE"
    assert report.baseline_mean == 0.9
    assert report.candidate_mean == 0.9
