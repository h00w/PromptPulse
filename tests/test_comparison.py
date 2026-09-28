from promptpulse.comparison import compare_runs
from promptpulse.data import load_dataset


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
