from promptpulse.data import load_dataset
from promptpulse.evaluation import evaluate_response, policy_compliance
import json
import pytest


def test_dataset_schema_and_unique_ids():
    rows = load_dataset()
    assert len(rows) >= 4
    assert len({row["id"] for row in rows}) == len(rows)


def test_malformed_policy_terms_cannot_be_scored_as_character_checks(tmp_path):
    row = dict(load_dataset()[0], required_terms="30 days")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="required_terms"):
        load_dataset(path)


def test_empty_reference_is_rejected_before_evaluation(tmp_path):
    row = dict(load_dataset()[0], reference_context=" ")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="reference_context"):
        load_dataset(path)


def test_expected_answers_clear_deterministic_gate():
    for row in load_dataset():
        result = evaluate_response(
            user_query=row["user_query"],
            reference_context=row["reference_context"],
            response=row["expected_answer"],
            required_terms=row["required_terms"],
            forbidden_terms=row["forbidden_terms"],
            pass_threshold=0.50,
        )
        assert result.policy_compliance == 1.0, (row["id"], result)
        assert result.pulse_score >= 0.50, (row["id"], result)


def test_policy_violation_fails_even_if_answer_is_relevant():
    row = load_dataset()[0]
    bad = "You can return it within 60 days."
    result = evaluate_response(
        user_query=row["user_query"],
        reference_context=row["reference_context"],
        response=bad,
        required_terms=row["required_terms"],
        forbidden_terms=row["forbidden_terms"],
    )
    assert result.policy_compliance < 1.0
    assert result.passed is False


def test_empty_answer_fails():
    result = evaluate_response(
        user_query="What is the refund policy?",
        reference_context="Refunds are available for 30 days.",
        response="",
    )
    assert result.passed is False
    assert result.pulse_score < 0.70


@pytest.mark.parametrize("field", ["required_terms", "forbidden_terms"])
@pytest.mark.parametrize("terms", ["30 days", b"safe", [""], ["  "], [None], [1], None, 1])
def test_direct_policy_checks_reject_malformed_terms(field, terms):
    with pytest.raises(ValueError, match=field):
        policy_compliance("Refunds are available for 30 days.", **{field: terms})


def test_policy_checks_accept_term_generators_without_losing_checks():
    score, failures = policy_compliance(
        "Refunds are available for 30 days.",
        required_terms=(term for term in ["30 days"]),
        forbidden_terms=(term for term in ["60 days"]),
    )
    assert score == 1.0
    assert failures == []


def test_policy_terms_do_not_match_inside_other_words_or_numbers():
    score, failures = policy_compliance("Refund in 130 days; policy is unsafe.", required_terms=["30 days"], forbidden_terms=["safe"])
    assert score == 0.5
    assert failures == ["Missing required term: 30 days"]
    score, failures = policy_compliance("Refund in 30 days; policy is safe.", required_terms=["30 days"], forbidden_terms=["safe"])
    assert score == 0.5
    assert failures == ["Forbidden term present: safe"]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, True, False, "0.7", None])
def test_invalid_evaluation_threshold_cannot_change_release_decision(threshold):
    row = load_dataset()[0]
    with pytest.raises(ValueError, match="pass_threshold"):
        evaluate_response(
            user_query=row["user_query"], reference_context=row["reference_context"],
            response=row["expected_answer"], pass_threshold=threshold,
        )
