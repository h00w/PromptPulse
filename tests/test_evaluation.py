from promptpulse.data import load_dataset
from promptpulse.evaluation import evaluate_response


def test_dataset_schema_and_unique_ids():
    rows = load_dataset()
    assert len(rows) >= 4
    assert len({row["id"] for row in rows}) == len(rows)


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
