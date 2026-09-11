"""Optional LLM-as-a-judge regression tests.

These tests run only when OPENAI_API_KEY is configured. The deterministic
PromptPulse tests always run, so contributors can validate PRs without a paid
judge API.
"""
from __future__ import annotations

import os

import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not configured; skipping DeepEval judge tests.", allow_module_level=True)

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from promptpulse.data import load_dataset


@pytest.mark.parametrize("row", load_dataset(), ids=lambda row: row["id"])
def test_expected_answer_with_deepeval(row):
    test_case = LLMTestCase(
        input=row["user_query"],
        actual_output=row["expected_answer"],
        context=[row["reference_context"]],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.70),
        HallucinationMetric(threshold=0.70),
    ]
    assert_test(test_case, metrics)
