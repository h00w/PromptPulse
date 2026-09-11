from types import SimpleNamespace

import promptpulse.inference as inference


class _FakeCompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, *, model, **kwargs):
        self.calls.append(model)
        if model == "unsupported/example-model":
            raise RuntimeError(
                "Bad request: {'code': 'model_not_supported', "
                "'message': 'not supported by any provider you have enabled'}"
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Grounded fallback response")
                )
            ]
        )


class _FakeInferenceClient:
    calls: list[str] = []

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(self.calls))


def test_unsupported_preferred_model_uses_fallback(monkeypatch):
    _FakeInferenceClient.calls = []
    monkeypatch.setattr(inference, "InferenceClient", _FakeInferenceClient)
    monkeypatch.setattr(inference, "get_hf_token", lambda: "test-token")
    monkeypatch.setattr(
        inference,
        "FALLBACK_MODELS",
        ("supported/fallback-model",),
    )

    result = inference.generate_response_with_fallback(
        "Question?",
        "Approved context.",
        inference.GenerationConfig(model="unsupported/example-model"),
    )

    assert result.text == "Grounded fallback response"
    assert result.model == "supported/fallback-model"
    assert result.attempted_models == (
        "unsupported/example-model",
        "supported/fallback-model",
    )
    assert _FakeInferenceClient.calls == [
        "unsupported/example-model",
        "supported/fallback-model",
    ]


def test_non_compatibility_error_is_not_hidden(monkeypatch):
    class _AuthFailureCompletions:
        def create(self, **kwargs):
            raise RuntimeError("401 Unauthorized")

    class _AuthFailureClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_AuthFailureCompletions())

    monkeypatch.setattr(inference, "InferenceClient", _AuthFailureClient)
    monkeypatch.setattr(inference, "get_hf_token", lambda: "test-token")

    try:
        inference.generate_response_with_fallback(
            "Question?",
            "Approved context.",
            inference.GenerationConfig(model="some/model"),
        )
    except RuntimeError as exc:
        assert "401 Unauthorized" in str(exc)
    else:
        raise AssertionError("Authentication failures must be surfaced immediately")
