"""Phase 2 gateway tests: LLM_ENABLED switch, model failover, schema sanitising, usage reporting.

All tests use a fake Gemini client; nothing here touches the network.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from src.config import get_settings
from src.harness import llm
from src.harness.llm import (
    LLMGatewayError,
    LLMResult,
    generate_structured,
    generate_text,
    has_valid_gemini_key,
    sanitize_json_schema,
)
from src.harness.subagent import IsolatedSubAgentWorker
from src.schemas.evidence import StructuredEvidenceCard


class _Answer(BaseModel):
    answer: str


class _FakeInteractions:
    """Fake ``client.interactions``: fails for models in ``failing`` and records every call."""

    def __init__(self, failing: set[str], output_text: str) -> None:
        self.failing = failing
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs["model"] in self.failing:
            raise RuntimeError(f"503 UNAVAILABLE for {kwargs['model']}")
        usage = SimpleNamespace(total_input_tokens=11, total_output_tokens=7)
        return SimpleNamespace(output_text=self.output_text, usage=usage)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, failing: set[str], output_text: str
) -> _FakeInteractions:
    interactions = _FakeInteractions(failing, output_text)
    monkeypatch.setattr(llm, "_get_client", lambda api_key: SimpleNamespace(interactions=interactions))
    return interactions


def test_llm_disabled_by_default_in_tests() -> None:
    """conftest sets LLM_ENABLED=false, so no component may reach the API by default."""
    get_settings.cache_clear()
    assert get_settings().llm_enabled is False
    assert has_valid_gemini_key() is False
    assert IsolatedSubAgentWorker()._should_use_llm() is False


def test_failover_reports_the_model_that_actually_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_client(monkeypatch, {"gemini-3.8-flash"}, "hello")
    result = generate_text("hi", model="gemini-3.8-flash")

    assert isinstance(result, LLMResult)
    assert result.value == "hello"
    assert result.model == "gemini-3.5-flash-lite"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert [c["model"] for c in fake.calls] == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert all("timeout" in c for c in fake.calls)


def test_all_models_failing_raises_gateway_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_client(monkeypatch, set(llm._FALLBACK_MODELS), "unused")
    with pytest.raises(LLMGatewayError):
        generate_text("hi", model="gemini-3.8-flash")
    assert len(fake.calls) == len(llm._FALLBACK_MODELS)


def test_generate_structured_validates_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_client(monkeypatch, set(), '{"answer": "42"}')
    result = generate_structured("q", _Answer, system_instruction="be brief")

    assert result.value == _Answer(answer="42")
    assert result.model == get_settings().default_llm_model
    sent = fake.calls[0]
    assert sent["system_instruction"] == "be brief"
    assert sent["response_format"]["mime_type"] == "application/json"
    assert "additionalProperties" not in str(sent["response_format"]["schema"])


def test_generate_structured_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(monkeypatch, set(), '{"wrong_field": 1}')
    with pytest.raises(LLMGatewayError):
        generate_structured("q", _Answer)


def test_sanitize_json_schema_inlines_refs_and_strips_unsupported_keys() -> None:
    schema = sanitize_json_schema(StructuredEvidenceCard.model_json_schema())
    text = str(schema)
    assert "$defs" not in text and "$ref" not in text
    assert "additionalProperties" not in text and "'title'" not in text
    claim_props = schema["properties"]["claims"]["items"]["properties"]
    assert "source_chunk_id" in claim_props
