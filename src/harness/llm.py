"""Gemini LLM Gateway using the official ``google-genai`` Interactions API with Structured Outputs.

Data Flow:
    Prompt + Pydantic ``BaseModel`` schema -> sanitized JSON Schema (inlined ``$defs``, stripped
    ``additionalProperties``) -> ``client.interactions.create(...)`` with
    ``response_format={"type": "text", "mime_type": "application/json", "schema": ...}`` ->
    validated Pydantic instance wrapped in ``LLMResult(value, input_tokens, output_tokens, model)``.

Design Rationale:
    1. Uses the ``google-genai`` Interactions API and ``gemini-3.8-flash`` by default, with a
       model cascade (``gemini-3.5-flash-lite`` -> ``gemini-3.1-flash-lite``) when the primary model
       is overloaded (503) or unavailable.
    2. ``LLMResult.model`` reports the model that *actually* answered, so tracing and benchmarks
       never attribute a fallback model's output to the primary model. Every failover is logged at
       WARNING level.
    3. The SDK's internal retry/backoff is disabled so a failover happens quickly. This touches a
       private SDK module, so it is isolated in ``_disable_sdk_retries`` and degrades gracefully
       (retries simply stay enabled) if the SDK layout changes.
    4. ``LLM_ENABLED=false`` turns the whole gateway off (``has_valid_gemini_key`` returns False),
       which keeps unit tests hermetic and lets the Harness run offline.
    5. Pydantic JSON schemas are sanitized because the API rejects ``additionalProperties``.
    6. ``truststore`` makes Windows' OS certificate store work for TLS.
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple, TypeVar

import httpx
import truststore
from pydantic import BaseModel, ValidationError

from src.config import HarnessSettings, get_settings

truststore.inject_into_ssl()
logger = logging.getLogger(__name__)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

_FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)


class LLMResult(NamedTuple):
    """Result of one gateway call, including which model really produced it."""

    value: Any
    input_tokens: int
    output_tokens: int
    model: str


class LLMGatewayError(RuntimeError):
    """Raised when the Gemini Interactions API call fails across all candidate models."""


def _sdk_error_types() -> tuple[type[BaseException], ...]:
    """Return the SDK's own API error class (private module) when importable."""
    try:
        from google.genai._gaos.lib.compat_errors import APIError

        return (APIError,)
    except ImportError:
        return ()


_RECOVERABLE_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    RuntimeError,
    ValueError,
    OSError,
    *_sdk_error_types(),
)


def has_valid_gemini_key(settings: HarnessSettings | None = None) -> bool:
    """Return True if the LLM is enabled and a non-placeholder Gemini API key is configured."""
    cfg = settings or get_settings()
    if not cfg.llm_enabled:
        return False
    key = (cfg.gemini_api_key or "").strip()
    return bool(key and "your_" not in key and key != "test")


def sanitize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs``/``$ref`` and strip unsupported keywords for the Gemini Interactions API."""
    defs = schema.get("$defs", {})

    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = str(node["$ref"]).split("/")[-1]
                if ref_name in defs:
                    return _clean(defs[ref_name])
            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k in {"$defs", "additionalProperties", "title"}:
                    continue
                cleaned[k] = _clean(v)
            return cleaned
        if isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(schema)


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> Any:
    """Create (once per API key) a Gemini client with fast-failover retry settings."""
    from google import genai

    client = genai.Client(api_key=api_key)
    _disable_sdk_retries(client)
    return client


def _disable_sdk_retries(client: Any) -> None:
    """Turn off the SDK's internal retry/backoff so failover to the next model is quick.

    Uses a private SDK module; if that layout changes we keep the default retries.
    """
    try:
        from google.genai._gaos.utils import retries as gaos_retries

        client.interactions.sdk_configuration.retry_config = gaos_retries.RetryConfig(
            "none", [], False
        )
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not disable SDK retries (%s); failover may be slower.", exc)


def _candidate_models(primary: str) -> list[str]:
    candidates = [primary]
    for m in _FALLBACK_MODELS:
        if m not in candidates:
            candidates.append(m)
    return candidates


def _invoke_with_fallback(
    client: Any,
    primary_model: str,
    base_kwargs: dict[str, Any],
    timeout_sec: float,
) -> tuple[Any, str]:
    """Call ``client.interactions.create`` across candidate models; return ``(interaction, model)``."""
    last_exc: BaseException | None = None
    for candidate in _candidate_models(primary_model):
        try:
            interaction = client.interactions.create(
                **base_kwargs, model=candidate, timeout=timeout_sec
            )
        except _RECOVERABLE_ERRORS as exc:
            last_exc = exc
            logger.warning("Model %s failed (%s); trying next fallback model.", candidate, exc)
            continue
        if candidate != primary_model:
            logger.warning("Primary model %s unavailable; answered by %s.", primary_model, candidate)
        return interaction, candidate
    raise LLMGatewayError(str(last_exc)) from last_exc


def _usage_tokens(interaction: Any) -> tuple[int, int]:
    usage = getattr(interaction, "usage", None)
    return (
        int(getattr(usage, "total_input_tokens", 0) or 0),
        int(getattr(usage, "total_output_tokens", 0) or 0),
    )


def generate_structured(
    prompt: str,
    schema: type[M],
    *,
    system_instruction: str | None = None,
    model: str | None = None,
    settings: HarnessSettings | None = None,
) -> LLMResult:
    """Call Gemini with a Pydantic JSON schema; ``LLMResult.value`` is a validated ``schema`` instance."""
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {
        "input": prompt,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": sanitize_json_schema(schema.model_json_schema()),
        },
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction, used_model = _invoke_with_fallback(
        _get_client(cfg.gemini_api_key),
        model or cfg.default_llm_model,
        kwargs,
        cfg.llm_timeout_seconds,
    )
    try:
        parsed = schema.model_validate_json(interaction.output_text or "{}")
    except ValidationError as exc:
        raise LLMGatewayError(f"Model {used_model} returned invalid JSON: {exc}") from exc
    in_tok, out_tok = _usage_tokens(interaction)
    return LLMResult(parsed, in_tok, out_tok, used_model)


def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    model: str | None = None,
    settings: HarnessSettings | None = None,
) -> LLMResult:
    """Call Gemini for plain-text/Markdown generation; ``LLMResult.value`` is the stripped text."""
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {"input": prompt}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction, used_model = _invoke_with_fallback(
        _get_client(cfg.gemini_api_key),
        model or cfg.default_llm_model,
        kwargs,
        cfg.llm_timeout_seconds,
    )
    in_tok, out_tok = _usage_tokens(interaction)
    return LLMResult((interaction.output_text or "").strip(), in_tok, out_tok, used_model)
