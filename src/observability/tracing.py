"""Layer 4: Langfuse Observability & Tracing Setup following Langfuse Best Practices.

Data Flow & Design Rationale:
    1. Loads `.env` credentials into `os.environ` BEFORE initializing the Langfuse OpenTelemetry client.
    2. Provides a singleton `get_langfuse_client()` and safe wrappers (`update_span_io`, `flush_traces`,
       `trace_harness_session`) that work seamlessly both when Langfuse Cloud is reachable and during
       offline unit testing.
    3. Enforces Langfuse Multi-Agent Best Practices:
       - Propagates `session_id` and `tags` via `propagate_attributes` for multi-session memory evaluation.
       - Explicitly sets clean `input`, `output`, and `metadata` via `update_current_span` so `self` or
         internal state buffers do not clutter the Langfuse UI.
       - Assigns semantic observation types (`agent`, `retriever`, `guardrail`, `chain`, `generation`).
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import truststore
from dotenv import load_dotenv

# Use native Windows OS Certificate Store for all SSL/TLS connections (httpx, requests, OTel, Gemini)
truststore.inject_into_ssl()

# IMPORTANT (Langfuse Best Practice): Load environment variables BEFORE importing/initializing Langfuse
load_dotenv()

from langfuse import Langfuse, get_client, observe, propagate_attributes

from src.config import get_settings


def _ensure_langfuse_env() -> bool:
    """Ensure Langfuse environment variables are populated from HarnessSettings if present."""
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False
    settings = get_settings()
    if settings.langfuse_public_key and "your_" not in settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key and "your_" not in settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    effective_url = os.environ.get("LANGFUSE_BASE_URL") or settings.langfuse_base_url or settings.langfuse_host
    if effective_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", effective_url)
        os.environ.setdefault("LANGFUSE_HOST", effective_url)
    return is_langfuse_configured()


def is_langfuse_configured() -> bool:
    """Return True if valid Langfuse public and secret keys are configured."""
    settings = get_settings()
    has_pk = bool(settings.langfuse_public_key and "your_" not in settings.langfuse_public_key)
    has_sk = bool(settings.langfuse_secret_key and "your_" not in settings.langfuse_secret_key)
    return has_pk and has_sk


def get_langfuse_client() -> Langfuse:
    """Return the initialized global Langfuse client."""
    _ensure_langfuse_env()
    return get_client()


def update_span_io(
    *,
    name: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
    status_message: str | None = None,
) -> None:
    """Explicitly set clean input, output, metadata, and name on the current Langfuse span."""
    if not _ensure_langfuse_env():
        return
    client = get_client()
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if input_data is not None:
        kwargs["input"] = input_data
    if output_data is not None:
        kwargs["output"] = output_data
    if metadata is not None:
        kwargs["metadata"] = metadata
    if level is not None:
        kwargs["level"] = level
    if status_message is not None:
        kwargs["status_message"] = status_message
    if kwargs:
        client.update_current_span(**kwargs)


def record_generation_metrics(
    *,
    name: str,
    model: str,
    input_data: Any,
    output_data: Any,
    input_tokens: int,
    output_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update the active `generation` observation with model name and token usage details."""
    if not _ensure_langfuse_env():
        return
    client = get_client()
    client.update_current_generation(
        name=name,
        model=model,
        input=input_data,
        output=output_data,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        metadata=metadata or {},
    )


@contextmanager
def trace_harness_session(
    *,
    session_id: str,
    user_id: str = "gr2-researcher",
    trace_name: str = "deep-research-harness-run",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Propagate session_id, user_id, trace_name, and tags across all nested observations."""
    _ensure_langfuse_env()
    effective_tags = tags or ["gr2-deep-research", "agent-harness", "3-tier-memory"]
    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=effective_tags,
        metadata=metadata or {},
    ):
        yield


def flush_traces() -> None:
    """Flush all pending OpenTelemetry spans to Langfuse before process exit."""
    if _ensure_langfuse_env():
        get_client().flush()


__all__ = [
    "flush_traces",
    "get_langfuse_client",
    "is_langfuse_configured",
    "observe",
    "record_generation_metrics",
    "trace_harness_session",
    "update_span_io",
]
