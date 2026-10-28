"""Pytest configuration: Disable remote OpenTelemetry network export during offline unit tests."""

import os

os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
