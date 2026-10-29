"""Application configuration loaded from environment variables via Pydantic v2 Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    """Centralized settings for the Agent Harness, MCP tools, and 3-Tier Memory Engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    gemini_api_key: str = Field(default="", description="Google Gemini API Key")
    default_llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Default foundation model for planning and sub-agent extraction",
    )

    # Observability (Langfuse)
    langfuse_public_key: str = Field(default="", description="Langfuse Public Key")
    langfuse_secret_key: str = Field(default="", description="Langfuse Secret Key")
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse Base URL (US or EU cloud or self-hosted)",
    )
    langfuse_host: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse host URL alias",
    )

    # Storage & 3-Tier Memory Engine
    sqlite_db_path: Path = Field(
        default=Path("./data/cognitive_memory.db"),
        description="Path to SQLite database storing Temporal Semantic & Episodic tables",
    )
    qdrant_local_path: Path = Field(
        default=Path("./data/qdrant_storage"),
        description="Local path for Qdrant vector storage",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Optional remote Qdrant server URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional Qdrant API Key",
    )

    # Embeddings & Retrieval
    embedding_backend: Literal["local", "gemini", "hash"] = Field(
        default="local",
        description="Embedder: 'local' (sentence-transformers), 'gemini' (API) or 'hash' (offline tests)",
    )
    local_embedding_model: str = Field(
        default="BAAI/bge-m3", description="sentence-transformers model for the 'local' backend"
    )
    embedding_model: str = Field(default="gemini-embedding-001", description="Gemini embedding model")
    embedding_dim: int = Field(
        default=768, ge=16, description="Vector size for 'gemini'/'hash' backends (local infers it)"
    )
    rag_corpus_path: Path = Field(
        default=Path("./data/corpus"),
        description="Directory of .jsonl chunk files ingested by the Hybrid RAG server",
    )
    rag_raw_dir: Path = Field(
        default=Path("./data/raw/wikipedia_ai_ml"),
        description="Directory of raw .txt documents consumed by the ingestion CLI",
    )
    reranker_model: str | None = Field(
        default=None,
        description="Optional Cross-Encoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2'); None disables reranking",
    )

    # Harness & Context Controller Parameters
    token_budget_limit: int = Field(
        default=32000,
        ge=1024,
        description="Maximum token budget for the Main Agent Working Memory",
    )
    compaction_threshold: float = Field(
        default=0.70,
        ge=0.1,
        le=0.95,
        description="Fraction of token_budget_limit that triggers Auto-Compaction",
    )
    grounding_confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required by the Anti-Poisoning Verification Gate",
    )
    max_research_iterations: int = Field(
        default=4,
        ge=1,
        description="Maximum Planner -> Sub-agent -> Verifier loop iterations",
    )


@lru_cache(maxsize=1)
def get_settings() -> HarnessSettings:
    """Return a cached instance of HarnessSettings."""
    return HarnessSettings()
