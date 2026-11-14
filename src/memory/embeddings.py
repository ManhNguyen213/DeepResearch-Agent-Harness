"""Pluggable text embedders shared by the Hybrid RAG server and the Temporal Semantic Store.

Data Flow:
    text -> ``Embedder.embed`` -> L2-normalised float vector -> Qdrant (cosine distance).

Design Rationale:
    Retrieval and memory code depend only on the small ``Embedder`` protocol, so unit tests
    run fully offline with the deterministic ``HashingEmbedder`` while production uses
    ``GeminiEmbedder``. No black-box vector-store wrapper is involved.
"""

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from src.config import HarnessSettings, get_settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector of length ``dim`` per input text."""
        ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class HashingEmbedder:
    """Deterministic, offline bag-of-words hashing embedder (feature hashing + L2 norm).

    Texts sharing tokens get high cosine similarity, which is sufficient for tests and
    offline development. It has no semantic generalisation across synonyms.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via signed feature hashing of lower-cased word tokens."""
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            out.append(_normalize(vec))
        return out


class GeminiEmbedder:
    """Embedder backed by the Gemini embedding API (``google-genai`` SDK)."""

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the configured Gemini embedding model."""
        from google.genai import types

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )
        return [_normalize(list(e.values or [])) for e in (response.embeddings or [])]


class SentenceTransformerEmbedder:
    """Local embedder via ``sentence-transformers`` (default: BAAI/bge-m3, as used in GR1).

    Runs offline after the first model download and needs no API quota.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        import os

        # Prefer the local cache without any Hub round-trips, but scope the offline flags to this
        # load only: leaving them set would silently break other Hub downloads in the process
        # (e.g. enabling the Cross-Encoder reranker later). The flags must be set *before*
        # sentence_transformers/huggingface_hub are first imported, because they read them then.
        offline_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        saved = {k: os.environ.get(k) for k in offline_keys}
        for k in offline_keys:
            os.environ[k] = "1"
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
            except OSError:
                for k in offline_keys:
                    os.environ[k] = "0"
                self._model = SentenceTransformer(model_name)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dim = int(get_dim() or 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; vectors are normalised by the model call."""
        vectors = self._model.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


_shared_embedder_cache: dict[tuple[str, str, int], Embedder] = {}


def get_embedder(settings: HarnessSettings | None = None) -> Embedder:
    """Build the embedder selected by ``settings.embedding_backend``."""
    settings = settings or get_settings()
    if settings.embedding_backend == "local":
        return SentenceTransformerEmbedder(settings.local_embedding_model)
    if settings.embedding_backend == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=gemini requires GEMINI_API_KEY to be set.")
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return HashingEmbedder(dim=settings.embedding_dim)


def get_shared_embedder(settings: HarnessSettings | None = None) -> Embedder:
    """Return a cached singleton embedder per (backend, model, dim) so BAAI/bge-m3 loads only once."""
    cfg = settings or get_settings()
    model_key = (
        cfg.local_embedding_model if cfg.embedding_backend == "local" else cfg.embedding_model
    )
    cache_key = (cfg.embedding_backend, model_key, cfg.embedding_dim)
    if cache_key not in _shared_embedder_cache:
        _shared_embedder_cache[cache_key] = get_embedder(cfg)
    return _shared_embedder_cache[cache_key]
