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
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
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
