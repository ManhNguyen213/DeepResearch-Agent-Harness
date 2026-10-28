"""Ingestion: raw ``.txt`` documents -> deterministic, citation-ready chunks in JSONL.

Data Flow:
    data/raw/<corpus>/*.txt -> paragraph-aware chunking (with overlap) -> ``RetrievedChunk`` rows
    -> data/corpus/<name>.jsonl (consumed by ``HybridRetriever``).

Design Rationale:
    Re-engineered from GR1's ``split_text_chunks`` (fixed character windows). Chunks now respect
    paragraph/sentence boundaries and get a stable ``chunk_id = doc_<slug>_c<NNNN>`` that depends
    only on document name and chunk position, so citations survive re-ingestion. Wikipedia pages
    have no publication date, so ``published_date`` stays None unless a ``dates.json`` mapping
    ``{filename_stem: "YYYY-MM-DD"}`` exists next to the raw files.
"""

import argparse
import json
import re
from pathlib import Path

from src.config import get_settings
from src.schemas.evidence import RetrievedChunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"[ \t]+")


def slugify(name: str) -> str:
    """Lower-case ASCII slug used inside ``chunk_id``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _units(text: str, max_chars: int) -> list[str]:
    """Split text into paragraphs, further split into sentences if a paragraph is too long."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = _WS_RE.sub(" ", para).strip()
        if len(para) < 3:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return units


def chunk_text(text: str, max_chars: int = 1500, overlap_chars: int = 150) -> list[str]:
    """Pack paragraph/sentence units into chunks of at most ~``max_chars`` characters.

    The tail of each chunk (up to ``overlap_chars``, whole units only) is repeated at the start of
    the next one to preserve context across boundaries.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in _units(text, max_chars):
        if current and size + len(unit) + 1 > max_chars:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev)
            current, size = tail, tail_size
        current.append(unit)
        size += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def ingest_directory(
    raw_dir: Path | str,
    out_file: Path | str,
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> int:
    """Chunk all ``*.txt`` files in ``raw_dir`` and write them to ``out_file`` (JSONL).

    Returns:
        Number of chunks written.
    """
    raw_dir = Path(raw_dir)
    dates_file = raw_dir / "dates.json"
    dates: dict[str, str] = json.loads(dates_file.read_text("utf-8")) if dates_file.exists() else {}
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for file in sorted(raw_dir.glob("*.txt")):
            slug = slugify(file.stem)
            title = file.stem.replace("_", " ")
            text = file.read_text(encoding="utf-8", errors="ignore")
            for idx, content in enumerate(chunk_text(text, max_chars, overlap_chars)):
                chunk = RetrievedChunk(
                    chunk_id=f"doc_{slug}_c{idx:04d}",
                    content=content,
                    document_title=title,
                    published_date=dates.get(file.stem),
                )
                fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
                total += 1
    return total


def main() -> None:
    """CLI: ``uv run python -m src.retrieval.ingest``."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk raw .txt documents into a JSONL corpus.")
    parser.add_argument("--raw-dir", default=str(settings.rag_raw_dir))
    parser.add_argument("--out", default=str(settings.rag_corpus_path / "wikipedia_ai_ml.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--overlap-chars", type=int, default=150)
    args = parser.parse_args()
    count = ingest_directory(args.raw_dir, args.out, args.max_chars, args.overlap_chars)
    print(f"Wrote {count} chunks to {args.out}")


if __name__ == "__main__":
    main()
