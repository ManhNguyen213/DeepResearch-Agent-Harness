"""Isolated Sub-agent Worker implementing Sub-agent Context Isolation (Context Forking)."""

import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_settings
from src.harness.compactor import ContextBudgetController
from src.harness.llm import generate_structured, has_valid_gemini_key
from src.harness.tools import hybrid_search
from src.observability.tracing import observe, record_generation_metrics, update_span_io
from src.schemas.evidence import EvidenceClaim, RetrievedChunk, StructuredEvidenceCard
from src.schemas.state import SubQueryTask

logger = logging.getLogger(__name__)

# Heuristic (non-LLM) claims are NOT entailment-checked. Keep their confidence below the
# Verification Gate threshold (0.80) so they can never be verified or committed to memory.
HEURISTIC_CLAIM_CONFIDENCE = 0.40


class _SubAgentLLMExtraction(BaseModel):
    """Structured output schema produced by Gemini inside the isolated sub-agent context."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        description="Concise synthesis (150-300 words) answering the sub-query with inline [chunk_id] citations",
    )
    claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description="Atomic factual claims grounded strictly in the provided source_chunk_ids",
    )


class IsolatedSubAgentWorker:
    """Executes a SubQueryTask inside an isolated context window and returns a StructuredEvidenceCard.

    Data Flow & Design Rationale:
        1. Initializes a strictly local ``sub_messages`` list that never touches ``HarnessState``.
        2. Calls the MCP tool ``hybrid_search(query=task.query_text, top_k=top_k)`` to retrieve
           5-10 raw document chunks (~3,000-5,000 tokens).
        3. Extracts a compact ``StructuredEvidenceCard`` (~300 tokens) + atomic ``EvidenceClaim``
           records with deterministic ``source_chunk_id`` and ``valid_from`` dates using Gemini
           Structured Outputs (or deterministic offline extraction during unit tests).
        4. Explicitly discards ``raw_chunks`` and ``sub_messages`` so only the compact
           ``StructuredEvidenceCard`` crosses back into the Main Agent's Working Memory.
    """

    def __init__(self, top_k: int = 5, use_llm: bool | None = None) -> None:
        self.top_k = top_k
        self.use_llm = use_llm

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        return has_valid_gemini_key()

    @staticmethod
    def _build_sub_messages(
        task: SubQueryTask,
        raw_chunks: list[RetrievedChunk],
    ) -> list[dict[str, str]]:
        """Construct the isolated sub-agent message list (never merged into Main Agent state)."""
        formatted_chunks = "\n\n".join(
            f"[chunk_id={c.chunk_id} | published_date={c.published_date or 'unknown'} | "
            f"title={c.document_title}]\n{c.content}"
            for c in raw_chunks
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an Isolated Research Sub-agent. Read the raw retrieved chunks and "
                    "extract a concise synthesis (150-300 words) with inline [chunk_id] citations "
                    "plus atomic EvidenceClaim items. Every claim MUST use an exact chunk_id "
                    "from the provided chunks and preserve valid_from dates when stated."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sub-Query ID: {task.sub_query_id}\n"
                    f"Sub-Query: {task.query_text}\n"
                    f"Target Entities: {', '.join(task.target_entities) or 'General'}\n\n"
                    f"Raw Retrieved Chunks:\n{formatted_chunks}"
                ),
            },
        ]

    @staticmethod
    def _estimate_card_tokens(summary: str, claims: list[EvidenceClaim]) -> int:
        claims_text = " ".join(
            f"{c.entity} {c.attribute} {c.value} {c.source_chunk_id}" for c in claims
        )
        return ContextBudgetController.estimate_tokens(f"{summary} {claims_text}")

    @observe(
        name="subagent-evidence-extraction",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    def _extract_structured_card(
        self,
        task: SubQueryTask,
        raw_chunks: list[RetrievedChunk],
        sub_messages: list[dict[str, str]],
    ) -> StructuredEvidenceCard:
        """Extract a compact StructuredEvidenceCard from isolated sub_messages and record metrics."""
        settings = get_settings()
        raw_text = "\n\n".join(m["content"] for m in sub_messages)
        raw_tokens = ContextBudgetController.estimate_tokens(raw_text)
        chunk_ids = [c.chunk_id for c in raw_chunks]

        extracted_claims: list[EvidenceClaim] = []
        summary = ""
        used_model: str | None = None
        in_tok = raw_tokens
        out_tok = 0
        llm_wanted = self._should_use_llm()

        if llm_wanted and raw_chunks:
            try:
                result = generate_structured(
                    sub_messages[1]["content"],
                    _SubAgentLLMExtraction,
                    system_instruction=sub_messages[0]["content"],
                    settings=settings,
                )
                summary = result.value.summary
                extracted_claims = result.value.claims
                used_model = result.model
                in_tok = result.input_tokens or in_tok
                out_tok = result.output_tokens
            except (ValidationError, RuntimeError, ValueError, OSError) as exc:
                logger.warning(
                    "Sub-agent %s: LLM extraction failed (%s); using DEGRADED heuristic card.",
                    task.sub_query_id,
                    exc,
                )
                summary = ""

        degraded = not summary
        if degraded:
            if not llm_wanted:
                logger.debug("Sub-agent %s: LLM disabled; using heuristic card.", task.sub_query_id)
            extracted_claims = [
                EvidenceClaim(
                    entity=task.target_entities[0] if task.target_entities else "General_Entity",
                    attribute="research_finding",
                    value=chunk.content[:160].strip(),
                    source_chunk_id=chunk.chunk_id,
                    valid_from=chunk.published_date,
                    confidence_score=HEURISTIC_CLAIM_CONFIDENCE,
                )
                for chunk in raw_chunks
            ]
            snippets = [
                f"{c.content[:90].strip()} [{c.chunk_id}]" for c in raw_chunks[:3]
            ]
            summary = (
                f"[DEGRADED heuristic] {len(raw_chunks)} chunks for '{task.query_text}': "
                + " | ".join(snippets)
            )

        card_tokens = ContextBudgetController.estimate_tokens(summary)
        if not out_tok:
            out_tok = card_tokens

        card = StructuredEvidenceCard(
            sub_query_id=task.sub_query_id,
            sub_query=task.query_text,
            summary=summary,
            claims=extracted_claims,
            retrieved_chunk_ids=chunk_ids,
            raw_tokens_consumed=raw_tokens,
            card_token_estimate=card_tokens,
            degraded=degraded,
            llm_model=used_model,
        )

        record_generation_metrics(
            name=f"evidence-card-extraction-{task.sub_query_id}",
            model=used_model or "heuristic-fallback",
            input_data={
                "sub_query": task.query_text,
                "retrieved_chunk_ids": chunk_ids,
                "raw_chunks_count": len(raw_chunks),
            },
            output_data=card.model_dump(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            metadata={
                "context_isolation_savings_tokens": max(0, raw_tokens - card_tokens),
            },
        )
        return card

    @observe(
        name="researcher-subagent",
        as_type="agent",
        capture_input=False,
        capture_output=False,
    )
    def execute_isolated(
        self,
        task: SubQueryTask,
        raw_chunks: list[RetrievedChunk] | None = None,
    ) -> StructuredEvidenceCard:
        """Process raw chunks in an isolated context and return only a StructuredEvidenceCard."""
        distinct_agent_name = f"researcher-subagent-{task.sub_query_id}"
        if raw_chunks is None:
            raw_dicts = hybrid_search(query=task.query_text, top_k=self.top_k)
            raw_chunks = [RetrievedChunk.model_validate(d) for d in raw_dicts]

        # Private sub-agent context window (Context Forking)
        sub_messages = self._build_sub_messages(task, raw_chunks)
        card = self._extract_structured_card(task, raw_chunks, sub_messages)

        # Explicitly clear private buffers before returning to Main Agent
        sub_messages.clear()
        raw_chunks = []

        update_span_io(
            name=distinct_agent_name,
            input_data={
                "sub_query_id": task.sub_query_id,
                "query_text": task.query_text,
                "target_entities": task.target_entities,
            },
            output_data={
                "sub_query_id": card.sub_query_id,
                "summary": card.summary,
                "claims_extracted": len(card.claims),
                "retrieved_chunk_ids": card.retrieved_chunk_ids,
            },
            metadata={
                "raw_tokens_discarded": card.raw_tokens_consumed,
                "card_token_estimate": card.card_token_estimate,
            },
        )
        return card
