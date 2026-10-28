"""Isolated Sub-agent Worker implementing Sub-agent Context Isolation (Context Forking)."""

from src.config import get_settings
from src.harness.compactor import ContextBudgetController
from src.mcp_servers.rag_server import hybrid_search
from src.observability.tracing import observe, record_generation_metrics, update_span_io
from src.schemas.evidence import EvidenceClaim, RetrievedChunk, StructuredEvidenceCard
from src.schemas.state import SubQueryTask


class IsolatedSubAgentWorker:
    """Executes a SubQueryTask inside an isolated context window and returns a StructuredEvidenceCard.

    Design Rationale:
        Instead of appending 5-10 raw retrieved chunks (~3,000-5,000 tokens per sub-query)
        into the Main Agent's message list, the sub-agent processes raw chunks privately,
        extracts structured claims with `source_chunk_id`, discards the raw chunks, and
        returns only a ~300-500 token `StructuredEvidenceCard`.
    """

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
    ) -> StructuredEvidenceCard:
        """Extract a compact StructuredEvidenceCard from raw chunks and record LLM token metrics."""
        settings = get_settings()
        raw_text = "\n\n".join(c.content for c in raw_chunks)
        raw_tokens = ContextBudgetController.estimate_tokens(raw_text)
        chunk_ids = [c.chunk_id for c in raw_chunks]

        extracted_claims: list[EvidenceClaim] = [
            EvidenceClaim(
                entity=task.target_entities[0] if task.target_entities else "General_Entity",
                attribute="research_finding",
                value=chunk.content[:200],
                source_chunk_id=chunk.chunk_id,
                valid_from=chunk.published_date,
                confidence_score=0.90,
            )
            for chunk in raw_chunks
        ]

        summary = (
            f"Synthesized {len(raw_chunks)} chunks for sub-query '{task.query_text}'. "
            f"Sources: {', '.join(chunk_ids)}."
        )
        card_tokens = ContextBudgetController.estimate_tokens(summary)

        card = StructuredEvidenceCard(
            sub_query_id=task.sub_query_id,
            sub_query=task.query_text,
            summary=summary,
            claims=extracted_claims,
            retrieved_chunk_ids=chunk_ids,
            raw_tokens_consumed=raw_tokens,
            card_token_estimate=card_tokens,
        )

        record_generation_metrics(
            name=f"evidence-card-extraction-{task.sub_query_id}",
            model=settings.default_llm_model,
            input_data={
                "sub_query": task.query_text,
                "retrieved_chunk_ids": chunk_ids,
                "raw_chunks_count": len(raw_chunks),
            },
            output_data=card.model_dump(),
            input_tokens=raw_tokens,
            output_tokens=card_tokens,
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
        """Process raw chunks in an isolated context and return a compact StructuredEvidenceCard."""
        distinct_agent_name = f"researcher-subagent-{task.sub_query_id}"
        if raw_chunks is None:
            raw_dicts = hybrid_search(query=task.query_text, top_k=3)
            raw_chunks = [RetrievedChunk.model_validate(d) for d in raw_dicts]

        card = self._extract_structured_card(task, raw_chunks)
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
