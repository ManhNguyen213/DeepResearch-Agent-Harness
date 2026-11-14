"""Query Planner node: Decomposes complex multi-hop queries into SubQueryTask items.

Data Flow:
    1. Checks `TemporalSemanticStore` (via `search_temporal_memory` / `query_temporal_memory`)
       for verified facts already known from previous sessions so the Planner avoids redundant
       retrieval and records pre-loaded facts in `scratchpad` / `verified_claims`.
    2. Injects loaded `ProceduralRule` instructions and any prior `StateSnapshot` context into the
       planning prompt.
    3. Uses Gemini Structured Outputs (`PlanDecompositionSchema`) to decompose complex questions
       into 2-4 independent `SubQueryTask` objects (with deterministic heuristic decomposition
       when running offline in unit tests).
"""

import logging
import re
import sqlite3

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_settings
from src.harness.compactor import ContextBudgetController
from src.harness.llm import generate_structured, has_valid_gemini_key
from src.harness.tools import query_temporal_memory, search_temporal_memory
from src.observability.tracing import observe, record_generation_metrics, update_span_io
from src.schemas.evidence import EvidenceClaim
from src.schemas.state import HarnessState, SubQueryTask

logger = logging.getLogger(__name__)


class PlannedSubQueryItem(BaseModel):
    """Schema item for Gemini structured query decomposition."""

    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(..., description="Targeted, self-contained search query")
    target_entities: list[str] = Field(
        default_factory=list,
        description="Primary entities or concepts targeted by this sub-query",
    )


class PlanDecompositionSchema(BaseModel):
    """Top-level structured output returned by Gemini when decomposing a multi-hop query."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description="Brief explanation of how the user query and existing memory facts shaped the plan",
    )
    sub_queries: list[PlannedSubQueryItem] = Field(
        ...,
        description="2 to 4 independent sub-queries covering missing aspects of the user query",
    )


_STOPWORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "analyze",
    "compare",
    "explain",
    "describe",
    "between",
    "across",
    "from",
    "with",
    "into",
    "about",
    "the",
    "and",
    "for",
}


class QueryPlanner:
    """Decomposes a user research query into 2-4 targeted sub-queries informed by 3-Tier Memory."""

    def __init__(self, use_llm: bool | None = None, max_sub_queries: int = 4) -> None:
        self.use_llm = use_llm
        self.max_sub_queries = max_sub_queries

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        return has_valid_gemini_key()

    @staticmethod
    def _extract_candidate_entities(query: str) -> list[str]:
        """Extract capitalized entity tokens/phrases for fast Temporal Memory lookup."""
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", query)
        entities: list[str] = []
        for tok in tokens:
            if tok.lower() not in _STOPWORDS and tok not in entities:
                entities.append(tok)
        return entities[:5]

    def lookup_temporal_memory(self, query: str) -> list[dict]:
        """Check Temporal Semantic Memory for existing ACTIVE facts relevant to the user query."""
        found_by_id: dict[str, dict] = {}
        try:
            for hit in search_temporal_memory(query=query, top_k=5, active_only=True):
                if float(hit.get("score", 0.0)) >= 0.25:
                    found_by_id[str(hit["memory_id"])] = hit
        except (sqlite3.Error, RuntimeError, ValueError, OSError) as exc:
            logger.debug("Semantic memory search skipped: %s", exc)

        for ent in self._extract_candidate_entities(query):
            try:
                for rec in query_temporal_memory(entity=ent, active_only=True):
                    found_by_id[str(rec["memory_id"])] = rec
            except (sqlite3.Error, RuntimeError, ValueError, OSError) as exc:
                logger.debug("Entity memory lookup skipped for %s: %s", ent, exc)
        return list(found_by_id.values())

    def _heuristic_decompose(
        self,
        user_query: str,
        known_facts: list[dict],
        iteration: int,
    ) -> list[SubQueryTask]:
        """Deterministic 2-4 sub-query decomposition used when offline or as fallback."""
        entities = self._extract_candidate_entities(user_query)
        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", user_query)))

        raw_clauses = [
            c.strip(" ,.;:?!-")
            for c in re.split(
                r"\b(?:and|vs\.?|versus|compared to|as well as|while)\b|[?;]",
                user_query,
                flags=re.IGNORECASE,
            )
            if len(c.strip()) > 8
        ]

        planned_texts: list[tuple[str, list[str]]] = []
        if len(years) >= 2:
            for yr in years[: self.max_sub_queries]:
                planned_texts.append((f"{yr}: {user_query}", entities))
        elif len(raw_clauses) >= 2:
            for clause in raw_clauses[: self.max_sub_queries]:
                clause_ents = self._extract_candidate_entities(clause) or entities
                planned_texts.append((clause, clause_ents))
        elif len(entities) >= 2:
            for ent in entities[: self.max_sub_queries]:
                planned_texts.append((f"{ent}: {user_query}", [ent]))

        if len(planned_texts) < 2:
            primary_ent = entities or ["General_Topic"]
            planned_texts = [
                (f"Core definitions, architecture, and primary facts: {user_query}", primary_ent),
                (f"Historical evolution, comparisons, and empirical details: {user_query}", primary_ent),
            ]

        # Only clean search text goes into query_text: it is fed verbatim to BM25 and dense search.
        tasks: list[SubQueryTask] = []
        prefix = f"sq_iter{iteration}_" if iteration > 1 else "sq_"
        for idx, (q_text, q_ents) in enumerate(planned_texts[: self.max_sub_queries], start=1):
            tasks.append(
                SubQueryTask(
                    sub_query_id=f"{prefix}{idx}",
                    query_text=q_text,
                    target_entities=q_ents,
                    completed=False,
                )
            )
        return tasks

    @observe(
        name="planner-llm-decomposition",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    def _decompose_with_llm(
        self,
        state: HarnessState,
        known_facts: list[dict],
        next_iter: int,
    ) -> tuple[list[SubQueryTask], str]:
        """Decompose user query into 2-4 SubQueryTask items via Gemini Structured Output."""
        settings = get_settings()
        memory_block = (
            "\n".join(
                f"- {f['entity']}.{f['attribute']} = {f['value']} "
                f"(valid_from={f['valid_from']}, chunk={f['source_chunk_id']})"
                for f in known_facts
            )
            if known_facts
            else "None"
        )
        rules_block = "\n".join(f"- {r}" for r in state.procedural_rules) or "None"
        snapshots_block = (
            "\n".join(s.condensed_reasoning for s in state.snapshots) or "None"
        )

        prompt = (
            f"User Research Query: {state.user_query}\n\n"
            f"Already Verified Facts in Temporal Semantic Memory (DO NOT re-query these exact facts):\n"
            f"{memory_block}\n\n"
            f"Procedural Memory Search Strategies:\n{rules_block}\n\n"
            f"Prior Compacted Context Snapshots:\n{snapshots_block}\n\n"
            f"Decompose the remaining research needs into 2 to {self.max_sub_queries} independent, "
            f"specific sub-queries suitable for hybrid dense+BM25 search."
        )
        result = generate_structured(
            prompt,
            PlanDecompositionSchema,
            system_instruction=(
                "You are the Query Planner of a Deep Research Agent Harness. "
                "Break complex multi-hop questions into 2 to 4 focused sub-queries, "
                "skipping facts already present in Temporal Semantic Memory."
            ),
            settings=settings,
        )
        parsed: PlanDecompositionSchema = result.value

        prefix = f"sq_iter{next_iter}_" if next_iter > 1 else "sq_"
        items = parsed.sub_queries[: self.max_sub_queries]
        tasks = [
            SubQueryTask(
                sub_query_id=f"{prefix}{idx}",
                query_text=item.query_text,
                target_entities=item.target_entities,
                completed=False,
            )
            for idx, item in enumerate(items, start=1)
        ]
        record_generation_metrics(
            name=f"query-decomposition-iter-{next_iter}",
            model=result.model,
            input_data={"prompt": prompt, "known_facts_count": len(known_facts)},
            output_data=parsed.model_dump(),
            input_tokens=result.input_tokens or ContextBudgetController.estimate_tokens(prompt),
            output_tokens=result.output_tokens
            or ContextBudgetController.estimate_tokens(parsed.reasoning),
        )
        return tasks, parsed.reasoning

    @observe(
        name="query-planner",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def plan(self, state: HarnessState) -> dict:
        """Check Temporal Memory and generate 2-4 SubQueryTask items for Isolated Sub-agents."""
        next_iter = state.iteration_count + 1

        # If caller pre-populated pending sub_queries and we are on iteration 0, run them directly
        pending_existing = [sq for sq in state.sub_queries if not sq.completed]
        if pending_existing and state.iteration_count == 0:
            update_span_io(
                input_data={
                    "user_query": state.user_query,
                    "existing_sub_queries_count": len(pending_existing),
                },
                output_data={
                    "iteration_count": next_iter,
                    "sub_queries": [sq.model_dump() for sq in state.sub_queries],
                },
            )
            return {"iteration_count": next_iter}

        known_facts = self.lookup_temporal_memory(state.user_query)
        memory_claims = [
            EvidenceClaim(
                entity=str(f["entity"]),
                attribute=str(f["attribute"]),
                value=str(f["value"]),
                source_chunk_id=str(f["source_chunk_id"]),
                valid_from=f.get("valid_from"),
                confidence_score=float(f.get("confidence_score", 0.95)),
            )
            for f in known_facts
        ]

        planner_note = ""
        tasks: list[SubQueryTask] = []
        if self._should_use_llm():
            try:
                tasks, planner_note = self._decompose_with_llm(state, known_facts, next_iter)
            except (ValidationError, RuntimeError, ValueError, OSError) as exc:
                logger.warning("LLM planner failed (%s); using heuristic decomposition.", exc)
                tasks = []

        if not tasks:
            tasks = self._heuristic_decompose(state.user_query, known_facts, next_iter)
            planner_note = (
                f"Decomposed query into {len(tasks)} sub-queries "
                f"(reused {len(known_facts)} facts from Temporal Semantic Memory)."
            )

        existing_ids = {sq.sub_query_id for sq in state.sub_queries}
        merged_sub_queries = [
            sq.model_copy(update={"completed": True}) for sq in state.sub_queries
        ] + [t for t in tasks if t.sub_query_id not in existing_ids]

        scratchpad_parts = [p for p in [state.scratchpad, planner_note] if p]
        if known_facts:
            mem_summary = "; ".join(
                f"{f['entity']}.{f['attribute']}={f['value']} [{f['source_chunk_id']}]"
                for f in known_facts
            )
            scratchpad_parts.append(f"Memory Hits: {mem_summary}")
        updated_scratchpad = "\n".join(scratchpad_parts)

        existing_claim_keys = {
            (c.entity, c.attribute, c.source_chunk_id) for c in state.verified_claims
        }
        merged_verified = list(state.verified_claims) + [
            mc
            for mc in memory_claims
            if (mc.entity, mc.attribute, mc.source_chunk_id) not in existing_claim_keys
        ]

        update_span_io(
            input_data={
                "user_query": state.user_query,
                "procedural_rules": state.procedural_rules,
                "temporal_memory_hits": len(known_facts),
            },
            output_data={
                "iteration_count": next_iter,
                "new_sub_queries": [t.model_dump() for t in tasks],
            },
        )
        return {
            "sub_queries": merged_sub_queries,
            "verified_claims": merged_verified,
            "scratchpad": updated_scratchpad,
            "iteration_count": next_iter,
        }
