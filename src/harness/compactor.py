"""Context & Budget Controller: Token tracking, Procedural Memory loading, and Auto-Compaction."""

import logging
import sqlite3

from src.config import get_settings
from src.harness.llm import generate_text, has_valid_gemini_key
from src.harness.tools import get_procedural_rules
from src.observability.tracing import observe, record_generation_metrics, update_span_io
from src.schemas.state import HarnessState, StateSnapshot

logger = logging.getLogger(__name__)


class ContextBudgetController:
    """Monitors Working Memory token usage, loads Procedural Memory, and executes Auto-Compaction.

    Data Flow:
        1. At session start (`iteration_count == 0` and empty `procedural_rules`), queries
           `EpisodicProceduralStore` via MCP tool `get_procedural_rules(query=state.user_query)`
           and injects matched strategies into `HarnessState.procedural_rules`.
        2. Computes token footprint of `HarnessState` (`user_query`, `scratchpad`,
           `procedural_rules`, `evidence_cards`, and `snapshots`).
        3. Checks whether `current_tokens >= token_budget_limit * compaction_threshold` (default 70%).
        4. If triggered, compresses `scratchpad` and older `evidence_cards` into a compact
           `StateSnapshot` (using Gemini LLM summarization when enabled, or deterministic synthesis
           in offline mode) while deterministically preserving 100% of `preserved_citations`.
    """

    def __init__(
        self,
        token_budget_limit: int = 32000,
        compaction_threshold: float = 0.70,
        use_llm: bool | None = None,
    ) -> None:
        self.token_budget_limit = token_budget_limit
        self.compaction_threshold = compaction_threshold
        self.use_llm = use_llm

    @property
    def trigger_token_limit(self) -> int:
        """Return the exact token threshold that triggers Auto-Compaction."""
        return int(self.token_budget_limit * self.compaction_threshold)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast heuristic token estimator (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def compute_state_tokens(self, state: HarnessState) -> int:
        """Compute total estimated tokens currently occupying Working Memory."""
        total = self.estimate_tokens(state.user_query) + self.estimate_tokens(state.scratchpad)
        for rule in state.procedural_rules:
            total += self.estimate_tokens(rule)
        for card in state.evidence_cards:
            total += card.card_token_estimate or self.estimate_tokens(card.summary)
        for snap in state.snapshots:
            total += snap.tokens_after
        return total

    def should_compact(self, state: HarnessState) -> bool:
        """Return True if Working Memory exceeds the compaction threshold."""
        current_tokens = self.compute_state_tokens(state)
        return current_tokens >= self.trigger_token_limit

    def load_procedural_memory(self, state: HarnessState, limit: int = 5) -> list[str]:
        """Load matching ProceduralRule instructions from Tier-3 Memory at session start."""
        if state.procedural_rules:
            return state.procedural_rules
        try:
            rule_dicts = get_procedural_rules(query=state.user_query, limit=limit)
            return [
                f"[{r['domain_pattern']}] {r['strategy_instruction']}"
                for r in rule_dicts
                if r.get("strategy_instruction")
            ]
        except (sqlite3.Error, RuntimeError, OSError) as exc:
            logger.debug("Procedural memory lookup skipped: %s", exc)
            return []

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        return has_valid_gemini_key()

    @observe(
        name="auto-compaction-llm-summarizer",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    def _summarize_with_llm(
        self,
        state: HarnessState,
        fallback_summary: str,
        preserved_citations: list[str],
    ) -> str:
        """Condense scratchpad and evidence cards via Gemini while retaining inline citations."""
        settings = get_settings()
        cards_text = "\n".join(
            f"- [{c.sub_query_id}] ({c.sub_query}): {c.summary} "
            f"[Citations: {', '.join(c.retrieved_chunk_ids)}]"
            for c in state.evidence_cards
        )
        prompt = (
            f"Research Query: {state.user_query}\n"
            f"Scratchpad:\n{state.scratchpad[:2000]}\n\n"
            f"Evidence Cards:\n{cards_text}\n\n"
            f"Compress all findings above into a single high-density paragraph (<= 120 words). "
            f"Keep key numbers, dates, and inline [chunk_id] citations."
        )
        try:
            result = generate_text(
                prompt,
                system_instruction=(
                    "You are a Context Compactor for an autonomous research agent. "
                    "Produce a concise, factual summary preserving dates, metrics, and [chunk_id] tags."
                ),
                settings=settings,
            )
            if result.value:
                record_generation_metrics(
                    name=f"compaction-step-{state.iteration_count}",
                    model=result.model,
                    input_data={"prompt": prompt, "preserved_citations": preserved_citations},
                    output_data={"condensed_reasoning": result.value},
                    input_tokens=result.input_tokens or self.estimate_tokens(prompt),
                    output_tokens=result.output_tokens or self.estimate_tokens(result.value),
                )
                return f"Compacted Step {state.iteration_count}: {result.value}"
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("LLM compaction failed (%s); using deterministic fallback summary.", exc)
        return fallback_summary

    @observe(
        name="context-budget-controller",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def compact_state(self, state: HarnessState) -> HarnessState:
        """Load procedural rules at session start and compress Working Memory when >= 70% budget."""
        loaded_rules = state.procedural_rules
        if state.iteration_count == 0 and not loaded_rules:
            loaded_rules = self.load_procedural_memory(state)
            if loaded_rules:
                state = state.model_copy(update={"procedural_rules": loaded_rules})

        tokens_before = self.compute_state_tokens(state)
        if tokens_before < self.trigger_token_limit:
            update_span_io(
                input_data={
                    "iteration": state.iteration_count,
                    "tokens_before": tokens_before,
                    "trigger_token_limit": self.trigger_token_limit,
                    "procedural_rules_loaded": len(loaded_rules),
                },
                output_data={
                    "compaction_triggered": False,
                    "tokens_after": tokens_before,
                },
                metadata={
                    "token_budget_limit": self.token_budget_limit,
                    "compaction_threshold": self.compaction_threshold,
                },
            )
            return state.model_copy(update={"current_token_estimate": tokens_before})

        # Deterministic 100% citation preservation across verified claims, prior snapshots, and cards
        preserved_citations = sorted(
            {claim.source_chunk_id for claim in state.verified_claims}
            | {cid for snap in state.snapshots for cid in snap.preserved_citations}
            | {cid for card in state.evidence_cards for cid in card.retrieved_chunk_ids}
        )

        bullet_summaries = [
            f"[{card.sub_query_id}] {card.summary[:200]} (Citations: {', '.join(card.retrieved_chunk_ids)})"
            for card in state.evidence_cards
        ]
        if not bullet_summaries and state.scratchpad:
            bullet_summaries = [state.scratchpad[:240]]

        fallback_reasoning = (
            f"Compacted Step {state.iteration_count}: " + " | ".join(bullet_summaries)
        )
        condensed_reasoning = (
            self._summarize_with_llm(state, fallback_reasoning, preserved_citations)
            if self._should_use_llm()
            else fallback_reasoning
        )
        tokens_after_snap = self.estimate_tokens(condensed_reasoning)

        snapshot = StateSnapshot(
            compaction_step=state.iteration_count,
            condensed_reasoning=condensed_reasoning,
            preserved_citations=preserved_citations,
            tokens_before=tokens_before,
            tokens_after=tokens_after_snap,
        )

        updated_state = state.model_copy(
            update={
                "snapshots": [*state.snapshots, snapshot],
                "scratchpad": "",
                "evidence_cards": [],
            }
        )
        new_total = self.compute_state_tokens(updated_state)
        update_span_io(
            input_data={
                "iteration": state.iteration_count,
                "tokens_before": tokens_before,
                "trigger_token_limit": self.trigger_token_limit,
                "evidence_cards_count": len(state.evidence_cards),
            },
            output_data={
                "compaction_triggered": True,
                "tokens_after": new_total,
                "tokens_saved": tokens_before - new_total,
                "preserved_citations": preserved_citations,
            },
            metadata={
                "token_budget_limit": self.token_budget_limit,
                "compaction_threshold": self.compaction_threshold,
            },
        )
        return updated_state.model_copy(update={"current_token_estimate": new_total})
