"""Context & Budget Controller: Token tracking, Procedural Memory loading, and Auto-Compaction."""

from src.observability.tracing import observe, update_span_io
from src.schemas.state import HarnessState, StateSnapshot


class ContextBudgetController:
    """Monitors Working Memory token usage and executes Auto-Compaction when exceeding threshold.

    Data Flow:
        1. Approximates or counts tokens in `HarnessState` (scratchpad + evidence_cards + snapshots).
        2. Checks whether `current_tokens >= token_budget_limit * compaction_threshold`.
        3. If triggered, compresses older `evidence_cards` and `scratchpad` text into a compact
           `StateSnapshot` while preserving every verified `source_chunk_id`.
    """

    def __init__(
        self,
        token_budget_limit: int = 32000,
        compaction_threshold: float = 0.70,
    ) -> None:
        self.token_budget_limit = token_budget_limit
        self.compaction_threshold = compaction_threshold

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

    @observe(
        name="context-budget-controller",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def compact_state(self, state: HarnessState) -> HarnessState:
        """Compress Working Memory into a StateSnapshot while preserving all verified citations."""
        tokens_before = self.compute_state_tokens(state)
        if tokens_before < self.trigger_token_limit:
            update_span_io(
                input_data={
                    "iteration": state.iteration_count,
                    "tokens_before": tokens_before,
                    "trigger_token_limit": self.trigger_token_limit,
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

        preserved_citations = sorted(
            {claim.source_chunk_id for claim in state.verified_claims}
            | {cid for snap in state.snapshots for cid in snap.preserved_citations}
            | {cid for card in state.evidence_cards for cid in card.retrieved_chunk_ids}
        )

        bullet_summaries = [
            f"[{card.sub_query_id}] {card.summary[:200]} (Citations: {', '.join(card.retrieved_chunk_ids)})"
            for card in state.evidence_cards
        ]
        condensed_reasoning = (
            f"Compacted Step {state.iteration_count}: " + " | ".join(bullet_summaries)
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
