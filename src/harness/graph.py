"""LangGraph StateGraph orchestrator for the 4-Layer Deep Research Agent Harness.

Data Flow:
    START -> ``context_controller`` (loads Procedural Memory & runs Auto-Compaction if >= 70% budget)
          -> ``planner`` (checks Temporal Semantic Memory & decomposes query into 2-4 ``SubQueryTask``)
          -> [conditional fan-out via ``Send("subagent_worker", task)`` x N in parallel]
          -> ``verification_gate`` (validates citations & updates ``current_token_estimate``)
          -> (conditional loop back to ``context_controller`` if ``needs_more_research`` else ``synthesizer``)
          -> ``synthesizer`` -> END.

Design Rationale:
    Using LangGraph's ``Send`` API with the ``merge_or_clear_evidence_cards`` reducer runs each
    ``SubQueryTask`` in an isolated ``subagent_worker`` node invocation. Raw retrieved chunks
    (~3,000-5,000 tokens per sub-query) are discarded inside ``IsolatedSubAgentWorker``, and only
    compact ``StructuredEvidenceCard`` objects (~300 tokens) are merged into ``HarnessState``.
"""

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import get_settings
from src.harness.compactor import ContextBudgetController
from src.harness.llm import generate_text, has_valid_gemini_key
from src.harness.planner import QueryPlanner
from src.harness.subagent import IsolatedSubAgentWorker
from src.harness.verifier import VerificationGate
from src.observability.tracing import (
    observe,
    record_generation_metrics,
    trace_harness_session,
    update_span_io,
)
from src.schemas.state import CLEAR_CARDS_SENTINEL, HarnessState, SubQueryTask

logger = logging.getLogger(__name__)


def build_harness_graph(
    *,
    token_budget_limit: int | None = None,
    compaction_threshold: float | None = None,
    top_k_per_subquery: int = 5,
    use_llm: bool | None = None,
) -> StateGraph:
    """Construct the explicit LangGraph StateGraph for the Deep Research Agent Harness."""
    settings = get_settings()
    compactor = ContextBudgetController(
        token_budget_limit=token_budget_limit or settings.token_budget_limit,
        compaction_threshold=compaction_threshold or settings.compaction_threshold,
        use_llm=use_llm,
    )
    planner = QueryPlanner(use_llm=use_llm)
    subagent_worker = IsolatedSubAgentWorker(top_k=top_k_per_subquery, use_llm=use_llm)
    verifier = VerificationGate(grounding_threshold=settings.grounding_confidence_threshold)

    def context_controller_node(state: HarnessState) -> dict[str, Any]:
        compacted = compactor.compact_state(state)
        was_compacted = len(compacted.snapshots) > len(state.snapshots)
        return {
            "procedural_rules": compacted.procedural_rules,
            "snapshots": compacted.snapshots,
            "scratchpad": compacted.scratchpad,
            "evidence_cards": CLEAR_CARDS_SENTINEL if was_compacted else compacted.evidence_cards,
            "current_token_estimate": compacted.current_token_estimate,
        }

    def planner_node(state: HarnessState) -> dict[str, Any]:
        plan_updates = planner.plan(state)
        updated_state = state.model_copy(update=plan_updates)
        plan_updates["current_token_estimate"] = compactor.compute_state_tokens(updated_state)
        return plan_updates

    def dispatch_subagents(state: HarnessState) -> list[Send] | str:
        """Fan-out pending SubQueryTask items in parallel via LangGraph Send API."""
        pending_tasks = [sq for sq in state.sub_queries if not sq.completed]
        if not pending_tasks:
            return "verification_gate"
        return [Send("subagent_worker", task) for task in pending_tasks]

    def subagent_worker_node(task: SubQueryTask | dict[str, Any] | HarnessState) -> dict[str, Any]:
        """Isolated Sub-agent node invoked via Send('subagent_worker', SubQueryTask).

        No lock is held here: LangGraph runs the ``Send`` branches in parallel threads and the
        retriever / stores are safe for concurrent reads (covered by a regression test).
        """
        if isinstance(task, HarnessState):
            cards = [
                subagent_worker.execute_isolated(task=sq)
                for sq in task.sub_queries
                if not sq.completed
            ]
            return {"evidence_cards": cards}
        sub_task = task if isinstance(task, SubQueryTask) else SubQueryTask.model_validate(task)
        card = subagent_worker.execute_isolated(task=sub_task)
        return {"evidence_cards": [card]}

    def verification_gate_node(state: HarnessState) -> dict[str, Any]:
        card_claims, rejected = verifier.verify_cards(state.evidence_cards)
        seen = {(c.entity, c.attribute, c.source_chunk_id) for c in state.verified_claims}
        merged_verified = list(state.verified_claims) + [
            c for c in card_claims if (c.entity, c.attribute, c.source_chunk_id) not in seen
        ]
        completed_sub_queries = [
            sq.model_copy(update={"completed": True}) for sq in state.sub_queries
        ]
        next_state = state.model_copy(
            update={
                "sub_queries": completed_sub_queries,
                "verified_claims": merged_verified,
                "rejected_claims": list(state.rejected_claims) + rejected,
            }
        )
        return {
            "sub_queries": completed_sub_queries,
            "verified_claims": merged_verified,
            "rejected_claims": next_state.rejected_claims,
            "current_token_estimate": compactor.compute_state_tokens(next_state),
            "needs_more_research": state.needs_more_research,
        }

    @observe(
        name="report-synthesizer",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def synthesizer_node(state: HarnessState) -> dict[str, Any]:
        citations = sorted(
            {c.source_chunk_id for c in state.verified_claims}
            | {cid for snap in state.snapshots for cid in snap.preserved_citations}
        )
        should_call_llm = use_llm if use_llm is not None else has_valid_gemini_key()
        report = ""

        if should_call_llm and (state.evidence_cards or state.verified_claims or state.snapshots):
            cards_digest = "\n".join(
                f"- [{c.sub_query_id}] {c.sub_query}: {c.summary}" for c in state.evidence_cards
            )
            snaps_digest = "\n".join(s.condensed_reasoning for s in state.snapshots)
            claims_digest = "\n".join(
                f"- {c.entity}.{c.attribute}: {c.value} [{c.source_chunk_id}]"
                for c in state.verified_claims[:20]
            )
            prompt = (
                f"Research Question: {state.user_query}\n\n"
                f"Compacted Snapshots:\n{snaps_digest or 'None'}\n\n"
                f"Structured Evidence Cards:\n{cards_digest or 'None'}\n\n"
                f"Verified Claims:\n{claims_digest or 'None'}\n\n"
                f"Write a structured, well-cited synthesis report answering the research question. "
                f"Cite sources inline using [chunk_id]."
            )
            try:
                result = generate_text(
                    prompt,
                    system_instruction=(
                        "You are the Lead Synthesizer of a Deep Research Agent Harness. "
                        "Produce an accurate, well-structured Markdown report grounded strictly in "
                        "the provided verified claims and evidence cards with [chunk_id] citations."
                    ),
                    settings=settings,
                )
                if result.value:
                    report = result.value
                    record_generation_metrics(
                        name="final-report-synthesis",
                        model=result.model,
                        input_data={"user_query": state.user_query, "citations": citations},
                        output_data={"final_report": report},
                        input_tokens=result.input_tokens or compactor.estimate_tokens(prompt),
                        output_tokens=result.output_tokens or compactor.estimate_tokens(report),
                    )
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("LLM report synthesis failed (%s); using template report.", exc)
                report = ""

        if not report:
            card_lines = [
                f"- **{c.sub_query_id} ({c.sub_query}):** {c.summary}"
                for c in state.evidence_cards
            ]
            snap_lines = [f"- {s.condensed_reasoning}" for s in state.snapshots]
            findings_section = "\n".join(snap_lines + card_lines) or "No evidence cards collected."
            report = (
                f"### Deep Research Synthesis\n\n"
                f"**Query:** {state.user_query}\n\n"
                f"**Key Findings:**\n{findings_section}\n\n"
                f"**Verified Claims ({len(state.verified_claims)}):** "
                f"Grounded in chunks: {', '.join(citations) if citations else 'None'}."
            )

        token_est = compactor.compute_state_tokens(state)
        update_span_io(
            input_data={
                "user_query": state.user_query,
                "verified_claims_count": len(state.verified_claims),
                "citations": citations,
            },
            output_data={
                "final_report": report,
                "current_token_estimate": token_est,
            },
        )
        return {
            "final_report": report,
            "current_token_estimate": token_est,
        }

    def route_after_verification(
        state: HarnessState,
    ) -> Literal["context_controller", "synthesizer"]:
        if state.needs_more_research and state.iteration_count < settings.max_research_iterations:
            return "context_controller"
        return "synthesizer"

    workflow = StateGraph(HarnessState)
    workflow.add_node("context_controller", context_controller_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("subagent_worker", subagent_worker_node)
    workflow.add_node("verification_gate", verification_gate_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "context_controller")
    workflow.add_edge("context_controller", "planner")
    workflow.add_conditional_edges(
        "planner",
        dispatch_subagents,
        ["subagent_worker", "verification_gate"],
    )
    workflow.add_edge("subagent_worker", "verification_gate")
    workflow.add_conditional_edges(
        "verification_gate",
        route_after_verification,
        {
            "context_controller": "context_controller",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_edge("synthesizer", END)
    return workflow


@observe(
    name="deep-research-main-agent",
    as_type="agent",
    capture_input=False,
    capture_output=False,
)
def _execute_traced_graph(initial_state: HarnessState, **graph_kwargs: Any) -> dict[str, Any]:
    """Internal helper wrapped in an `agent` root span."""
    compiled = build_harness_graph(**graph_kwargs).compile()
    result = compiled.invoke(initial_state)
    update_span_io(
        input_data={
            "user_query": initial_state.user_query,
            "session_id": initial_state.session_id,
        },
        output_data={
            "final_report": result.get("final_report", ""),
            "verified_claims_count": len(result.get("verified_claims", [])),
            "rejected_claims_count": len(result.get("rejected_claims", [])),
            "iteration_count": result.get("iteration_count", 0),
            "current_token_estimate": result.get("current_token_estimate", 0),
        },
        metadata={
            "sub_queries_count": len(result.get("sub_queries", [])),
            "evidence_cards_count": len(result.get("evidence_cards", [])),
            "snapshots_count": len(result.get("snapshots", [])),
        },
    )
    return result


def run_harness_workflow(initial_state: HarnessState, **graph_kwargs: Any) -> dict[str, Any]:
    """Execute the full Deep Research Agent Harness with Langfuse session & trace propagation."""
    with trace_harness_session(
        session_id=initial_state.session_id,
        trace_name="deep-research-harness-run",
        metadata={"initial_query": initial_state.user_query},
    ):
        return _execute_traced_graph(initial_state, **graph_kwargs)
