"""LangGraph StateGraph orchestrator for the 4-Layer Deep Research Agent Harness."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.harness.compactor import ContextBudgetController
from src.harness.planner import QueryPlanner
from src.harness.subagent import IsolatedSubAgentWorker
from src.harness.verifier import VerificationGate
from src.observability.tracing import observe, trace_harness_session, update_span_io
from src.schemas.state import HarnessState


def build_harness_graph() -> StateGraph:
    """Construct the explicit LangGraph StateGraph for the Deep Research Agent Harness."""
    settings = get_settings()
    compactor = ContextBudgetController(
        token_budget_limit=settings.token_budget_limit,
        compaction_threshold=settings.compaction_threshold,
    )
    planner = QueryPlanner()
    subagent_worker = IsolatedSubAgentWorker()
    verifier = VerificationGate(grounding_threshold=settings.grounding_confidence_threshold)

    def context_controller_node(state: HarnessState) -> dict:
        compacted = compactor.compact_state(state)
        return {
            "snapshots": compacted.snapshots,
            "scratchpad": compacted.scratchpad,
            "evidence_cards": compacted.evidence_cards,
            "current_token_estimate": compacted.current_token_estimate,
        }

    def planner_node(state: HarnessState) -> dict:
        return planner.plan(state)

    def subagent_worker_node(state: HarnessState) -> dict:
        new_cards = []
        for task in state.sub_queries:
            if not task.completed:
                card = subagent_worker.execute_isolated(task=task)
                new_cards.append(card)
        return {"evidence_cards": new_cards}

    def verification_gate_node(state: HarnessState) -> dict:
        verified, rejected = verifier.verify_cards(state.evidence_cards)
        return {
            "verified_claims": verified,
            "rejected_claims": rejected,
            "needs_more_research": False,
        }

    @observe(
        name="report-synthesizer",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def synthesizer_node(state: HarnessState) -> dict:
        citations = sorted({c.source_chunk_id for c in state.verified_claims})
        report = (
            f"### Deep Research Synthesis\n\n"
            f"**Query:** {state.user_query}\n\n"
            f"**Verified Claims ({len(state.verified_claims)}):** "
            f"Grounded in chunks: {', '.join(citations) if citations else 'None'}."
        )
        update_span_io(
            input_data={
                "user_query": state.user_query,
                "verified_claims_count": len(state.verified_claims),
                "citations": citations,
            },
            output_data={"final_report": report},
        )
        return {"final_report": report}

    def route_after_verification(
        state: HarnessState,
    ) -> Literal["planner", "synthesizer"]:
        if state.needs_more_research and state.iteration_count < settings.max_research_iterations:
            return "planner"
        return "synthesizer"

    workflow = StateGraph(HarnessState)
    workflow.add_node("context_controller", context_controller_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("subagent_workers", subagent_worker_node)
    workflow.add_node("verification_gate", verification_gate_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "context_controller")
    workflow.add_edge("context_controller", "planner")
    workflow.add_edge("planner", "subagent_workers")
    workflow.add_edge("subagent_workers", "verification_gate")
    workflow.add_conditional_edges(
        "verification_gate",
        route_after_verification,
        {
            "planner": "planner",
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
def _execute_traced_graph(initial_state: HarnessState) -> dict:
    """Internal helper wrapped in an `agent` root span."""
    compiled = build_harness_graph().compile()
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
        },
        metadata={
            "sub_queries_count": len(result.get("sub_queries", [])),
            "evidence_cards_count": len(result.get("evidence_cards", [])),
        },
    )
    return result


def run_harness_workflow(initial_state: HarnessState) -> dict:
    """Execute the full Deep Research Agent Harness with Langfuse session & trace propagation."""
    with trace_harness_session(
        session_id=initial_state.session_id,
        trace_name="deep-research-harness-run",
        metadata={"initial_query": initial_state.user_query},
    ):
        return _execute_traced_graph(initial_state)
