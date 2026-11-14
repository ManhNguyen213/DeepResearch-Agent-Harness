"""Phase 2 tests: ContextBudgetController (Procedural Memory + Auto-Compaction), QueryPlanner
(Temporal Semantic Memory pre-check + 2-4 sub-query decomposition), IsolatedSubAgentWorker
(Context Forking savings > 80%), and LangGraph StateGraph with parallel Send API."""

from pathlib import Path

import pytest

from src.harness.compactor import ContextBudgetController
from src.harness.graph import build_harness_graph, run_harness_workflow
from src.harness.planner import QueryPlanner
from src.harness.subagent import IsolatedSubAgentWorker
from src.mcp_servers import memory_server
from src.schemas.evidence import RetrievedChunk
from src.schemas.memory import ProceduralRule
from src.schemas.state import HarnessState, SubQueryTask

# The offline ``isolated_env`` fixture lives in tests/conftest.py.


def test_compactor_loads_procedural_memory_at_session_start(isolated_env: Path) -> None:
    """Verify 2.1: ContextBudgetController loads matching ProceduralRule items at iteration 0."""
    ep_store = memory_server.get_episodic_store()
    ep_store.save_procedural_rule(
        ProceduralRule(
            rule_id="rule_capex_01",
            domain_pattern="fpt_ai_capex_investment",
            strategy_instruction="Always compare 2024 and 2025 AI Capex figures using date_filter.",
            effectiveness_score=0.95,
        )
    )

    compactor = ContextBudgetController(token_budget_limit=32000, use_llm=False)
    initial_state = HarnessState(
        user_query="Analyze FPT AI capex investment changes from 2024 to 2025",
        iteration_count=0,
    )
    updated = compactor.compact_state(initial_state)

    assert len(updated.procedural_rules) == 1
    assert "fpt_ai_capex_investment" in updated.procedural_rules[0]
    assert "date_filter" in updated.procedural_rules[0]
    assert updated.current_token_estimate > 0


def test_planner_checks_temporal_memory_and_decomposes_multi_hop(isolated_env: Path) -> None:
    """Verify 2.2: QueryPlanner reuses existing Temporal Semantic Memory and generates 2-4 SubQueryTasks."""
    memory_server.commit_temporal_memory(
        entity="FPT",
        attribute="AI_Capex_2024",
        value="100M USD in Q1 2024",
        valid_from="2024-03-01",
        source_chunk_id="doc_fpt_2024_q1_p05",
        confidence_score=0.95,
    )

    planner = QueryPlanner(use_llm=False)
    state = HarnessState(
        user_query=(
            "Compare FPT AI Capex in 2024 and 2025, analyze GPU cluster expansion, "
            "and evaluate revenue impact in Japan and Vietnam"
        )
    )
    plan_out = planner.plan(state)

    sub_queries = plan_out["sub_queries"]
    assert 2 <= len(sub_queries) <= 4
    assert all(isinstance(sq, SubQueryTask) and not sq.completed for sq in sub_queries)
    assert any(c.source_chunk_id == "doc_fpt_2024_q1_p05" for c in plan_out["verified_claims"])
    assert "Memory Hits:" in plan_out["scratchpad"]


def test_subagent_context_isolation_saves_over_80_percent_tokens(
    isolated_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 2.3 & Exit Criteria 1: 4 SubQueryTasks x 5 chunks (~14,000 raw tokens) -> Main Agent < 2,000 tokens."""
    # Each chunk is ~2,800 chars (~700 tokens) -> 5 chunks = ~3,500 tokens per task -> ~14,000 tokens across 4 tasks
    heavy_paragraph = (
        "FPT Corporation financial and AI infrastructure disclosure covering GPU clusters, "
        "data center capex allocation, enterprise cloud adoption, and regional revenue growth. "
    ) * 16

    def fake_heavy_hybrid_search(
        query: str, top_k: int = 5, date_filter: str | None = None
    ) -> list[dict]:
        return [
            RetrievedChunk(
                chunk_id=f"doc_heavy_{idx:02d}",
                content=f"[{query}] Section {idx}: {heavy_paragraph}",
                document_title="FPT_Annual_Report_2025.pdf",
                published_date="2025-06-01",
                score=0.92,
            ).model_dump()
            for idx in range(1, top_k + 1)
        ]

    monkeypatch.setattr("src.harness.subagent.hybrid_search", fake_heavy_hybrid_search)

    initial_state = HarnessState(
        user_query=(
            "Analyze FPT 2023 strategy, 2024 AI capex, 2025 factory expansion, "
            "and 2026 technology revenue outlook?"
        )
    )
    result = run_harness_workflow(initial_state, top_k_per_subquery=5, use_llm=False)

    cards = result["evidence_cards"]
    assert len(cards) == 4

    total_raw_tokens = sum(c.raw_tokens_consumed for c in cards)
    main_agent_tokens = result["current_token_estimate"]

    # Raw tokens across 4 sub-agents must be ~14,000 tokens (>= 12,000), while Main Agent stays < 2,000 tokens
    assert total_raw_tokens >= 12000
    assert main_agent_tokens < 2000
    savings_ratio = 1.0 - (main_agent_tokens / total_raw_tokens)
    assert savings_ratio >= 0.80


def test_stategraph_auto_compaction_on_second_iteration_preserves_all_citations(
    isolated_env: Path,
) -> None:
    """Verify 2.4 & Exit Criteria 2: Lower TOKEN_BUDGET_LIMIT to 2000, trigger compaction on loop 2, keep 100% citations."""
    bloated_scratchpad = (
        "Detailed intermediate multi-hop reasoning trace from iteration 1 covering "
        "historical AI capex and infrastructure milestones across regions. "
    ) * 65  # ~1,500 tokens -> exceeds 70% of 2,000 token budget (1,400 trigger limit)

    # Pre-populate state as entering Iteration 2 after VerificationGate requested more research
    worker = IsolatedSubAgentWorker(top_k=3, use_llm=False)
    prior_card = worker.execute_isolated(
        SubQueryTask(
            sub_query_id="sq_iter1_1",
            query_text="FPT AI capex 2024",
            target_entities=["FPT"],
        )
    )
    expected_prior_citations = set(prior_card.retrieved_chunk_ids)
    assert expected_prior_citations

    state_iter2 = HarnessState(
        user_query="Compare FPT AI capex in 2024 and 2025",
        scratchpad=bloated_scratchpad,
        evidence_cards=[prior_card],
        verified_claims=prior_card.claims,
        iteration_count=1,
        needs_more_research=False,
    )

    graph = build_harness_graph(
        token_budget_limit=2000,
        compaction_threshold=0.70,
        top_k_per_subquery=2,
        use_llm=False,
    ).compile()
    final_state = graph.invoke(state_iter2)

    assert len(final_state["snapshots"]) == 1
    snapshot = final_state["snapshots"][0]
    assert snapshot.tokens_before >= 1400
    assert final_state["current_token_estimate"] < 1400
    assert expected_prior_citations.issubset(set(snapshot.preserved_citations))
    assert final_state["final_report"].startswith("### Deep Research Synthesis")
