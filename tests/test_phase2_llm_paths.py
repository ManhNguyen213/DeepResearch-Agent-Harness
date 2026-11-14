"""Phase 2 tests for LLM code paths (via fakes), degraded fallbacks, parallelism and thread safety.

No test here reaches the network: the Gemini gateway functions are monkeypatched in the module
that imports them, and components are built with ``use_llm=True`` explicitly.
"""

import time
from pathlib import Path
from typing import Any

import pytest

from src.harness.compactor import ContextBudgetController
from src.harness.graph import build_harness_graph
from src.harness.llm import LLMGatewayError, LLMResult
from src.harness.planner import PlanDecompositionSchema, PlannedSubQueryItem, QueryPlanner
from src.harness.subagent import (
    HEURISTIC_CLAIM_CONFIDENCE,
    IsolatedSubAgentWorker,
    _SubAgentLLMExtraction,
)
from src.harness.verifier import VerificationGate
from src.schemas.evidence import EvidenceClaim, RetrievedChunk, StructuredEvidenceCard
from src.schemas.state import HarnessState, SubQueryTask

MODEL = "fake-model"


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content=f"Evidence paragraph {i} about the topic. " * 20,
            published_date="2025-01-01",
        )
        for i in (1, 2, 3)
    ]


def _task(task_id: str = "sq_1") -> SubQueryTask:
    return SubQueryTask(sub_query_id=task_id, query_text="topic", target_entities=["Topic"])


def _extraction(*chunk_ids: str) -> _SubAgentLLMExtraction:
    return _SubAgentLLMExtraction(
        summary="LLM summary [c1]",
        claims=[
            EvidenceClaim(
                entity="Topic",
                attribute="finding",
                value=f"fact from {cid}",
                source_chunk_id=cid,
                confidence_score=0.95,
            )
            for cid in chunk_ids
        ],
    )


def _raise_gateway_error(*_args: Any, **_kwargs: Any) -> Any:
    raise LLMGatewayError("503 UNAVAILABLE")


# --------------------------------------------------------------------------- sub-agent


def test_subagent_llm_path_uses_llm_output_and_records_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.harness.subagent.generate_structured",
        lambda *a, **k: LLMResult(_extraction("c1"), 321, 45, MODEL),
    )
    card = IsolatedSubAgentWorker(use_llm=True).execute_isolated(_task(), _chunks())

    assert card.summary == "LLM summary [c1]"
    assert card.llm_model == MODEL
    assert card.degraded is False
    assert [c.source_chunk_id for c in card.claims] == ["c1"]
    assert card.retrieved_chunk_ids == ["c1", "c2", "c3"]
    assert card.raw_tokens_consumed > card.card_token_estimate


def test_subagent_llm_failure_gives_degraded_card_the_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.harness.subagent.generate_structured", _raise_gateway_error)
    card = IsolatedSubAgentWorker(use_llm=True).execute_isolated(_task(), _chunks())

    assert card.degraded is True
    assert card.llm_model is None
    assert card.summary.startswith("[DEGRADED heuristic]")
    assert card.claims and all(c.confidence_score == HEURISTIC_CLAIM_CONFIDENCE for c in card.claims)

    verified, rejected = VerificationGate(grounding_threshold=0.80).verify_cards([card])
    assert verified == []
    assert len(rejected) == len(card.claims)


# --------------------------------------------------------------------------- planner


def test_planner_llm_path_builds_tasks_from_structured_output(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PlanDecompositionSchema(
        reasoning="split by year",
        sub_queries=[
            PlannedSubQueryItem(query_text=f"query {i}", target_entities=["A"]) for i in (1, 2, 3)
        ],
    )
    monkeypatch.setattr(
        "src.harness.planner.generate_structured", lambda *a, **k: LLMResult(plan, 9, 9, MODEL)
    )
    out = QueryPlanner(use_llm=True).plan(HarnessState(user_query="anything"))

    assert [t.sub_query_id for t in out["sub_queries"]] == ["sq_1", "sq_2", "sq_3"]
    assert [t.query_text for t in out["sub_queries"]] == ["query 1", "query 2", "query 3"]
    assert "split by year" in out["scratchpad"]


def test_planner_llm_failure_falls_back_to_clean_heuristic_queries(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.harness.planner.generate_structured", _raise_gateway_error)
    query = "Compare FPT AI capex in 2024 and 2025"
    out = QueryPlanner(use_llm=True).plan(HarnessState(user_query=query))

    tasks = out["sub_queries"]
    assert 2 <= len(tasks) <= 4
    # Search text must stay clean: it is fed verbatim to BM25 / dense retrieval.
    assert all("excluding" not in t.query_text and "(" not in t.query_text for t in tasks)
    assert all(query in t.query_text for t in tasks)


# --------------------------------------------------------------------------- compactor


def _bloated_state(card: StructuredEvidenceCard) -> HarnessState:
    return HarnessState(
        user_query="Analyze the topic",
        scratchpad="Intermediate reasoning step. " * 300,
        evidence_cards=[card],
        verified_claims=card.claims,
        iteration_count=2,
    )


def test_compactor_llm_summary_is_used_and_citations_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.harness.compactor.generate_text",
        lambda *a, **k: LLMResult("Condensed findings [c1]", 10, 5, MODEL),
    )
    card = IsolatedSubAgentWorker(use_llm=False).execute_isolated(_task(), _chunks())
    compactor = ContextBudgetController(token_budget_limit=2000, use_llm=True)

    snapshot = compactor.compact_state(_bloated_state(card)).snapshots[0]

    assert "Condensed findings [c1]" in snapshot.condensed_reasoning
    assert {"c1", "c2", "c3"} <= set(snapshot.preserved_citations)


def test_compactor_llm_failure_uses_deterministic_summary_and_keeps_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.harness.compactor.generate_text", _raise_gateway_error)
    card = IsolatedSubAgentWorker(use_llm=False).execute_isolated(_task(), _chunks())
    compactor = ContextBudgetController(token_budget_limit=2000, use_llm=True)

    compacted = compactor.compact_state(_bloated_state(card))
    snapshot = compacted.snapshots[0]

    assert snapshot.condensed_reasoning.startswith("Compacted Step 2")
    assert {"c1", "c2", "c3"} <= set(snapshot.preserved_citations)
    assert compacted.current_token_estimate < snapshot.tokens_before


# --------------------------------------------------------------------------- full graph


def test_graph_end_to_end_with_fake_llm_rejects_fabricated_citations(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PlanDecompositionSchema(
        reasoning="two angles",
        sub_queries=[
            PlannedSubQueryItem(query_text="angle one", target_entities=["A"]),
            PlannedSubQueryItem(query_text="angle two", target_entities=["B"]),
        ],
    )
    monkeypatch.setattr(
        "src.harness.planner.generate_structured", lambda *a, **k: LLMResult(plan, 5, 5, MODEL)
    )
    monkeypatch.setattr(
        "src.harness.subagent.hybrid_search",
        lambda query, top_k=5, date_filter=None: [c.model_dump() for c in _chunks()],
    )
    # The fake sub-agent cites one real chunk (c1) and one fabricated chunk ("ghost").
    monkeypatch.setattr(
        "src.harness.subagent.generate_structured",
        lambda *a, **k: LLMResult(_extraction("c1", "ghost"), 400, 60, MODEL),
    )
    monkeypatch.setattr(
        "src.harness.graph.generate_text",
        lambda *a, **k: LLMResult("FINAL REPORT [c1]", 50, 20, MODEL),
    )

    graph = build_harness_graph(use_llm=True).compile()
    result = graph.invoke(HarnessState(user_query="Compare A and B"))

    assert result["final_report"] == "FINAL REPORT [c1]"
    assert len(result["evidence_cards"]) == 2
    assert all(c.llm_model == MODEL and not c.degraded for c in result["evidence_cards"])
    assert {c.source_chunk_id for c in result["verified_claims"]} == {"c1"}
    assert {c.source_chunk_id for c in result["rejected_claims"]} == {"ghost"}


def test_offline_degraded_run_never_verifies_new_claims(isolated_env: Path) -> None:
    graph = build_harness_graph(use_llm=False).compile()
    result = graph.invoke(HarnessState(user_query="FPT AI capex investment 2024 and 2025"))

    assert result["evidence_cards"] and all(c.degraded for c in result["evidence_cards"])
    assert result["verified_claims"] == []
    assert result["rejected_claims"]


def test_preverified_temporal_memory_claims_survive_verification_gate(
    isolated_env: Path,
) -> None:
    """Verify M4: Facts pre-loaded from TemporalSemanticStore by QueryPlanner stay in
    verified_claims across verification_gate_node even when new cards are degraded/rejected.
    """
    from src.mcp_servers import memory_server

    memory_server.commit_temporal_memory(
        entity="FPT",
        attribute="AI_Capex_2024",
        value="100M USD in Q1 2024",
        valid_from="2024-03-01",
        source_chunk_id="prior_session_chunk_01",
        confidence_score=0.95,
    )

    graph = build_harness_graph(use_llm=False).compile()
    result = graph.invoke(HarnessState(user_query="Compare FPT AI Capex in 2024 and 2025"))

    verified_ids = {c.source_chunk_id for c in result["verified_claims"]}
    rejected_ids = {c.source_chunk_id for c in result["rejected_claims"]}
    assert "prior_session_chunk_01" in verified_ids
    assert "prior_session_chunk_01" not in rejected_ids


# --------------------------------------------------------------------------- parallelism

FOUR_YEAR_QUERY = "FPT AI capex investment in 2023, 2024, 2025 and 2026"


def test_subagents_run_in_parallel(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for the removed subagent_lock: 4 x 0.5s workers must overlap."""

    def slow_execute(self: IsolatedSubAgentWorker, task: SubQueryTask, raw_chunks: Any = None):
        time.sleep(0.5)
        return StructuredEvidenceCard(
            sub_query_id=task.sub_query_id, sub_query=task.query_text, summary="x"
        )

    monkeypatch.setattr(IsolatedSubAgentWorker, "execute_isolated", slow_execute)
    graph = build_harness_graph(use_llm=False).compile()

    start = time.perf_counter()
    result = graph.invoke(HarnessState(user_query=FOUR_YEAR_QUERY))
    elapsed = time.perf_counter() - start

    assert len(result["evidence_cards"]) == 4
    assert elapsed < 1.5, f"sub-agents appear serialised: {elapsed:.2f}s for 4 x 0.5s"


@pytest.mark.parametrize("round_index", range(3))
def test_parallel_subagents_with_real_retrieval_are_thread_safe(
    isolated_env: Path, round_index: int
) -> None:
    """Concurrent first use of the retriever / stores must not race or corrupt results."""
    graph = build_harness_graph(use_llm=False).compile()
    result = graph.invoke(HarnessState(user_query=FOUR_YEAR_QUERY))

    assert len(result["evidence_cards"]) == 4
    assert all(card.retrieved_chunk_ids for card in result["evidence_cards"])
