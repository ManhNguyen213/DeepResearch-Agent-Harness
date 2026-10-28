"""Unit tests for Pydantic v2 schemas, Temporal Conflict Resolution, Verification Gate, and Compaction."""

from src.harness.compactor import ContextBudgetController
from src.harness.subagent import IsolatedSubAgentWorker
from src.harness.verifier import VerificationGate
from src.memory.temporal_store import TemporalSemanticStore
from src.schemas.evidence import EvidenceClaim, RetrievedChunk, StructuredEvidenceCard
from src.schemas.memory import MemoryStatus, TemporalFactRecord
from src.schemas.state import HarnessState, SubQueryTask


def test_temporal_conflict_resolution_supersedes_old_fact() -> None:
    """Verify that committing a newer fact marks the older fact as SUPERSEDED without deleting it."""
    store = TemporalSemanticStore(db_path=":memory:")

    fact_2024 = TemporalFactRecord(
        memory_id="mem_2024_q1",
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        value="Invested 100M USD in AI infrastructure",
        valid_from="2024-03-01",
        source_chunk_id="doc_fpt_2024_q1_p05",
        confidence_score=0.92,
    )
    store.commit_fact(fact_2024)

    fact_2025 = TemporalFactRecord(
        memory_id="mem_2025_q2",
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        value="Expanded investment to 200M USD for AI factories in Vietnam and Japan",
        valid_from="2025-06-01",
        source_chunk_id="doc_fpt_2025_q2_p14",
        confidence_score=0.96,
    )
    store.commit_fact(fact_2025)

    active_facts = store.get_facts(
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        active_only=True,
    )
    assert len(active_facts) == 1
    assert active_facts[0].memory_id == "mem_2025_q2"
    assert active_facts[0].status == MemoryStatus.ACTIVE
    assert active_facts[0].valid_to is None

    all_history = store.get_facts(
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        active_only=False,
    )
    assert len(all_history) == 2
    old_record = next(r for r in all_history if r.memory_id == "mem_2024_q1")
    assert old_record.status == MemoryStatus.SUPERSEDED
    assert old_record.valid_to == "2025-06-01"
    store.close()


def test_anti_poisoning_verification_gate() -> None:
    """Verify that claims with hallucinated chunk IDs or low confidence are rejected."""
    gate = VerificationGate(grounding_threshold=0.80)

    valid_claim = EvidenceClaim(
        entity="FPT_Corporation",
        attribute="AI_Capex_Investment",
        value="200M USD",
        source_chunk_id="chunk_real_01",
        confidence_score=0.95,
    )
    hallucinated_chunk_claim = EvidenceClaim(
        entity="FPT_Corporation",
        attribute="Revenue",
        value="999B USD",
        source_chunk_id="chunk_hallucinated_99",
        confidence_score=0.95,
    )
    low_confidence_claim = EvidenceClaim(
        entity="FPT_Corporation",
        attribute="Margin",
        value="Uncertain projection",
        source_chunk_id="chunk_real_01",
        confidence_score=0.55,
    )

    card = StructuredEvidenceCard(
        sub_query_id="sq_1",
        sub_query="What is FPT's AI investment?",
        summary="FPT invested 200M USD [chunk_real_01].",
        claims=[valid_claim, hallucinated_chunk_claim, low_confidence_claim],
        retrieved_chunk_ids=["chunk_real_01"],
    )

    verified, rejected = gate.verify_cards([card])
    assert len(verified) == 1
    assert verified[0].source_chunk_id == "chunk_real_01"
    assert len(rejected) == 2


def test_subagent_isolation_and_auto_compaction_budget() -> None:
    """Verify IsolatedSubAgentWorker output compactness and ContextBudgetController Auto-Compaction."""
    worker = IsolatedSubAgentWorker()
    task = SubQueryTask(
        sub_query_id="sq_1",
        query_text="Analyze AI Capex",
        target_entities=["FPT_Corporation"],
    )
    raw_chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            content="Detailed financial disclosure text. " * 60,
            published_date="2025-06-01",
        )
        for i in range(5)
    ]

    card = worker.execute_isolated(task, raw_chunks)
    assert card.raw_tokens_consumed > card.card_token_estimate
    assert len(card.retrieved_chunk_ids) == 5

    compactor = ContextBudgetController(token_budget_limit=2000, compaction_threshold=0.70)
    bloated_state = HarnessState(
        user_query="Analyze AI Capex across 2024-2025",
        scratchpad="Intermediate reasoning step. " * 300,
        evidence_cards=[card],
        verified_claims=card.claims,
        iteration_count=2,
    )

    assert compactor.should_compact(bloated_state) is True
    compacted_state = compactor.compact_state(bloated_state)
    assert len(compacted_state.snapshots) == 1
    assert compacted_state.evidence_cards == []
    assert compacted_state.scratchpad == ""
    assert set(card.retrieved_chunk_ids).issubset(
        set(compacted_state.snapshots[0].preserved_citations)
    )
    assert compacted_state.current_token_estimate < compacted_state.snapshots[0].tokens_before
