"""Anti-Poisoning Verification & Critic Gate for deterministic citation and entailment checking."""

from src.observability.tracing import observe, update_span_io
from src.schemas.evidence import EvidenceClaim, StructuredEvidenceCard


class VerificationGate:
    """Validates candidate claims before allowing report synthesis or Long-Term Memory commits.

    Two mandatory conditions prevent Memory Poisoning:
        1. Deterministic Citation Check: `claim.source_chunk_id` must exist in the set of
           `retrieved_chunk_ids` actually returned by MCP tools in the current session.
        2. Entailment / Groundedness Threshold: `claim.confidence_score >= grounding_threshold`.
    """

    def __init__(self, grounding_threshold: float = 0.80) -> None:
        self.grounding_threshold = grounding_threshold

    @observe(
        name="anti-poisoning-verification-gate",
        as_type="guardrail",
        capture_input=False,
        capture_output=False,
    )
    def verify_cards(
        self,
        evidence_cards: list[StructuredEvidenceCard],
    ) -> tuple[list[EvidenceClaim], list[EvidenceClaim]]:
        """Partition all claims from evidence cards into (verified_claims, rejected_claims)."""
        valid_chunk_ids: set[str] = {
            chunk_id for card in evidence_cards for chunk_id in card.retrieved_chunk_ids
        }

        verified: list[EvidenceClaim] = []
        rejected: list[EvidenceClaim] = []

        for card in evidence_cards:
            for claim in card.claims:
                has_valid_citation = claim.source_chunk_id in valid_chunk_ids
                meets_confidence = claim.confidence_score >= self.grounding_threshold
                if has_valid_citation and meets_confidence:
                    verified.append(claim)
                else:
                    rejected.append(claim)

        update_span_io(
            input_data={
                "evidence_cards_count": len(evidence_cards),
                "valid_chunk_ids": sorted(valid_chunk_ids),
                "grounding_threshold": self.grounding_threshold,
            },
            output_data={
                "verified_claims_count": len(verified),
                "rejected_claims_count": len(rejected),
                "verified_chunk_ids": sorted({c.source_chunk_id for c in verified}),
            },
            level="WARNING" if rejected else "DEFAULT",
            status_message=(
                f"Rejected {len(rejected)} ungrounded/hallucinated claims"
                if rejected
                else "All claims passed verification"
            ),
        )
        return verified, rejected
