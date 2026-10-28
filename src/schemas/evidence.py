"""Pydantic v2 schemas for Isolated Sub-agent Workers and Structured Evidence Cards."""

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """Raw document chunk returned by an MCP Retrieval Tool (consumed only by Sub-agents)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Unique deterministic identifier of the source chunk")
    content: str = Field(..., description="Raw text content of the chunk")
    document_title: str = Field(default="", description="Source document title or filename")
    published_date: str | None = Field(
        default=None,
        description="ISO-8601 date string (YYYY-MM-DD) associated with the document",
    )
    score: float = Field(default=0.0, description="Hybrid retrieval or reranker relevance score")


class EvidenceClaim(BaseModel):
    """A single grounded factual claim extracted by a Researcher Sub-agent."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(..., description="Primary entity subject of the claim (e.g., FPT_Corporation)")
    attribute: str = Field(..., description="Normalized attribute or relation (e.g., AI_Capex_Investment)")
    value: str = Field(..., description="Concise factual assertion extracted from the source chunk")
    source_chunk_id: str = Field(
        ...,
        description="Exact chunk_id from which this claim was extracted",
    )
    valid_from: str | None = Field(
        default=None,
        description="Effective start date (YYYY-MM-DD) of the fact if stated in the source",
    )
    confidence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Sub-agent semantic entailment/groundedness score",
    )


class StructuredEvidenceCard(BaseModel):
    """Compact output (~300-500 tokens) returned by an Isolated Sub-agent Worker to the Main Agent.

    Raw document chunks are discarded inside the sub-agent context window; only this
    structured card crosses the boundary back into the Main Agent's Working Memory.
    """

    model_config = ConfigDict(extra="forbid")

    sub_query_id: str = Field(..., description="ID of the SubQueryTask handled by the sub-agent")
    sub_query: str = Field(..., description="The sub-question investigated")
    summary: str = Field(
        ...,
        description="Concise synthesis (150-300 words) answering the sub-query with inline [chunk_id] citations",
    )
    claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description="Structured atomic claims extracted for verification and memory commitment",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Complete list of deterministic chunk_ids returned by MCP tools during this sub-task",
    )
    raw_tokens_consumed: int = Field(
        default=0,
        ge=0,
        description="Number of raw context tokens processed and discarded inside the isolated sub-agent",
    )
    card_token_estimate: int = Field(
        default=0,
        ge=0,
        description="Estimated token size of this compact evidence card",
    )
