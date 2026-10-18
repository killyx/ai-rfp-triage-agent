"""Pydantic v2 schemas for OpenAI Structured Outputs, RAG retrieval, scoring, and HITL state."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LeadClassification(str, Enum):
    """Decision branches for incoming B2B RFP leads."""

    HIGH_FIT = "HIGH_FIT"
    LOW_FIT = "LOW_FIT"
    AMBIGUOUS = "AMBIGUOUS"


class HITLStatus(str, Enum):
    """Human-in-the-Loop approval gate states."""

    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED_AND_SENT = "APPROVED_AND_SENT"
    REJECTED = "REJECTED"


class RFPExtraction(BaseModel):
    """
    Strict JSON Schema model used with OpenAI Structured Outputs (`gpt-4o-mini`)
    to extract structured parameters from raw emails or PDF RFP documents.
    """

    model_config = ConfigDict(extra="forbid")

    client_name: str = Field(
        ...,
        description="Full name of the prospect or primary contact person, or 'Unknown' if not stated.",
    )
    company: str = Field(
        ...,
        description="Name of the client organization or company submitting the RFP.",
    )
    estimated_budget_usd: int | None = Field(
        default=None,
        description="Extracted project budget normalized to USD integer, or null if not disclosed.",
    )
    timeline_weeks: int | None = Field(
        default=None,
        description="Expected delivery timeline normalized to integer weeks, or null if unspecified.",
    )
    required_tech_stack: list[str] = Field(
        default_factory=list,
        description="List of specific programming languages, frameworks, cloud providers, or tools mentioned.",
    )
    scope_summary: str = Field(
        ...,
        min_length=10,
        description="Concise 1-3 sentence technical summary of the requested solution and deliverables.",
    )
    missing_information_flags: list[str] = Field(
        default_factory=list,
        description="Specific missing critical details (e.g., 'missing_budget', 'missing_timeline', 'vague_functional_scope', 'unspecified_tech_stack').",
    )

    @field_validator("required_tech_stack", mode="before")
    @classmethod
    def clean_tech_stack(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            seen: list[str] = []
            for item in value:
                cleaned = str(item).strip()
                if cleaned and cleaned not in seen:
                    seen.append(cleaned)
            return seen
        return [str(value).strip()]

    @field_validator("estimated_budget_usd")
    @classmethod
    def validate_non_negative_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("estimated_budget_usd cannot be negative")
        return value

    @field_validator("timeline_weeks")
    @classmethod
    def validate_positive_timeline(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("timeline_weeks must be greater than zero when provided")
        return value


class MatchedCaseStudy(BaseModel):
    """A historical software house project retrieved via vector similarity search."""

    id: str
    title: str
    industry: str
    budget_usd: int
    timeline_weeks: int
    tech_stack: list[str]
    summary: str
    business_outcomes: list[str]
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    matched_technologies: list[str] = Field(default_factory=list)


class TechStackVerification(BaseModel):
    """Verification of the client's requested technologies against software house capabilities."""

    requested_stack: list[str]
    supported_matches: list[str]
    unsupported_flagged: list[str]
    unrecognized_items: list[str]
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    recommended_roles: list[str]
    estimated_weekly_burn_usd: int


class RAGRetrievalResult(BaseModel):
    """Aggregated output from the Knowledge Retrieval (RAG) pipeline step."""

    matched_case_studies: list[MatchedCaseStudy]
    tech_verification: TechStackVerification
    top_similarity_score: float = Field(..., ge=0.0, le=1.0)
    policy_minimum_budget_usd: int
    meets_minimum_budget: bool | None


class ScoreBreakdown(BaseModel):
    """Explainable multi-factor lead qualification score (0-100) and routing classification."""

    budget_score: int = Field(..., ge=0, le=35, description="0-35 points based on budget vs $10k floor and scope fit.")
    tech_stack_score: int = Field(..., ge=0, le=30, description="0-30 points based on supported vs unsupported tech.")
    scope_clarity_score: int = Field(..., ge=0, le=20, description="0-20 points based on completeness and RAG similarity.")
    timeline_realism_score: int = Field(..., ge=0, le=15, description="0-15 points based on timeline feasibility.")
    total_score: int = Field(..., ge=0, le=100)
    classification: LeadClassification
    disqualification_reasons: list[str] = Field(default_factory=list)
    reasoning_summary: str


class ProposalDraft(BaseModel):
    """Generated response email / commercial proposal tailored to the lead classification."""

    subject: str
    recipient_name: str
    company: str
    classification: LeadClassification
    estimated_price_range_usd: str
    recommended_duration_weeks: str
    referenced_case_studies: list[str] = Field(default_factory=list)
    body_markdown: str
    recommended_next_step: str


class ModelCascadeTelemetry(BaseModel):
    """Cost and latency telemetry demonstrating model cascading (gpt-4o-mini -> gpt-4o)."""

    extraction_model: str
    synthesis_model: str
    execution_mode: Literal["openai_live", "deterministic_structured_engine"]
    extraction_tokens: int
    synthesis_tokens: int
    estimated_cost_usd: float
    monolithic_frontier_cost_usd: float
    cost_savings_percentage: float
    processing_latency_ms: int


class SlackBlockKitPayload(BaseModel):
    """Slack Block Kit interactive notification card payload for Human-in-the-Loop triage."""

    channel: str = "#sales-rfp-triage"
    text: str
    blocks: list[dict[str, Any]]


class LeadTriageRecord(BaseModel):
    """Complete end-to-end record of a processed B2B lead with HITL governance state."""

    lead_id: str
    created_at: str
    source_type: Literal["text_paste", "pdf_upload", "sample_preset", "n8n_webhook"]
    source_filename: str | None = None
    raw_text: str
    extraction: RFPExtraction
    rag_result: RAGRetrievalResult
    scoring: ScoreBreakdown
    proposal_draft: ProposalDraft
    slack_payload: SlackBlockKitPayload
    telemetry: ModelCascadeTelemetry
    hitl_status: HITLStatus = HITLStatus.PENDING_APPROVAL
    dispatch_blocked: bool = True
    approved_by: str | None = None
    approved_at: str | None = None
    reviewer_notes: str | None = None
    crm_deal_id: str


class LeadTextIngestRequest(BaseModel):
    """Request payload for submitting raw email or RFP text."""

    raw_text: str = Field(..., min_length=15, description="Raw email body or RFP document text.")
    source_name: str | None = Field(default="manual_paste.txt")


class LeadApprovalRequest(BaseModel):
    """Webhook payload for Human-in-the-Loop approval, draft editing, or rejection."""

    lead_id: str
    action: Literal["approve", "edit_and_approve", "reject"]
    reviewer_name: str = Field(default="Staff Solutions Architect")
    edited_draft_markdown: str | None = None
    reviewer_notes: str | None = None


class CRMAuditEntry(BaseModel):
    """Simulated HubSpot / Airtable CRM audit log entry."""

    entry_id: str
    timestamp: str
    lead_id: str
    company: str
    crm_provider: str = "HubSpot Enterprise CRM"
    event_type: str
    deal_stage: str
    deal_value_usd: int | None
    hitl_gate_status: str
    outbound_email_dispatched: bool
    details: str
