"""Simulated HubSpot / Airtable CRM state store and Human-in-the-Loop (HITL) enforcement gate."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.schemas import (
    CRMAuditEntry,
    HITLStatus,
    LeadApprovalRequest,
    LeadClassification,
    LeadTriageRecord,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class CRMSimulator:
    """
    In-memory + audit-logged simulator for HubSpot CRM deals and Slack HITL approvals.
    Guarantees that NO automated proposal or decline leaves the system until an authorized
    reviewer invokes the `/api/lead/approve` webhook.
    """

    def __init__(self) -> None:
        self._leads: dict[str, LeadTriageRecord] = {}
        self._audit_log: list[CRMAuditEntry] = []

    def register_triaged_lead(self, record: LeadTriageRecord) -> LeadTriageRecord:
        """Store a newly triaged lead in PENDING_APPROVAL state and record a CRM staging audit entry."""
        record.hitl_status = HITLStatus.PENDING_APPROVAL
        record.dispatch_blocked = True
        self._leads[record.lead_id] = record

        stage_map = {
            LeadClassification.HIGH_FIT: "Qualified — Proposal Staged (Awaiting VP Sign-off)",
            LeadClassification.AMBIGUOUS: "Discovery — Scoping Questions Staged (Awaiting Sign-off)",
            LeadClassification.LOW_FIT: "Disqualified — Polite Decline Staged (Awaiting Sign-off)",
        }

        audit_entry = CRMAuditEntry(
            entry_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            timestamp=_utc_now_iso(),
            lead_id=record.lead_id,
            company=record.extraction.company,
            crm_provider="HubSpot Enterprise CRM",
            event_type="DEAL_STAGED_HITL_LOCKED",
            deal_stage=stage_map[record.scoring.classification],
            deal_value_usd=record.extraction.estimated_budget_usd,
            hitl_gate_status=HITLStatus.PENDING_APPROVAL.value,
            outbound_email_dispatched=False,
            details=(
                f"Lead scored {record.scoring.total_score}/100 ({record.scoring.classification.value}). "
                f"Draft generated via {record.telemetry.synthesis_model}. Outbound dispatch LOCKED pending human review."
            ),
        )
        self._audit_log.insert(0, audit_entry)
        return record

    def process_hitl_decision(self, req: LeadApprovalRequest) -> tuple[LeadTriageRecord, CRMAuditEntry]:
        """
        Process Human-in-the-Loop webhook action (`approve`, `edit_and_approve`, or `reject`).
        Unlocks outbound email dispatch ONLY when approved.
        """
        if req.lead_id not in self._leads:
            raise KeyError(f"Lead ID '{req.lead_id}' not found in triage store.")

        record = self._leads[req.lead_id]
        now_str = _utc_now_iso()

        if req.edited_draft_markdown and req.edited_draft_markdown.strip():
            record.proposal_draft.body_markdown = req.edited_draft_markdown.strip()

        record.approved_by = req.reviewer_name
        record.approved_at = now_str
        record.reviewer_notes = req.reviewer_notes

        if req.action in ("approve", "edit_and_approve"):
            record.hitl_status = HITLStatus.APPROVED_AND_SENT
            record.dispatch_blocked = False
            event_type = "HITL_APPROVED_AND_DISPATCHED" if req.action == "approve" else "HITL_EDITED_AND_DISPATCHED"

            if record.scoring.classification == LeadClassification.HIGH_FIT:
                deal_stage = "Proposal Sent — Executive Follow-Up Scheduled"
            elif record.scoring.classification == LeadClassification.AMBIGUOUS:
                deal_stage = "Discovery Questionnaire Dispatched"
            else:
                deal_stage = "Closed Lost — Polite Referral Sent"

            details = (
                f"Approved by {req.reviewer_name}. Outbound proposal email dispatched to "
                f"{record.extraction.client_name} ({record.extraction.company}) and synced to HubSpot Deal {record.crm_deal_id}."
            )
            dispatched = True
        else:
            record.hitl_status = HITLStatus.REJECTED
            record.dispatch_blocked = True
            event_type = "HITL_REJECTED_BY_REVIEWER"
            deal_stage = "Closed Archived — Blocked by Human Reviewer"
            details = (
                f"Rejected by {req.reviewer_name}. Outbound communication permanently suppressed. "
                f"Notes: {req.reviewer_notes or 'No outbound response sent.'}"
            )
            dispatched = False

        audit_entry = CRMAuditEntry(
            entry_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            timestamp=now_str,
            lead_id=record.lead_id,
            company=record.extraction.company,
            crm_provider="HubSpot Enterprise CRM",
            event_type=event_type,
            deal_stage=deal_stage,
            deal_value_usd=record.extraction.estimated_budget_usd,
            hitl_gate_status=record.hitl_status.value,
            outbound_email_dispatched=dispatched,
            details=details,
        )
        self._audit_log.insert(0, audit_entry)
        return record, audit_entry

    def get_lead(self, lead_id: str) -> LeadTriageRecord | None:
        return self._leads.get(lead_id)

    def list_leads(self) -> list[LeadTriageRecord]:
        return list(self._leads.values())

    def list_audit_log(self) -> list[CRMAuditEntry]:
        return list(self._audit_log)


crm_simulator = CRMSimulator()
