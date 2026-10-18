"""Comprehensive unit and integration tests for the Autonomous B2B RFP & Lead Triage Agent."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import settings
from app.main import app
from app.schemas import LeadClassification, RFPExtraction
from app.services.pdf_parser import (
    PDFExtractionError,
    extract_text_from_pdf_bytes,
    extract_text_from_pdf_path,
)
from app.services.vector_store import vector_store

client = TestClient(app)


class TestPydanticStructuredSchemas:
    """Validate strict JSON Schema constraints for OpenAI Structured Outputs."""

    def test_valid_rfp_extraction_schema(self) -> None:
        payload = {
            "client_name": "Marcus Vance",
            "company": "NovaFin Capital Partners Ltd.",
            "estimated_budget_usd": 85000,
            "timeline_weeks": 14,
            "required_tech_stack": ["Python", "FastAPI", "Python", "React", "pgvector"],
            "scope_summary": "Autonomous KYB & credit underwriting document intelligence platform with HITL workbench.",
            "missing_information_flags": [],
        }
        model = RFPExtraction.model_validate(payload)
        assert model.client_name == "Marcus Vance"
        assert model.estimated_budget_usd == 85000
        # Verify deduplication validator on required_tech_stack
        assert model.required_tech_stack == ["Python", "FastAPI", "React", "pgvector"]

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RFPExtraction.model_validate(
                {
                    "client_name": "Jane Doe",
                    "company": "Acme Corp",
                    "estimated_budget_usd": 50000,
                    "timeline_weeks": 10,
                    "required_tech_stack": ["Python"],
                    "scope_summary": "Build an internal RAG assistant for legal contract review.",
                    "missing_information_flags": [],
                    "unexpected_hallucinated_field": "should_fail",
                }
            )

    def test_negative_budget_and_zero_timeline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RFPExtraction(
                client_name="Jane Doe",
                company="Acme Corp",
                estimated_budget_usd=-500,
                timeline_weeks=8,
                required_tech_stack=["Python"],
                scope_summary="Valid scope summary text for testing.",
                missing_information_flags=[],
            )

        with pytest.raises(ValidationError):
            RFPExtraction(
                client_name="Jane Doe",
                company="Acme Corp",
                estimated_budget_usd=25000,
                timeline_weeks=0,
                required_tech_stack=["Python"],
                scope_summary="Valid scope summary text for testing.",
                missing_information_flags=[],
            )


class TestPDFParserAndRAGStore:
    """Validate PyMuPDF/pdfplumber PDF ingestion and RAG vector store matching."""

    def test_parse_ideal_fintech_pdf_sample(self) -> None:
        pdf_path = settings.mock_samples_dir / "01_ideal_enterprise_fintech_rfp.pdf"
        assert pdf_path.exists(), "Seeded PDF sample must exist on disk"

        parsed = extract_text_from_pdf_path(pdf_path)
        assert parsed.page_count >= 1
        assert "NovaFin Capital" in parsed.text
        assert "$85,000" in parsed.text
        assert "FastAPI" in parsed.text

    def test_corrupted_or_invalid_pdf_raises_error(self) -> None:
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf_bytes(b"not-a-valid-pdf-header", filename="broken.pdf")

        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf_bytes(b"", filename="empty.pdf")

    def test_rag_tech_stack_verification_flags_unsupported(self) -> None:
        verification = vector_store.verify_tech_stack(
            ["Python", "FastAPI", "WordPress", "WooCommerce", "Unity3D"]
        )
        assert "Python" in verification.supported_matches
        assert "FastAPI" in verification.supported_matches
        assert "WordPress" in verification.unsupported_flagged
        assert "WooCommerce" in verification.unsupported_flagged
        assert "Unity3D" in verification.unsupported_flagged


class TestEndToEndTriagePipelineAndHITL:
    """Validate all 3 B2B lead scenarios and Human-in-the-Loop approval gate via FastAPI TestClient."""

    def test_scenario_1_high_fit_fintech_pdf_upload(self) -> None:
        pdf_path = settings.mock_samples_dir / "01_ideal_enterprise_fintech_rfp.pdf"
        with pdf_path.open("rb") as f:
            response = client.post(
                "/api/triage/pdf",
                files={"file": ("01_ideal_enterprise_fintech_rfp.pdf", f, "application/pdf")},
            )
        assert response.status_code == 200
        data = response.json()

        assert data["extraction"]["estimated_budget_usd"] == 85000
        assert data["extraction"]["timeline_weeks"] == 14
        assert "FastAPI" in data["extraction"]["required_tech_stack"]
        assert data["scoring"]["classification"] == LeadClassification.HIGH_FIT.value
        assert data["scoring"]["total_score"] > 75
        # Verify Model Cascading escalated High-Fit synthesis to gpt-4o
        assert data["telemetry"]["extraction_model"] == "gpt-4o-mini"
        assert data["telemetry"]["synthesis_model"] == "gpt-4o"
        # Verify HITL Gate locks outbound dispatch initially
        assert data["hitl_status"] == "PENDING_APPROVAL"
        assert data["dispatch_blocked"] is True

    def test_scenario_2_low_budget_lead_disqualified(self) -> None:
        response = client.post("/api/triage/sample/low_budget_lead")
        assert response.status_code == 200
        data = response.json()

        assert data["extraction"]["estimated_budget_usd"] == 3500
        assert data["rag_result"]["meets_minimum_budget"] is False
        assert data["scoring"]["classification"] == LeadClassification.LOW_FIT.value
        assert data["scoring"]["total_score"] < 75
        # Verify Model Cascading kept Low-Fit decline on cost-effective gpt-4o-mini
        assert data["telemetry"]["synthesis_model"] == "gpt-4o-mini"
        assert data["telemetry"]["cost_savings_percentage"] > 90.0

    def test_scenario_3_ambiguous_lead_triggers_discovery_questions(self) -> None:
        response = client.post("/api/triage/sample/ambiguous_lead")
        assert response.status_code == 200
        data = response.json()

        assert data["extraction"]["estimated_budget_usd"] is None
        assert "missing_budget" in data["extraction"]["missing_information_flags"]
        assert data["scoring"]["classification"] == LeadClassification.AMBIGUOUS.value
        assert "Information Needed" in data["proposal_draft"]["body_markdown"]

    def test_hitl_approval_and_edit_workflow_unlocks_dispatch(self) -> None:
        # 1. Create a triaged lead
        ingest_resp = client.post("/api/triage/sample/ideal_fintech_pdf")
        assert ingest_resp.status_code == 200
        lead_id = ingest_resp.json()["lead_id"]

        # 2. Approve with an edited markdown draft via /api/lead/approve
        custom_draft = "Dear Marcus,\n\nCustom approved executive proposal for NovaFin Capital ($85,000 USD)."
        approve_resp = client.post(
            "/api/lead/approve",
            json={
                "lead_id": lead_id,
                "action": "edit_and_approve",
                "reviewer_name": "Daniel K. — Staff AI Engineer",
                "edited_draft_markdown": custom_draft,
                "reviewer_notes": "Verified architecture capacity.",
            },
        )
        assert approve_resp.status_code == 200
        res_json = approve_resp.json()

        assert res_json["hitl_status"] == "APPROVED_AND_SENT"
        assert res_json["dispatch_blocked"] is False
        assert res_json["lead"]["proposal_draft"]["body_markdown"] == custom_draft
        assert res_json["audit_entry"]["outbound_email_dispatched"] is True

        # 3. Verify CRM audit log recorded both staging and approval events
        audit_resp = client.get("/api/crm/audit-log")
        assert audit_resp.status_code == 200
        events = [e for e in audit_resp.json()["audit_log"] if e["lead_id"] == lead_id]
        assert len(events) >= 2
        assert events[0]["event_type"] == "HITL_EDITED_AND_DISPATCHED"

    def test_dashboard_ui_and_health_endpoints(self) -> None:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"
        assert health.json()["minimum_budget_policy_usd"] == 10000

        ui = client.get("/")
        assert ui.status_code == 200
        assert "Autonomous B2B RFP &amp; Lead Triage Agent" in ui.text
        assert "edit-draft-dialog" in ui.text
