"""FastAPI main application for the Autonomous B2B RFP & Lead Triage Agent."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, settings
from app.schemas import (
    LeadApprovalRequest,
    LeadTextIngestRequest,
    LeadTriageRecord,
)
from app.services.crm_simulator import crm_simulator
from app.services.llm_agent import rfp_agent
from app.services.pdf_parser import (
    PDFExtractionError,
    extract_text_from_pdf_bytes,
    extract_text_from_pdf_path,
)
from app.services.vector_store import vector_store

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade B2B AI RFP & Lead Triage Agent featuring OpenAI Structured Outputs, "
        "Model Cascading (gpt-4o-mini -> gpt-4o), RAG Knowledge Retrieval, and Slack Block Kit "
        "Human-in-the-Loop (HITL) approval governance."
    ),
)

STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SAMPLE_PRESETS: dict[str, dict[str, str]] = {
    "ideal_fintech_pdf": {
        "id": "ideal_fintech_pdf",
        "label": "Scenario 1: High-Fit Enterprise FinTech RFP ($85k PDF)",
        "badge": "HIGH_FIT",
        "filename": "01_ideal_enterprise_fintech_rfp.pdf",
        "fallback_txt": "01_ideal_enterprise_fintech_rfp.txt",
        "description": "Series B Commercial FinTech requesting a Python/FastAPI + React + pgvector Document Intelligence platform ($85,000 budget, 14 weeks).",
    },
    "low_budget_lead": {
        "id": "low_budget_lead",
        "label": "Scenario 2: Low-Budget & Legacy Stack Lead ($3.5k)",
        "badge": "LOW_FIT",
        "filename": "02_low_budget_ecommerce_lead.txt",
        "fallback_txt": "02_low_budget_ecommerce_lead.txt",
        "description": "E-commerce founder requesting WordPress/WooCommerce + Unity3D AR mobile apps + AI chatbot in 4 weeks for $3,500 fixed price.",
    },
    "ambiguous_lead": {
        "id": "ambiguous_lead",
        "label": "Scenario 3: Ambiguous Logistics Inquiry (Missing Scope/Budget)",
        "badge": "AMBIGUOUS",
        "filename": "03_ambiguous_stealth_startup_lead.txt",
        "fallback_txt": "03_ambiguous_stealth_startup_lead.txt",
        "description": "European freight network requesting AI agents & RAG for back-office dispatch, but missing budget envelope and delivery timeline.",
    },
}


async def execute_full_triage_pipeline(
    raw_text: str,
    source_type: Literal["text_paste", "pdf_upload", "sample_preset", "n8n_webhook"],
    source_filename: str | None = None,
) -> LeadTriageRecord:
    """Run the 4-stage AI RFP Triage pipeline and stage the result behind the HITL approval gate."""
    start_time = time.perf_counter()

    # Step 1: Ingestion & Extraction (gpt-4o-mini Structured Outputs)
    extraction, exec_mode, extraction_tokens = await rfp_agent.extract_rfp_parameters(raw_text)

    # Step 2: Knowledge Retrieval (RAG over software house case studies & rate card)
    rag_result = vector_store.retrieve_context(extraction=extraction, raw_text=raw_text)

    # Step 3: Multi-factor Scoring & Cascaded Proposal Synthesis (gpt-4o for High-Fit, gpt-4o-mini otherwise)
    scoring = rfp_agent.calculate_lead_score(extraction=extraction, rag_result=rag_result)
    proposal, telemetry = await rfp_agent.generate_proposal_and_telemetry(
        extraction=extraction,
        rag_result=rag_result,
        scoring=scoring,
        extraction_mode=exec_mode,
        extraction_tokens=extraction_tokens,
        start_time=start_time,
    )

    lead_id = f"LEAD-{uuid.uuid4().hex[:8].upper()}"
    crm_deal_id = f"HS-DEAL-{uuid.uuid4().int % 900000 + 100000}"

    # Step 4: Build Slack Block Kit HITL Notification & Lock Dispatch until Approved
    slack_payload = rfp_agent.build_slack_block_kit_payload(
        lead_id=lead_id,
        extraction=extraction,
        rag_result=rag_result,
        scoring=scoring,
        proposal=proposal,
        telemetry=telemetry,
    )

    record = LeadTriageRecord(
        lead_id=lead_id,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        source_type=source_type,
        source_filename=source_filename,
        raw_text=raw_text,
        extraction=extraction,
        rag_result=rag_result,
        scoring=scoring,
        proposal_draft=proposal,
        slack_payload=slack_payload,
        telemetry=telemetry,
        crm_deal_id=crm_deal_id,
    )
    return crm_simulator.register_triaged_lead(record)


@app.get("/", response_class=HTMLResponse)
async def render_simulator_dashboard(request: Request) -> HTMLResponse:
    """Serve the interactive AI RFP & Lead Triage Simulator Web Dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "min_budget_usd": settings.min_budget_usd,
            "high_fit_threshold": settings.high_fit_score_threshold,
            "extraction_model": settings.extraction_model,
            "proposal_model": settings.proposal_model,
            "has_live_key": settings.has_live_openai_key,
            "samples": list(SAMPLE_PRESETS.values()),
            "knowledge_base": vector_store.knowledge_base,
        },
    )


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """System readiness and configuration telemetry endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "openai_live_configured": settings.has_live_openai_key,
        "models": {
            "tier_1_extraction": settings.extraction_model,
            "tier_2_proposal": settings.proposal_model,
        },
        "indexed_case_studies": len(vector_store.knowledge_base.get("case_studies", [])),
        "minimum_budget_policy_usd": settings.min_budget_usd,
    }


@app.get("/api/knowledge-base")
async def get_knowledge_base() -> dict[str, Any]:
    """Return the software house RAG knowledge base (rates, tech stack, case studies)."""
    return vector_store.knowledge_base


@app.get("/api/samples")
async def list_mock_samples() -> dict[str, Any]:
    """Return available pre-configured demo scenarios and their raw text."""
    enriched: list[dict[str, Any]] = []
    for key, meta in SAMPLE_PRESETS.items():
        txt_path = settings.mock_samples_dir / meta["fallback_txt"]
        content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        enriched.append({**meta, "raw_text": content})
    return {"samples": enriched}


@app.post("/api/triage/text", response_model=LeadTriageRecord)
async def triage_raw_text(payload: LeadTextIngestRequest) -> LeadTriageRecord:
    """Ingest raw email or RFP text, extract structured JSON, run RAG + scoring, and stage for HITL."""
    return await execute_full_triage_pipeline(
        raw_text=payload.raw_text,
        source_type="text_paste",
        source_filename=payload.source_name or "manual_paste.txt",
    )


@app.post("/api/triage/pdf", response_model=LeadTriageRecord)
async def triage_pdf_upload(file: UploadFile = File(...)) -> LeadTriageRecord:
    """Ingest a PDF RFP attachment, parse via PyMuPDF/pdfplumber, and execute the full triage pipeline."""
    filename = file.filename or "uploaded_rfp.pdf"
    pdf_bytes = await file.read()
    try:
        parsed_doc = extract_text_from_pdf_bytes(pdf_bytes, filename=filename)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await execute_full_triage_pipeline(
        raw_text=parsed_doc.text,
        source_type="pdf_upload",
        source_filename=f"{filename} ({parsed_doc.page_count}p via {parsed_doc.engine_used})",
    )


@app.post("/api/triage/sample/{sample_id}", response_model=LeadTriageRecord)
async def triage_preset_sample(sample_id: str) -> LeadTriageRecord:
    """Run the triage pipeline on one of the 3 curated portfolio demo scenarios."""
    if sample_id not in SAMPLE_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown sample preset '{sample_id}'.")

    meta = SAMPLE_PRESETS[sample_id]
    target_path = settings.mock_samples_dir / meta["filename"]

    if target_path.suffix.lower() == ".pdf" and target_path.exists():
        parsed = extract_text_from_pdf_path(target_path)
        raw_text = parsed.text
        source_label = f"{meta['filename']} ({parsed.page_count}p PDF via {parsed.engine_used})"
    else:
        fallback_path = settings.mock_samples_dir / meta["fallback_txt"]
        if not fallback_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample file missing at {fallback_path}")
        raw_text = fallback_path.read_text(encoding="utf-8")
        source_label = meta["fallback_txt"]

    return await execute_full_triage_pipeline(
        raw_text=raw_text,
        source_type="sample_preset",
        source_filename=source_label,
    )


@app.post("/api/lead/approve")
async def approve_or_reject_lead(payload: LeadApprovalRequest) -> dict[str, Any]:
    """
    Human-in-the-Loop (HITL) webhook endpoint triggered from Slack Block Kit interactive buttons
    or the Simulator Workbench. Unlocks outbound dispatch ONLY upon explicit human approval.
    """
    try:
        updated_lead, audit_entry = crm_simulator.process_hitl_decision(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "success",
        "lead_id": updated_lead.lead_id,
        "hitl_status": updated_lead.hitl_status,
        "dispatch_blocked": updated_lead.dispatch_blocked,
        "lead": updated_lead,
        "audit_entry": audit_entry,
    }


@app.get("/api/leads")
async def list_triaged_leads() -> dict[str, Any]:
    """Return all triaged leads in reverse chronological order."""
    return {"leads": crm_simulator.list_leads()}


@app.get("/api/crm/audit-log")
async def get_crm_audit_log() -> dict[str, Any]:
    """Return the simulated HubSpot / Airtable CRM audit trail."""
    return {"audit_log": crm_simulator.list_audit_log()}
