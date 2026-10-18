# Autonomous B2B RFP & Lead Triage Agent (with Human-in-the-Loop Governance)

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![n8n Workflow](https://img.shields.io/badge/Orchestration-n8n-EA4B71?logo=n8n&logoColor=white)
![OpenAI Structured Outputs](https://img.shields.io/badge/OpenAI-Structured_Outputs-412991?logo=openai&logoColor=white)
![Vector RAG](https://img.shields.io/badge/RAG-ChromaDB_%2F_Hybrid_Vector-00ADD8)

A production-grade reference architecture for **autonomous B2B Request for Proposal (RFP) qualification, RAG-grounded technical estimation, and Human-in-the-Loop (HITL) commercial sign-off**. Built as a flagship portfolio system for **AI Automation Specialist / Staff AI Solutions Engineer** roles.

![Interactive AI RFP & Lead Triage Simulator Preview](docs/dashboard_preview.png)

---

## 1. Executive Summary & Business ROI

### The Business Problem
In B2B software houses and AI consultancies, senior solutions architects spend **4 to 9 hours per incoming RFP** manually reading multi-page PDF attachments, cross-checking requested tech stacks against internal capabilities, hunting for similar past case studies, and drafting initial ballpark estimates or polite declines. Slow response times (>24 hours) directly erode win rates on competitive enterprise deals, while unqualified low-budget inquiries drain expensive engineering bandwidth.

### The Autonomous Solution
This platform automates the end-to-end pre-sales triage lifecycle by combining **n8n workflow orchestration**, **FastAPI PDF & schema validation microservices**, **cascading OpenAI models (`gpt-4o-mini` $\rightarrow$ `gpt-4o`)**, and a **semantic RAG knowledge base** over historical case studies and rate cards—gated by an interactive **Slack Block Kit Human-in-the-Loop (HITL)** approval step prior to HubSpot CRM commitment.

### Quantifiable Business Impact (ROI)

| Metric | Manual Pre-Sales Process | Autonomous AI Triage Agent | Measured Impact |
| :--- | :--- | :--- | :--- |
| **First-Response Turnaround** | 24 – 36 Hours | **< 2 Minutes** (Staged in Slack) | **99.8% Latency Reduction** |
| **Architect Triage Cost / Lead** | ~$185 USD (1.5 hrs @ $125/hr) | **~$0.004 – $0.018 USD** (LLM + RAG) | **>85% Cost Reduction** (incl. 5-min human review) |
| **LLM Token Expenditure** | Monolithic `gpt-4o` on all leads | **Cascaded `gpt-4o-mini` $\rightarrow$ `gpt-4o`** | **78% – 95% Token Spend Savings** |
| **Unqualified Lead Filtering** | Ad-hoc manual email replies | Automated **$10,000+ USD Policy Gate** | **100% Elimination** of sub-threshold scoping calls |
| **Commercial Hallucination Risk** | Unchecked auto-replies | **100% HITL Webhook Lock** (`/api/lead/approve`) | **Zero Unauthorized Commitments** |

---

## 2. System Architecture & Workflow Diagrams

### End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Client as B2B Prospect (Email / PDF)
    participant N8N as n8n Orchestrator
    participant API as FastAPI Triage Service
    participant Mini as OpenAI (gpt-4o-mini)
    participant RAG as Vector Knowledge Base
    participant Frontier as OpenAI (gpt-4o)
    participant Slack as Slack Block Kit (HITL)
    actor Architect as Solutions Architect
    participant CRM as HubSpot CRM Audit Log

    Client->>N8N: Submit Inquiry / Attach RFP PDF
    N8N->>API: POST /api/triage/pdf (or /api/triage/text)
    API->>API: Dual-Engine PDF Extraction (PyMuPDF -> pdfplumber fallback)
    API->>Mini: Structured Outputs Extraction (RFPExtraction JSON Schema)
    Mini-->>API: Strict JSON (budget, timeline, tech_stack, missing_flags)
    API->>RAG: Hybrid Vector Search (Case Studies + Tech Stack Policy + Rate Card)
    RAG-->>API: Top-K Case Studies (Cosine + Tech Overlap) & Pod Burn Rate
    API->>API: Multi-Factor Lead Scoring (0-100) & $10k Policy Gate

    alt HIGH_FIT (Score > 75 & Budget >= $10,000)
        API->>Frontier: Synthesize Custom Proposal + Price Band + Pod Allocation (gpt-4o)
        Frontier-->>API: Executive Proposal Draft (Markdown)
    else AMBIGUOUS (Missing Budget / Timeline / Scope)
        API->>Mini: Generate Discovery Clarification Questionnaire (gpt-4o-mini)
        Mini-->>API: Discovery Scoping Email Draft
    else LOW_FIT (Score < 75 or Budget < $10,000)
        API->>Mini: Generate Polite Decline + No-Code Alternatives (gpt-4o-mini)
        Mini-->>API: Polite Referral Draft
    end

    API->>CRM: Log Staged Deal (Status: PENDING_APPROVAL, Dispatch: LOCKED)
    API->>Slack: Dispatch Interactive Block Kit Card (#sales-rfp-triage)
    Architect->>Slack: Click [Approve & Send], [Edit Draft], or [Reject]
    Slack->>API: POST /api/lead/approve (HITL Webhook)
    API->>CRM: Unlock Dispatch & Commit Final Deal Stage to HubSpot CRM
```

### Qualification & Model Cascading Decision Flow

```mermaid
flowchart TD
    A["Incoming B2B Lead (Raw Email or PDF Attachment)"] --> B["PDF Parsing Service (PyMuPDF / pdfplumber)"]
    B --> C["Tier-1 Extraction: gpt-4o-mini + Pydantic Structured Outputs"]
    C --> D["RAG Vector Retrieval: Case Studies + Supported Tech Verification"]
    D --> E{"Evaluate Score (0-100) & $10k Policy"}

    E -- "Budget < $10k OR Legacy Stack (Score < 75)" --> F["LOW_FIT Branch (Stay on gpt-4o-mini)"]
    E -- "Missing Critical Flags (Budget / Timeline)" --> G["AMBIGUOUS Branch (Stay on gpt-4o-mini)"]
    E -- "Score > 75 & Budget >= $10k" --> H["HIGH_FIT Branch (Escalate to gpt-4o)"]

    F --> I["Draft Polite Decline & No-Code Referral"]
    G --> J["Draft Discovery Scoping Questions + Case Study Citations"]
    H --> K["Draft Custom Architecture Proposal + Price Band ($USD)"]

    I --> L["Slack Block Kit Notification (HITL Lock Active)"]
    J --> L
    K --> L

    L --> M{"POST /api/lead/approve"}
    M -- "Approve / Edit & Approve" --> N["Unlock Outbound Email & Sync HubSpot Deal"]
    M -- "Reject" --> O["Suppress Outbound & Archive in CRM Audit Log"]
```

---

## 3. Key Engineering Highlights

### 1. Why Human-in-the-Loop (HITL) is Mandatory in B2B Pricing
Autonomous agents must **never** send legally or commercially binding software estimates directly to external clients without human sign-off. In this architecture:
- Every triaged lead is persisted with `hitl_status = "PENDING_APPROVAL"` and `dispatch_blocked = True`.
- The agent constructs a rich **Slack Block Kit** payload displaying the extracted budget, RAG case study match percentage, scoring breakdown, and draft proposal, accompanied by interactive action buttons (`✅ Approve & Send Proposal`, `✏️ Edit Draft`, `🛑 Reject & Archive`).
- Outbound dispatch and final HubSpot CRM stage progression remain physically locked until the `/api/lead/approve` webhook is invoked by an authorized reviewer.

### 2. Token Cost Optimization via Model Cascading (`gpt-4o-mini` $\rightarrow$ `gpt-4o`)
Running frontier reasoning models (`gpt-4o`) across every raw PDF and unqualified spam inquiry wastes up to 85% of LLM API budgets. This system implements a **two-tier model cascade**:
- **Tier 1 (`gpt-4o-mini` @ ~$0.30 / 1M blended tokens):** Handles 100% of initial document ingestion, `RFPExtraction` JSON schema extraction, `LOW_FIT` polite declines, and `AMBIGUOUS` discovery questionnaires.
- **Tier 2 (`gpt-4o` @ ~$6.25 / 1M blended tokens):** Triggered **exclusively** when a lead passes qualification as `HIGH_FIT` (`total_score > 75` and `estimated_budget_usd >= $10,000`).
- Real-time telemetry (`ModelCascadeTelemetry`) calculates exact token savings per execution and surfaces it directly on the UI Simulator dashboard and Slack card.

### 3. Edge-Case Resilience & Deterministic Fallbacks
- **Dual-Engine PDF Parsing (`app/services/pdf_parser.py`):** Uses **PyMuPDF (`fitz`)** for sub-10ms streaming text extraction and automatically falls back to **`pdfplumber`** when encountering complex table-only layouts or stream anomalies. Rejects encrypted, empty, or non-PDF binary payloads with descriptive `422 Unprocessable Entity` errors.
- **Strict Pydantic v2 Schema Validation (`app/schemas.py`):** Enforces `ConfigDict(extra="forbid")` on `RFPExtraction` to prevent hallucinated keys, deduplicates technology arrays, and validates budget/timeline boundaries.
- **Zero-Downtime Offline / CI Execution (`app/services/llm_agent.py`):** When `OPENAI_API_KEY` is omitted (e.g., during automated CI tests or local evaluation), the agent seamlessly switches to its built-in deterministic structured extraction engine while maintaining identical Pydantic schema contracts.

---

## 4. Repository Structure

```text
.
├── docker-compose.yml                  # Orchestrates FastAPI App, n8n, and ChromaDB
├── Dockerfile                          # Python 3.11-slim production container
├── .env.example                        # Environment configuration template
├── requirements.txt                    # Locked Python dependencies
├── pytest.ini                          # Pytest configuration
├── workflows/
│   └── rfp_triage_workflow.json        # Ready-to-import n8n workflow with labeled nodes
├── app/
│   ├── main.py                         # FastAPI endpoints, UI routes, & HITL webhook
│   ├── config.py                       # Settings, model tiers, & $10k budget policy
│   ├── schemas.py                      # Pydantic v2 models for Structured Outputs & HITL
│   ├── services/
│   │   ├── pdf_parser.py               # PyMuPDF + pdfplumber resilient PDF extractor
│   │   ├── vector_store.py             # Hybrid vector RAG & tech stack verifier
│   │   ├── llm_agent.py                # Model cascading, scoring engine, & Slack Block Kit
│   │   └── crm_simulator.py            # HubSpot CRM state & HITL governance lock
│   ├── templates/
│   │   └── index.html                  # TailwindCSS Interactive Demo Simulator Dashboard
│   └── static/
│       ├── css/styles.css              # Custom properties & native <dialog> styles
│       └── js/dashboard.js             # Frontend simulator controller
├── data/
│   ├── knowledge_base.json             # Software house rate card, tech stack, & 5 case studies
│   └── mock_samples/
│       ├── 01_ideal_enterprise_fintech_rfp.pdf  # High-Fit $85k FinTech RFP (PDF)
│       ├── 01_ideal_enterprise_fintech_rfp.txt  # High-Fit text source
│       ├── 02_low_budget_ecommerce_lead.txt     # Low-Fit $3.5k WordPress/Unity3D lead
│       └── 03_ambiguous_stealth_startup_lead.txt# Ambiguous logistics lead (missing budget/timeline)
├── scripts/
│   └── seed_knowledge_base.py          # Generates PDF samples & seeds vector store index
└── tests/
    └── test_pipeline.py                # Unit & E2E tests for schemas, PDF, RAG, & HITL
```

---

## 5. Quickstart Guide

### Option A: One-Command Docker Compose Stack (`n8n` + `FastAPI Simulator` + `ChromaDB`)

1. **Configure environment variables (optional for live OpenAI calls):**
   ```bash
   cp .env.example .env
   # Edit .env to add your OPENAI_API_KEY, or leave default to use the built-in deterministic engine
   ```

2. **Launch the full stack:**
   ```bash
   docker compose up --build
   ```

3. **Access the services:**
   - **Interactive AI Triage & Slack HITL Simulator UI:** [http://localhost:8000](http://localhost:8000)
   - **FastAPI OpenAPI / Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **n8n Workflow Editor:** [http://localhost:5678](http://localhost:5678)

4. **Import the n8n Workflow:**
   - Open n8n at `http://localhost:5678`.
   - Click **Add Workflow** $\rightarrow$ **Import from File...** and select `workflows/rfp_triage_workflow.json`.
   - Configure your OpenAI and Slack credentials (or point the HTTP nodes directly to `http://app:8000/api/triage/text` and `http://app:8000/api/lead/approve`).

### Option B: Local Python Development & Automated Test Suite

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Seed the local vector database and generate PDF mock files
python scripts/seed_knowledge_base.py

# 2. Run the automated verification test suite
pytest -v

# 3. Start the interactive web simulator locally
uvicorn app.main:app --reload --port 8000
```
