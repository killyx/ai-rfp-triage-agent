"""LLM Agent service implementing OpenAI Structured Outputs, Model Cascading, Scoring, and Slack Block Kit generation."""

from __future__ import annotations

import re
import time
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.schemas import (
    LeadClassification,
    ModelCascadeTelemetry,
    ProposalDraft,
    RAGRetrievalResult,
    RFPExtraction,
    ScoreBreakdown,
    SlackBlockKitPayload,
)

# Known tech keywords for deterministic extraction when running without a live OpenAI API key
KNOWN_TECH_PATTERNS: list[tuple[str, str]] = [
    (r"\bpython\b", "Python"),
    (r"\bfastapi\b", "FastAPI"),
    (r"\bdjango\b", "Django"),
    (r"\bpydantic\b", "Pydantic"),
    (r"\bpostgresql\b|\bpostgres\b", "PostgreSQL"),
    (r"\bpgvector\b", "pgvector"),
    (r"\bchromadb\b|\bchroma\b", "ChromaDB"),
    (r"\bqdrant\b", "Qdrant"),
    (r"\bredis\b", "Redis"),
    (r"\bkafka\b", "Kafka"),
    (r"\bopenai\b|\bgpt-4o\b", "OpenAI"),
    (r"\blangchain\b", "LangChain"),
    (r"\bllamaindex\b", "LlamaIndex"),
    (r"\bn8n\b", "n8n"),
    (r"\breact\b", "React"),
    (r"\bnext\.?js\b", "Next.js"),
    (r"\btypescript\b", "TypeScript"),
    (r"\btailwind(?:css)?\b", "TailwindCSS"),
    (r"\baws\b|\becs\b|\brds\b", "AWS"),
    (r"\bgcp\b|\bgoogle cloud\b", "GCP"),
    (r"\bdocker\b", "Docker"),
    (r"\bkubernetes\b|\bk8s\b", "Kubernetes"),
    (r"\bterraform\b", "Terraform"),
    (r"\bpymupdf\b|\bpdfplumber\b", "PyMuPDF"),
    (r"\brag\b", "RAG"),
    (r"\bstructured outputs\b", "Structured Outputs"),
    (r"\bwordpress\b", "WordPress"),
    (r"\bwoocommerce\b", "WooCommerce"),
    (r"\bphp\b", "PHP Legacy"),
    (r"\bunity3d\b|\bunity\b", "Unity3D"),
    (r"\bshopify\b", "Shopify Liquid Themes"),
    (r"\bmagento\b", "Magento"),
    (r"\bcobol\b", "COBOL"),
]


class RFPTriageAgent:
    """
    Production AI Agent orchestrating:
    1. Tier-1 Fast Extraction (`gpt-4o-mini` with JSON Schema Structured Outputs)
    2. Multi-Factor Scoring & Qualification (`HIGH_FIT`, `LOW_FIT`, `AMBIGUOUS`)
    3. Tier-2 Cascaded Proposal Synthesis (`gpt-4o` for High-Fit, `gpt-4o-mini` for Low/Ambiguous)
    4. Slack Block Kit interactive card construction for Human-in-the-Loop sign-off
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        if settings.has_live_openai_key:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def extract_rfp_parameters(self, raw_text: str) -> tuple[RFPExtraction, str, int]:
        """
        Extract structured RFP parameters using `gpt-4o-mini` Structured Outputs (`beta.chat.completions.parse`).
        Falls back to the deterministic structured parser if no live API key is configured or on network error.
        """
        if self._client is not None:
            try:
                completion = await self._client.beta.chat.completions.parse(
                    model=settings.extraction_model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Staff AI Solutions Architect performing structured B2B RFP extraction. "
                                "Extract client_name, company, estimated_budget_usd (integer USD or null if missing), "
                                "timeline_weeks (integer weeks or null if missing), required_tech_stack, scope_summary, "
                                "and missing_information_flags ('missing_budget', 'missing_timeline', "
                                "'vague_functional_scope', 'unspecified_tech_stack')."
                            ),
                        },
                        {"role": "user", "content": raw_text},
                    ],
                    response_format=RFPExtraction,
                )
                parsed = completion.choices[0].message.parsed
                tokens_used = completion.usage.total_tokens if completion.usage else 650
                if parsed is not None:
                    return parsed, "openai_live", tokens_used
            except Exception:
                # Graceful resilience fallback to deterministic structured extraction
                pass

        extracted = self._deterministic_extract(raw_text)
        approx_tokens = max(220, len(raw_text) // 3)
        return extracted, "deterministic_structured_engine", approx_tokens

    def _deterministic_extract(self, raw_text: str) -> RFPExtraction:
        """High-precision deterministic parser guaranteeing schema compliance in offline/CI environments."""
        text = raw_text.strip()

        # 1. Extract Client Name
        client_name = "Unknown Contact"
        client_patterns = [
            r"(?:Client Contact|Contact Person|Prepared by|From):\s*([A-Z][a-zA-Z\s.-]+?)(?:,|\n|<|\()",
            r"(?:Warm regards|Best regards|Best|Sincerely|Regards|Thanks),\s*\n+([A-Z][a-zA-Z\s.-]{2,35})",
        ]
        for pat in client_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) >= 3 and "@" not in candidate:
                    client_name = candidate
                    break

        # 2. Extract Company Name
        company = "Unidentified Organization"
        company_patterns = [
            r"Company:\s*([^\n(]+)",
            r"Organization:\s*([^\n(]+)",
            r"(?:Founder|Director|VP|CTO|CEO)[^\n]*,\s*([A-Z][A-Za-z0-9\s&.-]{2,45})",
        ]
        for pat in company_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                break
        if company == "Unidentified Organization":
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                company = lines[-1][:50] if len(lines[-1]) < 45 else "Prospective B2B Client"

        # 3. Extract Estimated Budget (USD)
        estimated_budget_usd: int | None = None
        # Check explicit "don't have a fixed budget" / "no fixed budget" / "brak budżetu"
        no_budget_phrases = re.search(
            r"(?:don't have a fixed budget|no fixed budget|without a fixed budget|budget is not locked|tbd budget|brak budżetu)",
            text,
            re.IGNORECASE,
        )
        if not no_budget_phrases:
            budget_patterns = [
                r"(?:Budget|Allocated Engineering Budget|maximum total budget)[^$\d\n]{0,40}\$\s*([\d,]+(?:\.\d+)?)\s*(k|K|thousand|USD)?",
                r"\$\s*([\d,]+)\s*(?:USD|fixed price)",
                r"\$\s*(\d{1,3}(?:,\d{3})+|\d+)\s*(k|K)\b",
            ]
            for pat in budget_patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    raw_num = float(match.group(1).replace(",", ""))
                    suffix = (match.group(2) or "").lower() if match.lastindex and match.lastindex >= 2 else ""
                    if suffix == "k" or suffix == "thousand":
                        raw_num *= 1000
                    estimated_budget_usd = int(raw_num)
                    break

        # 4. Extract Timeline (Weeks)
        timeline_weeks: int | None = None
        no_timeline_phrases = re.search(
            r"(?:timeline locked down yet|no fixed timeline|without a timeline|brak timeline)",
            text,
            re.IGNORECASE,
        )
        if not no_timeline_phrases:
            weeks_match = re.search(
                r"(?:Timeline|Target Delivery Timeline|launched and live in|within|in)\s*(?:[^\d\n]{0,25})?(\d{1,2})\s*weeks?\b",
                text,
                re.IGNORECASE,
            )
            if weeks_match:
                timeline_weeks = int(weeks_match.group(1))
            else:
                months_match = re.search(r"(\d{1,2})\s*months?\b", text, re.IGNORECASE)
                if months_match:
                    timeline_weeks = int(months_match.group(1)) * 4

        # 5. Extract Required Tech Stack
        tech_stack: list[str] = []
        for regex, canonical in KNOWN_TECH_PATTERNS:
            if re.search(regex, text, re.IGNORECASE):
                if canonical not in tech_stack:
                    tech_stack.append(canonical)

        # 6. Build Scope Summary
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if len(p.strip()) > 45
            and not p.strip().lower().startswith(("subject:", "from:", "client contact:", "request for proposal"))
        ]
        if paragraphs:
            summary_candidate = paragraphs[0].replace("\n", " ")
            if len(paragraphs) > 1 and len(summary_candidate) < 140:
                summary_candidate = f"{summary_candidate} {paragraphs[1].replace(chr(10), ' ')}"
            scope_summary = re.sub(r"\s+", " ", summary_candidate)[:360].strip()
        else:
            scope_summary = re.sub(r"\s+", " ", text)[:280].strip()

        # 7. Identify Missing Information Flags
        missing_flags: list[str] = []
        if estimated_budget_usd is None:
            missing_flags.append("missing_budget")
        if timeline_weeks is None:
            missing_flags.append("missing_timeline")
        if not tech_stack:
            missing_flags.append("unspecified_tech_stack")
        if len(text) < 550 or "haven't finalized" in text.lower() or "ballpark estimate" in text.lower():
            missing_flags.append("vague_functional_scope")

        return RFPExtraction(
            client_name=client_name,
            company=company,
            estimated_budget_usd=estimated_budget_usd,
            timeline_weeks=timeline_weeks,
            required_tech_stack=tech_stack,
            scope_summary=scope_summary,
            missing_information_flags=missing_flags,
        )

    def calculate_lead_score(
        self,
        extraction: RFPExtraction,
        rag_result: RAGRetrievalResult,
    ) -> ScoreBreakdown:
        """
        Compute an explainable 0-100 lead qualification score and classify into:
        - HIGH_FIT (Score > 75 and meets minimum $10k budget)
        - LOW_FIT (Explicit budget < $10k OR unsupported legacy tech stack with Score < 75)
        - AMBIGUOUS (Missing budget/timeline/scope specifications requiring discovery clarification)
        """
        disqualifications: list[str] = []
        min_budget = rag_result.policy_minimum_budget_usd

        # 1. Budget Score (0 - 35 pts)
        if extraction.estimated_budget_usd is None:
            budget_score = 12  # Unknown budget — neither confirmed high nor disqualified low
        elif extraction.estimated_budget_usd < min_budget:
            budget_score = 4
            disqualifications.append(
                f"Budget (${extraction.estimated_budget_usd:,}) is below company minimum policy (${min_budget:,})."
            )
        elif extraction.estimated_budget_usd >= 50000:
            budget_score = 35
        elif extraction.estimated_budget_usd >= 25000:
            budget_score = 30
        else:
            budget_score = 24

        # 2. Tech Stack Score (0 - 30 pts)
        tv = rag_result.tech_verification
        if tv.unsupported_flagged:
            disqualifications.append(
                f"Requested out-of-scope technologies: {', '.join(tv.unsupported_flagged)}."
            )
        if not tv.requested_stack:
            tech_score = 14
        else:
            base_tech = int(round(tv.coverage_ratio * 28))
            penalty = len(tv.unsupported_flagged) * 9
            bonus = 2 if len(tv.supported_matches) >= 4 else 0
            tech_score = max(2, min(30, base_tech - penalty + bonus))

        # 3. Scope Clarity & RAG Domain Alignment (0 - 20 pts)
        missing_count = len(extraction.missing_information_flags)
        rag_boost = int(round(rag_result.top_similarity_score * 10))
        clarity_base = max(2, 12 - (missing_count * 4))
        scope_clarity_score = max(3, min(20, clarity_base + rag_boost))

        # 4. Timeline Realism Score (0 - 15 pts)
        if extraction.timeline_weeks is None:
            timeline_score = 6
        elif extraction.timeline_weeks < 6 and (
            len(tv.requested_stack) >= 3 or (extraction.estimated_budget_usd or 0) < 15000
        ):
            timeline_score = 3
            disqualifications.append(
                f"Unrealistic delivery timeline ({extraction.timeline_weeks} weeks) for requested multi-system scope."
            )
        elif 8 <= extraction.timeline_weeks <= 26:
            timeline_score = 15
        else:
            timeline_score = 11

        total_score = min(100, budget_score + tech_score + scope_clarity_score + timeline_score)

        # Classification routing rules per business specification:
        # Rule A: Explicit budget below $10k or dominant unsupported tech -> LOW_FIT
        if (
            extraction.estimated_budget_usd is not None
            and extraction.estimated_budget_usd < min_budget
        ) or len(tv.unsupported_flagged) >= 2:
            classification = LeadClassification.LOW_FIT
        # Rule B: Missing critical information (budget/timeline/scope) -> AMBIGUOUS
        elif (
            "missing_budget" in extraction.missing_information_flags
            or "missing_timeline" in extraction.missing_information_flags
            or "vague_functional_scope" in extraction.missing_information_flags
        ) and total_score <= settings.high_fit_score_threshold:
            classification = LeadClassification.AMBIGUOUS
        # Rule C: Score > 75 -> HIGH_FIT
        elif total_score > settings.high_fit_score_threshold:
            classification = LeadClassification.HIGH_FIT
        else:
            classification = LeadClassification.LOW_FIT

        if classification == LeadClassification.HIGH_FIT:
            reasoning = (
                f"High-Fit Enterprise Lead (Score {total_score}/100): Approved budget of "
                f"${extraction.estimated_budget_usd or 0:,} exceeds the ${min_budget:,} floor, "
                f"{len(tv.supported_matches)} core technologies match our production capabilities, "
                f"and RAG matched case study '{rag_result.matched_case_studies[0].title}' "
                f"at {int(rag_result.top_similarity_score * 100)}% similarity."
            )
        elif classification == LeadClassification.AMBIGUOUS:
            reasoning = (
                f"Ambiguous / Incomplete Brief (Score {total_score}/100): Strong domain alignment "
                f"with '{rag_result.matched_case_studies[0].title}', but missing critical parameters "
                f"({', '.join(extraction.missing_information_flags)}). Routed to Discovery Clarification flow."
            )
        else:
            reasoning = (
                f"Low-Fit Lead (Score {total_score}/100): Disqualified due to "
                f"{' '.join(disqualifications) or 'score below 75 threshold'}. "
                "Routed to polite decline with alternative no-code/partner recommendations."
            )

        return ScoreBreakdown(
            budget_score=budget_score,
            tech_stack_score=tech_score,
            scope_clarity_score=scope_clarity_score,
            timeline_realism_score=timeline_score,
            total_score=total_score,
            classification=classification,
            disqualification_reasons=disqualifications,
            reasoning_summary=reasoning,
        )

    async def generate_proposal_and_telemetry(
        self,
        extraction: RFPExtraction,
        rag_result: RAGRetrievalResult,
        scoring: ScoreBreakdown,
        extraction_mode: str,
        extraction_tokens: int,
        start_time: float,
    ) -> tuple[ProposalDraft, ModelCascadeTelemetry]:
        """
        Model Cascading Step:
        - HIGH_FIT leads escalate to the frontier model (`gpt-4o`) to craft a custom commercial proposal
          with architecture recommendations, squad allocation, and price bands.
        - LOW_FIT and AMBIGUOUS leads stay on the cost-efficient model (`gpt-4o-mini`) to generate
          polite declines or discovery clarification emails, saving ~80% in LLM token spend.
        """
        if scoring.classification == LeadClassification.HIGH_FIT:
            synthesis_model = settings.proposal_model  # gpt-4o
        else:
            synthesis_model = settings.extraction_model  # gpt-4o-mini

        proposal = self._build_structured_proposal(extraction, rag_result, scoring)
        synthesis_tokens = 780 if scoring.classification == LeadClassification.HIGH_FIT else 410

        # If live OpenAI key is present, refine body_markdown with the selected model tier
        if self._client is not None:
            try:
                resp = await self._client.chat.completions.create(
                    model=synthesis_model,
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Staff AI Solutions Engineer at ApexVector Labs. "
                                "Write a concise, executive-ready B2B response in Markdown based on the provided triage context."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Classification: {scoring.classification.value}\n"
                                f"Client: {extraction.client_name} ({extraction.company})\n"
                                f"Price Band: {proposal.estimated_price_range_usd}\n"
                                f"Base Draft:\n{proposal.body_markdown}"
                            ),
                        },
                    ],
                )
                if resp.choices and resp.choices[0].message.content:
                    proposal.body_markdown = resp.choices[0].message.content.strip()
                if resp.usage:
                    synthesis_tokens = resp.usage.total_tokens
            except Exception:
                pass

        # Calculate exact model cascading cost telemetry
        # Pricing per 1M tokens (blended in/out): gpt-4o-mini ~$0.30/1M, gpt-4o ~$6.25/1M
        mini_cost_per_token = 0.30 / 1_000_000
        frontier_cost_per_token = 6.25 / 1_000_000

        total_tokens = extraction_tokens + synthesis_tokens
        monolithic_cost = round(total_tokens * frontier_cost_per_token, 5)

        if synthesis_model == settings.proposal_model:
            actual_cost = round(
                (extraction_tokens * mini_cost_per_token) + (synthesis_tokens * frontier_cost_per_token),
                5,
            )
        else:
            actual_cost = round(total_tokens * mini_cost_per_token, 5)

        savings_pct = round(
            max(0.0, (1.0 - (actual_cost / max(0.00001, monolithic_cost))) * 100.0), 1
        )
        latency_ms = max(45, int((time.perf_counter() - start_time) * 1000))

        telemetry = ModelCascadeTelemetry(
            extraction_model=settings.extraction_model,
            synthesis_model=synthesis_model,
            execution_mode="openai_live" if extraction_mode == "openai_live" else "deterministic_structured_engine",
            extraction_tokens=extraction_tokens,
            synthesis_tokens=synthesis_tokens,
            estimated_cost_usd=actual_cost,
            monolithic_frontier_cost_usd=monolithic_cost,
            cost_savings_percentage=savings_pct,
            processing_latency_ms=latency_ms,
        )
        return proposal, telemetry

    def _build_structured_proposal(
        self,
        extraction: RFPExtraction,
        rag_result: RAGRetrievalResult,
        scoring: ScoreBreakdown,
    ) -> ProposalDraft:
        top_cs = rag_result.matched_case_studies[0] if rag_result.matched_case_studies else None
        second_cs = rag_result.matched_case_studies[1] if len(rag_result.matched_case_studies) > 1 else None
        cs_refs = [cs.title for cs in rag_result.matched_case_studies[:2]]

        if scoring.classification == LeadClassification.HIGH_FIT:
            weeks = extraction.timeline_weeks or 14
            low_est = max(65000, int((extraction.estimated_budget_usd or 80000) * 0.9))
            high_est = int((extraction.estimated_budget_usd or 80000) * 1.1)
            price_range = f"${low_est:,} – ${high_est:,} USD"
            duration_str = f"{weeks} Weeks (2-Week Architecture Discovery + {max(4, weeks - 2)}-Week Production Delivery)"
            roles_bullets = "\n".join(f"- **{r}**" for r in rag_result.tech_verification.recommended_roles)
            cs_outcomes = (
                "\n".join(f"  - {o}" for o in top_cs.business_outcomes)
                if top_cs
                else "  - 85% reduction in manual document processing latency"
            )

            body = (
                f"Dear {extraction.client_name},\n\n"
                f"Thank you for sharing the RFP for **{extraction.company}**. Having reviewed your technical requirements "
                f"and target architecture ({', '.join(rag_result.tech_verification.supported_matches[:6])}), "
                f"our engineering leadership confirmed a **High-Fit Alignment ({scoring.total_score}/100)** with our core AI & FinTech practice.\n\n"
                f"### 1. Relevant Proven Track Record (RAG Case Study Match)\n"
                f"We recently delivered **{top_cs.title if top_cs else 'Enterprise Document Intelligence'}** "
                f"({int(rag_result.top_similarity_score * 100)}% architectural similarity), achieving:\n"
                f"{cs_outcomes}\n\n"
                f"### 2. Proposed Architecture & Delivery Strategy\n"
                f"- **Ingestion & Parsing Layer:** High-throughput PDF extraction (PyMuPDF + table layout parser) with schema-enforced OpenAI Structured Outputs.\n"
                f"- **Cost-Optimized Model Cascading:** Tier-1 extraction via `gpt-4o-mini` escalating to `gpt-4o` strictly for complex reasoning & synthesis.\n"
                f"- **Human-in-the-Loop Workbench:** React + TypeScript compliance gate and Slack Block Kit notifications prior to HubSpot CRM commit.\n\n"
                f"### 3. Commercial Estimate & Dedicated Pod\n"
                f"- **Estimated Investment Band:** **{price_range}**\n"
                f"- **Target Timeline:** **{duration_str}**\n"
                f"- **Proposed Engineering Pod:**\n{roles_bullets}\n\n"
                f"We have reserved capacity for a **Technical Architecture Deep-Dive** this Thursday. Please let us know if 15:00 CET works for your team.\n\n"
                f"Best regards,\n"
                f"**Solutions Architecture Team — ApexVector Labs**"
            )
            return ProposalDraft(
                subject=f"Proposal & Architecture Blueprint: {extraction.company} AI Platform ({price_range})",
                recipient_name=extraction.client_name,
                company=extraction.company,
                classification=LeadClassification.HIGH_FIT,
                estimated_price_range_usd=price_range,
                recommended_duration_weeks=duration_str,
                referenced_case_studies=cs_refs,
                body_markdown=body,
                recommended_next_step="Approve Proposal in Slack Block Kit -> Dispatch Executive PDF & Book Architecture Call",
            )

        if scoring.classification == LeadClassification.AMBIGUOUS:
            price_range = "$6,500 USD (2-Week Paid Discovery Sprint) | MVP Estimate: $35,000 – $75,000 USD (Pending Scope)"
            duration_str = "2 Weeks Discovery Phase + 8–12 Weeks MVP Rollout"
            missing_human = [flag.replace("_", " ").title() for flag in extraction.missing_information_flags]
            missing_list = "\n".join(f"{idx}. **{item}**" for idx, item in enumerate(missing_human, start=1))

            body = (
                f"Dear {extraction.client_name},\n\n"
                f"Thank you for reaching out from **{extraction.company}**. Your initiative to automate operations with AI agents and RAG "
                f"closely mirrors our recent delivery of **{top_cs.title if top_cs else 'AI Supply Chain Orchestrator'}** "
                f"{f'and **{second_cs.title}**' if second_cs else ''}.\n\n"
                f"### Information Needed for an Accurate Engineering Estimate\n"
                f"To provide you with a binding fixed-scope range and delivery roadmap, could you share a few additional details on:\n"
                f"{missing_list}\n"
                f"- **Document & Event Volume:** Approximate number of PDFs/emails processed per month and target ERP integration (e.g., SAP, Oracle, custom REST API).\n"
                f"- **Target Budget Envelope:** Please note our custom engineering engagements start at **${rag_result.policy_minimum_budget_usd:,} USD**, "
                f"with an optional **$6,500 / 2-week Architecture Discovery Sprint** to lock down specifications.\n\n"
                f"Would you be open to a 25-minute Scoping Workshop early next week to map out your MVP parameters?\n\n"
                f"Warm regards,\n"
                f"**Solutions Architecture Team — ApexVector Labs**"
            )
            return ProposalDraft(
                subject=f"Re: {extraction.company} AI Automation — Case Studies & Scoping Questions",
                recipient_name=extraction.client_name,
                company=extraction.company,
                classification=LeadClassification.AMBIGUOUS,
                estimated_price_range_usd=price_range,
                recommended_duration_weeks=duration_str,
                referenced_case_studies=cs_refs,
                body_markdown=body,
                recommended_next_step="Approve Discovery Questionnaire in Slack -> Trigger HubSpot 'Scoping Needed' Sequence",
            )

        # LOW_FIT polite decline with alternative recommendations
        price_range = f"Declined (Below ${rag_result.policy_minimum_budget_usd:,} USD Minimum Policy / Stack Mismatch)"
        duration_str = "N/A — Recommended Partner / No-Code Alternative"
        unsupported_note = (
            f"Additionally, our engineering pods specialize in Python/FastAPI, Cloud-Native AI, and React rather than "
            f"**{', '.join(rag_result.tech_verification.unsupported_flagged)}**."
            if rag_result.tech_verification.unsupported_flagged
            else ""
        )

        body = (
            f"Dear {extraction.client_name},\n\n"
            f"Thank you for considering **ApexVector Labs** for **{extraction.company}** and sharing your launch plans.\n\n"
            f"After reviewing your requirements, we won't be the right engineering partner for this stage of your project. "
            f"Our custom AI and cloud engineering engagements have a minimum project threshold of **${rag_result.policy_minimum_budget_usd:,} USD** "
            f"to cover dedicated architecture, automated QA, and production infrastructure. {unsupported_note}\n\n"
            f"### Recommended Alternatives to Hit Your Launch Goal\n"
            f"To launch within your **${extraction.estimated_budget_usd or 3500:,} USD** budget and **{extraction.timeline_weeks or 4}-week** window, we recommend:\n"
            f"1. **No-Code / Low-Code Marketplace Stack:** Utilize **Sharetribe**, **Bubble.io**, or a pre-built **Shopify Multi-Vendor** template instead of custom development from scratch.\n"
            f"2. **Plug-and-Play AI Chatbot:** Embed **Chatbase** or **OpenAI Assistants Widget** directly via snippet (<$50/month) rather than training a custom model.\n"
            f"3. **Certified Freelance Collective:** Reach out to vetted WordPress/Mobile specialists on **Codementor** or **Toptal** for rapid template customization.\n\n"
            f"We wish you tremendous success with your upcoming launch, and we would be glad to reconnect once you are scaling toward Series A custom infrastructure.\n\n"
            f"Best regards,\n"
            f"**Partner Relations — ApexVector Labs**"
        )
        return ProposalDraft(
            subject=f"Regarding your {extraction.company} project inquiry & recommended resources",
            recipient_name=extraction.client_name,
            company=extraction.company,
            classification=LeadClassification.LOW_FIT,
            estimated_price_range_usd=price_range,
            recommended_duration_weeks=duration_str,
            referenced_case_studies=[],
            body_markdown=body,
            recommended_next_step="Approve Polite Decline -> Log Closed-Lost (Low Fit) in HubSpot CRM",
        )

    def build_slack_block_kit_payload(
        self,
        lead_id: str,
        extraction: RFPExtraction,
        rag_result: RAGRetrievalResult,
        scoring: ScoreBreakdown,
        proposal: ProposalDraft,
        telemetry: ModelCascadeTelemetry,
    ) -> SlackBlockKitPayload:
        """
        Construct an authentic Slack Block Kit interactive payload for the `#sales-rfp-triage` channel.
        Includes Human-in-the-Loop action buttons bound to `/api/lead/approve`.
        """
        badge_map = {
            LeadClassification.HIGH_FIT: "🟢 HIGH-FIT ENTERPRISE LEAD",
            LeadClassification.AMBIGUOUS: "🟡 AMBIGUOUS BRIEF (DISCOVERY NEEDED)",
            LeadClassification.LOW_FIT: "🔴 LOW-FIT / BELOW THRESHOLD",
        }
        badge = badge_map[scoring.classification]
        budget_str = (
            f"${extraction.estimated_budget_usd:,} USD"
            if extraction.estimated_budget_usd is not None
            else "⚠️ Not Disclosed"
        )
        timeline_str = (
            f"{extraction.timeline_weeks} Weeks"
            if extraction.timeline_weeks is not None
            else "⚠️ Unspecified"
        )
        top_match = (
            f"{rag_result.matched_case_studies[0].title} ({int(rag_result.top_similarity_score * 100)}%)"
            if rag_result.matched_case_studies
            else "None"
        )

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{badge} — {extraction.company}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Client Contact:*\n{extraction.client_name}"},
                    {"type": "mrkdwn", "text": f"*Qualification Score:*\n*{scoring.total_score}/100* ({scoring.classification.value})"},
                    {"type": "mrkdwn", "text": f"*Extracted Budget:*\n{budget_str}"},
                    {"type": "mrkdwn", "text": f"*Target Timeline:*\n{timeline_str}"},
                    {"type": "mrkdwn", "text": f"*Top RAG Case Study:*\n{top_match}"},
                    {"type": "mrkdwn", "text": f"*Model Cascade Savings:*\n{telemetry.cost_savings_percentage}% (`{telemetry.extraction_model}` → `{telemetry.synthesis_model}`)"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Proposed Action & Commercial Band:*\n`{proposal.estimated_price_range_usd}`\n>{scoring.reasoning_summary}",
                },
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🔒 *Human-in-the-Loop Gate Active:* Outbound email & HubSpot deal commit are *BLOCKED* until approved (`lead_id={lead_id}`).",
                    }
                ],
            },
            {
                "type": "actions",
                "block_id": f"hitl_actions_{lead_id}",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve & Send Proposal", "emoji": True},
                        "style": "primary",
                        "value": lead_id,
                        "action_id": "approve_lead",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✏️ Edit Draft", "emoji": True},
                        "value": lead_id,
                        "action_id": "edit_draft",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🛑 Reject & Archive", "emoji": True},
                        "style": "danger",
                        "value": lead_id,
                        "action_id": "reject_lead",
                    },
                ],
            },
        ]

        return SlackBlockKitPayload(
            channel="#sales-rfp-triage",
            text=f"[{scoring.classification.value}] New RFP from {extraction.company} ({scoring.total_score}/100) awaiting HITL approval.",
            blocks=blocks,
        )


rfp_agent = RFPTriageAgent()
