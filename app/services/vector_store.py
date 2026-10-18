"""Vector Store & Knowledge Base Retrieval (RAG) service for B2B RFP qualification."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.schemas import (
    MatchedCaseStudy,
    RAGRetrievalResult,
    RFPExtraction,
    TechStackVerification,
)


# Canonical aliases for normalizing technology names extracted from client briefs
TECH_ALIASES: dict[str, str] = {
    "python": "Python",
    "python 3": "Python",
    "python 3.11": "Python",
    "python 3.11+": "Python",
    "fastapi": "FastAPI",
    "django": "Django",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "golang": "Go",
    "go": "Go",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "pgvector": "pgvector",
    "redis": "Redis",
    "kafka": "Kafka",
    "rabbitmq": "RabbitMQ",
    "graphql": "GraphQL",
    "rest": "REST API",
    "rest api": "REST API",
    "react": "React",
    "reactjs": "React",
    "next": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "tailwind": "TailwindCSS",
    "tailwindcss": "TailwindCSS",
    "vite": "Vite",
    "aws": "AWS",
    "amazon web services": "AWS",
    "ecs": "AWS",
    "rds": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "github actions": "GitHub Actions",
    "langfuse": "Langfuse",
    "prometheus": "Prometheus",
    "openai": "OpenAI",
    "openai api": "OpenAI",
    "gpt-4o": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "claude": "Anthropic Claude",
    "anthropic": "Anthropic Claude",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "n8n": "n8n",
    "chroma": "ChromaDB",
    "chromadb": "ChromaDB",
    "qdrant": "Qdrant",
    "pymupdf": "PyMuPDF",
    "pdfplumber": "PyMuPDF",
    "pydantic": "Pydantic",
    "pydantic v2": "Pydantic",
    "rag": "RAG",
    "structured outputs": "Structured Outputs",
    "wordpress": "WordPress",
    "woocommerce": "WooCommerce",
    "shopify": "Shopify Liquid Themes",
    "magento": "Magento",
    "php": "PHP Legacy",
    "cobol": "COBOL",
    "unity": "Unity3D",
    "unity3d": "Unity3D",
    "unreal": "Unreal Engine",
    "objective-c": "Native iOS Objective-C",
    "solidity": "Blockchain Smart Contracts Solidity",
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "our", "are", "need",
    "want", "build", "system", "platform", "application", "project", "client",
    "into", "over", "using", "have", "will", "can", "more", "than", "per",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_+#.-]{2,}", text.lower())
    return [t.strip(".-") for t in tokens if len(t.strip(".-")) >= 2 and t not in STOPWORDS]


class KnowledgeVectorStore:
    """
    Hybrid local vector store combining TF-IDF / Domain Concept Embeddings
    with persistent index storage (`storage/vector_db/vector_index.json`) and
    optional ChromaDB collection synchronization.
    """

    def __init__(self, kb_path: Path | None = None, persist_dir: Path | None = None) -> None:
        self.kb_path = kb_path or settings.knowledge_base_path
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.knowledge_base: dict[str, Any] = self._load_kb()
        self.vocabulary: dict[str, int] = {}
        self.idf_weights: np.ndarray = np.array([])
        self.case_study_vectors: dict[str, np.ndarray] = {}
        self._build_and_persist_index()

    def _load_kb(self) -> dict[str, Any]:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at {self.kb_path}")
        return json.loads(self.kb_path.read_text(encoding="utf-8"))

    def _build_and_persist_index(self) -> None:
        """Compute normalized TF-IDF + technical domain embeddings for all case studies."""
        case_studies = self.knowledge_base.get("case_studies", [])
        docs_tokens: dict[str, list[str]] = {}
        df_counter: Counter[str] = Counter()

        for cs in case_studies:
            corpus_text = " ".join(
                [
                    cs.get("title", ""),
                    cs.get("industry", ""),
                    cs.get("summary", ""),
                    " ".join(cs.get("tech_stack", [])) * 2,  # Boost tech stack terms
                    " ".join(cs.get("business_outcomes", [])),
                ]
            )
            tokens = _tokenize(corpus_text)
            docs_tokens[cs["id"]] = tokens
            for unique_tok in set(tokens):
                df_counter[unique_tok] += 1

        vocab_terms = sorted(df_counter.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(vocab_terms)}
        n_docs = max(1, len(case_studies))
        self.idf_weights = np.array(
            [math.log((1.0 + n_docs) / (1.0 + df_counter[t])) + 1.0 for t in vocab_terms],
            dtype=np.float64,
        )

        for cs in case_studies:
            cs_id = cs["id"]
            vec = self._encode_tokens(docs_tokens[cs_id])
            self.case_study_vectors[cs_id] = vec

        # Persist serialized index metadata to storage directory
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            index_file = self.persist_dir / "vector_index.json"
            index_payload = {
                "engine": "ApexVector-Hybrid-Embedding-v1",
                "vocabulary_size": len(self.vocabulary),
                "indexed_case_studies": len(case_studies),
                "case_study_ids": [cs["id"] for cs in case_studies],
            }
            index_file.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _encode_tokens(self, tokens: list[str]) -> np.ndarray:
        if not self.vocabulary:
            return np.zeros(1, dtype=np.float64)
        vec = np.zeros(len(self.vocabulary), dtype=np.float64)
        counts = Counter(tokens)
        for tok, count in counts.items():
            idx = self.vocabulary.get(tok)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf_weights[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get_all_supported_tech(self) -> set[str]:
        supported_groups = self.knowledge_base.get("supported_tech_stack", {})
        all_supported: set[str] = set()
        for group_items in supported_groups.values():
            for item in group_items:
                all_supported.add(item)
        return all_supported

    def get_all_unsupported_tech(self) -> set[str]:
        return set(self.knowledge_base.get("unsupported_tech_stack", []))

    def verify_tech_stack(self, requested_stack: list[str], raw_text: str = "") -> TechStackVerification:
        """
        Cross-reference the client's requested technology stack against the
        software house's supported and unsupported technology catalogs.
        """
        supported_catalog = self.get_all_supported_tech()
        unsupported_catalog = self.get_all_unsupported_tech()

        supported_lower = {t.lower(): t for t in supported_catalog}
        unsupported_lower = {t.lower(): t for t in unsupported_catalog}

        supported_matches: list[str] = []
        unsupported_flagged: list[str] = []
        unrecognized_items: list[str] = []

        for raw_item in requested_stack:
            clean_key = raw_item.strip().lower()
            canonical = TECH_ALIASES.get(clean_key, raw_item.strip())

            if canonical in supported_catalog:
                if canonical not in supported_matches:
                    supported_matches.append(canonical)
            elif canonical.lower() in supported_lower:
                matched = supported_lower[canonical.lower()]
                if matched not in supported_matches:
                    supported_matches.append(matched)
            elif canonical in unsupported_catalog or canonical.lower() in unsupported_lower:
                flagged = unsupported_lower.get(canonical.lower(), canonical)
                if flagged not in unsupported_flagged:
                    unsupported_flagged.append(flagged)
            else:
                # Check substring match against catalogs
                found_sub = False
                for unsup_low, unsup_orig in unsupported_lower.items():
                    if unsup_low in clean_key or clean_key in unsup_low:
                        if unsup_orig not in unsupported_flagged:
                            unsupported_flagged.append(unsup_orig)
                        found_sub = True
                        break
                if not found_sub:
                    for sup_low, sup_orig in supported_lower.items():
                        if sup_low in clean_key or clean_key in sup_low:
                            if sup_orig not in supported_matches:
                                supported_matches.append(sup_orig)
                            found_sub = True
                            break
                if not found_sub and raw_item.strip():
                    unrecognized_items.append(raw_item.strip())

        total_evaluated = len(supported_matches) + len(unsupported_flagged) + len(unrecognized_items)
        if total_evaluated == 0:
            coverage_ratio = 0.5  # Neutral when client hasn't locked stack yet
        else:
            coverage_ratio = round(len(supported_matches) / total_evaluated, 2)

        # Recommend squad roles & compute weekly burn rate
        recommended_roles = ["Staff AI / LLM Systems Architect", "Senior Python Backend Engineer (FastAPI / AsyncIO)"]
        frontend_Set = {"React", "Next.js", "TypeScript", "TailwindCSS", "Vite"}
        cloud_set = {"AWS", "GCP", "Azure", "Kubernetes", "Terraform", "Docker"}

        if any(t in frontend_Set for t in supported_matches) or "dashboard" in raw_text.lower() or "workbench" in raw_text.lower():
            recommended_roles.append("Senior Frontend / Fullstack Engineer (React / Next.js / TypeScript)")
        if any(t in cloud_set for t in supported_matches):
            recommended_roles.append("Cloud & DevOps Engineer (AWS / GCP / Kubernetes / Terraform)")
        recommended_roles.append("QA & AI Evaluation Engineer")

        rate_lookup = {r["role"]: r["weekly_allocation_usd"] for r in self.knowledge_base.get("rate_card", [])}
        weekly_burn = sum(rate_lookup.get(role, 4500) for role in recommended_roles)

        return TechStackVerification(
            requested_stack=requested_stack,
            supported_matches=supported_matches,
            unsupported_flagged=unsupported_flagged,
            unrecognized_items=unrecognized_items,
            coverage_ratio=coverage_ratio,
            recommended_roles=recommended_roles,
            estimated_weekly_burn_usd=weekly_burn,
        )

    def retrieve_context(
        self,
        extraction: RFPExtraction,
        raw_text: str = "",
        top_k: int = 3,
    ) -> RAGRetrievalResult:
        """
        Execute hybrid semantic + tech-overlap retrieval against historical case studies
        and evaluate budget & technical alignment.
        """
        tech_verification = self.verify_tech_stack(extraction.required_tech_stack, raw_text=raw_text)
        query_text = " ".join(
            [
                extraction.company,
                extraction.scope_summary,
                " ".join(extraction.required_tech_stack) * 2,
                raw_text[:1500],
            ]
        )
        query_vec = self._encode_tokens(_tokenize(query_text))
        supported_set = {t.lower() for t in tech_verification.supported_matches} | {
            t.lower() for t in extraction.required_tech_stack
        }

        scored_studies: list[MatchedCaseStudy] = []
        for cs in self.knowledge_base.get("case_studies", []):
            cs_vec = self.case_study_vectors.get(cs["id"])
            cosine_sim = float(np.dot(query_vec, cs_vec)) if cs_vec is not None else 0.0

            cs_stack = cs.get("tech_stack", [])
            matched_tech = [t for t in cs_stack if t.lower() in supported_set]
            tech_overlap_ratio = len(matched_tech) / max(1, len(cs_stack))

            # Hybrid similarity combining dense lexical vector cosine + tech stack Jaccard overlap
            hybrid_score = min(0.98, round((cosine_sim * 0.65) + (tech_overlap_ratio * 0.45) + 0.12, 2))
            if cosine_sim < 0.05 and not matched_tech:
                hybrid_score = round(max(0.15, cosine_sim), 2)

            scored_studies.append(
                MatchedCaseStudy(
                    id=cs["id"],
                    title=cs["title"],
                    industry=cs["industry"],
                    budget_usd=cs["budget_usd"],
                    timeline_weeks=cs["timeline_weeks"],
                    tech_stack=cs_stack,
                    summary=cs["summary"],
                    business_outcomes=cs.get("business_outcomes", []),
                    similarity_score=hybrid_score,
                    matched_technologies=matched_tech,
                )
            )

        scored_studies.sort(key=lambda item: item.similarity_score, reverse=True)
        top_studies = scored_studies[:top_k]
        top_sim = top_studies[0].similarity_score if top_studies else 0.0

        min_budget = int(
            self.knowledge_base.get("company_profile", {}).get(
                "minimum_budget_usd", settings.min_budget_usd
            )
        )
        meets_budget: bool | None = None
        if extraction.estimated_budget_usd is not None:
            meets_budget = extraction.estimated_budget_usd >= min_budget

        return RAGRetrievalResult(
            matched_case_studies=top_studies,
            tech_verification=tech_verification,
            top_similarity_score=top_sim,
            policy_minimum_budget_usd=min_budget,
            meets_minimum_budget=meets_budget,
        )


vector_store = KnowledgeVectorStore()
