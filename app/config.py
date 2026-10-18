"""Centralized configuration and engineering constants for the RFP Triage Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables with sensible defaults."""

    app_name: str = "ApexVector AI — Autonomous B2B RFP & Lead Triage Agent"
    app_version: str = "1.0.0"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    extraction_model: str = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    proposal_model: str = os.getenv("PROPOSAL_MODEL", "gpt-4o")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Business rules & thresholds
    min_budget_usd: int = int(os.getenv("MIN_BUDGET_USD", "10000"))
    high_fit_score_threshold: int = int(os.getenv("HIGH_FIT_SCORE_THRESHOLD", "75"))

    # Storage paths
    knowledge_base_path: Path = Path(
        os.getenv("KNOWLEDGE_BASE_PATH", str(BASE_DIR / "data" / "knowledge_base.json"))
    )
    chroma_persist_dir: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "storage" / "vector_db"))
    )
    mock_samples_dir: Path = BASE_DIR / "data" / "mock_samples"

    # n8n Webhook URL
    n8n_webhook_url: str = os.getenv(
        "N8N_WEBHOOK_URL", "http://localhost:5678/webhook/rfp-ingest"
    )

    @property
    def has_live_openai_key(self) -> bool:
        """Return True if a real OpenAI API key is configured."""
        key = self.openai_api_key.strip()
        return bool(key and key.startswith("sk-") and "your-openai-api-key" not in key)


settings = Settings()
