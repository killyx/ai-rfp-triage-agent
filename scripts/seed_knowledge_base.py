"""Seed script to build the local vector index and generate PDF mock samples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.services.pdf_parser import render_text_to_pdf_file
from app.services.vector_store import KnowledgeVectorStore


def seed_all() -> None:
    """Generate sample PDFs from text templates and persist the RAG vector index."""
    samples_dir = settings.mock_samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Generate 01_ideal_enterprise_fintech_rfp.pdf from its text source
    ideal_txt = samples_dir / "01_ideal_enterprise_fintech_rfp.txt"
    ideal_pdf = samples_dir / "01_ideal_enterprise_fintech_rfp.pdf"
    if ideal_txt.exists():
        render_text_to_pdf_file(
            text=ideal_txt.read_text(encoding="utf-8"),
            output_path=ideal_pdf,
            title="CONFIDENTIAL B2B RFP — NovaFin Capital Partners Ltd.",
        )

    # Initialize vector store to compute and persist embeddings to storage/vector_db/vector_index.json
    store = KnowledgeVectorStore()
    cs_count = len(store.knowledge_base.get("case_studies", []))
    supported_count = len(store.get_all_supported_tech())

    print(f"[SEED] Generated PDF sample: {ideal_pdf} ({ideal_pdf.stat().st_size} bytes)")
    print(
        f"[SEED] Persisted RAG Vector Index at {store.persist_dir / 'vector_index.json'} "
        f"({cs_count} case studies, {supported_count} supported technologies, "
        f"vocabulary size={len(store.vocabulary)})"
    )


if __name__ == "__main__":
    seed_all()
