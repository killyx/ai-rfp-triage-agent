"""Resilient PDF ingestion and text extraction service using PyMuPDF (fitz) and pdfplumber."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


class PDFExtractionError(ValueError):
    """Raised when a PDF file is corrupted, password-protected, or contains no extractable text."""


@dataclass
class ParsedPDFDocument:
    """Result container for parsed PDF files."""

    text: str
    page_count: int
    engine_used: str
    character_count: int


def normalize_extracted_text(raw_text: str) -> str:
    """Clean control characters, normalize line breaks, and collapse excessive blank lines."""
    cleaned = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> ParsedPDFDocument:
    """
    Extract text from raw PDF bytes using PyMuPDF (fitz) as primary fast engine,
    falling back to pdfplumber for complex layouts or table-heavy pages.
    """
    if not pdf_bytes or len(pdf_bytes) < 16:
        raise PDFExtractionError(f"Uploaded file '{filename}' is empty or too small to be a valid PDF.")

    if not pdf_bytes.startswith(b"%PDF"):
        raise PDFExtractionError(f"Uploaded file '{filename}' does not have a valid %PDF header.")

    pages_text: list[str] = []
    page_count = 0
    engine_used = "PyMuPDF (fitz)"

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise PDFExtractionError(f"PDF '{filename}' is encrypted/password-protected.")
            page_count = len(doc)
            for page in doc:
                page_str = page.get_text("text") or ""
                if page_str.strip():
                    pages_text.append(page_str)
    except PDFExtractionError:
        raise
    except Exception as fitz_err:
        # Fallback to pdfplumber if PyMuPDF encounters stream errors
        try:
            engine_used = "pdfplumber"
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_str = page.extract_text() or ""
                    if page_str.strip():
                        pages_text.append(page_str)
        except Exception as plumber_err:
            raise PDFExtractionError(
                f"Failed to parse PDF '{filename}' with both PyMuPDF ({fitz_err}) and pdfplumber ({plumber_err})."
            ) from plumber_err

    # If PyMuPDF opened the document but extracted no text, try pdfplumber table/layout extraction
    if not pages_text and page_count > 0:
        try:
            engine_used = "pdfplumber (fallback)"
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_str = page.extract_text() or ""
                    if page_str.strip():
                        pages_text.append(page_str)
        except Exception:
            pass

    combined_text = normalize_extracted_text("\n\n".join(pages_text))
    if len(combined_text) < 15:
        raise PDFExtractionError(
            f"PDF '{filename}' contains no extractable text ({len(combined_text)} chars). "
            "Scanned image-only PDFs require OCR pre-processing."
        )

    return ParsedPDFDocument(
        text=combined_text,
        page_count=page_count,
        engine_used=engine_used,
        character_count=len(combined_text),
    )


def extract_text_from_pdf_path(pdf_path: str | Path) -> ParsedPDFDocument:
    """Convenience helper to extract text from a PDF file on disk."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    return extract_text_from_pdf_bytes(path.read_bytes(), filename=path.name)


def render_text_to_pdf_file(text: str, output_path: str | Path, title: str = "B2B Request for Proposal") -> Path:
    """Render plain text into a clean, multi-page formatted PDF document using PyMuPDF."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    lines = text.splitlines()
    lines_per_page = 42

    for page_idx in range(0, max(1, len(lines)), lines_per_page):
        page = doc.new_page(width=595, height=842)  # A4 portrait
        # Header banner
        page.draw_rect(fitz.Rect(36, 28, 559, 62), color=(0.08, 0.12, 0.22), fill=(0.08, 0.12, 0.22))
        page.insert_text(
            fitz.Point(48, 49),
            f"{title}  |  Page {(page_idx // lines_per_page) + 1}",
            fontsize=10,
            fontname="helv",
            color=(0.92, 0.95, 1.0),
        )

        chunk = "\n".join(lines[page_idx : page_idx + lines_per_page])
        text_rect = fitz.Rect(42, 78, 553, 800)
        page.insert_textbox(
            text_rect,
            chunk,
            fontsize=10.2,
            fontname="helv",
            color=(0.1, 0.12, 0.16),
            align=fitz.TEXT_ALIGN_LEFT,
        )

    doc.save(str(out))
    doc.close()
    return out
