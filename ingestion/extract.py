"""
Text extraction: normalizes PDF / DOC / DOCX / TXT into a common structure:

    [{"page": <int>, "section": <str|None>, "text": <str>}, ...]

Keeping page/section info per extracted block lets the chunker preserve
provenance, which later gets written into chunk metadata.
"""

from __future__ import annotations
import os
import re
from typing import List, Dict, Optional


class ExtractionError(Exception):
    pass


def extract_text(file_path: str) -> List[Dict]:
    """Dispatch to the right extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".doc":
        return _extract_doc_legacy(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ExtractionError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _extract_pdf(file_path: str) -> List[Dict]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ExtractionError(
            "pypdf is required for PDF extraction. Install with `pip install pypdf`."
        ) from e

    reader = PdfReader(file_path)
    blocks = []
    current_section = None

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        # Heuristic: a short, capitalized standalone first line is treated as
        # a section heading for downstream metadata.
        first_line = text.strip().splitlines()[0].strip()
        if _looks_like_heading(first_line):
            current_section = first_line

        blocks.append({"page": page_num, "section": current_section, "text": text.strip()})

    if not blocks:
        raise ExtractionError(f"No extractable text found in PDF: {file_path}")
    return blocks


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _extract_docx(file_path: str) -> List[Dict]:
    try:
        import docx
    except ImportError as e:
        raise ExtractionError(
            "python-docx is required for DOCX extraction. Install with `pip install python-docx`."
        ) from e

    document = docx.Document(file_path)
    blocks = []
    current_section = None
    buffer: List[str] = []

    def flush():
        if buffer:
            blocks.append({
                "page": None,  # DOCX has no reliable page concept without rendering
                "section": current_section,
                "text": "\n".join(buffer).strip(),
            })
            buffer.clear()

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = (para.style.name or "").lower() if para.style else ""
        if style_name.startswith("heading") or style_name == "title":
            flush()
            current_section = text
            continue

        buffer.append(text)

    flush()

    if not blocks:
        raise ExtractionError(f"No extractable text found in DOCX: {file_path}")
    return blocks


# ---------------------------------------------------------------------------
# Legacy .doc (binary format) — requires an external converter
# ---------------------------------------------------------------------------
def _extract_doc_legacy(file_path: str) -> List[Dict]:
    """
    Legacy .doc files are a binary format that python-docx cannot read.
    We shell out to `antiword` or `libreoffice --headless` if available;
    otherwise raise a clear error asking the caller to pre-convert.
    """
    import shutil
    import subprocess

    if shutil.which("antiword"):
        result = subprocess.run(["antiword", file_path], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return [{"page": None, "section": None, "text": result.stdout.strip()}]

    if shutil.which("libreoffice"):
        # Convert to .txt in a temp dir, then read it back.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "txt", "--outdir", tmp, file_path],
                capture_output=True, text=True, timeout=120,
            )
            converted = os.path.join(tmp, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            if os.path.exists(converted):
                with open(converted, "r", encoding="utf-8", errors="ignore") as f:
                    return [{"page": None, "section": None, "text": f.read().strip()}]

    raise ExtractionError(
        f"Cannot extract legacy .doc file '{file_path}': install `antiword` or "
        f"`libreoffice`, or convert the file to .docx/.pdf/.txt before ingestion."
    )


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------
def _extract_txt(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        raise ExtractionError(f"Empty text file: {file_path}")
    return [{"page": None, "section": None, "text": text}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 80:
        return False
    words = line.split()
    if not words:
        return False
    capitalized_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    return capitalized_ratio > 0.6 and not line.endswith(".")
