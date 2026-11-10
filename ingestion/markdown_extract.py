"""
Extracts documents into Markdown instead of plain text — the key difference
from extract.py is that TABLES ARE PRESERVED as proper Markdown pipe tables
(| col | col | ... |) instead of being flattened into unstructured text,
which is what happens with a plain text extractor: cell content gets jumbled
together left-to-right with no column/row boundaries, effectively destroying
the table's structure.

Output shape matches extract.py's extract_text() — a list of
{"page": ..., "section": ..., "text": ...} blocks — so it's a drop-in
replacement everywhere downstream (chunker, metadata) already expects that
shape. The only difference is `text` is now Markdown, and tables inside it
are real Markdown tables.

`extract_to_markdown_blocks()` also gets saved to disk as an actual .md file
via `save_markdown_file()` — this is the "create a markdown file" step,
useful both for debugging (inspect_doc.py --markdown) and as an audit trail
of exactly what was chunked and embedded.
"""

from __future__ import annotations
import os
from typing import List, Dict

from config import MARKDOWN_DIR
from ingestion.extract import ExtractionError, _looks_like_heading, _extract_doc_legacy, _extract_txt
from ingestion.metadata import make_doc_id


def extract_to_markdown_blocks(file_path: str) -> List[Dict]:
    """Dispatch to the right markdown-aware extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _pdf_to_markdown_blocks(file_path)
    elif ext == ".docx":
        return _docx_to_markdown_blocks(file_path)
    elif ext == ".doc":
        # Legacy binary .doc has no table structure available without a
        # heavier converter; fall back to plain text (no tables to preserve
        # anyway in most .doc extraction paths).
        return _extract_doc_legacy(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)  # already plain text, nothing to convert
    else:
        raise ExtractionError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# PDF — text + tables via pdfplumber, tables excluded from the plain-text
# region so content isn't duplicated between the paragraph text and the table
# ---------------------------------------------------------------------------
def _pdf_to_markdown_blocks(file_path: str) -> List[Dict]:
    try:
        import pdfplumber
    except ImportError as e:
        raise ExtractionError(
            "pdfplumber is required for markdown/table-aware PDF extraction. "
            "Install with `pip install pdfplumber`."
        ) from e

    blocks = []
    current_section = None

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # Extract plain text only from the area OUTSIDE detected tables,
            # so table content isn't duplicated (once as raw jumbled text,
            # once as a proper markdown table).
            if table_bboxes:
                def _outside_tables(obj, bboxes=table_bboxes):
                    for (tx0, ttop, tx1, tbottom) in bboxes:
                        if obj.get("x0", -1) >= tx0 - 1 and obj.get("x1", 1e9) <= tx1 + 1 \
                                and obj.get("top", -1) >= ttop - 1 and obj.get("bottom", 1e9) <= tbottom + 1:
                            return False
                    return True
                text_source = page.filter(_outside_tables)
            else:
                text_source = page

            text = (text_source.extract_text() or "").strip()

            md_parts = []
            if text:
                first_line = text.splitlines()[0].strip()
                if _looks_like_heading(first_line):
                    current_section = first_line
                md_parts.append(text)

            for table in tables:
                rows = table.extract()
                md_table = _rows_to_markdown_table(rows)
                if md_table:
                    md_parts.append(md_table)

            if not md_parts:
                continue

            blocks.append({
                "page": page_num,
                "section": current_section,
                "text": "\n\n".join(md_parts),
            })

    if not blocks:
        raise ExtractionError(f"No extractable content found in PDF: {file_path}")
    return blocks


# ---------------------------------------------------------------------------
# DOCX — walk the document body in order so paragraphs and tables interleave
# exactly as they appear in the original file (python-docx's .paragraphs and
# .tables are separate lists with no ordering between them by default).
# ---------------------------------------------------------------------------
def _docx_to_markdown_blocks(file_path: str) -> List[Dict]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
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
                "text": "\n\n".join(buffer).strip(),
            })
            buffer.clear()

    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style_name = (para.style.name or "").lower() if para.style else ""
            if style_name.startswith("heading") or style_name == "title":
                flush()
                current_section = text
                continue
            buffer.append(text)

        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            md_table = _rows_to_markdown_table(rows)
            if md_table:
                buffer.append(md_table)

    flush()

    if not blocks:
        raise ExtractionError(f"No extractable content found in DOCX: {file_path}")
    return blocks


# ---------------------------------------------------------------------------
# Shared: rows -> Markdown pipe table
# ---------------------------------------------------------------------------
def _rows_to_markdown_table(rows) -> str:
    if not rows:
        return ""

    def clean(cell) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").replace("|", "\\|").strip()

    cleaned_rows = [[clean(c) for c in row] for row in rows]
    cleaned_rows = [r for r in cleaned_rows if any(c for c in r)]
    if not cleaned_rows:
        return ""

    header, *body_rows = cleaned_rows
    ncols = len(header)
    if ncols == 0:
        return ""

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in body_rows:
        row = (row + [""] * ncols)[:ncols]  # pad/truncate to header width
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save the extracted markdown to disk — this is the "create a markdown file" step
# ---------------------------------------------------------------------------
def save_markdown_file(file_path: str, blocks: List[Dict]) -> str:
    """
    Writes the extracted blocks out as a single .md file under
    data/extracted_markdown/, named by doc_id so it's stable across
    re-ingestions of the same file. Returns the path written.
    """
    os.makedirs(MARKDOWN_DIR, exist_ok=True)

    doc_id = make_doc_id(file_path)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(MARKDOWN_DIR, f"{doc_id}_{stem}.md")

    parts = []
    for block in blocks:
        header_bits = []
        if block.get("page") is not None:
            header_bits.append(f"Page {block['page']}")
        if block.get("section"):
            header_bits.append(block["section"])
        if header_bits:
            parts.append(f"## {' — '.join(header_bits)}")
        parts.append(block["text"])

    markdown_text = "\n\n".join(parts).strip() + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return out_path
