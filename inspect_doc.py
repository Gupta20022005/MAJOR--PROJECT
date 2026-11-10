"""
Inspect what the ingestion pipeline actually extracted / chunked / stored,
without needing to re-run embedding or write anything.

Usage:
    # Show raw extracted text (plain, tables NOT preserved as tables) — quick check.
    python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --raw

    # Show the Markdown extraction, WITH tables preserved as Markdown tables —
    # this is what actually gets chunked. Also writes the .md file to disk.
    python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --markdown

    # Show how it would be chunked (500 tok / 50 tok overlap), still no storage.
    python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --chunks

    # Show what's actually sitting in the vector store right now (i.e. what a
    # query can actually retrieve).
    python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --stored
"""

import argparse
from ingestion.extract import extract_text
from ingestion.markdown_extract import extract_to_markdown_blocks, save_markdown_file
from ingestion.chunker import chunk_extracted_blocks
from ingestion.metadata import make_doc_id
from ingestion.vector_store import VectorStore
from ingestion.classify import classify_document
from query.llm_providers import get_llm_provider
import os


def show_raw(file_path: str):
    blocks = extract_text(file_path)
    print(f"\n{len(blocks)} extracted block(s) from {file_path}\n")
    for i, block in enumerate(blocks):
        print(f"--- Block {i} | page={block.get('page')} | section={block.get('section')} ---")
        preview = block["text"][:500]
        print(preview + ("..." if len(block["text"]) > 500 else ""))
        print(f"[{len(block['text'])} characters total]\n")


def show_markdown(file_path: str):
    blocks = extract_to_markdown_blocks(file_path)
    md_path = save_markdown_file(file_path, blocks)
    print(f"\n{len(blocks)} markdown block(s) from {file_path}")
    print(f"Saved to: {md_path}\n")
    for i, block in enumerate(blocks):
        print(f"--- Block {i} | page={block.get('page')} | section={block.get('section')} ---")
        preview = block["text"][:800]
        print(preview + ("..." if len(block["text"]) > 800 else ""))
        print()


def show_classify(file_path: str):
    """Classify the document WITHOUT storing anything. Rules run first; the
    configured LLM is called only if the rules can't decide."""
    blocks = extract_to_markdown_blocks(file_path)
    text = "\n\n".join(b["text"] for b in blocks)
    d = classify_document(text, os.path.basename(file_path), get_llm_provider())
    print(f"\nDocument: {file_path}")
    print(f"  method       : {d.method}   (rule status: {d.rule_status})")
    print(f"  roles        : {d.roles}")
    print(f"  tier         : {d.tier}")
    print(f"  confidence   : {d.confidence if d.confidence is not None else 'n/a (decided by rules)'}")
    print(f"  needs review : {d.needs_review}")
    print(f"  reason       : {d.reason}\n")


def show_chunks(file_path: str):
    blocks = extract_to_markdown_blocks(file_path)
    chunks = chunk_extracted_blocks(blocks)
    print(f"\n{len(chunks)} chunk(s) would be produced from {file_path}\n")
    for c in chunks:
        print(f"--- Chunk {c['chunk_index']} | page={c.get('page')} | section={c.get('section')} ---")
        print(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        print()


def show_stored(file_path: str):
    doc_id = make_doc_id(file_path)
    store = VectorStore()
    chunks = store.get_by_doc_id(doc_id)
    if not chunks:
        print(f"\nNo chunks found in the vector store for doc_id={doc_id} ({file_path}).")
        print("Either it hasn't been ingested yet, or the path doesn't match what was ingested — "
              "doc_id is derived from the exact absolute file path.")
        return
    print(f"\n{len(chunks)} chunk(s) currently stored for {file_path} (doc_id={doc_id})\n")
    for c in chunks:
        m = c["metadata"]
        print(f"--- {c['id']} | page={m.get('page')} | section={m.get('section')} | "
              f"roles={m.get('allowed_roles')} | tier={m.get('sensitivity_tier')} | "
              f"method={m.get('classification_method')} | override={m.get('chunk_override')} | "
              f"review={m.get('needs_review')} ---")
        print(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the source document")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw", action="store_true", help="Show raw extracted text blocks (tables not preserved)")
    group.add_argument("--markdown", action="store_true", help="Show markdown extraction (tables preserved) and save the .md file")
    group.add_argument("--classify", action="store_true", help="Classify the document (rules first, LLM fallback) without storing")
    group.add_argument("--chunks", action="store_true", help="Show how it would be chunked (from markdown)")
    group.add_argument("--stored", action="store_true", help="Show what's actually stored in the vector store")
    args = parser.parse_args()

    if args.raw:
        show_raw(args.file)
    elif args.markdown:
        show_markdown(args.file)
    elif args.classify:
        show_classify(args.file)
    elif args.chunks:
        show_chunks(args.file)
    elif args.stored:
        show_stored(args.file)


if __name__ == "__main__":
    main()
