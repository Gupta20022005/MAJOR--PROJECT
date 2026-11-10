"""
Recursive character chunking, token-aware.

Strategy (same idea as LangChain's RecursiveCharacterTextSplitter):
  1. Try to split text on the "biggest" separator first (paragraph breaks).
  2. If a resulting piece is still bigger than the token budget, recursively
     split it on the next separator down (line breaks, sentence-ends, spaces,
     then raw characters as a last resort).
  3. Re-assemble pieces into chunks up to CHUNK_SIZE_TOKENS, carrying
     CHUNK_OVERLAP_TOKENS of trailing content into the next chunk so context
     isn't lost at chunk boundaries.
"""

from __future__ import annotations
from typing import List, Dict, Optional

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, RECURSIVE_SEPARATORS

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    def decode_tokens(tokens) -> str:
        return _ENC.decode(tokens)

    def encode_tokens(text: str):
        return _ENC.encode(text)

except ImportError:
    # Fallback: approximate tokens as whitespace-split words (~0.75 words/token
    # in English is closer, but a 1:1 word:token approximation is a safe,
    # dependency-free default for environments without tiktoken).
    def count_tokens(text: str) -> int:
        return len(text.split())

    def decode_tokens(tokens) -> str:
        return " ".join(tokens)

    def encode_tokens(text: str):
        return text.split()


def _split_on_separator(text: str, separators: List[str]) -> List[str]:
    """Recursively split text using the first separator that actually splits it."""
    if not separators:
        return [text]

    sep, rest = separators[0], separators[1:]

    if sep == "":
        # Last resort: split by raw characters.
        return list(text)

    if sep not in text:
        return _split_on_separator(text, rest)

    parts = text.split(sep)
    # Re-attach the separator (except for the last part) so joined text is faithful.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p != ""]


def _flatten_to_size(pieces: List[str], separators: List[str], max_tokens: int) -> List[str]:
    """Ensure every piece fits within max_tokens by recursively splitting oversized ones."""
    result = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            result.append(piece)
        elif len(separators) <= 1:
            result.append(piece)  # nothing finer to split on; accept as-is
        else:
            sub_pieces = _split_on_separator(piece, separators[1:])
            result.extend(_flatten_to_size(sub_pieces, separators[1:], max_tokens))
    return result


def recursive_character_chunk(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Split `text` into overlapping chunks of ~chunk_size_tokens tokens each,
    using a recursive separator strategy so splits happen at the most
    natural boundary possible (paragraph > line > sentence > word > char).
    """
    if not text or not text.strip():
        return []

    separators = separators or RECURSIVE_SEPARATORS
    raw_pieces = _split_on_separator(text, separators)
    pieces = _flatten_to_size(raw_pieces, separators, chunk_size_tokens)

    chunks: List[str] = []
    current_pieces: List[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if current_tokens + piece_tokens > chunk_size_tokens and current_pieces:
            # Finalize current chunk.
            chunk_text = "".join(current_pieces)
            chunks.append(chunk_text)

            # Build overlap: carry the trailing `chunk_overlap_tokens` worth of
            # tokens from this chunk into the next one.
            overlap_text = _tail_by_tokens(chunk_text, chunk_overlap_tokens)
            current_pieces = [overlap_text] if overlap_text else []
            current_tokens = count_tokens(overlap_text) if overlap_text else 0

        current_pieces.append(piece)
        current_tokens += piece_tokens

    if current_pieces:
        chunks.append("".join(current_pieces))

    return [c.strip() for c in chunks if c.strip()]


def _tail_by_tokens(text: str, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    tokens = encode_tokens(text)
    if len(tokens) <= n_tokens:
        return text
    tail_tokens = tokens[-n_tokens:]
    return decode_tokens(tail_tokens)


def chunk_extracted_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Takes the output of extract.extract_text() — a list of
    {page, section, text} blocks — and returns a flat list of chunk dicts:

        [{"text": ..., "page": ..., "section": ..., "chunk_index": ...}, ...]
    """
    chunks_out = []
    chunk_index = 0
    for block in blocks:
        for chunk_text in recursive_character_chunk(block["text"]):
            chunks_out.append({
                "text": chunk_text,
                "page": block.get("page"),
                "section": block.get("section"),
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return chunks_out
