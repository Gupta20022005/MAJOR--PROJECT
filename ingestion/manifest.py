"""
Tracks which files have already been ingested (path -> content hash + mtime)
so the pipeline can be triggered on every scan/watch event but only ever
does work for documents that are new or have changed since last ingestion.
"""

from __future__ import annotations
import hashlib
import json
import os
from typing import Dict

from config import INGESTION_MANIFEST_PATH, STATE_DIR


def _file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> Dict[str, Dict]:
    if not os.path.exists(INGESTION_MANIFEST_PATH):
        return {}
    with open(INGESTION_MANIFEST_PATH, "r") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Dict]):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(INGESTION_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def is_new_or_changed(file_path: str, manifest: Dict[str, Dict]) -> bool:
    entry = manifest.get(os.path.abspath(file_path))
    if entry is None:
        return True
    return entry.get("hash") != _file_hash(file_path)


def mark_ingested(file_path: str, manifest: Dict[str, Dict], chunk_count: int):
    manifest[os.path.abspath(file_path)] = {
        "hash": _file_hash(file_path),
        "chunk_count": chunk_count,
    }


def compute_hash(file_path: str) -> str:
    return _file_hash(file_path)
