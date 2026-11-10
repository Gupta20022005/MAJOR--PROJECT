"""
Central logging for the RAG pipeline.

Two separate logs, on purpose:

1. App log (`logs/app.log`, rotating): normal operational logging — the
   `logger.info/warning/error(...)` calls already scattered through
   ingestion/pipeline.py, query/pipeline.py, query/llm_providers.py, etc.
   Human-readable, for debugging.

2. Audit log (`logs/audit.log`, rotating, JSON-lines): one structured JSON
   object per security/activity event — authentication, role checks,
   queries, document access, ingestion. This is the "Audit Logs / Query
   Logs / Access Logs / Monitoring" box from the RB-RAG architecture: kept
   separate from app.log so it can be retained, shipped, or audited
   independently, and so app-log noise doesn't drown out security events.

Call `setup_logging()` once at process startup (already wired into
run_query.py, api.py, ingestion/pipeline.py, ingestion/watcher.py).
"""

from __future__ import annotations
import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Optional

from config import LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT

_app_logging_configured = False


def setup_logging() -> None:
    """Configures the root logger with console + rotating file output.
    Safe to call multiple times — only configures once."""
    global _app_logging_configured
    if _app_logging_configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _app_logging_configured = True


class AuditLogger:
    """
    Structured JSON-lines audit trail. Each call to `.log(event_type, **fields)`
    appends one JSON object to logs/audit.log, e.g.:

        {"timestamp": "...", "event_type": "auth_success", "user_id": "u_eng_1", "roles": [...]}
        {"timestamp": "...", "event_type": "query_submitted", "user_id": "u_eng_1", "query": "..."}
        {"timestamp": "...", "event_type": "retrieval", "user_id": "u_eng_1", "chunk_count": 3, "sources": [...]}
        {"timestamp": "...", "event_type": "document_ingested", "doc_id": "...", "source": "...", "chunk_count": 6}

    Kept as newline-delimited JSON (not a JSON array) so it's append-only,
    crash-safe, and trivially parseable with `jq` / `pandas.read_json(lines=True)`
    / any log-shipping tool.
    """

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # keep audit entries out of app.log / console

        if not self._logger.handlers:  # avoid duplicate handlers if instantiated twice
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(LOG_DIR, "audit.log"),
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON, one per line
            self._logger.addHandler(handler)

    def log(self, event_type: str, **fields) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }
        self._logger.info(json.dumps(record, default=str))


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Singleton accessor so every module shares one AuditLogger/handler."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
