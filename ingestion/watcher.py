"""
Two ways to satisfy "ingestion runs only when new documents are added":

1. EVENT-DRIVEN (recommended for a live folder / bucket): use `watchdog` to
   react to filesystem create/modify events in real time and trigger the
   pipeline immediately, debounced so a burst of file-copy events collapses
   into a single ingestion run.

2. POLLING (simplest, good for cron): call `run_once()` on a schedule (e.g.
   every N minutes via cron/Airflow/Lambda). The manifest diff inside
   IngestionPipeline.run_once() already guarantees it's a no-op when nothing
   changed, so polling frequently is cheap and safe.

Both paths funnel into the same IngestionPipeline.run_once(), so there is
exactly one code path that does the actual ingestion work.
"""

from __future__ import annotations
import logging
import threading
import time

from config import SOURCE_DOCS_DIR
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("ingestion.watcher")

DEBOUNCE_SECONDS = 5


def run_polling(interval_seconds: int = 60):
    """Simple blocking poll loop. Use for cron-less always-on deployments."""
    pipeline = IngestionPipeline()
    logger.info(f"Polling {SOURCE_DOCS_DIR} every {interval_seconds}s for new documents.")
    while True:
        pipeline.run_once()
        time.sleep(interval_seconds)


def run_event_driven(watch_dir: str = SOURCE_DOCS_DIR):
    """
    Real-time trigger using watchdog. Debounces bursts of events (e.g. a
    multi-file upload) into a single pipeline run.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as e:
        raise ImportError("watchdog is required for event-driven mode. Install with `pip install watchdog`.") from e

    pipeline = IngestionPipeline()
    debounce_timer = {"handle": None}
    lock = threading.Lock()

    def trigger_ingestion():
        with lock:
            logger.info("New document event detected — running ingestion pipeline.")
            pipeline.run_once()

    class NewDocHandler(FileSystemEventHandler):
        def _schedule(self, event):
            if event.is_directory:
                return
            with lock:
                if debounce_timer["handle"] is not None:
                    debounce_timer["handle"].cancel()
                debounce_timer["handle"] = threading.Timer(DEBOUNCE_SECONDS, trigger_ingestion)
                debounce_timer["handle"].start()

        def on_created(self, event):
            self._schedule(event)

        def on_modified(self, event):
            self._schedule(event)

    observer = Observer()
    observer.schedule(NewDocHandler(), watch_dir, recursive=True)
    observer.start()
    logger.info(f"Watching {watch_dir} for new documents (event-driven, debounce={DEBOUNCE_SECONDS}s).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "poll"
    if mode == "watch":
        run_event_driven()
    else:
        run_polling()
