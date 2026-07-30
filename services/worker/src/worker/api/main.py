"""Worker — scheduled jobs.

Phase 1 scaffold. The scheduler is real; the jobs it will run are declared but
call into the market-data CLI rather than reimplementing ingestion, so there is
exactly one ingest code path and the nightly run exercises the same code a human
runs by hand.

Exposes /health only. It is not an API — the endpoint exists so Compose can tell
whether the scheduler is alive.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import FastAPI

from worker.schedule import JobRegistry, build_scheduler

app = FastAPI(title="Indicant worker", version="2.0.0")

_registry = JobRegistry()
_scheduler = build_scheduler(_registry)


@app.on_event("startup")
def _start() -> None:
    # Off by default: a scheduler that starts automatically in every environment
    # will eventually hammer a public archive from someone's laptop.
    if os.environ.get("INDICANT_WORKER_ENABLED", "").lower() in {"1", "true", "yes"}:
        _scheduler.start()


@app.on_event("shutdown")
def _stop() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "worker",
        "scheduler_running": _scheduler.running,
        "enabled": os.environ.get("INDICANT_WORKER_ENABLED", "false"),
        "now": datetime.now(UTC).isoformat(),
        "jobs": [
            {
                "name": job.name,
                "schedule": job.schedule,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_status": job.last_status,
            }
            for job in _registry.jobs
        ],
    }
