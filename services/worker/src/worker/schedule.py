"""Job registry and scheduler.

Jobs shell out to the market-data CLI rather than importing its internals. That
looks indirect but it is deliberate: there is then exactly one ingest code path,
and the nightly run exercises the same command a human runs when debugging. A
second in-process path would drift from the CLI and only diverge under load.

Every job records its last status. A scheduler whose failures are invisible is
worse than no scheduler — see the ai_trade lesson about controls that cannot
fail visibly.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


@dataclass
class Job:
    name: str
    schedule: str
    command: list[str]
    trigger: CronTrigger
    last_run: datetime | None = None
    last_status: str | None = None
    last_output: str | None = None

    def run(self) -> str:
        self.last_run = datetime.now(UTC)
        try:
            proc = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=6 * 60 * 60,
                check=False,
            )
            self.last_output = (proc.stdout or "")[-4000:]
            self.last_status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
        except subprocess.TimeoutExpired:
            self.last_status = "timeout"
            self.last_output = None
        except Exception as exc:
            self.last_status = f"error:{type(exc).__name__}"
            self.last_output = str(exc)
        return self.last_status


@dataclass
class JobRegistry:
    jobs: list[Job] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.jobs:
            return
        self.jobs = [
            Job(
                name="nightly-ingest",
                schedule="Mon-Fri 20:30 IST (15:00 UTC)",
                command=["indicant-md", "ingest"],
                trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=0),
            ),
            Job(
                name="universe-refresh",
                schedule="Mon-Fri 20:45 IST (15:15 UTC)",
                command=["indicant-md", "universe"],
                trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=15),
            ),
            Job(
                name="continuity-sweep",
                schedule="Sat 02:00 UTC",
                # The Tier-4 sweep. Weekly rather than nightly because it scans
                # the whole lake, and a silent adjustment bug is a
                # days-not-hours problem.
                command=["indicant-md", "continuity"],
                trigger=CronTrigger(day_of_week="sat", hour=2, minute=0),
            ),
            Job(
                name="calendar-reconcile",
                schedule="Sat 03:00 UTC",
                command=["indicant-md", "calendar", "--learn"],
                trigger=CronTrigger(day_of_week="sat", hour=3, minute=0),
            ),
        ]

    def by_name(self, name: str) -> Job | None:
        return next((j for j in self.jobs if j.name == name), None)

    @property
    def failing(self) -> list[Job]:
        return [j for j in self.jobs if j.last_status and j.last_status != "ok"]


def build_scheduler(
    registry: JobRegistry,
    *,
    scheduler_factory: Callable[[], BackgroundScheduler] = BackgroundScheduler,
) -> BackgroundScheduler:
    scheduler = scheduler_factory()
    for job in registry.jobs:
        scheduler.add_job(
            job.run,
            trigger=job.trigger,
            id=job.name,
            name=job.name,
            # A missed window must not stack: if the host was asleep, run once.
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
    return scheduler
