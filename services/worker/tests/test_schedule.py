"""Worker scheduler tests.

The scheduler is the only real thing in this service at Phase 1, so it is the
only thing tested. Nothing here starts a scheduler or runs a subprocess.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from worker.schedule import JobRegistry, build_scheduler


class TestJobRegistry:
    def test_default_jobs_are_registered(self) -> None:
        names = {j.name for j in JobRegistry().jobs}
        assert names == {
            "nightly-ingest",
            "universe-refresh",
            "continuity-sweep",
            "calendar-reconcile",
        }

    def test_jobs_shell_out_to_the_cli(self) -> None:
        """One ingest code path — the nightly run must exercise the same command
        a human runs when debugging."""
        for job in JobRegistry().jobs:
            assert job.command[0] == "indicant-md"

    def test_no_job_uses_a_shell(self) -> None:
        """Fixed argv, no shell interpolation."""
        for job in JobRegistry().jobs:
            assert isinstance(job.command, list)
            assert all(isinstance(part, str) for part in job.command)

    def test_lookup_by_name(self) -> None:
        assert JobRegistry().by_name("nightly-ingest") is not None
        assert JobRegistry().by_name("nope") is None

    def test_fresh_registry_has_no_failures(self) -> None:
        assert JobRegistry().failing == []


class TestJobExecution:
    def test_success_records_ok(self) -> None:
        registry = JobRegistry()
        job = registry.by_name("nightly-ingest")
        assert job is not None
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "done", "")
            assert job.run() == "ok"
        assert job.last_run is not None
        assert job.last_output == "done"

    def test_non_zero_exit_is_recorded_not_swallowed(self) -> None:
        """A scheduler whose failures are invisible is worse than no scheduler."""
        registry = JobRegistry()
        job = registry.by_name("nightly-ingest")
        assert job is not None
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "boom")
            assert job.run() == "failed:1"
        assert registry.failing == [job]

    def test_timeout_is_recorded(self) -> None:
        registry = JobRegistry()
        job = registry.by_name("continuity-sweep")
        assert job is not None
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired([], 1)):
            assert job.run() == "timeout"

    def test_unexpected_exception_is_recorded(self) -> None:
        registry = JobRegistry()
        job = registry.by_name("nightly-ingest")
        assert job is not None
        with patch("subprocess.run", side_effect=OSError("no such binary")):
            assert job.run() == "error:OSError"

    def test_output_is_truncated(self) -> None:
        registry = JobRegistry()
        job = registry.by_name("nightly-ingest")
        assert job is not None
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "x" * 10_000, "")
            job.run()
        assert job.last_output is not None
        assert len(job.last_output) <= 4000


class TestScheduler:
    def test_every_job_is_added(self) -> None:
        registry = JobRegistry()
        scheduler = build_scheduler(registry)
        assert {j.id for j in scheduler.get_jobs()} == {j.name for j in registry.jobs}

    def test_missed_windows_coalesce_rather_than_stack(self) -> None:
        """If the host was asleep, run once — not once per missed window."""
        scheduler = build_scheduler(JobRegistry())
        for job in scheduler.get_jobs():
            assert job.coalesce is True
            assert job.max_instances == 1

    def test_scheduler_is_not_started_by_construction(self) -> None:
        assert not build_scheduler(JobRegistry()).running
