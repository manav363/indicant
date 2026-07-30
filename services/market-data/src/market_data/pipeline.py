"""Ingestion orchestration.

Wires fetch -> normalise -> gate -> store. Two entry points with different cost
profiles:

* ``ingest_day`` — nightly. One day, full context (prior close, history,
  trailing row counts) so every tier has what it needs.
* ``backfill`` — 20 years. Buffers by year and writes once per year, and is
  resumable because a 4,700-request run *will* be interrupted.

A missing file is not a failure. It is the expected signal for a holiday, and
the calendar decides whether it deserves an alert. Conflating the two turns a
backfill into 250 false alarms a year.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from indicant_contracts import CorporateAction, Dataset, Verdict

from market_data.ingest.bhavcopy import (
    BhavcopyFetcher,
    BhavcopyNotAvailable,
)
from market_data.ingest.calendar import TradingCalendarService
from market_data.normalize.canonical import SchemaError, normalise
from market_data.quality.gate import GateOutcome, QualityGate
from market_data.quality.quarantine import QuarantineStore
from market_data.quality.rules import RuleContext
from market_data.store.lake import Lake

# Trailing window for the row-count baseline and statistical rules.
TRAILING_WINDOW = 20
HISTORY_WINDOW_DAYS = 120


@dataclass
class DayResult:
    trade_date: date
    verdict: Verdict | None
    rows_accepted: int = 0
    rows_quarantined: int = 0
    skipped_reason: str | None = None
    error: str | None = None

    @property
    def ingested(self) -> bool:
        return self.verdict is not None and self.verdict.is_usable

    @property
    def is_holiday(self) -> bool:
        return self.skipped_reason == "not_available"


@dataclass
class BackfillResult:
    start: date
    end: date
    days: list[DayResult] = field(default_factory=list)

    @property
    def ingested_days(self) -> int:
        return sum(1 for d in self.days if d.ingested)

    @property
    def holidays(self) -> int:
        return sum(1 for d in self.days if d.is_holiday)

    @property
    def errors(self) -> list[DayResult]:
        return [d for d in self.days if d.error]

    @property
    def rejected(self) -> list[DayResult]:
        return [d for d in self.days if d.verdict is Verdict.REJECTED]

    @property
    def total_rows(self) -> int:
        return sum(d.rows_accepted for d in self.days)

    def summary(self) -> str:
        return (
            f"{self.start}..{self.end}: {self.ingested_days} days ingested, "
            f"{self.total_rows:,} rows, {self.holidays} no-file, "
            f"{len(self.rejected)} rejected, {len(self.errors)} errors"
        )


class IngestPipeline:
    def __init__(
        self,
        *,
        lake: Lake,
        fetcher: BhavcopyFetcher,
        calendar: TradingCalendarService | None = None,
        gate: QualityGate | None = None,
        corporate_actions: Sequence[CorporateAction] = (),
    ) -> None:
        self._lake = lake
        self._fetcher = fetcher
        self._calendar = calendar or TradingCalendarService.from_file()
        self._gate = gate or QualityGate()
        self._quarantine = QuarantineStore(lake)
        self._actions = list(corporate_actions)

    # ------------------------------------------------------------------ day

    def build_context(self, df: pd.DataFrame, trade_date: date) -> RuleContext:
        """Assemble everything the tiers need from the lake.

        Reads are bounded to a trailing window rather than the whole lake — a
        Tier-5 sigma estimate does not improve with 20 years of history, and
        scanning it per day would make the backfill quadratic.
        """
        prior_days = [d for d in self._lake.observed_trading_days() if d < trade_date]
        previous_close: pd.DataFrame | None = None
        previous_symbols: frozenset[str] = frozenset()
        trailing_counts: list[int] = []
        history: pd.DataFrame | None = None

        if prior_days:
            prev_day = prior_days[-1]
            prev_frame = self._lake.read_prices(start=prev_day, end=prev_day)
            if not prev_frame.empty:
                previous_close = prev_frame[["symbol", "close", "date"]]
                previous_symbols = frozenset(
                    prev_frame["symbol"].astype(str).str.upper()
                )

            window = prior_days[-TRAILING_WINDOW:]
            counts = self._lake.read_prices(start=window[0], end=window[-1])
            if not counts.empty:
                trailing_counts = (
                    counts.groupby("date").size().tolist()  # type: ignore[assignment]
                )

            hist_start = trade_date - timedelta(days=HISTORY_WINDOW_DAYS)
            history = self._lake.read_prices(start=hist_start, end=prior_days[-1])
            if history.empty:
                history = None

        return RuleContext(
            df=df,
            trade_date=trade_date,
            expected_trading_day=self._calendar.is_expected_trading_day(trade_date),
            previous_close=previous_close,
            previous_symbols=previous_symbols,
            trailing_row_counts=trailing_counts,
            history=history,
            corporate_actions=self._actions,
        )

    def ingest_day(self, trade_date: date, *, write: bool = True) -> DayResult:
        try:
            fetched = self._fetcher.fetch(trade_date)
        except BhavcopyNotAvailable:
            # Normal for a holiday. The calendar reconciliation decides whether
            # this specific date is suspicious.
            return DayResult(
                trade_date=trade_date, verdict=None, skipped_reason="not_available"
            )
        except Exception as exc:  # transport, zip corruption, etc.
            return DayResult(
                trade_date=trade_date, verdict=None, error=f"{type(exc).__name__}: {exc}"
            )

        try:
            raw = fetched.to_frame()
            canonical = normalise(raw, trade_date=trade_date, era=fetched.era)
        except SchemaError as exc:
            return DayResult(
                trade_date=trade_date,
                verdict=Verdict.REJECTED,
                error=f"schema: {exc} missing={exc.missing}",
            )
        except Exception as exc:
            return DayResult(
                trade_date=trade_date, verdict=None, error=f"{type(exc).__name__}: {exc}"
            )

        outcome = self._gate.run(self.build_context(canonical, trade_date))

        if write:
            self._persist(outcome)

        return DayResult(
            trade_date=trade_date,
            verdict=outcome.verdict,
            rows_accepted=len(outcome.accepted),
            rows_quarantined=len(outcome.quarantined),
        )

    def _persist(self, outcome: GateOutcome) -> None:
        self._quarantine.write_report(outcome.report)
        self._quarantine.write(outcome)
        if outcome.is_usable and not outcome.accepted.empty:
            self._lake.append_day(outcome.accepted, trade_date=outcome.report.trade_date)

    # ------------------------------------------------------------- backfill

    def backfill(
        self,
        *,
        start: date,
        end: date,
        resume: bool = True,
        progress=None,
    ) -> BackfillResult:
        """Ingest a date range, writing one parquet file per year.

        Resumability is not optional at this scale: ~4,700 sequential requests
        over a free public archive will be interrupted, and restarting from
        2006 every time makes the backfill effectively impossible to complete.
        """
        result = BackfillResult(start=start, end=end)
        done = self._load_state() if resume else set()

        for year, days in _by_year(start, end):
            buffered: list[pd.DataFrame] = []

            for day in days:
                if day in done:
                    continue
                if not self._calendar.is_expected_trading_day(day):
                    # Skip predicted holidays without a request. Over 20 years
                    # this avoids ~2,500 pointless round-trips.
                    continue

                day_result = self._ingest_for_backfill(day, buffered)
                result.days.append(day_result)
                if progress is not None:
                    progress(day_result)
                if day_result.verdict is not None or day_result.is_holiday:
                    done.add(day)

            if buffered:
                self._lake.write_year(buffered, year=year, dataset=Dataset.PRICES)
                self._save_state(done)

        self._save_state(done)
        return result

    def _ingest_for_backfill(
        self, day: date, buffer: list[pd.DataFrame]
    ) -> DayResult:
        """Like ingest_day but appends to the year buffer instead of writing.

        Context comes from the lake plus what is already buffered, so continuity
        checks still work inside a year that has not been flushed yet.
        """
        try:
            fetched = self._fetcher.fetch(day)
        except BhavcopyNotAvailable:
            return DayResult(trade_date=day, verdict=None, skipped_reason="not_available")
        except Exception as exc:
            return DayResult(
                trade_date=day, verdict=None, error=f"{type(exc).__name__}: {exc}"
            )

        try:
            canonical = normalise(fetched.to_frame(), trade_date=day, era=fetched.era)
        except SchemaError as exc:
            return DayResult(
                trade_date=day,
                verdict=Verdict.REJECTED,
                error=f"schema: {exc} missing={exc.missing}",
            )
        except Exception as exc:
            return DayResult(
                trade_date=day, verdict=None, error=f"{type(exc).__name__}: {exc}"
            )

        ctx = self._context_with_buffer(canonical, day, buffer)
        outcome = self._gate.run(ctx)

        self._quarantine.write_report(outcome.report)
        self._quarantine.write(outcome)

        if outcome.is_usable and not outcome.accepted.empty:
            buffer.append(outcome.accepted)

        return DayResult(
            trade_date=day,
            verdict=outcome.verdict,
            rows_accepted=len(outcome.accepted),
            rows_quarantined=len(outcome.quarantined),
        )

    def _context_with_buffer(
        self, df: pd.DataFrame, trade_date: date, buffer: list[pd.DataFrame]
    ) -> RuleContext:
        ctx = self.build_context(df, trade_date)
        if not buffer:
            return ctx

        recent = pd.concat(buffer[-TRAILING_WINDOW:], ignore_index=True)
        last_buffered = recent[recent["date"] == recent["date"].max()]

        return RuleContext(
            df=df,
            trade_date=trade_date,
            expected_trading_day=ctx.expected_trading_day,
            previous_close=last_buffered[["symbol", "close", "date"]],
            previous_symbols=frozenset(last_buffered["symbol"].astype(str).str.upper()),
            trailing_row_counts=recent.groupby("date").size().tolist(),
            history=recent,
            corporate_actions=self._actions,
        )

    # ----------------------------------------------------------- resume state

    def _load_state(self) -> set[date]:
        path = self._lake.paths.ingest_state
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text())
            return {date.fromisoformat(d) for d in payload.get("completed", [])}
        except (json.JSONDecodeError, ValueError):
            # A corrupt bookmark must not block a restart; the worst case is
            # re-ingesting days, which is idempotent.
            return set()

    def _save_state(self, done: set[date]) -> None:
        path = self._lake.paths.ingest_state
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"completed": sorted(d.isoformat() for d in done)}, indent=0)
        )


def _by_year(start: date, end: date) -> Iterator[tuple[int, list[date]]]:
    for year in range(start.year, end.year + 1):
        lo = max(start, date(year, 1, 1))
        hi = min(end, date(year, 12, 31))
        if lo > hi:
            continue
        days = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
        yield year, days
