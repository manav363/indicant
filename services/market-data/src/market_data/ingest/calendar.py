"""NSE trading calendar.

The calendar exists to answer one question: **if there is no bhavcopy for a
date, is that because the exchange was closed, or because ingestion broke?**
Without it, a silent ingestion failure is indistinguishable from a holiday and
the lake grows quiet holes.

Two sources, deliberately separate:

* **Observed** — dates for which a bhavcopy actually landed. Authoritative by
  construction: the exchange published a file, so it traded.
* **Expected** — weekday minus known holidays. A *prediction*, used only to
  decide whether a missing file deserves an alert.

Where they disagree, observed wins and the disagreement is reported. Fixed-date
national holidays are reliable and hardcoded; festival holidays follow lunar
calendars and must come from the curated list, so a year with no curated data
degrades to "weekends only" and will raise false suspicions on festival days.
That is the safe direction to fail — a false alert costs a glance, a missed
hole costs a corrupted backtest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import date, timedelta
from pathlib import Path

from indicant_contracts import TradingCalendar

# Fixed-date national holidays. Safe to hardcode: these do not move with a
# lunar calendar. When one falls on a weekend the exchange simply stays shut,
# which the weekend rule already covers.
_FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 26),   # Republic Day
    (5, 1),    # Maharashtra Day / Labour Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (12, 25),  # Christmas
)

_HOLIDAY_DATA = Path(__file__).parent / "data" / "nse_holidays.json"


class TradingCalendarService:
    """Predicts and records NSE trading days.

    `curated_holidays` maps ISO year -> list of ISO dates. Absent years fall
    back to weekends + fixed holidays only, and `has_curated_data` reports
    which years are trustworthy so callers can downgrade their confidence
    rather than silently trusting a guess.
    """

    def __init__(self, curated_holidays: dict[int, set[date]] | None = None) -> None:
        self._curated: dict[int, set[date]] = curated_holidays or {}

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_file(cls, path: Path | None = None) -> TradingCalendarService:
        path = path or _HOLIDAY_DATA
        if not path.exists():
            return cls({})
        raw: dict[str, list[str]] = json.loads(path.read_text())
        curated = {
            int(year): {date.fromisoformat(d) for d in days} for year, days in raw.items()
        }
        return cls(curated)

    def has_curated_data(self, year: int) -> bool:
        """False means holiday prediction for this year is weekends-only."""
        return year in self._curated

    # -------------------------------------------------------------- predicate

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5  # Saturday=5, Sunday=6

    def is_known_holiday(self, d: date) -> bool:
        if (d.month, d.day) in _FIXED_HOLIDAYS:
            return True
        return d in self._curated.get(d.year, set())

    def is_expected_trading_day(self, d: date) -> bool:
        """Best guess. Not authoritative — see module docstring."""
        return not self.is_weekend(d) and not self.is_known_holiday(d)

    # ------------------------------------------------------------- generation

    def expected_trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        return [d for d in _daterange(start, end) if self.is_expected_trading_day(d)]

    def build(self, start: date, end: date) -> TradingCalendar:
        return TradingCalendar(
            start=start,
            end=end,
            trading_days=tuple(self.expected_trading_days(start, end)),
        )

    # ---------------------------------------------------------- reconciliation

    def reconcile(
        self,
        *,
        start: date,
        end: date,
        observed: Iterable[date],
    ) -> CalendarReconciliation:
        """Compare prediction against what actually landed.

        `unexpected_trading_days` means the holiday list is wrong (a file
        exists for a date we thought was closed) — a data-quality issue in the
        calendar itself, fixable by curating that year.

        `missing_trading_days` is the one that matters: we expected a file and
        there is none. Either ingestion broke, or the day was a holiday we do
        not know about. Both need a human, which is why it is reported rather
        than absorbed.
        """
        observed_set = {d for d in observed if start <= d <= end}
        expected_set = set(self.expected_trading_days(start, end))
        return CalendarReconciliation(
            start=start,
            end=end,
            observed=observed_set,
            expected=expected_set,
            uncurated_years=tuple(
                sorted(
                    {d.year for d in _daterange(start, end)}
                    - {y for y in self._curated if self.has_curated_data(y)}
                )
            ),
        )

    def learn_from_observed(self, observed: Iterable[date], *, year: int) -> set[date]:
        """Derive a year's holiday list from what actually traded.

        Once a year is fully ingested, the observed set *is* the truth, so the
        weekday gaps in it are exactly the holidays. This turns the curated
        list into something the system maintains rather than something a human
        transcribes from a PDF every January.
        """
        observed_set = {d for d in observed if d.year == year}
        if not observed_set:
            raise ValueError(f"no observed trading days for {year}; cannot derive holidays")
        weekdays = {
            d
            for d in _daterange(date(year, 1, 1), date(year, 12, 31))
            if not self.is_weekend(d)
        }
        # Only trust the range we actually observed — a partially ingested year
        # would otherwise mark every future weekday as a holiday.
        upper = max(observed_set)
        lower = min(observed_set)
        derived = {d for d in weekdays - observed_set if lower <= d <= upper}
        self._curated[year] = derived
        return derived

    def to_dict(self) -> dict[str, list[str]]:
        return {
            str(year): sorted(d.isoformat() for d in days)
            for year, days in sorted(self._curated.items())
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or _HOLIDAY_DATA
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


class CalendarReconciliation:
    """Result of comparing predicted trading days against observed files."""

    def __init__(
        self,
        *,
        start: date,
        end: date,
        observed: set[date],
        expected: set[date],
        uncurated_years: tuple[int, ...] = (),
    ) -> None:
        self.start = start
        self.end = end
        self.observed = observed
        self.expected = expected
        self.uncurated_years = uncurated_years

    @property
    def missing_trading_days(self) -> tuple[date, ...]:
        """Expected a file, got none. Ingestion gap or unknown holiday."""
        return tuple(sorted(self.expected - self.observed))

    @property
    def unexpected_trading_days(self) -> tuple[date, ...]:
        """Got a file on a day we predicted closed. The holiday list is wrong."""
        return tuple(sorted(self.observed - self.expected))

    @property
    def coverage(self) -> float:
        if not self.expected:
            return 1.0
        return len(self.observed & self.expected) / len(self.expected)

    @property
    def is_clean(self) -> bool:
        return not self.missing_trading_days and not self.unexpected_trading_days

    def __repr__(self) -> str:
        return (
            f"CalendarReconciliation({self.start}..{self.end} "
            f"coverage={self.coverage:.4f} "
            f"missing={len(self.missing_trading_days)} "
            f"unexpected={len(self.unexpected_trading_days)})"
        )


def _daterange(start: date, end: date) -> Iterator[date]:
    current = start
    step = timedelta(days=1)
    while current <= end:
        yield current
        current += step
