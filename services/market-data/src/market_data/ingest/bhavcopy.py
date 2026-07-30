"""Bhavcopy fetching.

The fetcher is an interface, not a function, for one reason: **no test in this
repo may touch the network.** Fixtures drive the normalizers and the quality
gate; `HttpBhavcopyFetcher` is exercised only against a stubbed session.

URL shapes, by era:

* legacy equity   ``/content/historical/EQUITIES/2015/MAR/cm04MAR2015bhav.csv.zip``
* legacy delivery ``/products/content/sec_bhavdata_full_04032015.csv``
* UDiFF           ``/content/cm/BhavCopy_NSE_CM_0_0_0_20250102_F_0000.csv.zip``

A missing file is not an error. It is the normal signal for a holiday, and the
calendar decides whether it is suspicious. Conflating "404" with "failure" is
how a 20-year backfill turns into 250 false alarms a year.
"""

from __future__ import annotations

import contextlib
import io
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import pandas as pd
import requests

from market_data.normalize.canonical import Era, era_for
from market_data.settings import DEFAULT_HEADERS, NSE_ARCHIVE_BASE, NSE_BASE

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


class SourceKind(StrEnum):
    LEGACY_EQUITY = "legacy_equity"
    LEGACY_DELIVERY = "legacy_delivery"
    UDIFF_EQUITY = "udiff_equity"


@dataclass(frozen=True)
class FetchResult:
    """Raw bytes plus enough provenance to explain any row derived from them."""

    trade_date: date
    kind: SourceKind
    url: str
    content: bytes

    @property
    def era(self) -> Era:
        return Era.UDIFF if self.kind is SourceKind.UDIFF_EQUITY else Era.LEGACY

    def to_frame(self) -> pd.DataFrame:
        """Decode to a raw dataframe. Handles both zipped and plain CSV."""
        raw = self.content
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise ValueError(f"{self.url}: zip contains no CSV ({zf.namelist()})")
                raw = zf.read(names[0])
        return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_values=[""])


class BhavcopyNotAvailable(Exception):  # noqa: N818 - not an error; a holiday
    """The archive has no file for this date.

    Distinct from a transport failure on purpose — a holiday and a broken
    connection require completely different responses.
    """

    def __init__(self, trade_date: date, attempted: list[str]) -> None:
        super().__init__(f"no bhavcopy for {trade_date} (tried {len(attempted)} URLs)")
        self.trade_date = trade_date
        self.attempted = attempted


def legacy_equity_url(d: date) -> str:
    return (
        f"{NSE_ARCHIVE_BASE}/content/historical/EQUITIES/{d.year}/"
        f"{_MONTHS[d.month - 1]}/cm{d.day:02d}{_MONTHS[d.month - 1]}{d.year}bhav.csv.zip"
    )


def legacy_delivery_url(d: date) -> str:
    return (
        f"{NSE_ARCHIVE_BASE}/products/content/"
        f"sec_bhavdata_full_{d.day:02d}{d.month:02d}{d.year}.csv"
    )


def udiff_equity_url(d: date) -> str:
    return (
        f"{NSE_ARCHIVE_BASE}/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def candidate_sources(d: date) -> list[tuple[SourceKind, str]]:
    """Sources to try, richest first.

    The delivery report is preferred in the legacy era because it carries
    delivery quantity and percentage — a genuine institutional-participation
    feature the plain equity bhavcopy does not have.
    """
    if era_for(d) is Era.UDIFF:
        return [
            (SourceKind.UDIFF_EQUITY, udiff_equity_url(d)),
            (SourceKind.LEGACY_DELIVERY, legacy_delivery_url(d)),
        ]
    return [
        (SourceKind.LEGACY_DELIVERY, legacy_delivery_url(d)),
        (SourceKind.LEGACY_EQUITY, legacy_equity_url(d)),
    ]


class BhavcopyFetcher(ABC):
    @abstractmethod
    def fetch(self, trade_date: date) -> FetchResult:
        """Return the richest available source, or raise BhavcopyNotAvailable."""


class HttpBhavcopyFetcher(BhavcopyFetcher):
    """Fetches from the NSE archive with retries and a deliberate rate limit.

    NSE requires a warm-up request to set cookies before archive downloads
    succeed, and rejects requests without a browser-shaped User-Agent. Neither
    is a bot defence being circumvented — the archive is public and free, and
    the check also blocks ordinary scripted access to published files.
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 2.0,
        rate_limit_seconds: float = 0.35,
        sleep=time.sleep,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff = backoff
        self._rate_limit = rate_limit_seconds
        self._sleep = sleep
        self._warmed = False

    def _warm_up(self) -> None:
        if self._warmed:
            return
        # A failed warm-up is not fatal: the download may still succeed, and if
        # it does not it reports its own failure with a real URL attached.
        with contextlib.suppress(requests.RequestException):
            self._session.get(NSE_BASE, headers=DEFAULT_HEADERS, timeout=self._timeout)
        self._warmed = True

    def _get(self, url: str) -> bytes | None:
        """Return content, or None for 404. Raises on exhausted retries."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            if attempt:
                self._sleep(self._backoff**attempt)
            try:
                resp = self._session.get(url, headers=DEFAULT_HEADERS, timeout=self._timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 200 and resp.content:
                return bytes(resp.content)
            last_error = RuntimeError(f"HTTP {resp.status_code} for {url}")
        msg = f"failed to fetch {url} after {self._max_retries} attempts"
        raise RuntimeError(msg) from last_error

    def fetch(self, trade_date: date) -> FetchResult:
        self._warm_up()
        attempted: list[str] = []
        for kind, url in candidate_sources(trade_date):
            attempted.append(url)
            self._sleep(self._rate_limit)
            content = self._get(url)
            if content:
                return FetchResult(
                    trade_date=trade_date, kind=kind, url=url, content=content
                )
        raise BhavcopyNotAvailable(trade_date, attempted)


class LocalBhavcopyFetcher(BhavcopyFetcher):
    """Reads from a directory of already-downloaded files.

    Used by tests and by a re-run over an existing download cache, so a replay
    after a rule fix does not re-hit the archive.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _candidates(self, d: date) -> list[tuple[SourceKind, Path]]:
        stamp = d.strftime("%Y%m%d")
        return [
            (SourceKind.LEGACY_DELIVERY, self._root / f"sec_bhavdata_full_{stamp}.csv"),
            (SourceKind.UDIFF_EQUITY, self._root / f"BhavCopy_NSE_CM_{stamp}.csv"),
            (SourceKind.LEGACY_EQUITY, self._root / f"cm_{stamp}.csv"),
        ]

    def fetch(self, trade_date: date) -> FetchResult:
        attempted: list[str] = []
        for kind, path in self._candidates(trade_date):
            attempted.append(str(path))
            if path.exists():
                return FetchResult(
                    trade_date=trade_date,
                    kind=kind,
                    url=path.as_uri(),
                    content=path.read_bytes(),
                )
        raise BhavcopyNotAvailable(trade_date, attempted)
