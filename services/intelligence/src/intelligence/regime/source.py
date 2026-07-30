"""Where regime analysis gets its prices.

v1's `RegimeAggregator` called `fetch_ohlcv` directly, which meant a market-wide
regime request fanned out to ~50 live yfinance calls — from inside a request
path, with no bound on how long it could take and no way to test it offline.

The source is now an explicit dependency with no default, so reaching the
network is a decision someone made rather than something that happens. In
production it is `LakeRegimeSource`, which reads local parquet; in tests it is a
frame handed in directly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from intelligence.data.lake_client import LakeClient

# Regime classification needs enough history for ADX(14) plus a 252-day regime
# history array. Below this the classifier's output is not meaningful.
MIN_REGIME_ROWS = 260


class RegimeDataSource(Protocol):
    """What the aggregator needs, and nothing more."""

    def history(self, ticker: str, *, lookback_years: int) -> pd.DataFrame:
        """OHLCV indexed by date, oldest first. Raises if unavailable."""
        ...

    def default_tickers(self) -> list[str]:
        """Constituents to aggregate over."""
        ...


class LakeRegimeSource:
    """Reads from the local parquet lake. The production source.

    Constituents come from the point-in-time universe rather than a hardcoded
    NIFTY 50 list. A hardcoded list is survivorship bias with extra steps: it
    describes today's index and silently applies it to every historical date.
    """

    def __init__(
        self,
        client: LakeClient,
        *,
        as_of: date | None = None,
        max_constituents: int = 50,
    ) -> None:
        self._client = client
        self._as_of = as_of
        self._max = max_constituents

    def _resolve_as_of(self) -> date:
        if self._as_of is not None:
            return self._as_of
        days = self._client.trading_days()
        if not days:
            raise LakeEmptyError("the lake has no price data; run 'indicant-md backfill'")
        return days[-1]

    def history(self, ticker: str, *, lookback_years: int) -> pd.DataFrame:
        as_of = self._resolve_as_of()
        start = as_of - timedelta(days=int(lookback_years * 365.25))
        frame = self._client.read_panel(
            symbols=[ticker.upper()],
            start=start,
            end=as_of,
            adjusted=True,
            columns=["date", "symbol", "open", "high", "low", "close", "volume"],
        )
        if frame.empty:
            raise KeyError(f"{ticker}: no rows in the lake for {start}..{as_of}")
        if len(frame) < MIN_REGIME_ROWS:
            raise InsufficientHistoryError(
                f"{ticker}: {len(frame)} rows, need {MIN_REGIME_ROWS} for a "
                f"meaningful regime classification"
            )

        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"])
        # The ported classifier and feature code both expect a DatetimeIndex.
        return out.set_index("date").sort_index()

    def default_tickers(self) -> list[str]:
        as_of = self._resolve_as_of()
        eligible = self._client.eligible_symbols(as_of)
        if not eligible:
            raise LakeEmptyError(
                f"no eligible symbols as of {as_of}; run 'indicant-md universe'"
            )
        # Most liquid first — a market-wide read should be dominated by names
        # that actually carry the market, not by the long tail.
        return eligible[: self._max]


class FrameRegimeSource:
    """Serves pre-built frames. Used by tests so no suite touches the lake."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = {k.upper(): v for k, v in frames.items()}

    def history(self, ticker: str, *, lookback_years: int) -> pd.DataFrame:
        key = ticker.upper()
        if key not in self._frames:
            raise KeyError(f"{ticker}: no frame supplied")
        return self._frames[key]

    def default_tickers(self) -> list[str]:
        return sorted(self._frames)


class LakeEmptyError(RuntimeError):
    """The lake has nothing to analyse.

    Distinct from a per-symbol failure: this means the whole request is
    unanswerable, and returning a confident-looking 'neutral' regime for an
    empty lake would be a fabricated market call.
    """


class InsufficientHistoryError(ValueError):
    """Too few rows for a meaningful classification.

    Raised rather than classified-anyway: ADX over 30 rows produces a number,
    and that number is noise wearing the costume of a signal.
    """
