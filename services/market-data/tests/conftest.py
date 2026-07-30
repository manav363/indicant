"""Fixture builders.

Every fixture here is hand-built and small. Nothing in this suite touches the
network — the HTTP fetcher is exercised against a stubbed session, and the
normalizers and gate run on frames constructed in-process.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from indicant_contracts import CANONICAL_PRICE_COLUMNS, LakePaths

TRADE_DATE = date(2015, 3, 4)
PRIOR_DATE = date(2015, 3, 3)


def canonical_row(symbol: str = "RELIANCE", **overrides: object) -> dict[str, object]:
    """A single valid canonical row. Override any field to build a defect."""
    row: dict[str, object] = {
        "date": TRADE_DATE,
        "symbol": symbol,
        "series": "EQ",
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "prev_close": 99.0,
        "volume": 1_000_000,
        "turnover": 102_000_000.0,
        "trades": 5_000,
        "delivery_qty": 400_000,
        "delivery_pct": 40.0,
        "isin": "INE002A01018",
    }
    return row | overrides


def canonical_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """Canonical frame with correct dtypes and column order."""
    if rows is None:
        rows = [
            canonical_row("RELIANCE"),
            canonical_row("TCS", open=2500.0, high=2550.0, low=2480.0, close=2520.0,
                          prev_close=2495.0, turnover=2_520_000_000.0),
            canonical_row("INFY", open=1000.0, high=1020.0, low=995.0, close=1010.0,
                          prev_close=1005.0, turnover=1_010_000_000.0),
        ]
    df = pd.DataFrame(rows)
    return df[list(CANONICAL_PRICE_COLUMNS)]


def prior_close_frame(closes: dict[str, float] | None = None) -> pd.DataFrame:
    """Yesterday's closes, for continuity checks."""
    closes = closes or {"RELIANCE": 99.0, "TCS": 2495.0, "INFY": 1005.0}
    return pd.DataFrame(
        [{"symbol": s, "close": c, "date": PRIOR_DATE} for s, c in closes.items()]
    )


def history_frame(
    symbol: str = "RELIANCE",
    *,
    n: int = 30,
    close: float = 99.0,
    volume: int = 1_000_000,
    jitter: bool = True,
) -> pd.DataFrame:
    """A calm trailing history, so an injected outlier is unambiguous."""
    rows = []
    for i in range(n):
        drift = (0.1 if jitter else 0.0) * ((-1) ** i)
        rows.append(
            {
                "date": date(2015, 1, 2) + pd.Timedelta(days=i).to_pytimedelta(),
                "symbol": symbol,
                "series": "EQ",
                "close": close + drift,
                "volume": volume,
                "turnover": (close + drift) * volume,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def lake_paths(tmp_path) -> LakePaths:
    return LakePaths(root=tmp_path / "lake")


@pytest.fixture
def trade_date() -> date:
    return TRADE_DATE
