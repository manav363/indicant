"""
market_regime/data/fetcher.py
─────────────────────────────
Fetches OHLCV (Open, High, Low, Close, Volume) price data for NSE stocks
using yfinance as the data source.

Design decisions:
- All tickers must be in NSE format: "RELIANCE.NS", "TCS.NS" etc.
- Data is cached locally as parquet to avoid hammering Yahoo Finance.
- Retries with exponential backoff on network failures.
- Returns a strictly typed, validated DataFrame — callers can trust the schema.

DataFrame schema returned:
    index : DatetimeIndex (timezone-naive, date only)
    open  : float64
    high  : float64
    low   : float64
    close : float64
    volume: int64
    ticker: str       ← added column for multi-ticker DataFrames
"""

from __future__ import annotations

import time
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
# ── Ticker aliases ─────────────────────────────────────────────────────────────
# Some tickers are known under different symbols on Yahoo Finance.
# We try the alias automatically before failing.
TICKER_ALIASES: dict[str, str] = {
    "TATAMOTORS.NS": "TATAMOTORS.BO",
    "ZOMATO.NS":     "ZOMATO.BO",
    "NYKAA.NS":      "FSN.NS",
}

# ── Blocklist ──────────────────────────────────────────────────────────────────
# NSE occasionally puts dummy/placeholder tickers in index CSVs
# (e.g. during corporate restructuring). These will never have data.
TICKER_BLOCKLIST: set[str] = {
    "DUMMYVEDL1.NS", "DUMMYVEDL2.NS", "DUMMYVEDL3.NS", "DUMMYVEDL4.NS",
}
DEFAULT_CACHE_DIR = Path("./data/cache")
DEFAULT_LOOKBACK_YEARS = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0   # doubles each retry: 2s → 4s → 8s


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_hours: int = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single NSE ticker.

    Parameters
    ----------
    ticker : str
        NSE ticker in Yahoo Finance format, e.g. "RELIANCE.NS"
    start : date, optional
        Start date. If None, uses (end - lookback_years).
    end : date, optional
        End date. If None, uses today.
    lookback_years : int
        How many years back to fetch if start is not provided.
    cache_dir : Path
        Directory to store cached parquet files.
    cache_ttl_hours : int
        Cache is considered stale after this many hours.
    force_refresh : bool
        If True, bypass cache and always fetch fresh.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV DataFrame with DatetimeIndex.

    Raises
    ------
    ValueError
        If ticker format is wrong or returned data fails validation.
    RuntimeError
        If all retry attempts fail.
    """
    ticker = _normalise_ticker(ticker)

    # Reject blocklisted tickers immediately
    if ticker in TICKER_BLOCKLIST:
        raise ValueError(f"'{ticker}' is a placeholder ticker with no market data.")

    end = end or date.today()
    start = start or (end - timedelta(days=365 * lookback_years))

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_path = _cache_path(cache_dir, ticker, start, end)
    if not force_refresh and _cache_valid(cache_path, cache_ttl_hours):
        logger.debug("Cache hit for %s", ticker)
        return _load_cache(cache_path)

    # ── Fetch with retries ────────────────────────────────────────────────────
    df = _fetch_with_retries(ticker, start, end)

    # ── Validate + clean ──────────────────────────────────────────────────────
    df = _clean(df, ticker)
    _validate(df, ticker)

    # ── Persist to cache ──────────────────────────────────────────────────────
    _save_cache(df, cache_path)
    logger.info("Fetched %d rows for %s (%s → %s)", len(df), ticker, start, end)
    return df


def fetch_multiple(
    tickers: list[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    delay_between: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a list of tickers.

    Returns a dict mapping ticker → DataFrame.
    Tickers that fail are logged and excluded (no exception raised).
    A small delay between fetches avoids rate-limiting.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        try:
            df = fetch_ohlcv(
                ticker,
                start=start,
                end=end,
                lookback_years=lookback_years,
                cache_dir=cache_dir,
            )
            results[ticker] = df
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            failed.append(ticker)

        # Polite delay between requests (skip after last)
        if i < len(tickers) - 1:
            time.sleep(delay_between)

    if failed:
        logger.warning("Could not fetch %d tickers: %s", len(failed), failed)

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_ticker(ticker: str) -> str:
    """
    Ensure ticker ends with .NS (NSE) or .BO (BSE).
    Raises ValueError for obviously wrong formats.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    # If no exchange suffix, assume NSE
    if "." not in ticker:
        ticker = f"{ticker}.NS"
    suffix = ticker.split(".")[-1]
    if suffix not in {"NS", "BO"}:
        raise ValueError(
            f"Ticker '{ticker}' has unknown exchange suffix '.{suffix}'. "
            "Use '.NS' for NSE or '.BO' for BSE."
        )
    return ticker


def _fetch_with_retries(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Download data from yfinance with exponential backoff retries.

    Math: wait time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
    attempt 0 → 2s, attempt 1 → 4s, attempt 2 → 8s
    """
    last_exc: Exception = RuntimeError("Unknown error")

    # Try alias first if one exists
    tickers_to_try = [ticker]
    if ticker in TICKER_ALIASES:
        tickers_to_try.append(TICKER_ALIASES[ticker])
        logger.info("Will try alias %s for %s", TICKER_ALIASES[ticker], ticker)

    for attempt in range(MAX_RETRIES):
        # On retry, also try alias ticker
        current_ticker = tickers_to_try[min(attempt, len(tickers_to_try) - 1)]
        try:
            time.sleep(0.5)  # polite delay — avoids Yahoo rate limiting
            raw = yf.download(
                current_ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,    # adjusts for splits + dividends
                progress=False,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned empty DataFrame for '{current_ticker}'.")
            if current_ticker != ticker:
                logger.info("Used alias %s for %s", current_ticker, ticker)
            return raw

        except Exception as exc:
            last_exc = exc
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "Fetch attempt %d/%d failed for %s: %s. Retrying in %.1fs.",
                attempt + 1, MAX_RETRIES, ticker, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {MAX_RETRIES} fetch attempts failed for '{ticker}'."
    ) from last_exc


def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardise raw yfinance output into our canonical schema.

    Steps:
    1. Flatten MultiIndex columns (yfinance sometimes returns them)
    2. Lowercase column names
    3. Drop timezone from index, keep date only
    4. Remove rows where close is NaN or zero
    5. Remove duplicate dates (keep last)
    6. Sort ascending by date
    7. Add ticker column
    """
    # 1. Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # 2. Rename 'adj close' if present
    df = df.rename(columns={"adj close": "close"})

    # 3. Index → timezone-naive date
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    # 4. Drop bad rows
    df = df[df["close"].notna() & (df["close"] > 0)]

    # 5. Deduplicate
    df = df[~df.index.duplicated(keep="last")]

    # 6. Sort
    df = df.sort_index()

    # 7. Add ticker
    df["ticker"] = ticker

    # Keep only canonical columns
    keep = ["open", "high", "low", "close", "volume", "ticker"]
    df = df[[c for c in keep if c in df.columns]]

    return df


def _validate(df: pd.DataFrame, ticker: str) -> None:
    """
    Hard validation — raises ValueError if data looks wrong.

    Checks:
    - Required columns present
    - No negative prices
    - High >= Low for all rows
    - Volume >= 0
    - Minimum row count (at least 60 trading days ~ 3 months)
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{ticker}] Missing columns after cleaning: {missing}")

    if len(df) < 60:
        raise ValueError(
            f"[{ticker}] Only {len(df)} rows returned — need at least 60 trading days."
        )

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (df[col] < 0).any():
            raise ValueError(f"[{ticker}] Negative values found in column '{col}'.")

    if (df["high"] < df["low"]).any():
        n = (df["high"] < df["low"]).sum()
        raise ValueError(f"[{ticker}] {n} rows where high < low — data integrity issue.")

    if (df["volume"] < 0).any():
        raise ValueError(f"[{ticker}] Negative volume values found.")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, ticker: str, start: date, end: date) -> Path:
    safe_ticker = ticker.replace(".", "_")
    filename = f"{safe_ticker}_{start.isoformat()}_{end.isoformat()}.parquet"
    return cache_dir / filename


def _cache_valid(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < ttl_hours * 3600


def _load_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
