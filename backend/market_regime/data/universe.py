"""
market_regime/data/universe.py
───────────────────────────────
Loads and manages the NSE stock universe.

Strategy:
- Primary source: NSE India's publicly available index CSV files.
  These are free, no API key needed, and updated daily.
- Each major index (NIFTY 50, NIFTY 100, NIFTY 500) has a downloadable
  CSV at a stable NSE URL.
- We merge all indices, deduplicate, and return a clean list of tickers
  in Yahoo Finance format (e.g. "RELIANCE.NS").
- Results are cached locally so we don't hammer NSE on every run.

Why these indices:
- NIFTY 50  → large cap, most liquid, highest signal quality
- NIFTY 100 → adds mid-large cap
- NIFTY 500 → near-complete market coverage (~95% of market cap)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── NSE Index CSV URLs ────────────────────────────────────────────────────────
# NSE serves these as direct CSV downloads — no auth required.

NSE_INDEX_URLS: dict[str, str] = {
    "NIFTY50": (
        "https://archives.nseindia.com/content/indices/"
        "ind_nifty50list.csv"
    ),
    "NIFTY100": (
        "https://archives.nseindia.com/content/indices/"
        "ind_nifty100list.csv"
    ),
    "NIFTY500": (
        "https://archives.nseindia.com/content/indices/"
        "ind_nifty500list.csv"
    ),
    "NIFTYMIDCAP150": (
        "https://archives.nseindia.com/content/indices/"
        "ind_niftymidcap150list.csv"
    ),
    "NIFTYSMALLCAP250": (
        "https://archives.nseindia.com/content/indices/"
        "ind_niftysmallcap250list.csv"
    ),
}

# Fallback: well-known NIFTY 50 tickers hardcoded.
# Used when network is unavailable so the system degrades gracefully.
NIFTY50_FALLBACK = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "ZOMATO.NS",
]

DEFAULT_CACHE_DIR = Path("./data/cache")
UNIVERSE_CACHE_TTL_HOURS = 24


# ── Public API ────────────────────────────────────────────────────────────────

def load_universe(
    indices: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_hours: int = UNIVERSE_CACHE_TTL_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load the NSE stock universe for the given indices.

    Parameters
    ----------
    indices : list[str], optional
        Which NSE indices to include. Defaults to ["NIFTY50", "NIFTY500"].
        Valid options: NIFTY50, NIFTY100, NIFTY500, NIFTYMIDCAP150, NIFTYSMALLCAP250
    cache_dir : Path
        Where to cache the universe CSV.
    cache_ttl_hours : int
        How long the cache is valid.
    force_refresh : bool
        Bypass cache and re-download.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company_name, industry, index_membership
        Sorted by ticker alphabetically.
        All tickers are in Yahoo Finance format (e.g. "RELIANCE.NS").
    """
    indices = indices or ["NIFTY50", "NIFTY500"]

    cache_path = cache_dir / f"universe_{'_'.join(sorted(indices))}.parquet"

    if not force_refresh and _cache_valid(cache_path, cache_ttl_hours):
        logger.debug("Loading universe from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    frames: list[pd.DataFrame] = []

    for index_name in indices:
        if index_name not in NSE_INDEX_URLS:
            logger.warning("Unknown index '%s' — skipping.", index_name)
            continue

        df = _fetch_index(index_name, NSE_INDEX_URLS[index_name])
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error(
            "All index downloads failed — falling back to hardcoded NIFTY 50."
        )
        return _fallback_universe()

    universe = _merge_frames(frames)
    cache_dir.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(cache_path)

    logger.info(
        "Universe loaded: %d stocks across indices %s", len(universe), indices
    )
    return universe


def get_tickers(
    indices: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[str]:
    """
    Convenience function — returns just the list of tickers.

    Example
    -------
    >>> tickers = get_tickers(["NIFTY50"])
    >>> print(tickers[:3])
    ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS']
    """
    universe = load_universe(indices=indices, cache_dir=cache_dir)
    return sorted(universe["ticker"].tolist())


def search_ticker(query: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """
    Search the universe for tickers/companies matching a query string.
    Case-insensitive. Matches against ticker and company_name.

    Returns top matches sorted by relevance (exact ticker match first).
    """
    universe = load_universe(cache_dir=cache_dir)
    q = query.upper().strip()

    # Exact ticker match first
    exact = universe[universe["ticker"].str.startswith(q)]

    # Then company name contains
    name_match = universe[
        universe["company_name"].str.upper().str.contains(q, na=False)
        & ~universe.index.isin(exact.index)
    ]

    return pd.concat([exact, name_match]).head(20).reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_index(index_name: str, url: str) -> Optional[pd.DataFrame]:
    """
    Download an NSE index CSV and parse it.

    NSE CSV format (typical):
        Company Name, Industry, Symbol, Series, ISIN Code

    We extract: Symbol → ticker (.NS suffix), Company Name, Industry.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain",
        "Referer": "https://www.nseindia.com/",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))

            # Normalise column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Find symbol column (NSE uses 'symbol' column)
            symbol_col = next(
                (c for c in df.columns if "symbol" in c), None
            )
            name_col = next(
                (c for c in df.columns if "company" in c or "name" in c), None
            )
            industry_col = next(
                (c for c in df.columns if "industry" in c or "sector" in c), None
            )

            if symbol_col is None:
                logger.warning(
                    "[%s] Could not find symbol column. Columns: %s",
                    index_name, list(df.columns)
                )
                return None

            result = pd.DataFrame()
            result["ticker"] = (
                df[symbol_col].str.strip().str.upper() + ".NS"
            )
            result["company_name"] = (
                df[name_col].str.strip() if name_col else "Unknown"
            )
            result["industry"] = (
                df[industry_col].str.strip() if industry_col else "Unknown"
            )
            result["index_membership"] = index_name

            # Drop any rows where ticker looks malformed
            result = result[
                result["ticker"].str.match(r"^[A-Z0-9&\-\.]+\.NS$")
            ]

            logger.info(
                "[%s] Downloaded %d stocks from NSE.", index_name, len(result)
            )
            return result

        except Exception as exc:
            wait = 2.0 * (2 ** attempt)
            logger.warning(
                "[%s] Download attempt %d/3 failed: %s. Retrying in %.1fs.",
                index_name, attempt + 1, exc, wait,
            )
            time.sleep(wait)

    logger.error("[%s] All download attempts failed.", index_name)
    return None


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple index DataFrames, deduplicate by ticker.
    When a stock appears in multiple indices, keep the highest-tier index
    (NIFTY50 > NIFTY100 > NIFTY500 etc.) in index_membership,
    and join all memberships as a comma-separated string.
    """
    combined = pd.concat(frames, ignore_index=True)

    # Group by ticker to aggregate index memberships
    def aggregate(group: pd.DataFrame) -> pd.Series:
        # Note: ticker is the groupby key, not a column inside group
        # when include_groups=False — so we don't access group["ticker"]
        return pd.Series({
            "company_name": group["company_name"].iloc[0],
            "industry": group["industry"].iloc[0],
            "index_membership": ",".join(group["index_membership"].unique()),
        })

    merged = (
        combined
        .groupby("ticker", as_index=False)
        .apply(aggregate, include_groups=False)
        .reset_index(drop=True)
        .sort_values("ticker")
    )

    # Remove known dummy/placeholder tickers
    BLOCKLIST = {
        "DUMMYVEDL1.NS", "DUMMYVEDL2.NS", "DUMMYVEDL3.NS", "DUMMYVEDL4.NS",
    }
    merged = merged[~merged["ticker"].isin(BLOCKLIST)].reset_index(drop=True)

    return merged


def _fallback_universe() -> pd.DataFrame:
    """Return a minimal DataFrame using the hardcoded NIFTY 50 fallback list."""
    return pd.DataFrame({
        "ticker": NIFTY50_FALLBACK,
        "company_name": ["Unknown"] * len(NIFTY50_FALLBACK),
        "industry": ["Unknown"] * len(NIFTY50_FALLBACK),
        "index_membership": ["NIFTY50_FALLBACK"] * len(NIFTY50_FALLBACK),
    })


def _cache_valid(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < ttl_hours * 3600
