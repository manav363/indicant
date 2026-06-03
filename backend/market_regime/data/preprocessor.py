"""
market_regime/data/preprocessor.py
────────────────────────────────────
Transforms raw OHLCV data into a clean, feature-ready DataFrame.

Responsibilities:
1. Handle missing values (forward-fill, then drop leading NaNs)
2. Detect and handle outliers in price/volume
3. Add returns (log returns and simple returns)
4. Add basic derived columns used across all feature modules
5. Optionally normalise (z-score or min-max) for ML models

Math explained inline:

Log return:
    r_t = ln(P_t / P_{t-1})
    
    Why log returns instead of simple returns?
    - Log returns are additive over time: r_{0→T} = Σ r_t
    - More symmetric distribution (closer to normal)
    - Better numerical properties for ML
    - Simple return = exp(r_t) - 1

Z-score normalisation:
    z = (x - μ) / σ
    where μ = rolling mean, σ = rolling std

    We use ROLLING z-score (not global) to prevent lookahead bias —
    we only use information available up to time t when computing
    the normalisation at time t.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

NormMethod = Literal["zscore", "minmax", "none"]


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    norm_method: NormMethod = "none",
    norm_window: int = 252,        # ~1 trading year
    outlier_z_threshold: float = 5.0,
    min_rows: int = 60,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single stock's OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame from fetcher.py. Must have DatetimeIndex
        and columns: open, high, low, close, volume.
    norm_method : str
        How to normalise price/volume columns.
        "none"   → no normalisation (default, features handle their own)
        "zscore" → rolling z-score
        "minmax" → rolling min-max to [0, 1]
    norm_window : int
        Rolling window for normalisation (in trading days).
        Default 252 = 1 trading year.
    outlier_z_threshold : float
        Rows where |z-score of close| > this are flagged as outliers
        and their OHLCV values are forward-filled.
    min_rows : int
        Minimum rows required after cleaning. Raises if fewer remain.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with added columns:
        - log_return     : log(close_t / close_{t-1})
        - simple_return  : (close_t - close_{t-1}) / close_{t-1}
        - typical_price  : (high + low + close) / 3
        - price_range    : (high - low) / close   ← normalised daily range
        - is_outlier     : bool flag for detected outlier rows
    """
    df = df.copy()

    # Step 1: ensure correct dtypes
    df = _cast_dtypes(df)

    # Step 2: fill missing values
    df = _fill_missing(df)

    # Step 3: detect + handle outliers
    df, n_outliers = _handle_outliers(df, outlier_z_threshold)
    if n_outliers > 0:
        logger.warning("Replaced %d outlier rows via forward-fill.", n_outliers)

    # Step 4: add return columns
    df = _add_returns(df)

    # Step 5: add derived price columns
    df = _add_derived(df)

    # Step 6: drop leading NaNs (from returns computation)
    df = df.dropna(subset=["log_return", "simple_return"])

    # Step 7: validate minimum size
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} rows remain after preprocessing — "
            f"need at least {min_rows}."
        )

    # Step 8: optional normalisation
    if norm_method != "none":
        df = _normalise(df, method=norm_method, window=norm_window)

    logger.debug("Preprocessed: %d rows, %d columns.", len(df), len(df.columns))
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Standalone helper: compute log returns for a price series.

    Math: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Parameters
    ----------
    prices : pd.Series
        Price series (close prices).

    Returns
    -------
    pd.Series
        Log return series (NaN at index 0).
    """
    return np.log(prices / prices.shift(1))


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score of a series.

    Math: z_t = (x_t - μ_{t-w:t}) / σ_{t-w:t}

    Only uses past data (window ends at t, not t+future),
    so this is safe to use as a feature without lookahead bias.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling z-score series.
    """
    roll = series.rolling(window=window, min_periods=max(2, window // 4))
    return (series - roll.mean()) / roll.std()


def rolling_minmax(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling min-max normalisation to [0, 1].

    Math: x_norm_t = (x_t - min_{t-w:t}) / (max_{t-w:t} - min_{t-w:t})

    Safe from lookahead bias — only uses past window.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Values in [0, 1], NaN where window range is zero.
    """
    roll_min = series.rolling(window=window, min_periods=1).min()
    roll_max = series.rolling(window=window, min_periods=1).max()
    denom = roll_max - roll_min
    result = (series - roll_min) / denom
    return result.where(denom > 0, other=np.nan)


# ── Internal steps ────────────────────────────────────────────────────────────

def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast OHLCV columns to correct dtypes."""
    float_cols = ["open", "high", "low", "close"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in OHLCV columns.

    Strategy:
    - Forward-fill first (carry last known price forward)
    - Back-fill any remaining leading NaNs
    - Volume: fill NaN with 0 (market was closed / no data)
    """
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].ffill().bfill()

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)

    return df


def _handle_outliers(
    df: pd.DataFrame,
    z_threshold: float,
) -> tuple[pd.DataFrame, int]:
    """
    Detect price outliers using a global z-score on log returns.

    A row is an outlier if:
        |z-score of log return| > z_threshold

    z-score = (r_t - μ_r) / σ_r   (global, not rolling — for detection only)

    Outlier rows have their OHLCV values forward-filled (replaced with
    previous day's values). This prevents extreme price spikes (data errors,
    circuit breakers) from distorting features.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    mu = log_ret.mean()
    sigma = log_ret.std()

    if sigma == 0 or np.isnan(sigma):
        df["is_outlier"] = False
        return df, 0

    z = (log_ret - mu) / sigma
    outlier_mask = z.abs() > z_threshold

    df["is_outlier"] = outlier_mask
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        ohlcv = ["open", "high", "low", "close", "volume"]
        df.loc[outlier_mask, ohlcv] = np.nan
        df[ohlcv] = df[ohlcv].ffill()

    return df, int(n_outliers)


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return columns.

    Log return   : r_t = ln(P_t / P_{t-1})
    Simple return: R_t = (P_t - P_{t-1}) / P_{t-1} = exp(r_t) - 1

    Both are NaN at index 0 (no previous price to compare).
    """
    df["log_return"] = compute_log_returns(df["close"])
    df["simple_return"] = df["close"].pct_change()
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns that are useful across multiple feature modules.

    typical_price = (H + L + C) / 3
        Used in VWAP, money flow index.

    price_range = (H - L) / C
        Normalised daily range. Proxy for intraday volatility.
        Dividing by close makes it scale-invariant across stocks.
    """
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    return df


def _normalise(
    df: pd.DataFrame,
    method: NormMethod,
    window: int,
) -> pd.DataFrame:
    """
    Apply rolling normalisation to OHLCV + return columns.
    Skips boolean/string/ticker columns.
    """
    skip = {"ticker", "is_outlier"}
    numeric_cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        if method == "zscore":
            df[f"{col}_norm"] = rolling_zscore(df[col], window)
        elif method == "minmax":
            df[f"{col}_norm"] = rolling_minmax(df[col], window)

    return df
