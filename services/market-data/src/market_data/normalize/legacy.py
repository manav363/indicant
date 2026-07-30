"""Legacy bhavcopy normalizer — the primary path.

Covers 2006-01-01 through 2024-07-05, which is ~18 of the 20 backfill years.
Two file shapes live in this era:

* ``cmDDMMMYYYYbhav.csv`` — the equity bhavcopy. Turnover in **rupees**.
* ``sec_bhavdata_full_DDMMYYYY.csv`` — the delivery report. Adds delivery
  quantity and percentage, and reports turnover in **lakhs**.

The delivery file is strictly richer, so it is preferred when both exist. The
lakhs-to-rupees conversion is the single most dangerous line in this module: a
missed conversion is a silent 100,000x error that propagates into every
liquidity feature and every eligibility decision.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_data.normalize.canonical import (
    LAKHS_TO_RUPEES,
    finalise,
    require_columns,
)

# Equity bhavcopy: source column -> canonical column.
EQUITY_MAP: dict[str, str] = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TOTALTRADES": "trades",
    "ISIN": "isin",
}

# Delivery bhavcopy (sec_bhavdata_full). Note TURNOVER_LACS.
DELIVERY_MAP: dict[str, str] = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "CLOSE_PRICE": "close",
    "PREV_CLOSE": "prev_close",
    "TTL_TRD_QNTY": "volume",
    "TURNOVER_LACS": "turnover",
    "NO_OF_TRADES": "trades",
    "DELIV_QTY": "delivery_qty",
    "DELIV_PER": "delivery_pct",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """NSE ships leading spaces in delivery-file headers. Strip, don't guess."""
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def is_delivery_format(df: pd.DataFrame) -> bool:
    return "DELIV_QTY" in {str(c).strip().upper() for c in df.columns}


def normalise(df: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    cleaned = _clean_columns(df)
    if is_delivery_format(cleaned):
        return _normalise_delivery(cleaned, trade_date=trade_date)
    return _normalise_equity(cleaned, trade_date=trade_date)


def _normalise_equity(df: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    require_columns(df, EQUITY_MAP, era="legacy equity")
    out = df[list(EQUITY_MAP)].rename(columns=EQUITY_MAP)
    # TOTTRDVAL is already rupees — no conversion.
    return finalise(out, trade_date=trade_date)


def _normalise_delivery(df: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    require_columns(df, DELIVERY_MAP, era="legacy delivery")
    out = df[list(DELIVERY_MAP)].rename(columns=DELIVERY_MAP)

    # DELIV_QTY and DELIV_PER are '-' for series where delivery is not
    # reported (mostly non-EQ). Coerce rather than drop: the price data is
    # still good and Tier 2 does not require delivery.
    for col in ("delivery_qty", "delivery_pct", "turnover", "volume", "trades"):
        out[col] = pd.to_numeric(
            out[col].astype("string").str.strip().replace({"-": None, "": None}),
            errors="coerce",
        )

    # The conversion that must not be missed.
    out["turnover"] = out["turnover"] * LAKHS_TO_RUPEES

    return finalise(out, trade_date=trade_date)
