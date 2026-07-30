"""UDiFF bhavcopy normalizer — the recent path.

Applies from 2024-07-08 onward. UDiFF is a unified format covering every
segment, so an equity file contains derivatives-shaped columns
(``XpryDt``, ``StrkPric``, ``OptnTp``) that are empty for cash rows. Filtering
on those is what keeps futures and options out of the equity lake.

The October 2025 nomenclature update added four-digit years and session
indicators (I1/I2/F1/F2) to *filenames*, not to columns — so it affects the
fetcher's URL construction, not this module. `SsnId` is carried through the
filter because a file containing more than one session would otherwise
double-count volume.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_data.normalize.canonical import finalise, require_columns

UDIFF_MAP: dict[str, str] = {
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
    "ISIN": "isin",
}

# Cash-segment markers. UDiFF uses these to distinguish equity rows from
# derivatives rows inside one file.
CASH_SEGMENT = "CM"
CASH_INSTRUMENT_TYPES = frozenset({"STK", "EQ", "IDX"})


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def filter_cash_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only cash equity rows.

    Every filter is applied only if its column exists, so a future UDiFF
    revision that drops one of these degrades to a wider selection rather than
    an empty frame — and an empty frame would be caught by the Tier-3 row-count
    rule anyway.
    """
    out = df

    if "Sgmt" in out.columns:
        out = out[out["Sgmt"].astype("string").str.strip().str.upper() == CASH_SEGMENT]

    if "FinInstrmTp" in out.columns:
        out = out[
            out["FinInstrmTp"]
            .astype("string")
            .str.strip()
            .str.upper()
            .isin(CASH_INSTRUMENT_TYPES)
        ]

    # Cash rows have no expiry. Derivatives do. This is the most reliable
    # discriminator when instrument-type coding changes.
    if "XpryDt" in out.columns:
        expiry = out["XpryDt"].astype("string").str.strip()
        out = out[expiry.isna() | expiry.isin(["", "-"])]

    if "OptnTp" in out.columns:
        optn = out["OptnTp"].astype("string").str.strip()
        out = out[optn.isna() | optn.isin(["", "-", "XX"])]

    return out


def normalise(df: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    cleaned = _clean_columns(df)
    require_columns(cleaned, UDIFF_MAP, era="udiff")

    equity = filter_cash_equity(cleaned)
    out = equity[list(UDIFF_MAP)].rename(columns=UDIFF_MAP)

    # TtlTrfVal is in rupees — no unit conversion, unlike the legacy delivery
    # file. Delivery quantity is not part of UDiFF; it comes from the separate
    # delivery report and is merged upstream.
    return finalise(out, trade_date=trade_date)
