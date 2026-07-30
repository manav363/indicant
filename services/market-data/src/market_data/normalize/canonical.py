"""Canonical price schema and the dispatcher that gets any era into it.

Every bhavcopy format ends up as the same dataframe: the columns in
`CANONICAL_PRICE_COLUMNS`, the same dtypes, turnover in rupees. Downstream code
never learns that 2006 and 2026 files look nothing alike.

Unit handling is the trap worth naming. The legacy equity bhavcopy reports
turnover in rupees; the delivery bhavcopy reports it in *lakhs*. Getting that
wrong is a silent 100,000x error that every derived liquidity feature inherits,
so units are converted at the boundary and asserted by the quality gate's
turnover reconciliation rule.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandas as pd
from indicant_contracts import CANONICAL_PRICE_COLUMNS

LAKHS_TO_RUPEES = 100_000.0

CANONICAL_DTYPES: dict[str, str] = {
    "symbol": "string",
    "series": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "prev_close": "float64",
    "volume": "int64",
    "turnover": "float64",
    "trades": "Int64",
    "delivery_qty": "Int64",
    "delivery_pct": "float64",
    "isin": "string",
}


class Era(StrEnum):
    """Which bhavcopy format applies to a date.

    LEGACY covers 2006 through 2024-07-05 — roughly 18 of the 20 backfill
    years, which is why it is the primary code path and not the edge case.
    """

    LEGACY = "legacy"
    UDIFF = "udiff"


UDIFF_CUTOVER = date(2024, 7, 8)


def era_for(trade_date: date) -> Era:
    return Era.UDIFF if trade_date >= UDIFF_CUTOVER else Era.LEGACY


class SchemaError(ValueError):
    """Raised when a file does not look like the format its date implies.

    Carries the columns it did and did not find, because "schema mismatch" with
    no detail is not actionable when the upstream format changed silently.
    """

    def __init__(self, message: str, *, missing: list[str], found: list[str]) -> None:
        super().__init__(message)
        self.missing = missing
        self.found = found

    def as_evidence(self) -> dict[str, object]:
        return {"missing_columns": self.missing, "found_columns": self.found}


def require_columns(df: pd.DataFrame, required: dict[str, str], *, era: str) -> None:
    """Assert the source columns exist before mapping them.

    Fails loudly with both sides of the comparison rather than producing a
    dataframe full of NaN that looks plausible until it reaches a model.
    """
    found = [str(c) for c in df.columns]
    missing = [src for src in required if src not in df.columns]
    if missing:
        raise SchemaError(
            f"{era} bhavcopy is missing expected columns: {missing}",
            missing=missing,
            found=found,
        )


def finalise(df: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    """Apply canonical dtypes, column order and row ordering.

    Called by every normalizer as the last step so there is exactly one place
    that decides what a canonical frame looks like.
    """
    out = df.copy()

    out["symbol"] = out["symbol"].astype("string").str.strip().str.upper()
    out["series"] = out["series"].astype("string").str.strip().str.upper()
    if "isin" in out:
        out["isin"] = out["isin"].astype("string").str.strip()

    for col, dtype in CANONICAL_DTYPES.items():
        if col not in out.columns:
            out[col] = pd.NA
        if dtype in {"int64", "Int64"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            if dtype == "int64":
                # volume is non-nullable by contract; a null here means the
                # source row was malformed and Tier 2 will quarantine it.
                out[col] = out[col].fillna(0).astype("int64")
        elif dtype == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = out[col].astype(dtype)

    out["date"] = pd.Timestamp(trade_date).date()

    out = out[list(CANONICAL_PRICE_COLUMNS)]
    return out.sort_values(["symbol", "series"], kind="stable").reset_index(drop=True)


def normalise(
    df: pd.DataFrame,
    *,
    trade_date: date,
    era: Era | None = None,
) -> pd.DataFrame:
    """Dispatch to the right normalizer for the date, then canonicalise.

    Imported lazily to keep the two era modules independent of each other —
    neither should ever need to know the other exists.
    """
    from market_data.normalize import legacy, udiff

    resolved = era or era_for(trade_date)
    if resolved is Era.LEGACY:
        return legacy.normalise(df, trade_date=trade_date)
    return udiff.normalise(df, trade_date=trade_date)
