"""Date coercion.

One helper, because the naive version of this is a trap:

    if isinstance(value, date):   # WRONG
        return value

`pd.Timestamp` subclasses `datetime.date`, so that early-return hands a
Timestamp straight back. It then compares fine against other Timestamps and
raises `TypeError: Cannot compare Timestamp with datetime.date` only when it
meets a real date — which happens somewhere far from where the value entered.

DuckDB and parquet both hand back Timestamps, so every read boundary needs this.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def as_date(value: object) -> date:
    """Coerce anything date-like to a plain `datetime.date`.

    Order matters: Timestamp and datetime are checked before `date`, because
    both are subclasses of it.
    """
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or value is pd.NaT:
        raise ValueError("cannot coerce null to a date")
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def as_dates(values: object) -> list[date]:
    return [as_date(v) for v in values]  # type: ignore[union-attr]
