"""Parquet lake writer and reader. market-data is the only writer.

Two write paths, because a 20-year backfill and a nightly increment have
opposite cost profiles:

* ``write_year`` — used by the backfill. Buffers a whole year in memory and
  writes once. The naive alternative (read-modify-write the year file on every
  one of ~250 days) rewrites a growing 500k-row file 250 times per year, which
  is ~25 minutes of pure serialization across the backfill for no benefit.
* ``append_day`` — used nightly. One small read-modify-write is fine when it
  happens once.

Reads go through DuckDB because it globs parquet natively and pushes predicates
into the scan, so `intelligence` can pull a filtered million-row panel without
this service materialising it first.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from indicant_contracts import CANONICAL_PRICE_COLUMNS, Dataset, LakePaths

from market_data._dates import as_date

# Rows are unique on this key. Re-ingesting a date must replace, not duplicate.
PRICE_KEY = ("date", "symbol", "series")


class Lake:
    def __init__(self, paths: LakePaths) -> None:
        self.paths = paths

    # ------------------------------------------------------------------ write

    def write_year(
        self,
        frames: Iterable[pd.DataFrame],
        *,
        year: int,
        dataset: Dataset = Dataset.PRICES,
    ) -> Path:
        """Write one year in a single pass, merging with anything already there."""
        combined = _concat(frames)
        if combined.empty:
            raise ValueError(f"refusing to write an empty {dataset.value} year {year}")

        target = self.paths.file(dataset, when=date(year, 1, 1))
        if target.exists():
            combined = _concat([self._read_parquet(target), combined])

        return self._write_parquet(combined, target)

    def append_day(
        self,
        df: pd.DataFrame,
        *,
        trade_date: date,
        dataset: Dataset = Dataset.PRICES,
    ) -> Path:
        """Merge one trading day into its year partition (nightly path)."""
        if df.empty:
            raise ValueError(f"refusing to write an empty {dataset.value} for {trade_date}")
        return self.write_year([df], year=trade_date.year, dataset=dataset)

    def write_partition(
        self,
        df: pd.DataFrame,
        *,
        dataset: Dataset,
        when: date,
    ) -> Path:
        """Write a date-partitioned dataset (quarantine, quality runs, PIT)."""
        return self._write_parquet(df, self.paths.file(dataset, when=when))

    def _write_parquet(self, df: pd.DataFrame, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling then rename: a crash mid-write must not leave a
        # truncated parquet that reads as valid-but-short.
        tmp = target.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd")
        tmp.replace(target)
        return target

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    # ------------------------------------------------------------------- read

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(":memory:")

    def has_data(self, dataset: Dataset = Dataset.PRICES) -> bool:
        return any(self.paths.dataset_dir(dataset).rglob("*.parquet"))

    def read_prices(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        series: Sequence[str] | None = ("EQ",),
        dataset: Dataset = Dataset.PRICES,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Predicate-pushed read. Returns an empty canonical frame when the
        dataset does not exist yet, rather than raising — an empty lake is a
        valid state during a first run.
        """
        cols = list(columns) if columns else list(CANONICAL_PRICE_COLUMNS)
        if not self.has_data(dataset):
            return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

        where: list[str] = []
        params: list[object] = []
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            where.append(f"symbol IN ({placeholders})")
            params.extend(s.upper() for s in symbols)
        if series:
            placeholders = ", ".join("?" for _ in series)
            where.append(f"series IN ({placeholders})")
            params.extend(s.upper() for s in series)
        if start:
            where.append("date >= ?")
            params.append(start)
        if end:
            where.append("date <= ?")
            params.append(end)

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(cols)} FROM read_parquet(?, union_by_name := true) "
            f"{clause} ORDER BY symbol, date"
        )
        with self.connect() as con:
            return con.execute(sql, [self.paths.glob(dataset), *params]).df()

    def observed_trading_days(self, dataset: Dataset = Dataset.PRICES) -> list[date]:
        """Dates for which data actually landed. Authoritative trading calendar.

        The exchange published a file, therefore it traded. Nothing predicted
        here.
        """
        if not self.has_data(dataset):
            return []
        sql = (
            "SELECT DISTINCT date FROM read_parquet(?, union_by_name := true) ORDER BY date"
        )
        with self.connect() as con:
            rows = con.execute(sql, [self.paths.glob(dataset)]).fetchall()
        return [as_date(r[0]) for r in rows]

    def symbol_span(self, dataset: Dataset = Dataset.PRICES) -> pd.DataFrame:
        """First seen, last seen and observation count per symbol.

        `last_seen` well before the lake's end date is what identifies a
        delisted or suspended name — which is the mechanism that makes the
        historical universe survivorship-bias-free.
        """
        if not self.has_data(dataset):
            return pd.DataFrame(
                columns=["symbol", "series", "first_seen", "last_seen", "n_days", "isin"]
            )
        sql = """
            SELECT symbol,
                   any_value(series)  AS series,
                   min(date)          AS first_seen,
                   max(date)          AS last_seen,
                   count(*)           AS n_days,
                   any_value(isin)    AS isin
            FROM read_parquet(?, union_by_name := true)
            GROUP BY symbol
            ORDER BY symbol
        """
        with self.connect() as con:
            return con.execute(sql, [self.paths.glob(dataset)]).df()

    def row_count(self, dataset: Dataset = Dataset.PRICES) -> int:
        if not self.has_data(dataset):
            return 0
        with self.connect() as con:
            sql = "SELECT count(*) FROM read_parquet(?, union_by_name := true)"
            return int(con.execute(sql, [self.paths.glob(dataset)]).fetchone()[0])

    def read_dataset(self, dataset: Dataset) -> pd.DataFrame:
        if not self.has_data(dataset):
            return pd.DataFrame()
        with self.connect() as con:
            sql = "SELECT * FROM read_parquet(?, union_by_name := true)"
            return con.execute(sql, [self.paths.glob(dataset)]).df()


def group_by_year(frames: Iterable[pd.DataFrame]) -> dict[int, list[pd.DataFrame]]:
    """Bucket day-frames by year so a backfill can write once per year."""
    buckets: dict[int, list[pd.DataFrame]] = defaultdict(list)
    for frame in frames:
        if frame.empty:
            continue
        year = as_date(frame["date"].iloc[0]).year
        buckets[year].append(frame)
    return dict(buckets)


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [f for f in frames if not f.empty]
    if not usable:
        return pd.DataFrame(columns=list(CANONICAL_PRICE_COLUMNS))
    combined = pd.concat(usable, ignore_index=True)
    key = [c for c in PRICE_KEY if c in combined.columns]
    if key:
        # keep="last" so a re-ingest of a date replaces the older rows.
        combined = combined.drop_duplicates(subset=key, keep="last")
    return combined.sort_values(key or list(combined.columns)[:1], kind="stable").reset_index(
        drop=True
    )

