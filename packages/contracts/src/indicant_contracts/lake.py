"""Lake layout as a contract.

`market-data` is the single writer. `intelligence` reads these paths directly
for bulk data, because moving a million rows of OHLCV over JSON to train a
model is minutes of serialization for zero benefit — in an ML system the
store *is* the interface.

That only works if the layout is a published, frozen contract rather than an
implementation detail. Changing a partition key or a dataset name here breaks
both services' schema-freeze tests at build time, which is the point.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Dataset(StrEnum):
    """Every dataset in the lake. Nothing is written outside this set."""

    PRICES = "prices"
    ADJUSTED = "adjusted"
    CORP_ACTIONS = "corp_actions"
    SYMBOL_CHANGES = "symbol_changes"
    SYMBOL_META = "symbol_meta"
    CALENDAR = "calendar"
    QUALITY_RUNS = "quality/runs"
    QUALITY_SCORES = "quality/scores"
    QUARANTINE = "quality/quarantine"
    UNIVERSE_PIT = "universe_pit"


# Datasets partitioned by year. Chosen over per-day files because ~250 files
# per year x 20 years is 5,000 tiny parquet files, and parquet's per-file
# overhead dominates at that size.
_YEAR_PARTITIONED = frozenset({Dataset.PRICES, Dataset.ADJUSTED})

# Datasets partitioned by trade date. These are written per-ingestion and read
# per-incident, so one file per day is the right granularity.
_DATE_PARTITIONED = frozenset({Dataset.QUARANTINE, Dataset.QUALITY_RUNS, Dataset.UNIVERSE_PIT})


class LakePaths(BaseModel):
    """Resolves dataset paths under a lake root.

    Deliberately does no IO — it is a pure path calculator so both services
    can agree on layout without either importing storage code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path

    def dataset_dir(self, dataset: Dataset) -> Path:
        return self.root / dataset.value

    def partition(self, dataset: Dataset, *, when: date) -> Path:
        """Directory holding the partition that `when` belongs to."""
        base = self.dataset_dir(dataset)
        if dataset in _YEAR_PARTITIONED:
            return base / f"year={when.year}"
        if dataset in _DATE_PARTITIONED:
            return base / f"date={when.isoformat()}"
        return base

    def file(self, dataset: Dataset, *, when: date) -> Path:
        """Full parquet path for the partition `when` belongs to."""
        part = self.partition(dataset, when=when)
        if dataset in _YEAR_PARTITIONED:
            return part / f"{dataset.name.lower()}_{when.year}.parquet"
        if dataset in _DATE_PARTITIONED:
            return part / "data.parquet"
        return part / "data.parquet"

    def glob(self, dataset: Dataset) -> str:
        """Recursive glob for reading a whole dataset.

        Used by the intelligence service's lake client and by DuckDB's
        read_parquet, which is why it is a string and not a Path.
        """
        return str(self.dataset_dir(dataset) / "**" / "*.parquet")

    @property
    def catalog_db(self) -> Path:
        """DuckDB catalog holding metadata, symbol registry and PIT universe."""
        return self.root / "catalog.duckdb"

    @property
    def ingest_state(self) -> Path:
        """Resumable-backfill bookmark. A 20-year backfill will be interrupted."""
        return self.root / "_ingest_state.json"
