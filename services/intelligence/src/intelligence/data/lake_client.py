"""Read-only lake access.

This is the data plane. `market-data` writes; this service only ever reads, and
it reads parquet directly rather than over HTTP — a training panel is ~10^6 rows
and JSON serialization of that is minutes of pure waste.

The layout comes from `indicant_contracts.LakePaths`, which is frozen by a
schema test in the contracts package. That is what makes reading another
service's files a contract rather than a reach-across.

Deliberately has no write methods at all. Not "write methods that raise" —
absent, so a write is a NameError at import time rather than a runtime surprise.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import duckdb
import pandas as pd
from indicant_contracts import CANONICAL_PRICE_COLUMNS, Dataset, LakePaths


class LakeClient:
    def __init__(self, paths: LakePaths) -> None:
        self._paths = paths

    @property
    def paths(self) -> LakePaths:
        return self._paths

    def _has(self, dataset: Dataset) -> bool:
        return any(self._paths.dataset_dir(dataset).rglob("*.parquet"))

    @property
    def is_ready(self) -> bool:
        """Whether there is anything to train on. A service that reports healthy
        against an empty lake is up but useless.
        """
        return self._has(Dataset.PRICES)

    def read_panel(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        adjusted: bool = True,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read a (date x symbol) panel with predicates pushed into the scan.

        `adjusted=True` reads the corporate-action-adjusted dataset. Training on
        unadjusted prices puts a fake -50% return on every split date, so the
        default is the safe one and the caller has to opt out explicitly.
        """
        dataset = Dataset.ADJUSTED if adjusted else Dataset.PRICES
        if not self._has(dataset):
            if adjusted and self._has(Dataset.PRICES):
                raise LakeNotAdjusted(
                    "the adjusted dataset does not exist yet. Run "
                    "'indicant-md actions --file <corp_actions.csv>' then "
                    "'indicant-md adjust'. Passing adjusted=False works but "
                    "accepts a fabricated return on every split date."
                )
            return pd.DataFrame(
                {c: pd.Series(dtype="object") for c in (columns or CANONICAL_PRICE_COLUMNS)}
            )

        cols = list(columns) if columns else list(CANONICAL_PRICE_COLUMNS)
        where: list[str] = ["series = 'EQ'"]
        params: list[object] = []
        if symbols:
            where.append(f"symbol IN ({', '.join('?' for _ in symbols)})")
            params.extend(s.upper() for s in symbols)
        if start:
            where.append("date >= ?")
            params.append(start)
        if end:
            where.append("date <= ?")
            params.append(end)

        sql = (
            f"SELECT {', '.join(cols)} FROM read_parquet(?, union_by_name := true) "
            f"WHERE {' AND '.join(where)} ORDER BY date, symbol"
        )
        with duckdb.connect(":memory:") as con:
            return con.execute(sql, [self._paths.glob(dataset), *params]).df()

    def read_universe(self, as_of: date) -> pd.DataFrame:
        """The point-in-time universe written by market-data.

        Training must filter to this rather than to whatever symbols happen to
        be in the price data, or the panel silently includes companies that were
        not tradeable on the date being modelled.
        """
        path = self._paths.file(Dataset.UNIVERSE_PIT, when=as_of)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

    def eligible_symbols(self, as_of: date, *, strict: bool = True) -> list[str]:
        """Symbols the system can honestly answer for, as of a date.

        Raises when the point-in-time universe has never been computed, rather
        than returning an empty list. Those two states look identical to a
        caller and mean opposite things: "no stock qualifies today" versus
        "nobody has run `indicant-md universe` yet". The empty-list version
        silently turned a missing pipeline step into a search box that matched
        nothing and a screener with no rows, with no error anywhere.
        """
        universe = self.read_universe(as_of)
        if universe.empty:
            if strict:
                raise UniverseNotComputed(
                    f"no point-in-time universe for {as_of}. The lake has prices "
                    f"but the universe partition was never written — run "
                    f"'indicant-md universe --as-of {as_of}'."
                )
            return []
        return sorted(universe[universe["eligible"]]["symbol"].astype(str))

    def trading_days(self) -> list[date]:
        if not self._has(Dataset.PRICES):
            return []
        sql = "SELECT DISTINCT date FROM read_parquet(?, union_by_name := true) ORDER BY date"
        with duckdb.connect(":memory:") as con:
            rows = con.execute(sql, [self._paths.glob(Dataset.PRICES)]).fetchall()
        return [r[0] if isinstance(r[0], date) else pd.Timestamp(r[0]).date() for r in rows]

    def row_count(self, *, adjusted: bool = True) -> int:
        dataset = Dataset.ADJUSTED if adjusted else Dataset.PRICES
        if not self._has(dataset):
            return 0
        with duckdb.connect(":memory:") as con:
            sql = "SELECT count(*) FROM read_parquet(?, union_by_name := true)"
            return int(con.execute(sql, [self._paths.glob(dataset)]).fetchone()[0])


class UniverseNotComputed(RuntimeError):
    """The PIT universe partition is missing.

    Distinct from "nothing is eligible": one is a pipeline step that has not
    run, the other is a real answer about the market. Conflating them turns a
    missing step into an empty search box with no error.
    """


class LakeNotAdjusted(RuntimeError):
    """Raised rather than silently falling back to unadjusted prices.

    A silent fallback here would train a model on fabricated split-day returns
    and nothing downstream would ever indicate it happened.
    """
