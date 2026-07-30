"""Symbol registry and point-in-time universe.

The point-in-time universe is the mechanism that makes every backtest built on
this lake survivorship-bias-free. It answers "what could I have traded on
2015-03-04" using only information available on that date — which is a different
question from "what is in the index today", and confusing the two is the single
most common way a research backtest fails to reproduce.

The registry is derived from observed data, never from a configured list. A
configured list is how survivorship bias gets reintroduced after being removed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from indicant_contracts import (
    Dataset,
    EligibilityThresholds,
    ListingStatus,
    SymbolMeta,
    UniverseSnapshot,
)

from market_data._dates import as_date
from market_data.quality.scoring import QualityScorer
from market_data.store.lake import Lake

# A symbol unseen for this many calendar days is treated as delisted or
# suspended rather than merely quiet.
DELISTING_SILENCE_DAYS = 60


class Catalog:
    def __init__(self, lake: Lake, thresholds: EligibilityThresholds | None = None) -> None:
        self._lake = lake
        self._scorer = QualityScorer(thresholds)

    # ------------------------------------------------------------- registry

    def build_symbol_registry(self, *, as_of: date | None = None) -> list[SymbolMeta]:
        """Derive per-symbol metadata from what the lake actually contains.

        A symbol whose `last_seen` is well before `as_of` is marked delisted —
        and stays in history. That combination is what a survivorship-bias-free
        universe *is*.
        """
        span = self._lake.symbol_span()
        if span.empty:
            return []

        as_of = as_of or max(as_date(v) for v in span["last_seen"])
        cutoff = as_of - timedelta(days=DELISTING_SILENCE_DAYS)

        registry: list[SymbolMeta] = []
        for _, row in span.iterrows():
            last_seen = as_date(row["last_seen"])
            gone = last_seen < cutoff
            registry.append(
                SymbolMeta(
                    symbol=str(row["symbol"]),
                    isin=None if pd.isna(row.get("isin")) else str(row["isin"]),
                    series=str(row["series"]) if not pd.isna(row["series"]) else "EQ",  # type: ignore[arg-type]
                    status=ListingStatus.DELISTED if gone else ListingStatus.LISTED,
                    first_seen=as_date(row["first_seen"]),
                    last_seen=last_seen,
                    delisted_on=last_seen if gone else None,
                )
            )
        return registry

    def write_symbol_registry(self, *, as_of: date | None = None) -> int:
        registry = self.build_symbol_registry(as_of=as_of)
        if not registry:
            return 0
        frame = pd.DataFrame(
            [
                {
                    "symbol": m.symbol,
                    "isin": m.isin,
                    "series": str(m.series),
                    "status": str(m.status),
                    "first_seen": m.first_seen,
                    "last_seen": m.last_seen,
                    "delisted_on": m.delisted_on,
                }
                for m in registry
            ]
        )
        self._lake.write_partition(
            frame,
            dataset=Dataset.SYMBOL_META,
            when=as_of or max(m.last_seen for m in registry),
        )
        return len(registry)

    def delisted_symbols(self, *, as_of: date | None = None) -> list[str]:
        """The proof that survivorship bias is gone.

        If this list is empty over a 20-year window, the ingest is only seeing
        survivors and something is wrong.
        """
        return sorted(
            m.symbol
            for m in self.build_symbol_registry(as_of=as_of)
            if m.status is ListingStatus.DELISTED
        )

    # -------------------------------------------------------- PIT universe

    def universe_as_of(
        self,
        as_of: date,
        *,
        lookback_days: int = 1260,
        index_name: str | None = None,
    ) -> UniverseSnapshot:
        """Universe using only data available on `as_of`.

        The `end=as_of` filter is load-bearing. Reading the full history and
        then filtering would let a symbol's *future* liquidity influence whether
        it was eligible in the past.
        """
        start = as_of - timedelta(days=int(lookback_days * 1.5))
        prices = self._lake.read_prices(start=start, end=as_of)
        if prices.empty:
            return UniverseSnapshot(
                as_of=as_of, index_name=index_name, symbols=(), eligible_symbols=()
            )

        expected_days = prices["date"].nunique()
        scores = self._scorer.score_all(prices, as_of=as_of, expected_days=expected_days)
        return self._scorer.build_universe(scores, as_of=as_of, index_name=index_name)

    def write_universe(self, snapshot: UniverseSnapshot) -> int:
        frame = pd.DataFrame(
            [
                {
                    "as_of": snapshot.as_of,
                    "index_name": snapshot.index_name,
                    "symbol": s,
                    "eligible": s in set(snapshot.eligible_symbols),
                    "exclusion_reason": snapshot.excluded.get(s),
                }
                for s in snapshot.symbols
            ]
        )
        if frame.empty:
            return 0
        self._lake.write_partition(
            frame, dataset=Dataset.UNIVERSE_PIT, when=snapshot.as_of
        )
        return len(frame)

    def read_universe(self, as_of: date) -> pd.DataFrame:
        path = self._lake.paths.file(Dataset.UNIVERSE_PIT, when=as_of)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

