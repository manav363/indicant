"""L0 — panel construction.

This is the single change that makes statistical significance reachable.

v1 trained one model per ticker on ~1,200 rows and asked whether the result was
significant. It was not — p=0.68 for RELIANCE, p=1.00 for TCS. That is the
*expected* outcome, not a bug: you cannot resolve a sub-1% effect from 1,200
noisy samples, so the design foreclosed the result before any modelling started.

A pooled panel of ~1,700 symbols x ~4,700 days is ~10^6 rows. The literature is
unambiguous that this is the standard approach and that per-stock time-series
regression "has important limitations" (Gu/Kelly/Xiu pool ~30,000 US stocks).

Two leakage traps live here, and both are subtle enough to survive a code
review:

1. **Cross-sectional ranks must be computed per date.** Ranking across the whole
   panel at once leaks the future distribution into every historical row.
2. **The universe must be point-in-time.** Pooling today's eligible symbols
   across twenty years is survivorship bias with extra steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from intelligence.data.lake_client import LakeClient
from intelligence.features.technical import add_all_features

# Columns that describe the observation rather than the market state. Never
# ranked, never fed to a model.
IDENTITY_COLUMNS: tuple[str, ...] = ("date", "symbol", "series", "isin", "sector")

# Raw price/volume columns. Excluded from the feature set because their absolute
# level is not comparable across symbols — a Rs 50 stock and a Rs 5,000 stock
# are not 100x different in any way a model should learn.
RAW_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "prev_close",
    "volume", "turnover", "trades", "delivery_qty", "adj_factor", "adj_close",
)

# A cross-sectional rank computed over fewer than this many symbols is noise:
# with 3 symbols the percentiles are 0, 0.5, 1 regardless of the values.
MIN_SYMBOLS_FOR_RANK = 20

# Feature engineering needs this much history before its output is meaningful
# (the longest indicator window in technical.py plus settle time).
WARMUP_ROWS = 252


@dataclass
class PanelConfig:
    start: date | None = None
    end: date | None = None
    min_history_rows: int = WARMUP_ROWS
    add_cross_sectional: bool = True
    add_sector_neutral: bool = False
    symbols: Sequence[str] | None = None
    # Trim the warm-up rows after feature computation. Keeping them would feed
    # the model a block of NaN-heavy rows whose indicators had not converged.
    drop_warmup: bool = True


@dataclass
class PanelResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_symbols(self) -> int:
        return int(self.frame["symbol"].nunique()) if not self.frame.empty else 0

    @property
    def n_dates(self) -> int:
        return int(self.frame["date"].nunique()) if not self.frame.empty else 0

    def summary(self) -> str:
        return (
            f"panel: {self.n_rows:,} rows | {self.n_symbols} symbols | "
            f"{self.n_dates} dates | {len(self.feature_columns)} features | "
            f"{len(self.skipped)} symbols skipped"
        )


class PanelBuilder:
    """Builds a (date x symbol) feature panel from the lake."""

    def __init__(self, client: LakeClient) -> None:
        self._client = client

    def build(self, config: PanelConfig | None = None) -> PanelResult:
        cfg = config or PanelConfig()

        symbols = list(cfg.symbols) if cfg.symbols else self._pit_symbols(cfg)
        if not symbols:
            return PanelResult(frame=pd.DataFrame(), feature_columns=[])

        prices = self._client.read_panel(
            symbols=symbols, start=cfg.start, end=cfg.end, adjusted=True
        )
        if prices.empty:
            return PanelResult(frame=pd.DataFrame(), feature_columns=[])

        return self.build_from_prices(prices, cfg)

    def build_from_prices(
        self, prices: pd.DataFrame, config: PanelConfig | None = None
    ) -> PanelResult:
        """Feature-engineer a price frame into a panel.

        Split out from `build` so tests can drive it with a hand-made frame and
        never touch the lake.
        """
        cfg = config or PanelConfig()
        if prices.empty:
            return PanelResult(frame=pd.DataFrame(), feature_columns=[])

        featured, skipped = self._features_per_symbol(prices, cfg)
        if featured.empty:
            return PanelResult(frame=pd.DataFrame(), feature_columns=[], skipped=skipped)

        base_features = self._feature_columns(featured)

        if cfg.add_cross_sectional:
            featured = self._add_cross_sectional(featured, base_features)
        if cfg.add_sector_neutral and "sector" in featured.columns:
            featured = self._add_sector_neutral(featured, base_features)

        featured = featured.sort_values(["date", "symbol"]).reset_index(drop=True)
        return PanelResult(
            frame=featured,
            feature_columns=self._feature_columns(featured),
            skipped=skipped,
        )

    # ------------------------------------------------------------------ steps

    def _pit_symbols(self, cfg: PanelConfig) -> list[str]:
        """Eligible symbols as of the panel's END date.

        Using the end date rather than today is what keeps a historical panel
        honest — 'today's eligible names' applied to 2015 is exactly the
        survivorship bias the lake was rebuilt to remove.
        """
        as_of = cfg.end
        if as_of is None:
            days = self._client.trading_days()
            if not days:
                return []
            as_of = days[-1]
        return self._client.eligible_symbols(as_of)

    def _features_per_symbol(
        self, prices: pd.DataFrame, cfg: PanelConfig
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Run the 46 indicators per symbol, then stack.

        Per-symbol is not an optimisation choice — it is correctness. Every
        indicator is a rolling function of one series, and computing them over a
        stacked frame would roll across the symbol boundary, mixing RELIANCE's
        prices into TCS's moving average.
        """
        out: list[pd.DataFrame] = []
        skipped: dict[str, str] = {}

        for symbol, group in prices.groupby("symbol", sort=True):
            sym = str(symbol)
            if len(group) < cfg.min_history_rows:
                skipped[sym] = f"{len(group)} rows, need {cfg.min_history_rows}"
                continue

            frame = group.sort_values("date").copy()
            index_dates = pd.to_datetime(frame["date"])
            frame = frame.set_index(index_dates)

            try:
                featured = add_all_features(frame)
            except Exception as exc:
                # One bad symbol must not take down a 1,700-symbol panel, but it
                # must be reported rather than silently absent.
                skipped[sym] = f"{type(exc).__name__}: {exc}"
                continue

            featured = featured.reset_index(drop=True)
            featured["symbol"] = sym
            featured["date"] = frame["date"].to_numpy()

            if cfg.drop_warmup:
                featured = featured.iloc[cfg.min_history_rows - 1 :]

            out.append(featured)

        if not out:
            return pd.DataFrame(), skipped
        return pd.concat(out, ignore_index=True), skipped

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded = set(IDENTITY_COLUMNS) | set(RAW_COLUMNS)
        return [
            c
            for c in frame.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
        ]

    def _add_cross_sectional(
        self, frame: pd.DataFrame, features: Sequence[str]
    ) -> pd.DataFrame:
        """Percentile rank of each feature within its own date.

        This is what turns 'RSI is 68' into 'RSI is higher than 83% of the
        market today', which is the question a screener actually asks — you are
        choosing between stocks, not judging one in isolation.

        `groupby("date")` is load-bearing. Ranking the whole panel at once would
        leak the future distribution into every historical row.
        """
        out = frame.copy()
        by_date = out.groupby("date")

        # Guard the degenerate case: with a handful of symbols the percentiles
        # are an artefact of the count, not a market fact.
        counts = by_date["symbol"].transform("size")
        usable = counts >= MIN_SYMBOLS_FOR_RANK

        for col in features:
            ranked = by_date[col].rank(pct=True, method="average")
            out[f"{col}_xs"] = ranked.where(usable)

        return out

    def _add_sector_neutral(
        self, frame: pd.DataFrame, features: Sequence[str]
    ) -> pd.DataFrame:
        """Z-score within (date, sector).

        Separates 'this bank is strong' from 'banks are strong', which a
        market-wide rank conflates.
        """
        out = frame.copy()
        grouped = out.groupby(["date", "sector"])

        for col in features:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            out[f"{col}_sect"] = np.where(
                (std > 0) & std.notna(), (out[col] - mean) / std, np.nan
            )
        return out


def panel_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Symbols per date. A sudden drop means the panel has a hole that the
    cross-sectional ranks silently absorbed.
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "n_symbols"])
    return (
        panel.groupby("date")["symbol"]
        .nunique()
        .rename("n_symbols")
        .reset_index()
        .sort_values("date")
    )
