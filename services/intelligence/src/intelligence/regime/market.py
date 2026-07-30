"""
intelligence/regime/market.py
───────────────────────────────
Market-wide regime aggregator — produces a single regime signal for the
broader market (NIFTY 50 equal-weighted proxy) by running the shared
``RegimeClassifier`` on each constituent and aggregating the results.

Design
------
- Reuses ``classifier.py`` — never duplicates classification rules.
- The ``constituents_reporting`` field tracks how many NIFTY 50 stocks
  have valid data on the analysis date.  This prevents a false sense of
  confidence when only a few stocks are reporting.
- Results are cached in-memory for ``cache_ttl_minutes`` (default 15) to
  avoid hammering Yahoo Finance on every request.  The cache key is the
  current date + hour, so the regime signal updates at most once per hour.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from intelligence.data.preprocessor import preprocess
from intelligence.features.technical import add_all_features
from intelligence.regime.classifier import RegimeClassifier, RegimeResult
from intelligence.regime.source import RegimeDataSource

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class MarketRegimeResult:
    """Aggregated market-wide regime result."""

    analysis_date: str
    total_constituents: int
    constituents_reporting: int
    primary_regime: str
    regime_distribution: dict[str, int]
    market_adx: float | None
    composite_signal: str
    details: list[RegimeResult] = field(default_factory=list)
    cache_ttl_minutes: int = 15  # advertised so callers know the staleness window

    @property
    def reporting_ratio(self) -> float:
        if self.total_constituents == 0:
            return 0.0
        return self.constituents_reporting / self.total_constituents


# ── Aggregator ─────────────────────────────────────────────────────────────────

class RegimeAggregator:
    """
    Aggregates per-stock regime classifications into a market-wide signal.

    Uses the same ``RegimeClassifier`` as the per-stock endpoint —
    all regime rules are defined once in ``classifier.py``.

    Results are cached in-memory for ``cache_ttl_minutes`` (default 15).
    Call ``analyse(force_refresh=True)`` to bypass the cache.
    """

    def __init__(
        self,
        source: RegimeDataSource,
        tickers: list[str] | None = None,
        lookback_years: int = 5,
        cache_ttl_minutes: int = 15,
    ):
        # `source` is required and has no default. v1 defaulted to fetching from
        # yfinance, which meant a market-regime call could silently reach the
        # internet from inside a prediction. Reading is now always the caller's
        # explicit choice, and in production it is the local lake.
        self._source = source
        self.tickers = tickers if tickers is not None else source.default_tickers()
        self.lookback_years = lookback_years
        self._classifier = RegimeClassifier()
        self._cache: dict[str, tuple[float, MarketRegimeResult]] = {}
        self._cache_ttl_seconds = cache_ttl_minutes * 60
        self.failed_tickers: list[tuple[str, str]] = []

    def analyse(self, force_refresh: bool = False) -> MarketRegimeResult:
        """
        Run regime classification on all constituents and aggregate.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass the in-memory cache and re-fetch all data.

        Returns
        -------
        MarketRegimeResult
        """
        if not force_refresh:
            cached = self._get_cached()
            if cached is not None:
                return cached

        constituent_results: list[RegimeResult] = []
        constituents_reporting = 0

        for ticker in self.tickers:
            try:
                result = self._classify_single(ticker)
                constituent_results.append(result)
                constituents_reporting += 1
            except Exception as exc:
                # Recorded, not just logged at debug. A market call built from
                # 12 of 50 constituents is not the same object as one built from
                # 50, and `reporting_ratio` plus this list are what make the
                # difference visible instead of averaged away.
                self.failed_tickers.append((ticker, f"{type(exc).__name__}: {exc}"))
                logger.warning("regime: skipping %s (%s)", ticker, exc)
                continue

        if not constituent_results:
            logger.warning("No constituents could be classified.")
            result = MarketRegimeResult(
                analysis_date="",
                total_constituents=len(self.tickers),
                constituents_reporting=0,
                primary_regime="Unknown",
                regime_distribution={},
                market_adx=None,
                composite_signal="neutral",
                details=[],
            )
            self._set_cached(result)
            return result

        regime_distribution: dict[str, int] = {}
        adx_values: list[float] = []

        for r in constituent_results:
            regime_distribution[r.primary_regime] = (
                regime_distribution.get(r.primary_regime, 0) + 1
            )
            if r.adx is not None:
                adx_values.append(r.adx)

        primary_regime = max(regime_distribution, key=regime_distribution.get)  # type: ignore[arg-type]
        market_adx = float(np.median(adx_values)) if adx_values else None
        composite_signal = self._aggregate_signal(constituent_results)

        analysis_date = ""
        if constituent_results:
            last_history = constituent_results[-1].regime_history
            if last_history:
                analysis_date = last_history[-1]["date"]

        result = MarketRegimeResult(
            analysis_date=analysis_date,
            total_constituents=len(self.tickers),
            constituents_reporting=constituents_reporting,
            primary_regime=primary_regime,
            regime_distribution=regime_distribution,
            market_adx=round(market_adx, 2) if market_adx is not None else None,
            composite_signal=composite_signal,
            details=constituent_results,
        )
        self._set_cached(result)
        return result

    # ── Cache ──────────────────────────────────────────────────────────────

    def _cache_key(self) -> str:
        """Cache key = YYYY-MM-DD-HH so regime updates at most hourly."""
        return datetime.now().strftime("%Y-%m-%d-%H")

    def _get_cached(self) -> MarketRegimeResult | None:
        key = self._cache_key()
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.time() - ts > self._cache_ttl_seconds:
            del self._cache[key]
            return None
        return result

    def _set_cached(self, result: MarketRegimeResult) -> None:
        key = self._cache_key()
        self._cache[key] = (time.time(), result)
        # Evict stale entries to keep memory bounded
        now = time.time()
        stale_keys = [
            k for k, (ts, _) in self._cache.items()
            if now - ts > self._cache_ttl_seconds * 2
        ]
        for k in stale_keys:
            del self._cache[k]

    # ── Internal ──────────────────────────────────────────────────────────

    def _classify_single(self, ticker: str) -> RegimeResult:
        df = self._source.history(ticker, lookback_years=self.lookback_years)
        df = preprocess(df, min_rows=20)
        df = add_all_features(df)
        return self._classifier.classify(df, ticker=ticker)

    @staticmethod
    def _aggregate_signal(results: list[RegimeResult]) -> str:
        counts = {"risk_on": 0, "risk_off": 0, "neutral": 0}
        for r in results:
            counts[r.composite_signal] = counts.get(r.composite_signal, 0) + 1

        if counts["risk_on"] > counts["risk_off"] and counts["risk_on"] > counts["neutral"]:
            return "risk_on"
        elif counts["risk_off"] > counts["risk_on"] and counts["risk_off"] > counts["neutral"]:
            return "risk_off"
        else:
            return "neutral"

