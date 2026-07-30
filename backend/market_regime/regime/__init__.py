"""
market_regime/regime
────────────────────
Standalone regime detection module — classifies market into Bull, Bear,
RangeBound, HighVol, LowVol regimes.

Design:
  - ``classifier.py`` — shared ``RegimeClassifier`` used by both per-stock
    and market-wide aggregation.  Every regime rule lives here once.
  - ``market.py`` — ``RegimeAggregator`` that fetches NIFTY 50 constituents,
    runs ``classifier.py`` on each, and produces a market-wide signal with
    ``constituents_reporting`` metadata.
  - ``config.py`` — all threshold constants at a single source of truth.
"""

from market_regime.regime.classifier import RegimeClassifier, RegimeResult
from market_regime.regime.market import MarketRegimeResult, RegimeAggregator

__all__ = [
    "RegimeClassifier",
    "RegimeResult",
    "RegimeAggregator",
    "MarketRegimeResult",
]
