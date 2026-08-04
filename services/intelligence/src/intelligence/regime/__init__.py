"""
intelligence/regime
────────────────────
Standalone regime detection — Bull, Bear, RangeBound, HighVol, LowVol.

⚠️  NOT WIRED TO ANY ENDPOINT. Read this before trusting the module below.

This is v1 code, ported intact and still fully tested, but nothing in the
served API calls it. Both regime routes — ``/regime/{symbol}`` and
``/regime/market`` — go through ``serving.PredictionService._regime``, which
applies a much simpler two-condition rule (ADX for trending-or-not, then price
vs the 200-day average for direction) to the same feature panel a prediction
already built.

So there are two regime definitions in this codebase and only one of them runs.
The docstring here previously claimed that "every regime rule lives here once",
which was the opposite of true and is exactly how someone concludes the rich
taxonomy below is what the product shows.

This is the failure mode this project has hit before — ``adjust_all`` was
written, tested and never called; the v1 CLI was missing the ``--registry``
flag while 208 tests stayed green. A passing test suite proves the tested path
works, never that production reaches it.

Modules:
  - ``classifier.py`` — ``RegimeClassifier``, the richer taxonomy. Unreachable.
  - ``market.py`` — ``RegimeAggregator``, NIFTY-50 breadth. Unreachable;
    ``/regime/market`` counts ``serving`` regimes over the training universe.
  - ``config.py`` — threshold constants for the above.
"""

from intelligence.regime.classifier import RegimeClassifier, RegimeResult
from intelligence.regime.market import MarketRegimeResult, RegimeAggregator

__all__ = [
    "MarketRegimeResult",
    "RegimeAggregator",
    "RegimeClassifier",
    "RegimeResult",
]
