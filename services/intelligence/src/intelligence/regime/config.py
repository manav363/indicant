"""
intelligence/regime/config.py
───────────────────────────────
Threshold constants for regime classification.

Centralising all magic numbers here so both the per-stock classifier
and the market-wide aggregator share a single source of truth.
"""

from __future__ import annotations

# ── ADX thresholds (standard Welles Wilder values) ────────────────────────
ADX_WEAK = 20       # below this → no trend / ranging
ADX_STRONG = 40     # above this → strong trend
ADX_VERY_STRONG = 60  # above this → extremely strong (rare)

# ── Directional DI thresholds ─────────────────────────────────────────────
# plus_di vs minus_di difference considered meaningful
DI_MIN_DIFF = 0.0   # any positive/negative difference decides direction

# ── Volatility regime (percentile-based on realised vol 63d) ──────────────
RV_HIGH_PCTILE = 0.80   # above 80th percentile → high vol
RV_LOW_PCTILE = 0.20    # below 20th percentile → low vol

# ── Drawdown regime thresholds (as negative fractions) ────────────────────
DD_PEAK = -0.05         # above -5%  → peak / near high
DD_NORMAL = -0.15       # above -15% → normal pullback
DD_CORRECTION = -0.30   # above -30% → correction territory
# below -30% → bear market

# ── Trend consistency thresholds (fraction of positive returns) ───────────
TREND_CONSISTENT_HIGH = 0.55   # > 55% positive → uptrend
TREND_CONSISTENT_LOW = 0.45    # < 45% positive → downtrend

# ── Regime history window ─────────────────────────────────────────────────
REGIME_HISTORY_DAYS = 252  # trailing window for regime history
