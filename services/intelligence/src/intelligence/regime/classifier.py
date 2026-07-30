"""
intelligence/regime/classifier.py
───────────────────────────────────
Shared regime classification logic — the single source of truth for all
regime rules in the system.

Design
------
Every regime rule lives here and only here.  Both the per-stock prediction
endpoint and the market-wide ``RegimeAggregator`` reuse this class rather
than duplicating classification logic.

Usage
-----
    classifier = RegimeClassifier()
    result = classifier.classify(df)  # df must have regime_* features
    print(result.primary_regime)      # "Bull", "Bear", "RangeBound", …
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from intelligence.regime import config as cfg

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    """Single-stock regime classification result."""

    ticker: str
    primary_regime: str  # "Bull" | "Bear" | "RangeBound" | "HighVol" | "LowVol"
    regime_score: float  # confidence in the classification, 0..1
    adx: float | None
    trend_direction: str  # "up" | "down" | "sideways"
    volatility_regime: str  # "low" | "normal" | "high"
    drawdown_regime: str  # "peak" | "normal" | "correction" | "bear"
    composite_signal: str  # "risk_on" | "risk_off" | "neutral"
    regime_history: list[dict] = field(default_factory=list)
    # Each dict: {"date": str, "regime": str}


# ── Shared classifier ──────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Heuristic-based market regime classifier.

    Uses ADX for trend strength, +DI/-DI for direction, realised volatility
    percentiles for volatility regime, and drawdown from 252-day peak for
    drawdown regime.

    The classification rules are stable and interpretable — they map directly
    to standard technical analysis concepts.
    """

    # ── Required feature columns ─────────────────────────────────────────────
    # The classifier expects a DataFrame that has already been run through
    # features/technical.py add_regime_features() so these columns exist.
    REQUIRED_REGIME_COLS: set[str] = {
        "regime_adx",
        "regime_plus_di",
        "regime_minus_di",
    }

    def classify(self, df: pd.DataFrame, ticker: str = "") -> RegimeResult:
        """
        Classify the current market regime for a single stock.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the ``REQUIRED_REGIME_COLS`` columns.
            Ideally also ``regime_drawdown`` and ``regime_trend_consistency_63``
            for richer classification.
        ticker : str
            Stock ticker (used only in the result for identification).

        Returns
        -------
        RegimeResult
        """
        self._verify_columns(df)

        latest = df.iloc[-1]
        adx = latest.get("regime_adx", np.nan)
        plus_di = latest.get("regime_plus_di", np.nan)
        minus_di = latest.get("regime_minus_di", np.nan)
        drawdown = latest.get("regime_drawdown", None)
        trend_consistency = latest.get("regime_trend_consistency_63", None)

        # ── Trend direction (from +DI vs -DI) ───────────────────────────────
        if not pd.isna(plus_di) and not pd.isna(minus_di):
            trend_direction = (
                "up" if plus_di > minus_di + cfg.DI_MIN_DIFF
                else "down" if minus_di > plus_di + cfg.DI_MIN_DIFF
                else "sideways"
            )
        else:
            trend_direction = "sideways"

        # ── Primary regime (ADX-based) ──────────────────────────────────────
        if pd.isna(adx) or adx < cfg.ADX_WEAK:
            primary_regime = "RangeBound"
        elif adx >= cfg.ADX_WEAK and plus_di > minus_di:
            primary_regime = "Bull"
        elif adx >= cfg.ADX_WEAK and minus_di > plus_di:
            primary_regime = "Bear"
        else:
            # ADX >= 20 but +DI == -DI (extremely rare)
            primary_regime = "RangeBound"

        # ── Regime confidence score ──────────────────────────────────────────
        # How confident are we in the classification?
        # Based primarily on ADX strength, adjusted by trend consistency.
        regime_score = self._compute_confidence(
            adx, trend_consistency, primary_regime
        )

        # ── Volatility regime ────────────────────────────────────────────────
        volatility_regime = self._classify_volatility(df)

        # ── Drawdown regime ──────────────────────────────────────────────────
        drawdown_regime = self._classify_drawdown(drawdown)

        # ── Composite signal ─────────────────────────────────────────────────
        composite_signal = self._composite_signal(
            primary_regime, volatility_regime, drawdown_regime
        )

        # ── Regime history (last 252 rows) ───────────────────────────────────
        regime_history = self._build_history(df, ticker)

        return RegimeResult(
            ticker=ticker,
            primary_regime=primary_regime,
            regime_score=round(regime_score, 4),
            adx=round(adx, 2) if not pd.isna(adx) else None,
            trend_direction=trend_direction,
            volatility_regime=volatility_regime,
            drawdown_regime=drawdown_regime,
            composite_signal=composite_signal,
            regime_history=regime_history,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _verify_columns(df: pd.DataFrame) -> None:
        missing = RegimeClassifier.REQUIRED_REGIME_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required regime columns: {missing}. "
                "Run add_regime_features() first."
            )

    @staticmethod
    def _compute_confidence(
        adx: float, trend_consistency: float | None, regime: str
    ) -> float:
        """
        Compute a 0..1 confidence score for the classification.

        Heuristic:
        - RangeBound: confidence is inverse of ADX (low ADX = more confident
          it's ranging).
        - Bull/Bear: confidence scales with ADX, adjusted by trend consistency.
        """
        if pd.isna(adx):
            return 0.0

        if regime == "RangeBound":
            # More confident it's ranging when ADX is very low
            conf = max(0.0, 1.0 - adx / cfg.ADX_WEAK)
            return conf

        # Trending regimes — confidence scales with ADX strength
        # ADX 20 → 0.50, ADX 40 → 0.75, ADX 60 → 0.90
        conf = 0.5 + 0.4 * min(1.0, max(0.0, (adx - cfg.ADX_WEAK) / (cfg.ADX_VERY_STRONG - cfg.ADX_WEAK)))

        # Boost if trend consistency agrees
        if trend_consistency is not None and not pd.isna(trend_consistency):
            if (regime == "Bull" and trend_consistency > cfg.TREND_CONSISTENT_HIGH) or (regime == "Bear" and trend_consistency < cfg.TREND_CONSISTENT_LOW):
                conf = min(1.0, conf + 0.10)
            elif (regime == "Bull" and trend_consistency < cfg.TREND_CONSISTENT_LOW) or (regime == "Bear" and trend_consistency > cfg.TREND_CONSISTENT_HIGH):
                conf = max(0.0, conf - 0.10)

        return conf

    @staticmethod
    def _classify_volatility(df: pd.DataFrame) -> str:
        """
        Classify volatility regime using realised vol (rv_63) percentiles.

        Uses a rolling 252-day window for the percentile baseline.
        Falls back to 'normal' if rv_63 is not available.
        """
        if "rv_63" not in df.columns:
            return "normal"

        rv_63 = df["rv_63"].dropna()
        if len(rv_63) < 2:
            return "normal"

        latest_rv = rv_63.iloc[-1]
        baseline = rv_63.iloc[:-1]  # exclude current value from baseline

        if len(baseline) == 0:
            return "normal"

        percentile = (baseline < latest_rv).mean()

        if percentile >= cfg.RV_HIGH_PCTILE:
            return "high"
        elif percentile <= cfg.RV_LOW_PCTILE:
            return "low"
        else:
            return "normal"

    @staticmethod
    def _classify_drawdown(drawdown: float | None) -> str:
        """Classify drawdown regime from the drawdown fraction (always ≤ 0)."""
        if drawdown is None or pd.isna(drawdown):
            return "normal"

        if drawdown >= cfg.DD_PEAK:
            return "peak"
        elif drawdown >= cfg.DD_NORMAL:
            return "normal"
        elif drawdown >= cfg.DD_CORRECTION:
            return "correction"
        else:
            return "bear"

    @staticmethod
    def _composite_signal(
        primary_regime: str, volatility_regime: str, drawdown_regime: str
    ) -> str:
        """
        Derive a simple risk-on / risk-off / neutral composite signal.

        Rules:
        - Bull + normal/low vol + peak/normal drawdown → risk_on
        - Bear + high vol + correction/bear drawdown → risk_off
        - Everything else → neutral
        """
        risk_on = (
            primary_regime in ("Bull",)
            and volatility_regime in ("normal", "low")
            and drawdown_regime in ("peak", "normal")
        )
        risk_off = (
            primary_regime in ("Bear",)
            and volatility_regime in ("high",)
            and drawdown_regime in ("correction", "bear")
        )

        if risk_on:
            return "risk_on"
        elif risk_off:
            return "risk_off"
        else:
            return "neutral"

    @staticmethod
    def _build_history(df: pd.DataFrame, ticker: str) -> list[dict]:
        """
        Build a history of daily regime labels for the trailing N days.

        This powers the regime history sparkline in the frontend.
        Designed to be fast — vectorised, not row-by-row.
        """
        if len(df) < 2:
            return []

        # Use the last REGIME_HISTORY_DAYS rows
        lookback = min(cfg.REGIME_HISTORY_DAYS, len(df))
        df_slice = df.iloc[-lookback:]

        adx = df_slice["regime_adx"]
        plus_di = df_slice["regime_plus_di"]
        minus_di = df_slice["regime_minus_di"]

        # Vectorised regime assignment
        conditions = [
            (adx < cfg.ADX_WEAK),
            (adx >= cfg.ADX_WEAK) & (plus_di > minus_di),
            (adx >= cfg.ADX_WEAK) & (minus_di > plus_di),
        ]
        labels = ["RangeBound", "Bull", "Bear"]
        regimes = np.select(conditions, labels, default="RangeBound")

        # Use date index if available, else integer index
        dates = (
            df_slice.index.strftime("%Y-%m-%d")
            if isinstance(df_slice.index, pd.DatetimeIndex)
            else [str(i) for i in range(len(df_slice))]
        )

        return [
            {"date": d, "regime": r}
            for d, r in zip(dates[-cfg.REGIME_HISTORY_DAYS:], regimes[-cfg.REGIME_HISTORY_DAYS:], strict=True)
        ]
