"""
market_regime/api/routes/universe.py
──────────────────────────────────────
GET /api/universe  → ranked list of NSE stocks with signals
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from market_regime.api.schemas import UniverseResponse, UniverseStockSummary
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import load_universe
from market_regime.features.technical import add_all_features

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/universe", tags=["universe"])

FEATURE_PREFIXES = ("trend_", "momentum_", "volatility_", "volume_", "regime_")


@router.get("", response_model=UniverseResponse)
async def get_universe(
    index: str = Query(default="NIFTY50", description="NSE index to scan"),
    limit: int = Query(default=50, ge=1, le=200),
) -> UniverseResponse:
    """
    Return a ranked list of stocks from the NSE universe with
    key indicators and signals for the screener table.

    Note: This endpoint fetches data for many stocks sequentially.
    For NIFTY50 (~50 stocks) it takes ~30-60 seconds.
    In production this would be a cached background job.
    For now, we limit to top 10 for speed.
    """
    valid_indices = {"NIFTY50", "NIFTY100", "NIFTY500"}
    if index not in valid_indices:
        raise HTTPException(status_code=400, detail=f"index must be one of {valid_indices}")

    try:
        universe = load_universe(indices=[index])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load universe: {e}")

    tickers = universe["ticker"].tolist()[:min(limit, 15)]  # cap at 15 for speed

    stocks: list[UniverseStockSummary] = []

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, lookback_years=2)
            df = preprocess(df)
            df = add_all_features(df)
            latest = df.dropna(subset=["trend_sma_20"]).iloc[-1]

            def safe(col: str) -> float | None:
                val = latest.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return None
                return round(float(val), 4)

            rsi = safe("momentum_rsi_14") or 50.0
            adx = safe("regime_adx") or 20.0
            momentum_3m = safe("momentum_roc_3m")

            # Simple rule-based signal for screener (full ML per stock is too slow here)
            p_up_proxy = _simple_signal_proxy(latest)
            if p_up_proxy >= 0.6:
                signal, confidence = "BUY", p_up_proxy
            elif p_up_proxy <= 0.4:
                signal, confidence = "SELL", 1 - p_up_proxy
            else:
                signal, confidence = "HOLD", 0.5

            meta = universe[universe["ticker"] == ticker].iloc[0]

            stocks.append(UniverseStockSummary(
                ticker=ticker,
                company_name=str(meta["company_name"]),
                industry=str(meta["industry"]),
                current_price=round(float(latest["close"]), 2),
                signal=signal,
                confidence=round(confidence, 3),
                rsi_14=safe("momentum_rsi_14"),
                adx=safe("regime_adx"),
                drawdown=safe("regime_drawdown"),
                momentum_3m=momentum_3m,
            ))

        except Exception as e:
            logger.warning("Skipping %s in universe scan: %s", ticker, e)
            continue

    # Sort by confidence descending
    stocks.sort(key=lambda s: s.confidence, reverse=True)

    return UniverseResponse(
        stocks=stocks,
        total=len(stocks),
        generated_at=datetime.utcnow().isoformat(),
    )


def _simple_signal_proxy(row) -> float:
    """
    Fast heuristic signal (0-1) for universe screener.
    Uses RSI, momentum, ADX, and 52w position.
    Not as accurate as full ML — used only for bulk screener.
    """
    score = 0.5

    def get(col: str, default: float = 0.0) -> float:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    rsi = get("momentum_rsi_14", 50)
    momentum_3m = get("momentum_roc_3m", 0)
    adx = get("regime_adx", 20)
    pos_52w = get("regime_52w_position", 0.5)
    di_diff = get("regime_di_diff", 0)

    # RSI: oversold → bullish, overbought → bearish
    if rsi < 35:
        score += 0.10
    elif rsi > 70:
        score -= 0.10

    # Momentum: positive 3m → bullish
    if momentum_3m > 5:
        score += 0.10
    elif momentum_3m < -5:
        score -= 0.10

    # ADX: strong trend
    if adx > 25 and di_diff > 0:
        score += 0.10
    elif adx > 25 and di_diff < 0:
        score -= 0.10

    # 52w position: near lows → contrarian bullish
    if pos_52w < 0.2:
        score += 0.05

    return max(0.0, min(1.0, score))
