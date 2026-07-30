"""
market_regime/api/routes/regime.py
───────────────────────────────────
Regime detection endpoints.

GET /api/regime/market/summary — market-wide (NIFTY 50) regime aggregation
GET /api/regime/{ticker}       — per-stock regime classification
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from market_regime.api.schemas import (
    MarketRegimeDetail,
    MarketRegimeResponse,
    RegimeHistoryPoint,
    RegimeResponse,
)
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import load_universe
from market_regime.features.technical import add_all_features
from market_regime.regime.classifier import RegimeClassifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/regime", tags=["regime"])


@router.get("/market/summary", response_model=MarketRegimeResponse)
async def get_market_regime():
    """
    Aggregate regime classifications across NIFTY 50 constituents.

    Returns the majority regime, distribution across constituents,
    median ADX, and the `constituents_reporting` count that indicates
    data quality (how many of the 50 tickers had valid data).
    """
    try:
        universe = load_universe(indices=["NIFTY50"])
        tickers = sorted(universe["ticker"].tolist())
    except Exception:
        from market_regime.data.universe import NIFTY50_FALLBACK
        tickers = NIFTY50_FALLBACK.copy()

    classifier = RegimeClassifier()
    constituent_details: list[MarketRegimeDetail] = []
    regime_counts: dict[str, int] = {}
    adx_values: list[float] = []
    constituents_reporting = 0

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker)
            df = preprocess(df, min_rows=20)
            df = add_all_features(df)
            result = classifier.classify(df, ticker=ticker)
            constituent_details.append(MarketRegimeDetail(
                ticker=result.ticker,
                primary_regime=result.primary_regime,
                regime_confidence=result.regime_score,
                adx=result.adx,
                composite_signal=result.composite_signal,
            ))
            regime_counts[result.primary_regime] = (
                regime_counts.get(result.primary_regime, 0) + 1
            )
            if result.adx is not None:
                adx_values.append(result.adx)
            constituents_reporting += 1
        except Exception as exc:
            logger.debug("Skipping %s in market aggregation: %s", ticker, exc)
            continue

    if not constituent_details:
        raise HTTPException(
            status_code=503,
            detail="Could not classify any NIFTY 50 constituents. "
                   "Market data may be unavailable.",
        )

    primary_regime = max(regime_counts, key=regime_counts.get)  # type: ignore[arg-type]
    market_adx = round(float(sum(adx_values) / len(adx_values)), 2) if adx_values else None

    # Composite consensus
    on = sum(1 for d in constituent_details if d.composite_signal == "risk_on")
    off = sum(1 for d in constituent_details if d.composite_signal == "risk_off")
    composite_signal = "risk_on" if on > off else "risk_off" if off > on else "neutral"

    return MarketRegimeResponse(
        analysis_date=datetime.today().strftime("%Y-%m-%d"),
        total_constituents=len(tickers),
        constituents_reporting=constituents_reporting,
        primary_regime=primary_regime,
        regime_distribution=regime_counts,
        market_adx=market_adx,
        composite_signal=composite_signal,
        details=constituent_details,
    )


@router.get("/{ticker}", response_model=RegimeResponse)
async def get_stock_regime(ticker: str):
    """
    Classify the current market regime for a single stock.

    Returns the primary regime (Bull/Bear/RangeBound), confidence score,
    trend direction, volatility and drawdown regimes, composite signal,
    and trailing 252-day regime history.
    """
    ticker = ticker.strip().upper()
    if "." not in ticker:
        ticker = f"{ticker}.NS"

    try:
        df = fetch_ohlcv(ticker)
        df = preprocess(df, min_rows=60)
        df = add_all_features(df)
    except Exception as exc:
        logger.warning("Failed to process %s: %s", ticker, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch or process data for {ticker}: {exc}",
        ) from exc

    classifier = RegimeClassifier()
    result = classifier.classify(df, ticker=ticker)

    # Convert regime_history list[dict] to list[RegimeHistoryPoint].
    history = [
        RegimeHistoryPoint(date=h["date"], regime=h["regime"])
        for h in result.regime_history
    ]

    analysis_date = history[-1].date if history else datetime.today().strftime("%Y-%m-%d")

    return RegimeResponse(
        ticker=result.ticker,
        analysis_date=analysis_date,
        primary_regime=result.primary_regime,
        regime_confidence=result.regime_score,
        trend_direction=result.trend_direction,
        volatility_regime=result.volatility_regime,
        drawdown_regime=result.drawdown_regime,
        adx=result.adx,
        composite_signal=result.composite_signal,
        regime_history=history,
    )
