"""
market_regime/api/routes/stocks.py
────────────────────────────────────
Stock data endpoints:
    GET /api/stocks/search?q=RELIANCE   → search NSE universe
    GET /api/stocks/{ticker}/history    → OHLCV price history
    GET /api/stocks/{ticker}/indicators → latest technical indicators
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from market_regime.api.schemas import (
    OHLCVPoint,
    PriceHistoryResponse,
    StockSearchResponse,
    StockSearchResult,
    TechnicalIndicators,
)
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import load_universe, search_ticker
from market_regime.features.technical import add_all_features

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("/search", response_model=StockSearchResponse)
async def search_stocks(
    q: str = Query(..., min_length=1, max_length=50, description="Ticker or company name"),
    limit: int = Query(default=10, ge=1, le=50),
) -> StockSearchResponse:
    """
    Search the NSE universe for stocks matching a query.
    Used for the autocomplete search bar in the frontend.
    """
    try:
        results_df = search_ticker(q.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Universe search failed: {e}")

    results = [
        StockSearchResult(
            ticker=row["ticker"],
            company_name=row["company_name"],
            industry=row["industry"],
            index_membership=row["index_membership"],
        )
        for _, row in results_df.head(limit).iterrows()
    ]

    return StockSearchResponse(
        query=q,
        results=results,
        total=len(results),
    )


@router.get("/{ticker}/history", response_model=PriceHistoryResponse)
async def get_price_history(
    ticker: str,
    years: int = Query(default=3, ge=1, le=10),
) -> PriceHistoryResponse:
    """
    Fetch OHLCV price history for a stock.
    Used to render the price chart in the frontend.
    """
    ticker = _normalise_ticker(ticker)

    try:
        df = fetch_ohlcv(ticker, lookback_years=years)
        df = preprocess(df)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for '{ticker}': {e}")

    data = []
    for dt, row in df.iterrows():
        val = row.get("log_return")
        log_ret = None if (val is None or (isinstance(val, float) and np.isnan(val))) else round(float(val), 6)
        data.append(OHLCVPoint(
            date=dt.strftime("%Y-%m-%d"),
            open=round(float(row["open"]), 2),
            high=round(float(row["high"]), 2),
            low=round(float(row["low"]), 2),
            close=round(float(row["close"]), 2),
            volume=int(row["volume"]),
            log_return=log_ret,
        ))

    company_name = _get_company_name(ticker)

    return PriceHistoryResponse(
        ticker=ticker,
        company_name=company_name,
        data=data,
        period_start=data[0].date if data else "",
        period_end=data[-1].date if data else "",
        total_rows=len(data),
    )


@router.get("/{ticker}/indicators", response_model=TechnicalIndicators)
async def get_indicators(ticker: str) -> TechnicalIndicators:
    """
    Get the latest technical indicator values for a stock.
    Used for the FeaturePanel component in the frontend.
    """
    ticker = _normalise_ticker(ticker)

    try:
        df = fetch_ohlcv(ticker)
        df = preprocess(df)
        df = add_all_features(df)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not compute indicators: {e}")

    latest = df.dropna(subset=["trend_sma_20"]).iloc[-1]

    def safe(col: str) -> float | None:
        val = latest.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)

    return TechnicalIndicators(
        sma_20=safe("trend_sma_20"),
        sma_50=safe("trend_sma_50"),
        sma_200=safe("trend_sma_200"),
        ema_12=safe("trend_ema_12"),
        ema_26=safe("trend_ema_26"),
        macd=safe("trend_macd"),
        macd_signal=safe("trend_macd_signal"),
        macd_hist=safe("trend_macd_hist"),
        golden_cross=safe("trend_golden_cross"),
        rsi_14=safe("momentum_rsi_14"),
        rsi_28=safe("momentum_rsi_28"),
        stoch_k=safe("momentum_stoch_k"),
        stoch_d=safe("momentum_stoch_d"),
        roc_1m=safe("momentum_roc_1m"),
        roc_3m=safe("momentum_roc_3m"),
        roc_6m=safe("momentum_roc_6m"),
        roc_12m=safe("momentum_roc_12m"),
        bb_upper=safe("volatility_bb_upper"),
        bb_mid=safe("volatility_bb_mid"),
        bb_lower=safe("volatility_bb_lower"),
        bb_width=safe("volatility_bb_width"),
        bb_pct_b=safe("volatility_bb_pct_b"),
        atr_14=safe("volatility_atr_14"),
        atr_pct=safe("volatility_atr_pct"),
        rv_21=safe("volatility_rv_21"),
        rv_63=safe("volatility_rv_63"),
        obv=safe("volume_obv"),
        vwap_20=safe("volume_vwap_20"),
        volume_ratio=safe("volume_ratio"),
        adx=safe("regime_adx"),
        plus_di=safe("regime_plus_di"),
        minus_di=safe("regime_minus_di"),
        trend_consistency_21=safe("regime_trend_consistency_21"),
        trend_consistency_63=safe("regime_trend_consistency_63"),
        drawdown=safe("regime_drawdown"),
        position_52w=safe("regime_52w_position"),
    )


def _normalise_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if "." not in ticker:
        ticker = f"{ticker}.NS"
    return ticker


def _get_company_name(ticker: str) -> str:
    try:
        universe = load_universe()
        match = universe[universe["ticker"] == ticker]
        if not match.empty:
            return str(match.iloc[0]["company_name"])
    except Exception:
        pass
    return ticker.replace(".NS", "")
