"""
market_regime/api/routes/prediction.py
────────────────────────────────────────
/api/predict endpoint — the core of the API.

Flow:
    POST /api/predict
    { ticker, horizon_months, model }
        ↓
    fetch OHLCV → preprocess → add features → make labels
        ↓
    train model on all available data (walk-forward aware)
        ↓
    predict on most recent row
        ↓
    return PredictionResponse
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from market_regime.api.schemas import (
    FeatureImportanceItem,
    PredictionRequest,
    PredictionResponse,
    TechnicalIndicators,
)
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import load_universe
from market_regime.features.technical import add_all_features
from market_regime.models.gradient_boost import GradientBoostModel
from market_regime.models.logistic import LogisticRegressionScratch
from market_regime.validation.walk_forward import prepare_ml_dataset

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["prediction"])

# Feature column prefixes — everything except raw OHLCV
FEATURE_PREFIXES = ("trend_", "momentum_", "volatility_", "volume_", "regime_")


@router.post("", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Run full ML prediction pipeline for a single stock.

    1. Fetch historical OHLCV data
    2. Preprocess + add technical features
    3. Create forward-looking labels
    4. Train model on all available data
    5. Predict on the most recent data point
    6. Return structured prediction with indicators + feature importance
    """
    ticker = request.ticker
    horizon_days = request.horizon_months * 21   # ~21 trading days/month

    logger.info("Prediction request: ticker=%s, horizon=%d months", ticker, request.horizon_months)

    # ── Step 1: Fetch data ─────────────────────────────────────────────────
    try:
        df = fetch_ohlcv(ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for '{ticker}': {e}")

    # ── Step 2: Preprocess + features ──────────────────────────────────────
    try:
        df = preprocess(df)
        df = add_all_features(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature computation failed: {e}")

    # ── Step 3: Prepare dataset ────────────────────────────────────────────
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]

    try:
        X, y = prepare_ml_dataset(df, feat_cols, horizon_days=horizon_days)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dataset preparation failed: {e}")

    if len(X) < 200:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: only {len(X)} clean samples available."
        )

    # ── Step 4: Train model ────────────────────────────────────────────────
    try:
        if request.model == "logistic":
            model = LogisticRegressionScratch()
            model.fit(X.values, y.values, feat_cols)
        else:
            model = GradientBoostModel()
            model.fit(X.values, y.values, feat_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

    # ── Step 5: Predict on latest data point ──────────────────────────────
    # Use the most recent row from the FULL featured df (including rows
    # without labels — we're predicting the future, not validating)
    latest_features = df[feat_cols].dropna().iloc[[-1]]

    proba = model.predict_proba(latest_features.values)
    p_up = float(proba[0, 1])

    confidence_threshold = 0.55
    if p_up >= confidence_threshold:
        signal = "BUY"
    elif p_up <= (1 - confidence_threshold):
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = p_up if p_up >= 0.5 else 1 - p_up

    # ── Step 6: Build response ─────────────────────────────────────────────
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else latest_row
    price_change_1d = float(latest_row.get("simple_return", 0.0) or 0.0) * 100

    # Get company name from universe
    company_name = _get_company_name(ticker)

    # Technical indicators from latest row
    indicators = _extract_indicators(latest_row)

    # Feature importance
    top_features = _get_top_features(model, feat_cols, latest_features)

    warning = None
    if len(X) < 500:
        warning = "Limited historical data — prediction confidence may be lower."

    return PredictionResponse(
        ticker=ticker,
        company_name=company_name,
        signal=signal,
        confidence=round(confidence, 4),
        probability_up=round(p_up, 4),
        horizon_months=request.horizon_months,
        model_used=request.model,
        current_price=round(float(latest_row["close"]), 2),
        price_change_1d=round(price_change_1d, 2),
        indicators=indicators,
        top_features=top_features,
        analysis_date=date.today().isoformat(),
        warning=warning,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_company_name(ticker: str) -> str:
    try:
        universe = load_universe()
        match = universe[universe["ticker"] == ticker]
        if not match.empty:
            return str(match.iloc[0]["company_name"])
    except Exception:
        pass
    return ticker.replace(".NS", "").replace(".BO", "")


def _extract_indicators(row: pd.Series) -> TechnicalIndicators:
    """Extract latest indicator values from a DataFrame row."""
    def safe(col: str) -> float | None:
        val = row.get(col)
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


def _get_top_features(
    model,
    feat_cols: list[str],
    latest_features: pd.DataFrame,
    top_n: int = 8,
) -> list[FeatureImportanceItem]:
    """Extract top feature importances and label direction."""
    try:
        imp_df = model.feature_importance(top_n=top_n)
        result = []
        for _, row in imp_df.iterrows():
            feat = str(row["feature"])
            importance = float(row.get("importance", row.get("abs_weight", 0)))

            # Determine direction from feature value vs its importance sign
            weight = float(row.get("weight", row.get("importance", 0)))
            feat_val = 0.0
            if feat in latest_features.columns:
                feat_val = float(latest_features[feat].iloc[0] or 0)
            direction = "bullish" if (weight * feat_val) >= 0 else "bearish"

            result.append(FeatureImportanceItem(
                feature=feat,
                importance=round(importance, 4),
                direction=direction,
            ))
        return result
    except Exception as e:
        logger.warning("Could not extract feature importance: %s", e)
        return []
