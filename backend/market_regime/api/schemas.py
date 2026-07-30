"""
market_regime/api/schemas.py
──────────────────────────────
Pydantic v2 schemas for all API request and response models.

These define the contract between frontend and backend.
Every API response is validated against these schemas before being sent.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Stock Search ──────────────────────────────────────────────────────────────

class StockSearchResult(BaseModel):
    ticker: str
    company_name: str
    industry: str
    index_membership: str


class StockSearchResponse(BaseModel):
    query: str
    results: list[StockSearchResult]
    total: int


# ── OHLCV / Price Data ────────────────────────────────────────────────────────

class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    log_return: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    ticker: str
    company_name: str
    data: list[OHLCVPoint]
    period_start: str
    period_end: str
    total_rows: int


# ── Technical Indicators ──────────────────────────────────────────────────────

class TechnicalIndicators(BaseModel):
    # Trend
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    golden_cross: Optional[float] = None  # 1.0 or -1.0

    # Momentum
    rsi_14: Optional[float] = None
    rsi_28: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    roc_1m: Optional[float] = None
    roc_3m: Optional[float] = None
    roc_6m: Optional[float] = None
    roc_12m: Optional[float] = None

    # Volatility
    bb_upper: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_pct_b: Optional[float] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None
    rv_21: Optional[float] = None
    rv_63: Optional[float] = None

    # Volume
    obv: Optional[float] = None
    vwap_20: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Regime
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    trend_consistency_21: Optional[float] = None
    trend_consistency_63: Optional[float] = None
    drawdown: Optional[float] = None
    position_52w: Optional[float] = None


# ── Prediction ────────────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    direction: str   # "bullish" | "bearish"


class PredictionRequest(BaseModel):
    ticker: str
    horizon_months: int = Field(default=6, ge=1, le=24)
    model: str = Field(default="gradient_boost")

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if "." not in v:
            v = f"{v}.NS"
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"logistic", "gradient_boost"}
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    ticker: str
    company_name: str
    signal: str                    # "BUY" | "HOLD" | "SELL"
    confidence: float              # 0.5 → 1.0
    probability_up: float          # 0.0 → 1.0
    horizon_months: int
    model_used: str
    current_price: float
    price_change_1d: float         # % change today
    indicators: TechnicalIndicators
    top_features: list[FeatureImportanceItem]
    analysis_date: str
    warning: Optional[str] = None  # e.g. "low data quality"


# ── Universe / Screener ───────────────────────────────────────────────────────

class UniverseStockSummary(BaseModel):
    ticker: str
    company_name: str
    industry: str
    current_price: float
    signal: str
    confidence: float
    rsi_14: Optional[float] = None
    adx: Optional[float] = None
    drawdown: Optional[float] = None
    momentum_3m: Optional[float] = None


class UniverseResponse(BaseModel):
    stocks: list[UniverseStockSummary]
    total: int
    generated_at: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    version: str
    model_loaded: bool


# ── Regime Detection ─────────────────────────────────────────────────────────

class RegimeHistoryPoint(BaseModel):
    date: str
    regime: str              # "Bull" | "Bear" | "RangeBound"


class RegimeResponse(BaseModel):
    ticker: str
    analysis_date: str
    primary_regime: str      # "Bull" | "Bear" | "RangeBound"
    regime_confidence: float  # 0..1
    trend_direction: str     # "up" | "down" | "sideways"
    volatility_regime: str   # "low" | "normal" | "high"
    drawdown_regime: str     # "peak" | "normal" | "correction" | "bear"
    adx: Optional[float] = None
    composite_signal: str    # "risk_on" | "risk_off" | "neutral"
    regime_history: list[RegimeHistoryPoint]


class MarketRegimeDetail(BaseModel):
    ticker: str
    primary_regime: str
    regime_confidence: float
    adx: Optional[float] = None
    composite_signal: str


class MarketRegimeResponse(BaseModel):
    analysis_date: str
    total_constituents: int
    constituents_reporting: int
    primary_regime: str
    regime_distribution: dict[str, int]
    market_adx: Optional[float] = None
    composite_signal: str
    details: list[MarketRegimeDetail]
