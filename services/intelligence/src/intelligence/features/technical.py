"""
intelligence/features/technical.py
─────────────────────────────────────
Technical indicators implemented from scratch using numpy/pandas.

Philosophy:
- Every formula is written out explicitly — no black boxes.
- All indicators are computed using only PAST data at each point in time.
  This is called "causal" or "non-lookahead" computation.
- After the scratch implementation, we note where sklearn/ta-lib
  would give the same result (for validation and speed in production).

Indicators implemented:
    TREND      → SMA, EMA, MACD, MACD Signal, MACD Histogram
    MOMENTUM   → RSI, Stochastic %K, Stochastic %D
    VOLATILITY → Bollinger Bands (upper/mid/lower/width/%B), ATR
    VOLUME     → OBV, VWAP (rolling)
    REGIME     → ADX, rolling beta vs NIFTY (added separately)

All functions follow this contract:
    Input  : pd.DataFrame with columns [open, high, low, close, volume]
    Output : pd.DataFrame with NEW feature columns added (original cols kept)
    Index  : DatetimeIndex preserved, no rows dropped
    NaN    : First N rows will be NaN where N = lookback period. This is
             correct behaviour — do NOT fill these with zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator features to an OHLCV DataFrame.

    This is the main function called by the pipeline. It runs every
    indicator group in sequence and returns the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame from preprocessor. Must have columns:
        open, high, low, close, volume, log_return

    Returns
    -------
    pd.DataFrame
        Original DataFrame + all feature columns.
        Feature columns are prefixed by category:
        trend_*, momentum_*, volatility_*, volume_*, regime_*
    """
    df = df.copy()
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_regime_features(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, MACD and derived trend features."""
    close = df["close"]

    # ── Simple Moving Averages ─────────────────────────────────────────────
    # SMA(n)_t = (1/n) * Σ P_{t-n+1..t}
    # Arithmetic mean of the last n closing prices.
    # Lags behind price — the longer the window, the more lag.
    df["trend_sma_20"] = _sma(close, 20)
    df["trend_sma_50"] = _sma(close, 50)
    df["trend_sma_200"] = _sma(close, 200)

    # ── Exponential Moving Averages ────────────────────────────────────────
    # EMA(n)_t = α * P_t + (1-α) * EMA(n)_{t-1}
    # where α = 2 / (n + 1)
    #
    # EMA reacts faster to recent price changes than SMA.
    # At α=0.1 (n=19), today's price gets 10% weight,
    # yesterday's EMA gets 90% weight.
    df["trend_ema_12"] = _ema(close, 12)
    df["trend_ema_26"] = _ema(close, 26)
    df["trend_ema_50"] = _ema(close, 50)

    # ── Price vs MA (normalised distance) ─────────────────────────────────
    # How far is price from its moving average, as a fraction of MA?
    # Positive = price above MA (bullish), Negative = below (bearish)
    # This is scale-invariant — works across stocks of any price level.
    df["trend_price_vs_sma20"] = (close - df["trend_sma_20"]) / df["trend_sma_20"]
    df["trend_price_vs_sma50"] = (close - df["trend_sma_50"]) / df["trend_sma_50"]
    df["trend_price_vs_sma200"] = (close - df["trend_sma_200"]) / df["trend_sma_200"]

    # ── Golden/Death Cross signal ──────────────────────────────────────────
    # Golden cross: SMA50 crosses above SMA200 → bullish regime signal
    # Death cross:  SMA50 crosses below SMA200 → bearish regime signal
    # We encode as: +1 (golden), -1 (death), 0 (neither/NaN)
    sma50 = df["trend_sma_50"]
    sma200 = df["trend_sma_200"]
    df["trend_golden_cross"] = np.where(sma50 > sma200, 1.0, -1.0)
    df["trend_golden_cross"] = df["trend_golden_cross"].where(
        sma50.notna() & sma200.notna(), other=np.nan
    )

    # ── MACD ──────────────────────────────────────────────────────────────
    # MACD Line     = EMA(12) - EMA(26)
    # Signal Line   = EMA(9) of MACD Line
    # MACD Histogram = MACD Line - Signal Line
    #
    # Interpretation:
    # - MACD > 0: short-term momentum above long-term (bullish)
    # - MACD crossing Signal from below: buy signal
    # - Histogram growing: momentum accelerating
    #
    # Standard parameters: (12, 26, 9) — used by most practitioners
    macd_line = df["trend_ema_12"] - df["trend_ema_26"]
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line

    df["trend_macd"] = macd_line
    df["trend_macd_signal"] = signal_line
    df["trend_macd_hist"] = histogram

    # Normalise MACD by price level so it's comparable across stocks
    df["trend_macd_norm"] = macd_line / close

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, Stochastic Oscillator, and rate-of-change features."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── RSI (Relative Strength Index) ─────────────────────────────────────
    # RSI measures the speed and magnitude of price changes.
    #
    # Step 1: Compute daily price changes
    #   Δ_t = P_t - P_{t-1}
    #
    # Step 2: Separate gains and losses
    #   gain_t = max(Δ_t, 0)
    #   loss_t = max(-Δ_t, 0)   ← always positive
    #
    # Step 3: Smooth using Wilder's Moving Average (RMA)
    #   avg_gain_t = (avg_gain_{t-1} * (n-1) + gain_t) / n
    #   avg_loss_t = (avg_loss_{t-1} * (n-1) + loss_t) / n
    #   (This is equivalent to EMA with α = 1/n)
    #
    # Step 4: Relative Strength and RSI
    #   RS_t = avg_gain_t / avg_loss_t
    #   RSI_t = 100 - (100 / (1 + RS_t))
    #
    # Bounds: RSI ∈ [0, 100]
    # RSI > 70 → overbought (potential sell signal)
    # RSI < 30 → oversold  (potential buy signal)
    #
    # For long-term prediction, RSI(14) is standard.
    # We also compute RSI(28) for longer-term momentum.
    df["momentum_rsi_14"] = _rsi(close, 14)
    df["momentum_rsi_28"] = _rsi(close, 28)

    # ── Stochastic Oscillator ──────────────────────────────────────────────
    # Compares closing price to high-low range over n periods.
    #
    # %K_t = 100 * (C_t - L_n) / (H_n - L_n)
    # where:
    #   C_t = current close
    #   L_n = lowest low over last n periods
    #   H_n = highest high over last n periods
    #
    # %D_t = SMA(3) of %K_t   ← smoothed signal line
    #
    # %K ∈ [0, 100]
    # %K > 80 → overbought, %K < 20 → oversold
    # %K crossing %D → momentum shift signal
    k, d = _stochastic(high, low, close, k_period=14, d_period=3)
    df["momentum_stoch_k"] = k
    df["momentum_stoch_d"] = d
    df["momentum_stoch_diff"] = k - d   # positive = bullish momentum

    # ── Rate of Change (ROC) ──────────────────────────────────────────────
    # ROC_n_t = (P_t - P_{t-n}) / P_{t-n} * 100
    #
    # Measures percentage price change over n periods.
    # For long-term prediction, we use:
    # - 1 month (21 trading days)
    # - 3 months (63 trading days)
    # - 6 months (126 trading days)
    # - 12 months (252 trading days)
    for n, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
        df[f"momentum_roc_{label}"] = (close - close.shift(n)) / close.shift(n) * 100

    # ── Momentum Score (composite) ────────────────────────────────────────
    # Simple composite: average of z-scored 3m and 6m ROC.
    # Used as a single momentum signal for ranking stocks.
    roc_3m = df["momentum_roc_3m"]
    roc_6m = df["momentum_roc_6m"]
    df["momentum_composite"] = (
        _rolling_zscore(roc_3m, 252) + _rolling_zscore(roc_6m, 252)
    ) / 2

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Bollinger Bands, ATR, and realised volatility features."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── Bollinger Bands ────────────────────────────────────────────────────
    # Bollinger Bands place volatility-based envelopes around a moving average.
    #
    # Middle Band = SMA(20)
    # Upper Band  = SMA(20) + k * σ(20)
    # Lower Band  = SMA(20) - k * σ(20)
    #
    # where σ(20) = rolling standard deviation of close over 20 periods
    # Standard parameter: k = 2 (covers ~95% of price action if normal)
    #
    # Derived features:
    # %B = (Price - Lower) / (Upper - Lower)   ← where in the band is price?
    #      %B = 1.0 means price is at upper band
    #      %B = 0.0 means price is at lower band
    #      %B > 1.0 or < 0.0 means price is outside bands (rare)
    #
    # Bandwidth = (Upper - Lower) / Middle     ← how wide are the bands?
    #             High bandwidth = high volatility
    #             Low bandwidth = squeeze (often precedes breakout)
    sma20 = _sma(close, 20)
    std20 = close.rolling(window=20, min_periods=10).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20

    df["volatility_bb_upper"] = upper
    df["volatility_bb_mid"] = sma20
    df["volatility_bb_lower"] = lower
    df["volatility_bb_width"] = (upper - lower) / sma20
    df["volatility_bb_pct_b"] = (close - lower) / (upper - lower)

    # ── ATR (Average True Range) ───────────────────────────────────────────
    # ATR measures market volatility. Unlike std dev, it accounts for
    # gaps between sessions (overnight moves).
    #
    # True Range (TR) = max of:
    #   1. High_t - Low_t                    (today's range)
    #   2. |High_t - Close_{t-1}|            (gap up + today's range)
    #   3. |Low_t  - Close_{t-1}|            (gap down + today's range)
    #
    # ATR(n) = Wilder's MA of TR over n periods
    #        = (ATR_{t-1} * (n-1) + TR_t) / n
    #
    # ATR is in price units (₹). We normalise by close to make it
    # comparable across stocks:
    # ATR% = ATR / Close * 100
    atr14 = _atr(high, low, close, 14)
    atr21 = _atr(high, low, close, 21)
    df["volatility_atr_14"] = atr14
    df["volatility_atr_pct"] = atr14 / close * 100    # normalised
    df["volatility_atr_21"] = atr21

    # ── Realised Volatility ────────────────────────────────────────────────
    # Realised volatility = rolling std dev of log returns, annualised.
    #
    # σ_realised = std(log_returns, window=n) * sqrt(252)
    #
    # Multiplying by sqrt(252) annualises it (252 trading days/year).
    # This is the standard way to express volatility in finance.
    #
    # A stock with σ = 0.30 has 30% annualised volatility —
    # typical for mid-cap Indian stocks.
    if "log_return" in df.columns:
        log_ret = df["log_return"]
        df["volatility_rv_21"] = log_ret.rolling(21).std() * np.sqrt(252)
        df["volatility_rv_63"] = log_ret.rolling(63).std() * np.sqrt(252)

    # ── Volatility Regime ──────────────────────────────────────────────────
    # Is current volatility high or low relative to its own history?
    # Rolling z-score of ATR% over 252 days.
    # Positive = volatility above average (risk-off environment)
    # Negative = volatility below average (calm market)
    if "volatility_atr_pct" in df.columns:
        df["volatility_regime"] = _rolling_zscore(df["volatility_atr_pct"], 252)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OBV, VWAP, and volume trend features."""
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    # ── OBV (On-Balance Volume) ────────────────────────────────────────────
    # OBV tracks cumulative volume direction.
    #
    # OBV_t = OBV_{t-1} + V_t  if C_t > C_{t-1}  (up day)
    # OBV_t = OBV_{t-1} - V_t  if C_t < C_{t-1}  (down day)
    # OBV_t = OBV_{t-1}         if C_t = C_{t-1}  (flat day)
    #
    # Interpretation: if price rises with rising OBV → strong trend.
    # OBV diverging from price → potential trend reversal.
    #
    # We normalise OBV by its rolling mean to make it stationary.
    obv = _obv(close, volume)
    df["volume_obv"] = obv
    df["volume_obv_norm"] = _rolling_zscore(obv, 63)

    # ── Rolling VWAP ───────────────────────────────────────────────────────
    # VWAP = Volume Weighted Average Price
    #
    # True intraday VWAP requires tick data.
    # We approximate with daily data using a rolling window:
    #
    # VWAP(n)_t = Σ(TP_i * V_i) / Σ(V_i)  for i in [t-n+1, t]
    #
    # where TP_i = (H_i + L_i + C_i) / 3  (typical price)
    #
    # Interpretation: price above VWAP → buyers in control (bullish)
    #                 price below VWAP → sellers in control (bearish)
    if "typical_price" not in df.columns:
        df["typical_price"] = (high + low + close) / 3

    df["volume_vwap_20"] = _rolling_vwap(df["typical_price"], volume, 20)
    df["volume_price_vs_vwap"] = (close - df["volume_vwap_20"]) / df["volume_vwap_20"]

    # ── Volume Trend ───────────────────────────────────────────────────────
    # Is volume trending up or down relative to its own average?
    # Volume ratio = today's volume / 20-day average volume
    # > 1.5 → unusually high volume (confirms price moves)
    # < 0.5 → unusually low volume (price moves less reliable)
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    df["volume_ratio"] = volume / vol_ma20
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan)

    # ── Money Flow ────────────────────────────────────────────────────────
    # Simplified money flow: up-day volume vs down-day volume ratio
    # over a rolling window.
    #
    # up_volume   = volume on days where close > previous close
    # down_volume = volume on days where close < previous close
    # MFR = rolling(up_volume) / rolling(down_volume)
    price_change = close.diff()
    up_vol = volume.where(price_change > 0, 0.0)
    down_vol = volume.where(price_change < 0, 0.0)
    roll_up = up_vol.rolling(14, min_periods=5).sum()
    roll_down = down_vol.rolling(14, min_periods=5).sum()
    df["volume_mfr"] = roll_up / roll_down.replace(0, np.nan)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# REGIME INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime features.

    These features characterise the TYPE of market environment,
    not just the direction. They're the most important features
    for long-term prediction.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── ADX (Average Directional Index) ───────────────────────────────────
    # ADX measures trend STRENGTH, not direction.
    #
    # Step 1: Directional Movement
    #   +DM_t = max(H_t - H_{t-1}, 0)  if H_t-H_{t-1} > L_{t-1}-L_t, else 0
    #   -DM_t = max(L_{t-1} - L_t, 0)  if L_{t-1}-L_t > H_t-H_{t-1}, else 0
    #
    # Step 2: Smooth with ATR(14)
    #   +DI_14 = 100 * RMA(+DM, 14) / ATR(14)
    #   -DI_14 = 100 * RMA(-DM, 14) / ATR(14)
    #
    # Step 3: ADX
    #   DX_t = 100 * |+DI_t - -DI_t| / (+DI_t + -DI_t)
    #   ADX  = RMA(DX, 14)
    #
    # Interpretation:
    # ADX < 20 → weak/no trend (ranging market)
    # ADX 20-40 → developing trend
    # ADX > 40 → strong trend
    # ADX > 60 → extremely strong trend (rare)
    #
    # Note: ADX doesn't tell you UP or DOWN — use +DI vs -DI for that.
    adx, plus_di, minus_di = _adx(high, low, close, 14)
    df["regime_adx"] = adx
    df["regime_plus_di"] = plus_di
    df["regime_minus_di"] = minus_di
    df["regime_di_diff"] = plus_di - minus_di   # positive = uptrend

    # ── Trend Consistency ─────────────────────────────────────────────────
    # What fraction of the last n days had positive returns?
    # Near 1.0 → consistent uptrend
    # Near 0.0 → consistent downtrend
    # Near 0.5 → choppy/ranging
    if "log_return" in df.columns:
        log_ret = df["log_return"]
        df["regime_trend_consistency_21"] = (
            (log_ret > 0).rolling(21).mean()
        )
        df["regime_trend_consistency_63"] = (
            (log_ret > 0).rolling(63).mean()
        )

    # ── Drawdown from Peak ────────────────────────────────────────────────
    # How far is the current price from its rolling 252-day peak?
    # Drawdown = (Price - Peak) / Peak
    # Always <= 0. Near 0 = at or near 52-week high (bullish).
    # -0.20 = 20% below peak (bear market territory).
    rolling_peak = close.rolling(252, min_periods=21).max()
    df["regime_drawdown"] = (close - rolling_peak) / rolling_peak

    # ── 52-week High/Low Position ──────────────────────────────────────────
    # Where is price within its 52-week range?
    # 0 = at 52-week low, 1 = at 52-week high
    # Similar to Stochastic but over a year, so captures longer momentum.
    high_252 = high.rolling(252, min_periods=21).max()
    low_252 = low.rolling(252, min_periods=21).min()
    df["regime_52w_position"] = (close - low_252) / (high_252 - low_252)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SCRATCH IMPLEMENTATIONS (the actual math)
# ══════════════════════════════════════════════════════════════════════════════

def _sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.
    SMA_t = (1/n) * Σ P_{t-n+1..t}
    """
    return series.rolling(window=window, min_periods=window // 2).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    EMA_t = α * P_t + (1-α) * EMA_{t-1}
    where α = 2 / (span + 1)

    pandas ewm uses adjust=True by default which gives a slightly different
    formula at startup. We use adjust=False to match TradingView/Bloomberg.
    """
    return series.ewm(span=span, adjust=False, min_periods=span // 2).mean()


def _rma(series: pd.Series, window: int) -> pd.Series:
    """
    Wilder's Moving Average (RMA).
    Used internally by RSI and ATR.

    RMA_t = (RMA_{t-1} * (n-1) + x_t) / n
    Equivalent to EMA with α = 1/n
    """
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — from scratch.

    RSI = 100 - (100 / (1 + RS))
    RS  = avg_gain / avg_loss   (smoothed with Wilder's MA)
    """
    delta = close.diff()

    gain = delta.clip(lower=0)      # keep positive, zero out negative
    loss = (-delta).clip(lower=0)   # keep positive (flip sign), zero out positive

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    # Avoid division by zero: if avg_loss = 0, RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss = 0 (all gains), RSI should be 100
    rsi = rsi.fillna(100.0)

    return rsi


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator — from scratch.

    %K = 100 * (C - L_n) / (H_n - L_n)
    %D = SMA(d_period) of %K
    """
    low_n = low.rolling(k_period, min_periods=k_period // 2).min()
    high_n = high.rolling(k_period, min_periods=k_period // 2).max()

    denom = high_n - low_n
    k = 100 * (close - low_n) / denom.replace(0, np.nan)
    d = k.rolling(d_period, min_periods=1).mean()

    return k, d


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range — from scratch.

    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = Wilder's MA of TR
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True range is the max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return _rma(tr, period)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume — from scratch.

    OBV_t = OBV_{t-1} + sign(C_t - C_{t-1}) * V_t
    where sign: +1 if up, -1 if down, 0 if flat
    """
    direction = np.sign(close.diff()).fillna(0)
    signed_volume = direction * volume
    return signed_volume.cumsum()


def _rolling_vwap(
    typical_price: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """
    Rolling VWAP — from scratch.

    VWAP(n)_t = Σ(TP_i * V_i) / Σ(V_i)  for last n periods
    """
    tp_vol = typical_price * volume
    sum_tp_vol = tp_vol.rolling(window, min_periods=window // 2).sum()
    sum_vol = volume.rolling(window, min_periods=window // 2).sum()
    return sum_tp_vol / sum_vol.replace(0, np.nan)


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX, +DI, -DI — from scratch.

    Returns (adx, plus_di, minus_di)
    """
    # Directional movement
    up_move = high.diff()
    down_move = (-low).diff()   # = low_{t-1} - low_t, so positive means fell

    # +DM: up move when it's larger than down move AND positive
    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0
    )
    # -DM: down move when it's larger than up move AND positive
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0
    )

    plus_dm_s = pd.Series(plus_dm, index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)

    atr = _atr(high, low, close, period)

    # Smooth with Wilder's MA
    smooth_plus_dm = _rma(plus_dm_s, period)
    smooth_minus_dm = _rma(minus_dm_s, period)

    # Directional indicators (as percentages)
    plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = _rma(dx, period)

    return adx, plus_di, minus_di


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std
    Only uses past data — no lookahead.
    """
    roll = series.rolling(window=window, min_periods=max(2, window // 4))
    return (series - roll.mean()) / roll.std()
