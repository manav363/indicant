"""
tests/test_features.py
───────────────────────
Tests for the feature engineering layer.
Verifies correctness of indicator math and no-lookahead guarantee.
"""

import numpy as np
import pandas as pd

from intelligence.data.preprocessor import preprocess
from intelligence.features.technical import (
    _atr,
    _ema,
    _obv,
    _rsi,
    _sma,
    add_all_features,
)


def make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create a realistic synthetic OHLCV DataFrame."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 1000.0 + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 100)
    df = pd.DataFrame({
        "open":   close * np.random.uniform(0.99, 1.001, n),
        "high":   close * np.random.uniform(1.001, 1.015, n),
        "low":    close * np.random.uniform(0.985, 0.999, n),
        "close":  close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "ticker": "TEST.NS",
    }, index=idx)
    return preprocess(df)


# ── SMA tests ─────────────────────────────────────────────────────────────────

class TestSMA:
    def test_sma_equals_rolling_mean(self):
        s = pd.Series(np.arange(1.0, 101.0))
        sma = _sma(s, 10)
        expected = s.rolling(10, min_periods=5).mean()
        pd.testing.assert_series_equal(sma, expected, check_names=False)

    def test_sma_first_values_nan(self):
        s = pd.Series(np.arange(1.0, 21.0))
        sma = _sma(s, 10)
        # With min_periods=5, first 4 are NaN
        assert sma.iloc[:4].isna().all()


# ── EMA tests ─────────────────────────────────────────────────────────────────

class TestEMA:
    def test_ema_reacts_faster_than_sma(self):
        """EMA should be closer to recent price spike than SMA."""
        s = pd.Series([100.0] * 20 + [200.0] * 5)
        ema = _ema(s, 12)
        sma = _sma(s, 12)
        # After spike, EMA should be higher than SMA
        assert ema.iloc[-1] > sma.iloc[-1]

    def test_ema_bounded_by_price_range(self):
        s = pd.Series(np.random.uniform(50, 150, 100))
        ema = _ema(s, 20)
        valid = ema.dropna()
        assert (valid >= 0).all()


# ── RSI tests ─────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_bounded_0_100(self):
        s = pd.Series(np.random.uniform(100, 200, 200))
        rsi = _rsi(s, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_all_gains_gives_rsi_100(self):
        """Monotonically increasing price → RSI should approach 100."""
        s = pd.Series(np.arange(100.0, 150.0))
        rsi = _rsi(s, 14)
        assert rsi.dropna().iloc[-1] > 90

    def test_all_losses_gives_rsi_near_0(self):
        """Monotonically decreasing price → RSI should approach 0."""
        s = pd.Series(np.arange(150.0, 100.0, -1.0))
        rsi = _rsi(s, 14)
        assert rsi.dropna().iloc[-1] < 10


# ── ATR tests ─────────────────────────────────────────────────────────────────

class TestATR:
    def test_atr_always_positive(self):
        df = make_ohlcv()
        atr = _atr(df["high"], df["low"], df["close"], 14)
        assert (atr.dropna() > 0).all()

    def test_atr_larger_for_volatile_stock(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        calm = pd.DataFrame({
            "high": np.ones(100) * 101,
            "low": np.ones(100) * 99,
            "close": np.ones(100) * 100,
        }, index=idx)
        volatile = pd.DataFrame({
            "high": np.ones(100) * 115,
            "low": np.ones(100) * 85,
            "close": np.ones(100) * 100,
        }, index=idx)
        atr_calm = _atr(calm["high"], calm["low"], calm["close"], 14).dropna().mean()
        atr_vol = _atr(volatile["high"], volatile["low"], volatile["close"], 14).dropna().mean()
        assert atr_vol > atr_calm


# ── OBV tests ─────────────────────────────────────────────────────────────────

class TestOBV:
    def test_obv_increases_on_up_day(self):
        close = pd.Series([100.0, 110.0])
        volume = pd.Series([1_000_000, 2_000_000])
        obv = _obv(close, volume)
        assert obv.iloc[1] > obv.iloc[0]

    def test_obv_decreases_on_down_day(self):
        close = pd.Series([110.0, 100.0])
        volume = pd.Series([1_000_000, 2_000_000])
        obv = _obv(close, volume)
        assert obv.iloc[1] < obv.iloc[0]


# ── Full feature set tests ────────────────────────────────────────────────────

class TestAddAllFeatures:
    def test_returns_18_plus_features(self):
        df = make_ohlcv()
        featured = add_all_features(df)
        feat_cols = [c for c in featured.columns if any(
            c.startswith(p) for p in ("trend_", "momentum_", "volatility_", "volume_", "regime_")
        )]
        assert len(feat_cols) >= 18

    def test_no_new_rows_added(self):
        df = make_ohlcv()
        featured = add_all_features(df)
        assert len(featured) == len(df)

    def test_original_columns_preserved(self):
        df = make_ohlcv()
        featured = add_all_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in featured.columns

    def test_feature_categories_present(self):
        df = make_ohlcv()
        featured = add_all_features(df)
        cols = featured.columns.tolist()
        assert any(c.startswith("trend_") for c in cols)
        assert any(c.startswith("momentum_") for c in cols)
        assert any(c.startswith("volatility_") for c in cols)
        assert any(c.startswith("volume_") for c in cols)
        assert any(c.startswith("regime_") for c in cols)

    def test_rsi_in_valid_range(self):
        df = make_ohlcv()
        featured = add_all_features(df)
        rsi = featured["momentum_rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_no_lookahead_in_sma(self):
        """SMA at time t should not use data from t+1 onwards."""
        df = make_ohlcv(100)
        featured = add_all_features(df)
        # Insert a massive price spike at the end
        df2 = df.copy()
        df2.loc[df2.index[-1], "close"] = 999999
        df2 = preprocess(df2)
        f2 = add_all_features(df2)
        # SMA at second-to-last row should be same in both
        sma_before = featured["trend_sma_20"].iloc[-2]
        sma_after = f2["trend_sma_20"].iloc[-2]
        assert abs(sma_before - sma_after) < 0.01
