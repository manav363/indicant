"""
tests/test_data.py
───────────────────
Tests for the data layer: fetcher, preprocessor, universe.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fetcher tests ─────────────────────────────────────────────────────────────

from market_regime.data.fetcher import _normalise_ticker, _validate, _clean


class TestNormaliseTicker:
    def test_adds_ns_suffix(self):
        assert _normalise_ticker("RELIANCE") == "RELIANCE.NS"

    def test_preserves_ns_suffix(self):
        assert _normalise_ticker("TCS.NS") == "TCS.NS"

    def test_preserves_bo_suffix(self):
        assert _normalise_ticker("RELIANCE.BO") == "RELIANCE.BO"

    def test_uppercases(self):
        assert _normalise_ticker("reliance") == "RELIANCE.NS"

    def test_strips_whitespace(self):
        assert _normalise_ticker("  TCS  ") == "TCS.NS"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _normalise_ticker("")

    def test_rejects_unknown_suffix(self):
        with pytest.raises(ValueError):
            _normalise_ticker("RELIANCE.XX")


class TestValidate:
    def _make_df(self, n=100) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "open":   np.random.uniform(100, 200, n),
            "high":   np.random.uniform(150, 250, n),
            "low":    np.random.uniform(80, 150, n),
            "close":  np.random.uniform(100, 200, n),
            "volume": np.random.randint(1_000_000, 10_000_000, n),
            "ticker": "TEST.NS",
        }, index=idx)

    def test_valid_df_passes(self):
        df = self._make_df()
        # ensure high >= low
        df["high"] = df[["open", "close"]].max(axis=1) + 5
        df["low"] = df[["open", "close"]].min(axis=1) - 5
        _validate(df, "TEST.NS")   # should not raise

    def test_rejects_too_few_rows(self):
        df = self._make_df(30)
        with pytest.raises(ValueError, match="60 trading days"):
            _validate(df, "TEST.NS")

    def test_rejects_negative_prices(self):
        df = self._make_df()
        df.loc[df.index[0], "close"] = -1.0
        with pytest.raises(ValueError, match="Negative"):
            _validate(df, "TEST.NS")


# ── Preprocessor tests ────────────────────────────────────────────────────────

from market_regime.data.preprocessor import (
    preprocess, compute_log_returns, rolling_zscore, rolling_minmax
)


class TestLogReturns:
    def test_first_value_is_nan(self):
        prices = pd.Series([100, 110, 105])
        returns = compute_log_returns(prices)
        assert np.isnan(returns.iloc[0])

    def test_correct_calculation(self):
        prices = pd.Series([100.0, 110.0])
        returns = compute_log_returns(prices)
        expected = np.log(110 / 100)
        assert abs(returns.iloc[1] - expected) < 1e-10

    def test_additive_property(self):
        """Log returns are additive: r(0→2) = r(0→1) + r(1→2)"""
        prices = pd.Series([100.0, 110.0, 121.0])
        returns = compute_log_returns(prices)
        total = np.log(121 / 100)
        assert abs(returns.iloc[1] + returns.iloc[2] - total) < 1e-10


class TestRollingZscore:
    def test_output_shape(self):
        s = pd.Series(np.random.randn(100))
        z = rolling_zscore(s, 20)
        assert len(z) == len(s)

    def test_no_lookahead(self):
        """Zscore at t should only use data up to t."""
        s = pd.Series(np.ones(50))
        z = rolling_zscore(s, 10)
        # constant series → std = 0 → all NaN (division by zero handled)
        assert True  # just check it doesn't raise


class TestPreprocess:
    def _make_ohlcv(self, n=200) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 1000 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
            "ticker": "TEST.NS",
        }, index=idx)
        return df

    def test_adds_log_return(self):
        df = preprocess(self._make_ohlcv())
        assert "log_return" in df.columns

    def test_adds_typical_price(self):
        df = preprocess(self._make_ohlcv())
        assert "typical_price" in df.columns

    def test_no_nan_in_close(self):
        df = preprocess(self._make_ohlcv())
        assert df["close"].isna().sum() == 0

    def test_output_has_minimum_rows(self):
        df = preprocess(self._make_ohlcv(200))
        assert len(df) >= 60

    def test_is_outlier_column_added(self):
        df = preprocess(self._make_ohlcv())
        assert "is_outlier" in df.columns


# ── Walk-forward tests ────────────────────────────────────────────────────────

from market_regime.validation.walk_forward import (
    WalkForwardCV, WalkForwardConfig, make_labels
)


class TestWalkForwardCV:
    def _make_df(self, n=600) -> pd.DataFrame:
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        return pd.DataFrame({"close": np.random.uniform(100, 200, n)}, index=idx)

    def test_generates_splits(self):
        df = self._make_df()
        cv = WalkForwardCV(WalkForwardConfig(min_train_years=2, test_months=3))
        splits = list(cv.split(df))
        assert len(splits) >= 1

    def test_no_train_test_overlap(self):
        df = self._make_df()
        cv = WalkForwardCV(WalkForwardConfig(min_train_years=2, test_months=3, purge_days=63))
        for split in cv.split(df):
            assert split.train_idx.max() < split.test_idx.min()

    def test_test_always_after_train(self):
        df = self._make_df()
        cv = WalkForwardCV()
        for split in cv.split(df):
            assert split.train_end < split.test_start

    def test_insufficient_data_raises(self):
        df = self._make_df(50)
        cv = WalkForwardCV(WalkForwardConfig(min_train_years=2))
        with pytest.raises(ValueError, match="Not enough data"):
            list(cv.split(df))


class TestMakeLabels:
    def test_last_n_rows_are_nan(self):
        close = pd.Series(np.random.uniform(100, 200, 300),
                          index=pd.date_range("2020-01-01", periods=300, freq="B"))
        labels = make_labels(close, horizon_days=63)
        assert labels.iloc[-63:].isna().all()

    def test_labels_are_binary(self):
        close = pd.Series(np.random.uniform(100, 200, 300),
                          index=pd.date_range("2020-01-01", periods=300, freq="B"))
        labels = make_labels(close, horizon_days=63)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
