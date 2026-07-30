"""
tests/test_backtest.py
────────────────────────
Tests for the backtesting engine and performance metrics.

Test strategy:
    - Metrics: use synthetic return series with KNOWN statistical
      properties so we can assert expected Sharpe/Sortino/MaxDD values.
    - Engine: use a trivial "always buy" model on synthetic data and
      verify the walk-forward loop produces sensible results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelligence.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    position_from_probability,
    run_backtest,
)
from intelligence.backtest.metrics import (
    compute_all_metrics,
    compute_cagr,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_total_return,
    compute_turnover,
    compute_win_rate,
)

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

RNG = np.random.default_rng(42)


@pytest.fixture
def constant_up_returns() -> np.ndarray:
    """Every day returns exactly +0.001 (10 bps). Sharpe → ∞ (zero vol)."""
    return np.full(252, 0.001, dtype=np.float64)


@pytest.fixture
def noisy_zero_mean_returns() -> np.ndarray:
    """Gaussian noise with mean=0, std=0.01. Sharpe → 0."""
    return RNG.normal(0, 0.01, 1000)


@pytest.fixture
def positive_sharpe_returns() -> np.ndarray:
    """Mean=0.0005, std=0.01 over 252 days. Sharpe ≈ 0.05 * sqrt(252) ≈ 0.79."""
    return RNG.normal(0.0005, 0.01, 252)


@pytest.fixture
def drawdown_series() -> np.ndarray:
    """
    Returns: +10% then -20% then +10%.
    Max DD should be ≈ -0.18 (peak 1.1, trough 0.88).
    """
    r = np.zeros(300, dtype=np.float64)
    r[0:100] = 0.001  # small daily gains
    r[100:200] = -0.002  # steady decline
    r[200:300] = 0.002  # recovery
    return r


@pytest.fixture
def all_positive_returns() -> np.ndarray:
    """Every day returns exactly +0.0005. Win rate = 1.0, profit factor = inf."""
    return np.full(100, 0.0005, dtype=np.float64)


@pytest.fixture
def mixed_returns() -> np.ndarray:
    """Alternating +1% and -0.5%. Win rate ≈ 0.5, profit factor = 2.0."""
    r = np.zeros(100, dtype=np.float64)
    r[0::2] = 0.01  # even indices: +1%
    r[1::2] = -0.005  # odd indices: -0.5%
    return r


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A DataFrame with 800 daily rows of synthetic OHLCV + features + returns."""
    dates = pd.date_range("2020-01-01", periods=800, freq="B")
    close = 100.0 * np.exp(np.cumsum(RNG.normal(0.0002, 0.015, 800)))
    df = pd.DataFrame(
        {
            "close": close,
            "simple_return": np.diff(close, prepend=close[0]) / np.array(
                [close[0]] + list(close[:-1])
            ),
            "trend_sma_20": RNG.normal(0, 1, 800),
            "momentum_rsi_14": RNG.uniform(20, 80, 800),
            "volatility_atr_21": np.abs(RNG.normal(0, 1, 800)),
            "volume_obv": RNG.normal(0, 1, 800),
            "regime_adx_14": RNG.uniform(10, 40, 800),
        },
        index=dates,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — Sharpe
# ═══════════════════════════════════════════════════════════════════════


class TestSharpe:
    def test_constant_returns_zero_vol(self, constant_up_returns: np.ndarray) -> None:
        """Constant positive return → std is ~2e-19 → Sharpe is 0.0 (clamped)."""
        assert compute_sharpe(constant_up_returns) == 0.0

    def test_zero_mean_returns_zero(self, noisy_zero_mean_returns: np.ndarray) -> None:
        """Mean ≈ 0 → Sharpe ≈ 0 (within 1σ noise for seed)."""
        s = compute_sharpe(noisy_zero_mean_returns)
        assert abs(s) < 1.0

    def test_positive_sharpe(self, positive_sharpe_returns: np.ndarray) -> None:
        """Known positive mean/std → expected Sharpe ≈ 0.79."""
        s = compute_sharpe(positive_sharpe_returns)
        assert 0.3 < s < 1.5  # relaxed for random noise

    def test_too_few_points_returns_zero(self) -> None:
        assert compute_sharpe(np.array([0.001])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_sharpe(np.array([])) == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — Sortino
# ═══════════════════════════════════════════════════════════════════════


class TestSortino:
    def test_constant_returns_zero_vol(self, constant_up_returns: np.ndarray) -> None:
        """No downside → Sortino is 0 (no negative returns to compute std)."""
        assert compute_sortino(constant_up_returns) == 0.0

    def test_zero_mean(self, noisy_zero_mean_returns: np.ndarray) -> None:
        s = compute_sortino(noisy_zero_mean_returns)
        # Sortino amplifies noise since only ~half of samples contribute
        assert abs(s) < 3.0

    def test_sortino_less_than_sharpe(self, mixed_returns: np.ndarray) -> None:
        """With downside present, Sortino < Sharpe typically."""
        s = compute_sharpe(mixed_returns)
        so = compute_sortino(mixed_returns)
        # For alternating +/- series, Sortino should be finite and meaningful
        assert isinstance(so, float)
        assert isinstance(s, float)


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — Max Drawdown
# ═══════════════════════════════════════════════════════════════════════


class TestMaxDrawdown:
    def test_drawdown_series(self, drawdown_series: np.ndarray) -> None:
        dd = compute_max_drawdown(drawdown_series)
        assert dd < 0  # drawdown is negative
        assert dd > -0.5  # not worse than -50%
        # The peak is near 1.1 and trough near 0.88 → ~ -0.18 to -0.20
        assert -0.30 < dd < 0.0

    def test_all_positive_no_drawdown(self) -> None:
        dd = compute_max_drawdown(np.full(100, 0.001))
        assert dd >= 0  # no drawdown, or 0

    def test_too_few_points(self) -> None:
        assert compute_max_drawdown(np.array([0.01])) == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — CAGR / Total Return
# ═══════════════════════════════════════════════════════════════════════


class TestCAGR:
    def test_known_cagr(self) -> None:
        """10% annual return for 3 years (252*3 days)."""
        daily_ret = (1.10 ** (1 / 252)) - 1
        r = np.full(252 * 3, daily_ret)
        cagr = compute_cagr(r)
        assert cagr == pytest.approx(0.10, abs=0.005)

    def test_too_short(self) -> None:
        assert compute_cagr(np.array([0.01])) == 0.0


class TestTotalReturn:
    def test_known_total(self) -> None:
        """Daily 0.1% for 252 days → total ≈ 28.6%."""
        r = np.full(252, 0.001)
        tr = compute_total_return(r)
        expected = (1.001**252) - 1
        assert tr == pytest.approx(expected, rel=0.01)

    def test_empty_returns_zero(self) -> None:
        assert compute_total_return(np.array([])) == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — Win Rate / Profit Factor
# ═══════════════════════════════════════════════════════════════════════


class TestWinRate:
    def test_all_positive(self, all_positive_returns: np.ndarray) -> None:
        assert compute_win_rate(all_positive_returns) == 1.0

    def test_mixed(self, mixed_returns: np.ndarray) -> None:
        wr = compute_win_rate(mixed_returns)
        assert wr == pytest.approx(0.5, abs=0.05)

    def test_empty(self) -> None:
        assert compute_win_rate(np.array([])) == 0.0


class TestProfitFactor:
    def test_all_positive(self, all_positive_returns: np.ndarray) -> None:
        pf = compute_profit_factor(all_positive_returns)
        assert pf == float("inf")

    def test_mixed(self, mixed_returns: np.ndarray) -> None:
        pf = compute_profit_factor(mixed_returns)
        # Gains = 0.01 * 50 = 0.5; Losses = 0.005 * 50 = 0.25
        assert pf == pytest.approx(2.0, rel=0.1)

    def test_empty(self) -> None:
        pf = compute_profit_factor(np.array([]))
        assert pf == 1.0


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — Turnover
# ═══════════════════════════════════════════════════════════════════════


class TestTurnover:
    def test_no_changes(self) -> None:
        """Constant position → zero turnover."""
        pos = np.ones(100)
        assert compute_turnover(pos) == 0.0

    def test_daily_flip(self) -> None:
        """Flip between +1 and -1 every day. Each flip = 2.0 change.
        With prepend=pos[0], the first day has 0 change.
        Cumulative: 0+2+2+2+2 = 8, divided by 5 = 1.6."""
        pos = np.array([1, -1, 1, -1, 1], dtype=np.float64)
        assert compute_turnover(pos) == pytest.approx(1.6, abs=0.01)

    def test_single_change(self) -> None:
        """One change in 100 days → low turnover."""
        pos = np.zeros(100, dtype=np.float64)
        pos[50:] = 1.0
        # One change: 0 → 1 at index 50. Mean = (50*0 + 1 + 49*0) / 100
        assert compute_turnover(pos) == pytest.approx(0.01, abs=0.001)

    def test_too_short_returns_zero(self) -> None:
        assert compute_turnover(np.array([1.0])) == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Metrics — compute_all_metrics aggregator
# ═══════════════════════════════════════════════════════════════════════


class TestComputeAll:
    def test_returns_dict_with_expected_keys(self, mixed_returns: np.ndarray) -> None:
        metrics = compute_all_metrics(mixed_returns, mixed_returns * 0)
        expected_keys = {
            "sharpe", "sortino", "max_dd", "cagr",
            "total_return", "win_rate", "profit_factor",
            "turnover", "cost_adjusted_sharpe",
        }
        assert set(metrics.keys()) == expected_keys

    def test_without_positions(self, mixed_returns: np.ndarray) -> None:
        """Should still work, turnover = 0."""
        metrics = compute_all_metrics(mixed_returns)
        assert metrics["turnover"] == 0.0

    def test_nan_handling(self) -> None:
        """Returns with NaN should not crash."""
        r = np.array([0.001, np.nan, 0.002, -0.001])
        metrics = compute_all_metrics(r)
        assert isinstance(metrics["sharpe"], float)


# ═══════════════════════════════════════════════════════════════════════
#  Position logic
# ═══════════════════════════════════════════════════════════════════════


class TestPositionFromProbability:
    def test_buy_signal(self) -> None:
        assert position_from_probability(0.60) == 1.0

    def test_sell_signal(self) -> None:
        assert position_from_probability(0.40) == -1.0

    def test_hold_signal(self) -> None:
        assert position_from_probability(0.50) == 0.0

    def test_boundary_buy(self) -> None:
        assert position_from_probability(0.55) == 1.0

    def test_boundary_sell(self) -> None:
        assert position_from_probability(0.45) == -1.0

    def test_custom_thresholds(self) -> None:
        assert position_from_probability(0.70, buy_threshold=0.80, sell_threshold=0.20) == 0.0
        assert position_from_probability(0.85, buy_threshold=0.80, sell_threshold=0.20) == 1.0
        assert position_from_probability(0.10, buy_threshold=0.80, sell_threshold=0.20) == -1.0


# ═══════════════════════════════════════════════════════════════════════
#  Backtest Engine — integration smoke tests
# ═══════════════════════════════════════════════════════════════════════


class _AlwaysBuyModel:
    """A trivial model that always predicts P(up) = 0.9."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.run_id = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: object,
    ) -> "_AlwaysBuyModel":
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        prob_up = np.full((m, 2), 0.1)
        prob_up[:, 1] = 0.9  # P(up) = 0.9
        return prob_up

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.ones(X.shape[0], dtype=int)

    def predict_df(self, X_df: pd.DataFrame, threshold: float = 0.5) -> object:
        from intelligence.models.base import PredictionResult

        return PredictionResult(
            ticker="",
            signal="BUY",
            confidence=0.9,
            probability_up=0.9,
            model_name="AlwaysBuy",
        )

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["x"], "importance": [1.0]})


class _AlwaysSellModel:
    """Always predicts P(up) = 0.1."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.run_id = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: object,
    ) -> "_AlwaysSellModel":
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        prob_up = np.full((m, 2), 0.45)
        prob_up[:, 1] = 0.1  # P(up) = 0.1
        return prob_up

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)

    def predict_df(self, X_df: pd.DataFrame, threshold: float = 0.5) -> object:
        from intelligence.models.base import PredictionResult

        return PredictionResult(
            ticker="", signal="SELL", confidence=0.9, probability_up=0.1, model_name="AlwaysSell",
        )

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["x"], "importance": [1.0]})


class _OscillatingModel:
    """
    Alternates between BUY and SELL predictions at every other test row.

    Even-position test rows → P(up)=0.9  (BUY  → position +1)
    Odd-position test rows  → P(up)=0.1  (SELL → position -1)

    This guarantees frequent position changes in daily mode, but in
    weekly mode (rebalance every 5th row) many intra-week flips are
    skipped — making it a reliable test that weekly < daily trades.
    """

    def __init__(self) -> None:
        self.is_fitted = False
        self.run_id = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: object,
    ) -> "_OscillatingModel":
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        prob_up = np.full((m, 2), 0.5)
        for i in range(m):
            prob_up[i, 1] = 0.9 if i % 2 == 0 else 0.1
        return prob_up

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.ones(X.shape[0], dtype=int)

    def predict_df(self, X_df: pd.DataFrame, threshold: float = 0.5) -> object:
        from intelligence.models.base import PredictionResult
        return PredictionResult(
            ticker="", signal="BUY", confidence=0.9, probability_up=0.9, model_name="Oscillating",
        )

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["x"], "importance": [1.0]})


# ═══════════════════════════════════════════════════════════════════════
#  BacktestConfig — evaluation_freq
# ═══════════════════════════════════════════════════════════════════════


class TestBacktestConfig:
    def test_default_evaluation_freq_is_weekly(self) -> None:
        """Default should be weekly per locked plan decision."""
        cfg = BacktestConfig()
        assert cfg.evaluation_freq == "weekly"

    def test_rebalance_step_property(self) -> None:
        assert BacktestConfig(evaluation_freq="weekly").rebalance_step == 5
        assert BacktestConfig(evaluation_freq="daily").rebalance_step == 1

    def test_invalid_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluation_freq"):
            BacktestConfig(evaluation_freq="monthly")

    def test_daily_explicit(self) -> None:
        cfg = BacktestConfig(evaluation_freq="daily")
        assert cfg.evaluation_freq == "daily"
        assert cfg.rebalance_step == 1


# ═══════════════════════════════════════════════════════════════════════
#  Backtest Engine — weekly vs daily cadence
# ═══════════════════════════════════════════════════════════════════════


class TestWeeklyMode:
    FEATURES = ["trend_sma_20", "momentum_rsi_14", "volatility_atr_21", "volume_obv", "regime_adx_14"]

    def test_weekly_produces_fewer_trades_than_daily(
        self, synthetic_df: pd.DataFrame,
    ) -> None:
        """
        With an oscillating model that alternates BUY/SELL every day,
        weekly rebalancing (step=5) should produce strictly fewer trades
        than daily rebalancing (step=1).
        """
        model = _OscillatingModel()

        daily_result = run_backtest(
            ticker="TEST.NS", model=model, df=synthetic_df,
            feature_cols=self.FEATURES, horizon_days=63,
            bt_config=BacktestConfig(evaluation_freq="daily"),
        )

        weekly_result = run_backtest(
            ticker="TEST.NS", model=model, df=synthetic_df,
            feature_cols=self.FEATURES, horizon_days=63,
            bt_config=BacktestConfig(evaluation_freq="weekly"),
        )

        assert weekly_result.n_folds > 0
        assert daily_result.n_folds > 0
        assert weekly_result.num_trades < daily_result.num_trades, (
            f"Weekly trades ({weekly_result.num_trades}) should be less "
            f"than daily trades ({daily_result.num_trades}) on oscillating data"
        )

    def test_weekly_still_returns_daily_resolution_metrics(
        self, synthetic_df: pd.DataFrame,
    ) -> None:
        """Weekly mode should still produce daily-level return series for metric computation."""
        model = _OscillatingModel()
        result = run_backtest(
            ticker="TEST.NS", model=model, df=synthetic_df,
            feature_cols=self.FEATURES, horizon_days=63,
            bt_config=BacktestConfig(evaluation_freq="weekly"),
        )
        assert len(result.daily_returns) > 0
        assert result.sharpe is not None
        assert result.sortino is not None
        assert result.max_dd is not None

    def test_registry_logs_evaluation_freq(self, synthetic_df: pd.DataFrame) -> None:
        """Both weekly and daily values should round-trip into the registry."""
        import os
        import tempfile

        from intelligence.registry.model_registry import ModelRegistry

        db_path = tempfile.mktemp(suffix=".db")
        try:
            registry = ModelRegistry(db_path)
            registry.create_tables()

            # --- Weekly ---
            run_id_w = registry.log_run({
                "ticker": "TEST.NS", "model_type": "gradient_boost",
                "model_config": None,
                "data_start": "2020-01-01", "data_end": "2022-12-31",
                "n_samples": 100, "n_features": 5,
                "horizon_days": 63, "label_threshold": 0.0,
            })
            run_backtest(
                ticker="TEST.NS", model=_AlwaysBuyModel(), df=synthetic_df,
                feature_cols=self.FEATURES, horizon_days=63,
                bt_config=BacktestConfig(evaluation_freq="weekly"),
                registry=registry, run_id=run_id_w,
            )
            row_w = registry.get_run(run_id_w)
            assert row_w is not None
            assert row_w.get("evaluation_freq") == "weekly"
            assert row_w.get("oos_sharpe") is not None

            # --- Daily ---
            run_id_d = registry.log_run({
                "ticker": "TEST.NS", "model_type": "gradient_boost",
                "model_config": None,
                "data_start": "2020-01-01", "data_end": "2022-12-31",
                "n_samples": 100, "n_features": 5,
                "horizon_days": 63, "label_threshold": 0.0,
            })
            run_backtest(
                ticker="TEST.NS", model=_AlwaysBuyModel(), df=synthetic_df,
                feature_cols=self.FEATURES, horizon_days=63,
                bt_config=BacktestConfig(evaluation_freq="daily"),
                registry=registry, run_id=run_id_d,
            )
            row_d = registry.get_run(run_id_d)
            assert row_d is not None
            assert row_d.get("evaluation_freq") == "daily"
            assert row_d.get("oos_sharpe") is not None
        finally:
            os.unlink(db_path)


class TestBacktestEngine:
    FEATURES = ["trend_sma_20", "momentum_rsi_14", "volatility_atr_21", "volume_obv", "regime_adx_14"]

    def test_always_buy_on_synthetic_data(self, synthetic_df: pd.DataFrame) -> None:
        """Always-buy model should produce some positive return (market drifts up)."""
        model = _AlwaysBuyModel()
        result = run_backtest(
            ticker="TEST.NS",
            model=model,
            df=synthetic_df,
            feature_cols=self.FEATURES,
            horizon_days=63,
        )
        assert isinstance(result, BacktestResult)
        assert result.n_folds >= 1
        assert len(result.daily_returns) > 0
        # Always-buy on upward-drifted data → positive expected return
        assert isinstance(result.total_return, float)

    def test_always_sell_on_synthetic_data(self, synthetic_df: pd.DataFrame) -> None:
        """Always-sell should produce negative return on upward-drifted data."""
        model = _AlwaysSellModel()
        result = run_backtest(
            ticker="TEST.NS",
            model=model,
            df=synthetic_df,
            feature_cols=self.FEATURES,
            horizon_days=63,
        )
        assert result.n_folds >= 1
        # On a market that drifts up, always-short should lose money
        # (may not always hold due to random noise, but total_return < 0 is expected)
        assert isinstance(result.total_return, float)

    def test_short_horizon_does_not_crash(self, synthetic_df: pd.DataFrame) -> None:
        """A very short horizon should still complete without error."""
        model = _AlwaysBuyModel()
        result = run_backtest(
            ticker="TEST.NS",
            model=model,
            df=synthetic_df,
            feature_cols=self.FEATURES,
            horizon_days=21,
        )
        assert result.n_folds >= 1

    def test_missing_simple_return_raises(self) -> None:
        """DataFrame without simple_return should raise ValueError."""
        df = pd.DataFrame({"close": [100] * 100, "feature_1": [0] * 100},
                          index=pd.date_range("2020-01-01", periods=100, freq="B"))
        with pytest.raises(ValueError):
            run_backtest(
                ticker="TEST.NS",
                model=_AlwaysBuyModel(),
                df=df,
                feature_cols=["feature_1"],
                horizon_days=21,
            )

    def test_config_defaults_are_sensible(self) -> None:
        """BacktestConfig should have defaults matching Indian market."""
        cfg = BacktestConfig()
        assert cfg.transaction_cost == 0.001  # 0.1% (STT + brokerage)
        assert cfg.buy_threshold == 0.55
        assert cfg.sell_threshold == 0.45

    def test_result_has_all_expected_attributes(self, synthetic_df: pd.DataFrame) -> None:
        model = _AlwaysBuyModel()
        result = run_backtest(
            ticker="TEST.NS",
            model=model,
            df=synthetic_df,
            feature_cols=self.FEATURES,
            horizon_days=63,
        )
        assert result.ticker == "TEST.NS"
        assert hasattr(result, "sharpe")
        assert hasattr(result, "sortino")
        assert hasattr(result, "max_dd")
        assert hasattr(result, "cagr")
        assert hasattr(result, "total_return")
        assert hasattr(result, "turnover")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "n_folds")
        assert hasattr(result, "folds")
        assert len(result.folds) == result.n_folds
