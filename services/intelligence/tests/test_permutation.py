"""
tests/test_permutation.py
───────────────────────────
Tests for the permutation test module.
"""

import dataclasses
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from intelligence.backtest.permutation_test import (
    PermutationConfig,
    PermutationResult,
    PermutationTest,
)
from intelligence.validation.walk_forward import WalkForwardConfig

# ── Helpers ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _MockModelConfig:
    """Minimal mock model config (JSON-serializable dataclass)."""
    n_estimators: int = 10
    early_stopping_rounds: Any = None
    calibrate: bool = False
    random_state: int = 42


class _MockModel:
    """Minimal model mock that implements BaseModel interface for testing."""

    def __init__(self, config: Optional[_MockModelConfig] = None):
        self.config = config or _MockModelConfig()
        self.is_fitted = True
        self.feature_names: list[str] = []
        self.run_id: Optional[str] = None

    def fit(self, X, y, feature_names=None, registry=None, metadata=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.45), np.full(n, 0.55)])

    def predict(self, X, threshold=0.5):
        return np.ones(len(X), dtype=int)

    def predict_df(self, X_df, threshold=0.5):
        from intelligence.models.base import PredictionResult
        return PredictionResult(ticker="FAKE", signal="BUY", confidence=0.55, probability_up=0.55, model_name="fake")

    def feature_importance(self, top_n=15):
        return pd.DataFrame({"feature": ["f1"], "importance": [1.0]})


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic DataFrame with enough rows for walk-forward CV (min ~567)."""
    np.random.seed(42)
    n = 700
    dates = pd.date_range("2021-06-01", periods=n, freq="B")
    close_vals = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = pd.Series(close_vals, index=dates, name="close")
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
            "simple_return": np.random.randn(n) * 0.01,
        },
        index=dates,
    )
    df["trend_sma_20"] = close.rolling(20).mean()
    df["momentum_rsi_14"] = np.random.uniform(30, 70, n)
    df["volatility_atr"] = np.random.uniform(0.5, 2.0, n)
    return df


@pytest.fixture
def feature_cols() -> list[str]:
    return ["trend_sma_20", "momentum_rsi_14", "volatility_atr"]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPermutationConfig:
    def test_defaults(self):
        cfg = PermutationConfig()
        assert cfg.n_permutations == 200
        assert cfg.random_state == 42

    def test_custom_values(self):
        cfg = PermutationConfig(n_permutations=50, random_state=7)
        assert cfg.n_permutations == 50
        assert cfg.random_state == 7


class TestPermutationResult:
    def test_summary_keys(self):
        result = PermutationResult(
            ticker="TEST",
            horizon_days=126,
            actual_sharpe=0.5,
            null_sharpe_distribution=[0.1, 0.2, 0.3],
            p_value=0.04,
            n_permutations=200,
            n_permutations_completed=200,
            null_mean=0.15,
            null_std=0.1,
            null_95pct=0.35,
        )
        s = result.summary()
        assert s["permutation_p_value"] == 0.04
        assert s["n_permutations"] == 200
        assert s["null_sharpe_mean"] == 0.15
        assert s["null_sharpe_std"] == 0.1
        assert s["null_sharpe_95pct"] == 0.35

    def test_significant_flag_derived_from_p_value(self):
        r1 = PermutationResult(ticker="T", horizon_days=126, actual_sharpe=0.5, p_value=0.01, null_mean=0.0, null_std=0.1, null_95pct=0.2)
        assert r1.significant_at_5pct
        r2 = PermutationResult(ticker="T", horizon_days=126, actual_sharpe=0.5, p_value=0.10, null_mean=0.0, null_std=0.1, null_95pct=0.2)
        assert not r2.significant_at_5pct


class TestPermutationTestSmoke:
    """Smoke tests using a mock model to avoid XGBoost overhead."""

    def test_permutation_test_runs(self, sample_df, feature_cols):
        model = _MockModel()
        pt = PermutationTest(PermutationConfig(n_permutations=10))
        result = pt.run(
            ticker="FAKE",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
        )
        assert isinstance(result, PermutationResult)
        assert result.actual_sharpe is not None
        assert len(result.null_sharpe_distribution) == 10
        assert 0.0 <= result.p_value <= 1.0

    def test_permutation_loop_produces_results(self, sample_df, feature_cols):
        model = _MockModel()
        pt = PermutationTest(PermutationConfig(n_permutations=20))
        result = pt.run(
            ticker="FAKE",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
        )
        assert len(result.null_sharpe_distribution) == 20

    def test_p_value_between_zero_and_one(self, sample_df, feature_cols):
        model = _MockModel()
        pt = PermutationTest(PermutationConfig(n_permutations=15))
        result = pt.run(
            ticker="FAKE",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_registry_integration_round_trip(self, sample_df, feature_cols, tmp_path):
        from intelligence.registry.model_registry import ModelRegistry

        db_path = tmp_path / "test_perm.db"
        registry = ModelRegistry(db_path=str(db_path))
        registry.create_tables()

        run_id = registry.log_run({
            "ticker": "FAKE",
            "model_type": "gradient_boost",
            "model_config": _MockModelConfig(n_estimators=10),
            "data_start": "2021-06-01",
            "data_end": "2023-01-01",
            "n_samples": 700,
            "n_features": 3,
            "horizon_days": 21,
            "label_threshold": 0.0,
            "feature_list": feature_cols,
        })

        model = _MockModel()
        pt = PermutationTest(PermutationConfig(n_permutations=5))
        result = pt.run(
            ticker="FAKE",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
            registry=registry,
            run_id=run_id,
        )

        run = registry.get_run(run_id)
        assert run is not None
        assert run["permutation_p_value"] == pytest.approx(result.p_value, abs=0.01)
        assert run["n_permutations"] == 5

    def test_permutation_test_with_default_config(self, sample_df, feature_cols):
        model = _MockModel()
        pt = PermutationTest()
        result = pt.run(
            ticker="FAKE",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
        )
        assert isinstance(result, PermutationResult)
        assert len(result.null_sharpe_distribution) == 200

    def test_single_permutation(self, sample_df, feature_cols):
        model = _MockModel()
        pt = PermutationTest(PermutationConfig(n_permutations=1))
        result = pt.run(
            ticker="T",
            model=model,
            df=sample_df,
            feature_cols=feature_cols,
            horizon_days=21,
        )
        assert len(result.null_sharpe_distribution) == 1
        assert 0.0 <= result.p_value <= 1.0

    def test_custom_wf_config(self):
        """Runs with relaxed walk-forward constraints on smaller data."""
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        close_vals = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {"close": pd.Series(close_vals, index=dates),
             "simple_return": np.random.randn(n) * 0.01,
             "trend_sma_20": np.random.randn(n),
             "momentum_rsi_14": np.random.randn(n),
             "volatility_atr": np.random.randn(n)},
            index=dates,
        )
        model = _MockModel()
        wf_config = WalkForwardConfig(
            purge_days=10, embargo_days=2, min_train_years=0, test_months=1
        )
        pt = PermutationTest(PermutationConfig(n_permutations=3))
        result = pt.run(
            ticker="T",
            model=model,
            df=df,
            feature_cols=["trend_sma_20", "momentum_rsi_14", "volatility_atr"],
            horizon_days=10,
            wf_config=wf_config,
        )
        assert isinstance(result, PermutationResult)
