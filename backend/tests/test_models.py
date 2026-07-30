"""
tests/test_models.py
─────────────────────
Tests for the ML models layer.
"""

import numpy as np
import pandas as pd
import pytest

from market_regime.models.logistic import LogisticRegressionScratch, LogisticConfig
from market_regime.models.base import PredictionResult
from market_regime.signals.generator import generate_signal, ensemble_signal, Signal
from market_regime.risk.sizing import kelly_fraction, recommended_position


# ── Logistic Regression tests ─────────────────────────────────────────────────

class TestLogisticRegression:
    def _make_data(self, n=200, n_features=10):
        np.random.seed(42)
        X = np.random.randn(n, n_features)
        # Linearly separable: y=1 if first feature > 0
        y = (X[:, 0] > 0).astype(float)
        return X, y

    def test_fit_runs(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=50))
        model.fit(X, y)
        assert model.is_fitted

    def test_predict_proba_shape(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=50))
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=50))
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_binary(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=50))
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_learns_separable_data(self):
        """Should achieve > 70% accuracy on linearly separable data."""
        X, y = self._make_data(500, 5)
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=500, learning_rate=0.05))
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.70

    def test_feature_importance_length(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=50))
        model.fit(X, y)
        imp = model.feature_importance(top_n=5)
        assert len(imp) == 5

    def test_not_fitted_raises(self):
        model = LogisticRegressionScratch()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.random.randn(10, 5))

    def test_loss_decreases(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(LogisticConfig(n_epochs=200, learning_rate=0.05))
        model.fit(X, y)
        losses = model.loss_history
        # Loss at end should be lower than at start
        assert losses[-1] < losses[0]

    def test_rejects_non_binary_labels(self):
        X = np.random.randn(50, 5)
        y = np.random.uniform(0, 1, 50)  # continuous, not binary
        model = LogisticRegressionScratch()
        with pytest.raises(ValueError, match="0 and 1"):
            model.fit(X, y)

    def test_rejects_nan_features(self):
        X = np.random.randn(50, 5)
        X[0, 0] = np.nan
        y = np.random.randint(0, 2, 50).astype(float)
        model = LogisticRegressionScratch()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X, y)


# ── Sigmoid tests ─────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_sigmoid_zero(self):
        result = LogisticRegressionScratch.sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_sigmoid_large_positive(self):
        result = LogisticRegressionScratch.sigmoid(np.array([100.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_sigmoid_large_negative(self):
        result = LogisticRegressionScratch.sigmoid(np.array([-100.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_sigmoid_output_range(self):
        z = np.linspace(-10, 10, 1000)
        result = LogisticRegressionScratch.sigmoid(z)
        assert (result > 0).all() and (result < 1).all()


# ── Signal generator tests ────────────────────────────────────────────────────

class TestSignalGenerator:
    def test_high_probability_gives_buy(self):
        result = generate_signal(0.70)
        assert result.signal == Signal.BUY

    def test_low_probability_gives_sell(self):
        result = generate_signal(0.30)
        assert result.signal == Signal.SELL

    def test_middle_probability_gives_hold(self):
        result = generate_signal(0.50)
        assert result.signal == Signal.HOLD

    def test_confidence_between_half_and_one(self):
        for p in [0.3, 0.5, 0.7, 0.9]:
            result = generate_signal(p)
            assert 0.5 <= result.confidence <= 1.0

    def test_strong_signal_label(self):
        result = generate_signal(0.80)
        assert result.strength == "strong"

    def test_weak_signal_label(self):
        result = generate_signal(0.56)
        assert result.strength == "weak"

    def test_regime_misalignment_noted(self):
        result = generate_signal(0.70, adx=15.0)  # low ADX
        assert not result.regime_aligned
        assert len(result.notes) > 0


class TestEnsembleSignal:
    def test_equal_weight_average(self):
        result = ensemble_signal([0.6, 0.7])
        assert result.probability_up == pytest.approx(0.65, abs=0.01)

    def test_weighted_average(self):
        result = ensemble_signal([0.6, 0.8], weights=[0.3, 0.7])
        expected = 0.6 * 0.3 + 0.8 * 0.7
        assert result.probability_up == pytest.approx(expected, abs=0.01)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ensemble_signal([])


# ── Kelly Criterion tests ─────────────────────────────────────────────────────

class TestKellyCriterion:
    def test_no_edge_gives_zero(self):
        """When p=0.5 and win=loss, Kelly = 0."""
        f = kelly_fraction(0.5, win_return=0.10, loss_return=0.10)
        assert abs(f) < 0.01

    def test_strong_edge_gives_positive(self):
        f = kelly_fraction(0.70, win_return=0.15, loss_return=0.08)
        assert f > 0

    def test_bounded_minus_one_to_one(self):
        for p in [0.01, 0.5, 0.99]:
            f = kelly_fraction(p)
            assert -1.0 <= f <= 1.0

    def test_zero_confidence_gives_no_position(self):
        result = recommended_position("TEST.NS", p_win=0.40)
        assert result.recommended_fraction == 0.0

    def test_high_vol_reduces_position(self):
        # Use max_position=0.5 so the cap doesn't mask vol scaling
        low_vol = recommended_position("TEST.NS", p_win=0.65, annualised_volatility=0.15, max_position=0.5)
        high_vol = recommended_position("TEST.NS", p_win=0.65, annualised_volatility=0.45, max_position=0.5)
        assert high_vol.recommended_fraction < low_vol.recommended_fraction

    def test_max_position_cap(self):
        result = recommended_position("TEST.NS", p_win=0.99, max_position=0.10)
        assert result.recommended_fraction <= 0.10


# ── Gradient Boost tests ────────────────────────────────────────────────────────

class TestGradientBoost:
    def _make_data(self, n_samples=200, n_features=5):
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        # Linearly separable (y=1 if first feature > 0)
        y = (X[:, 0] > 0).astype(float)
        return X, y

    def test_fit_runs(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        assert model.is_fitted
        assert model.model is not None

    def test_predict_proba_shape(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_returns_binary(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_learns_separable_data(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=20, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(500, 5)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.80

    def test_predict_df_returns_prediction_result(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        feature_names = [f"feat_{i}" for i in range(5)]
        X_df = pd.DataFrame(X, columns=feature_names)
        result = model.predict_df(X_df)
        assert isinstance(result, PredictionResult)
        assert result.signal in ("BUY", "HOLD")
        assert 0.5 <= result.confidence <= 1.0
        assert 0.0 <= result.probability_up <= 1.0
        assert result.model_name == "GradientBoost_XGBoost"

    def test_feature_importance_returns_dataframe(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        feature_names = [f"feat_{i}" for i in range(5)]
        model.fit(X, y, feature_names=feature_names)
        imp = model.feature_importance(top_n=5)
        assert isinstance(imp, pd.DataFrame)
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) <= 5

    def test_feature_importance_names_mapped(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        feature_names = [f"feat_{i}" for i in range(5)]
        model.fit(X, y, feature_names=feature_names)
        imp = model.feature_importance(top_n=5)
        for feat in imp["feature"]:
            assert feat.startswith("feat_")

    def test_not_fitted_raises(self):
        from market_regime.models.gradient_boost import GradientBoostModel
        model = GradientBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.random.randn(10, 5))

    def test_not_fitted_importance_raises(self):
        from market_regime.models.gradient_boost import GradientBoostModel
        model = GradientBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.feature_importance()

    def test_calibration_enabled(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=True)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(500, 5)
        model.fit(X, y)
        assert model._calibrated
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_run_id_remains_none_without_registry(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y)
        assert model.run_id is None

    def test_invalid_registry_does_not_crash(self):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        model.fit(X, y, registry="not_a_registry")
        assert model.run_id is None

    def test_registry_logging_round_trip(self, tmp_path):
        from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
        from market_regime.registry.model_registry import ModelRegistry
        cfg = GradientBoostConfig(n_estimators=10, early_stopping_rounds=None, calibrate=False)
        model = GradientBoostModel(cfg)
        X, y = self._make_data(200, 5)
        db_path = tmp_path / "test_registry.db"
        registry = ModelRegistry(db_path=str(db_path))
        registry.create_tables()
        metadata = {"ticker": "TEST", "data_start": "2020-01-01", "data_end": "2025-01-01"}
        model.fit(X, y, registry=registry, metadata=metadata)
        assert model.run_id is not None
        run = registry.get_run(model.run_id)
        assert run is not None
        assert run["ticker"] == "TEST"
        assert run["model_type"] == "gradient_boost"
