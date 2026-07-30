"""
tests/test_registry.py
───────────────────────
Tests for the SQLite-based model registry.
"""

import os
import tempfile

import pytest

from intelligence.registry.model_registry import ModelRegistry

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path() -> str:
    """Return a temporary path; file does not exist yet."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        path = f.name
    # f is closed and deleted; we just keep the path string
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def registry(db_path: str) -> ModelRegistry:
    reg = ModelRegistry(db_path)
    reg.create_tables()
    return reg


@pytest.fixture
def sample_run_data() -> dict:
    return {
        "ticker": "RELIANCE.NS",
        "model_type": "gradient_boost",
        "model_config": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 4},
        "data_start": "2020-01-01",
        "data_end": "2025-12-31",
        "n_samples": 1200,
        "n_features": 46,
        "horizon_days": 126,
        "label_threshold": 0.0,
        "feature_list": ["trend_sma_20", "momentum_rsi_14", "regime_adx"],
    }


# ── Schema ────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_create_tables_creates_db_file(self, db_path: str):
        assert not os.path.exists(db_path)
        reg = ModelRegistry(db_path)
        reg.create_tables()
        assert os.path.exists(db_path)

    def test_create_tables_idempotent(self, registry: ModelRegistry):
        """Calling create_tables twice should not raise."""
        registry.create_tables()  # second call

    def test_empty_registry_has_no_runs(self, registry: ModelRegistry):
        assert registry.count_runs() == 0
        assert registry.list_runs() == []


# ── Write / Read ──────────────────────────────────────────────────────────────

class TestLogRun:
    def test_log_run_returns_run_id(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        assert isinstance(run_id, str)
        assert len(run_id) > 10

    def test_log_run_increments_count(self, registry: ModelRegistry, sample_run_data: dict):
        assert registry.count_runs() == 0
        registry.log_run(sample_run_data)
        assert registry.count_runs() == 1
        registry.log_run(sample_run_data)
        assert registry.count_runs() == 2

    def test_get_run_roundtrip(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        run = registry.get_run(run_id)

        assert run is not None
        assert run["run_id"] == run_id
        assert run["ticker"] == "RELIANCE.NS"
        assert run["model_type"] == "gradient_boost"
        assert run["n_samples"] == 1200
        assert run["n_features"] == 46
        assert run["horizon_days"] == 126

    def test_hyperparams_roundtrip(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        run = registry.get_run(run_id)

        hp = run["hyperparams"]
        assert isinstance(hp, dict)
        assert hp["n_estimators"] == 500
        assert hp["learning_rate"] == 0.05

    def test_feature_list_roundtrip(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        run = registry.get_run(run_id)

        fl = run["feature_list"]
        assert isinstance(fl, list)
        assert len(fl) == 3
        assert "trend_sma_20" in fl

    def test_default_status_is_trained(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        run = registry.get_run(run_id)
        assert run["status"] == "trained"

    def test_get_run_missing(self, registry: ModelRegistry):
        assert registry.get_run("nonexistent") is None


# ── Update ────────────────────────────────────────────────────────────────────

class TestUpdateRun:
    def test_update_single_field(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        registry.update_run(run_id, {"oos_sharpe": 0.93})

        run = registry.get_run(run_id)
        assert run["oos_sharpe"] == pytest.approx(0.93)

    def test_update_multiple_fields(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        registry.update_run(run_id, {
            "oos_sharpe": 0.93,
            "oos_sortino": 1.21,
            "oos_max_dd": -0.184,
            "status": "evaluated",
        })

        run = registry.get_run(run_id)
        assert run["oos_sharpe"] == pytest.approx(0.93)
        assert run["oos_sortino"] == pytest.approx(1.21)
        assert run["oos_max_dd"] == pytest.approx(-0.184)
        assert run["status"] == "evaluated"

    def test_update_only_allowed_fields(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        registry.update_run(run_id, {"oos_sharpe": 0.93, "malicious_field": "hack"})

        run = registry.get_run(run_id)
        assert run["oos_sharpe"] == pytest.approx(0.93)
        assert "malicious_field" not in run

    def test_update_empty_noop(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        registry.update_run(run_id, {})  # should not raise

    def test_update_nonexistent_run(self, registry: ModelRegistry):
        """Update on a non-existent run should not raise."""
        registry.update_run("ghost", {"oos_sharpe": 0.5})  # no-op


# ── List / Query ──────────────────────────────────────────────────────────────

class TestListRuns:
    def test_list_all_runs(self, registry: ModelRegistry, sample_run_data: dict):
        registry.log_run({**sample_run_data, "ticker": "RELIANCE.NS"})
        registry.log_run({**sample_run_data, "ticker": "TCS.NS"})

        runs = registry.list_runs()
        assert len(runs) == 2

    def test_list_filter_by_ticker(self, registry: ModelRegistry, sample_run_data: dict):
        registry.log_run({**sample_run_data, "ticker": "RELIANCE.NS"})
        registry.log_run({**sample_run_data, "ticker": "TCS.NS"})

        runs = registry.list_runs(ticker="RELIANCE.NS")
        assert len(runs) == 1
        assert runs[0]["ticker"] == "RELIANCE.NS"

    def test_list_returns_most_recent_first(self, registry: ModelRegistry, sample_run_data: dict):
        r1 = registry.log_run(sample_run_data)
        r2 = registry.log_run(sample_run_data)
        runs = registry.list_runs()
        # Most recent first
        assert runs[0]["run_id"] == r2
        assert runs[1]["run_id"] == r1

    def test_list_empty_when_no_match(self, registry: ModelRegistry):
        runs = registry.list_runs(ticker="NONEXISTENT.NS")
        assert runs == []


# ── Best Run ──────────────────────────────────────────────────────────────────

class TestBestRun:
    def test_get_best_run_by_sharpe(self, registry: ModelRegistry, sample_run_data: dict):
        r1 = registry.log_run(sample_run_data)
        r2 = registry.log_run(sample_run_data)
        registry.update_run(r1, {"oos_sharpe": 0.5})
        registry.update_run(r2, {"oos_sharpe": 1.2})

        best = registry.get_best_run("RELIANCE.NS", metric="oos_sharpe")
        assert best is not None
        assert best["run_id"] == r2
        assert best["oos_sharpe"] == pytest.approx(1.2)

    def test_get_best_run_no_evaluated_runs(self, registry: ModelRegistry, sample_run_data: dict):
        registry.log_run(sample_run_data)  # not evaluated
        best = registry.get_best_run("RELIANCE.NS")
        assert best is None

    def test_get_best_run_invalid_metric(self, registry: ModelRegistry):
        with pytest.raises(ValueError, match="metric must be one of"):
            registry.get_best_run("RELIANCE.NS", metric="invalid_metric")


# ── Delete ────────────────────────────────────────────────────────────────────

class TestDeleteRun:
    def test_delete_existing(self, registry: ModelRegistry, sample_run_data: dict):
        run_id = registry.log_run(sample_run_data)
        assert registry.count_runs() == 1

        deleted = registry.delete_run(run_id)
        assert deleted is True
        assert registry.count_runs() == 0
        assert registry.get_run(run_id) is None

    def test_delete_nonexistent(self, registry: ModelRegistry):
        deleted = registry.delete_run("ghost")
        assert deleted is False


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_log_run_without_config(self, registry: ModelRegistry):
        """Should handle missing model_config gracefully."""
        run_id = registry.log_run({
            "ticker": "TEST.NS",
            "model_type": "logistic",
            "data_start": "2020-01-01",
            "data_end": "2025-12-31",
            "n_samples": 500,
            "n_features": 10,
            "horizon_days": 63,
            "label_threshold": 0.0,
            "feature_list": ["f1", "f2"],
        })
        run = registry.get_run(run_id)
        assert run is not None
        assert run["hyperparams"] == {}

    def test_log_run_maximal_config(self, registry: ModelRegistry):
        """A dataclass config object should be serialised properly."""
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            n_estimators: int = 500
            learning_rate: float = 0.05
            name: str = "test"

        run_id = registry.log_run({
            "ticker": "TEST.NS",
            "model_type": "fake",
            "model_config": FakeConfig(),
            "data_start": "2020-01-01",
            "data_end": "2025-12-31",
            "n_samples": 500,
            "n_features": 3,
            "horizon_days": 63,
            "label_threshold": 0.0,
            "feature_list": ["a", "b", "c"],
        })
        run = registry.get_run(run_id)
        assert run["hyperparams"]["n_estimators"] == 500
        assert run["hyperparams"]["learning_rate"] == 0.05
        assert run["hyperparams"]["name"] == "test"

    def test_repeated_log_run_unique_ids(self, registry: ModelRegistry, sample_run_data: dict):
        ids = {registry.log_run(sample_run_data) for _ in range(10)}
        assert len(ids) == 10  # all unique
