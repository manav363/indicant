"""
intelligence/registry/model_registry.py
───────────────────────────────────────────
SQLite-based model registry for tracking training runs.

Replaces MLflow for this use case:
  - Portable single-file DB, no server to run
  - Queryable with plain SQL
  - Easy to diff across git branches
  - Every run is self-contained (hyperparams, data range, performance)

Schema defined in schema.sql.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Versioned model registry backed by SQLite.

    Usage:
        registry = ModelRegistry("model_registry.db")
        registry.create_tables()

        run_id = registry.log_run({
            "ticker": "RELIANCE.NS",
            "model_type": "gradient_boost",
            "model_config": config,           # dataclass → serialised
            "data_start": "2020-01-01",
            "data_end": "2025-12-31",
            "n_samples": 1200,
            "n_features": 46,
            "horizon_days": 126,
            "label_threshold": 0.0,
            "feature_list": ["trend_sma_20", ...],
        })

        registry.update_run(run_id, {"oos_sharpe": 0.93, "status": "evaluated"})
        run = registry.get_run(run_id)
        best = registry.get_best_run("RELIANCE.NS", metric="oos_sharpe")
    """

    def __init__(self, db_path: str = "model_registry.db") -> None:
        self.db_path = db_path

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Connection management ───────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection. WAL mode for concurrent reads."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ── Schema ──────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create tables from schema.sql if they don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        ddl = schema_path.read_text()
        conn = self._connect()
        try:
            conn.executescript(ddl)
            conn.commit()
            logger.debug("Registry tables created/verified at %s", self.db_path)
        finally:
            conn.close()

    # ── Write ───────────────────────────────────────────────────────────────

    def log_run(self, run_data: dict[str, Any]) -> str:
        """
        Insert a new training run and return its run_id.

        Expected keys in run_data:
            ticker, model_type, model_config (dataclass), data_start, data_end,
            n_samples, n_features, horizon_days, label_threshold, feature_list
        """
        run_id = _generate_run_id()
        now = _now_iso()

        # Serialise model config (dataclass → dict → JSON)
        config = run_data.pop("model_config", None)
        hyperparams = json.dumps(asdict(config) if hasattr(config, "__dataclass_fields__") else config or {})

        feature_list = run_data.pop("feature_list", [])
        feature_list_json = json.dumps(feature_list)

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO training_runs (
                    run_id, ticker, model_type, created_at,
                    data_start, data_end, n_samples, n_features,
                    horizon_days, label_threshold, hyperparams,
                    feature_list, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'trained')
                """,
                (
                    run_id,
                    run_data.get("ticker", ""),
                    run_data.get("model_type", ""),
                    now,
                    run_data.get("data_start", ""),
                    run_data.get("data_end", ""),
                    run_data.get("n_samples", 0),
                    run_data.get("n_features", 0),
                    run_data.get("horizon_days", 126),
                    run_data.get("label_threshold", 0.0),
                    hyperparams,
                    feature_list_json,
                ),
            )
            conn.commit()
            logger.info("Logged run %s for %s (%s)", run_id, run_data.get("ticker", ""), run_data.get("model_type", ""))
        finally:
            conn.close()

        return run_id

    def update_run(self, run_id: str, updates: dict[str, Any]) -> None:
        """
        Update fields on an existing run.

        Typical use: add backtest results after evaluation.
            registry.update_run(run_id, {
                "oos_sharpe": 0.93,
                "oos_sortino": 1.21,
                "status": "evaluated",
            })
        """
        if not updates:
            return

        allowed = {
            "oos_sharpe", "oos_sortino", "oos_max_dd", "oos_turnover",
            "cost_adjusted_sharpe", "accuracy", "precision", "recall",
            "model_artifact", "status", "evaluation_freq",
            "permutation_p_value", "n_permutations",
            "null_sharpe_mean", "null_sharpe_std", "null_sharpe_95pct",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}

        if not filtered:
            logger.warning("update_run: no valid fields in %s", set(updates))
            return

        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [run_id]

        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE training_runs SET {set_clause} WHERE run_id = ?",
                values,
            )
            conn.commit()
            logger.debug("Updated run %s: %s", run_id, filtered)
        finally:
            conn.close()

    # ── Read ────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a single run by ID, or None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    def list_runs(
        self,
        ticker: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List recent runs, optionally filtered by ticker.

        Results sorted by created_at descending.
        """
        conn = self._connect()
        try:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM training_runs WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_best_run(
        self,
        ticker: str,
        metric: str = "oos_sharpe",
    ) -> dict[str, Any] | None:
        """
        Find the best run for a ticker by a numeric performance metric.

        Metric must be one of: oos_sharpe, oos_sortino, cost_adjusted_sharpe.
        Returns None if no evaluated runs exist for this ticker.
        """
        valid_metrics = {"oos_sharpe", "oos_sortino", "cost_adjusted_sharpe"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        conn = self._connect()
        try:
            row = conn.execute(
                f"""
                SELECT * FROM training_runs
                WHERE ticker = ? AND {metric} IS NOT NULL
                ORDER BY {metric} DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    # ── Utilities ───────────────────────────────────────────────────────────

    def count_runs(self, ticker: str | None = None) -> int:
        """Total number of runs, optionally filtered by ticker."""
        conn = self._connect()
        try:
            if ticker:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM training_runs WHERE ticker = ?",
                    (ticker,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM training_runs").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID. Returns True if a row was deleted."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM training_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_run_id() -> str:
    """Short, sortable run ID: timestamp + 4 random hex chars."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}_{suffix}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, parsing JSON fields."""
    d = dict(row)

    # Parse JSON columns back to Python objects
    for json_col in ("hyperparams", "feature_list"):
        if json_col in d and isinstance(d[json_col], str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                d[json_col] = json.loads(d[json_col])

    return d
