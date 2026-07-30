#!/usr/bin/env python3
"""
scripts/check_indicator_drift.py
─────────────────────────────────
Compares current feature distributions against a cached baseline
to detect silent drift in indicator math or market regime changes.

First-run behaviour: bootstraps the baseline (saves current stats) and
exits 0 with an informational log — never false-alerts on first execution.

Usage:
    python scripts/check_indicator_drift.py              # run drift check
    python scripts/check_indicator_drift.py --force      # re-create baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime.data.fetcher import fetch_ohlcv
from market_regime.features.technical import add_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE_DIR = Path(__file__).resolve().parent / ".baselines"
BASELINE_FILE = BASELINE_DIR / "indicator_stats.json"
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
DRIFT_THRESHOLD = 3.0  # z-score limit before flagging drift


# ── helpers ──────────────────────────────────────────────────────────────


def _feature_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute mean + std for every numeric column in the feature DataFrame."""
    numeric = df.select_dtypes(include=[np.number])
    stats: dict[str, dict[str, float]] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 2:
            continue
        stats[col] = {
            "mean": float(np.mean(series)),
            "std": float(np.std(series, ddof=1)),
        }
    return stats


def _collect_current_stats() -> dict[str, dict]:
    """Fetch fresh data and compute per-ticker feature statistics."""
    all_stats: dict[str, dict] = {}
    for ticker in TICKERS:
        df = fetch_ohlcv(ticker, period="1y")
        if df is None or df.empty:
            logger.warning("No data for %s, skipping", ticker)
            continue
        df = add_all_features(df)
        stats = _feature_stats(df)
        all_stats[ticker] = {
            "n_features": len(stats),
            "features": stats,
        }
        logger.info("Collected %d features for %s", len(stats), ticker)
    return all_stats


def _hash(stats: dict) -> str:
    """Deterministic hash of the full stats dict (sorted keys)."""
    raw = json.dumps(stats, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Check indicator drift against baseline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-create baseline even if one exists",
    )
    args = parser.parse_args()

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    current = _collect_current_stats()
    current_hash = _hash(current)

    # ── Bootstrap path: first run or --force ─────────────────────────────
    if args.force or not BASELINE_FILE.exists():
        baseline = {
            "tickers": current,
            "hash": current_hash,
            "created_at": pd.Timestamp.now().isoformat(),
            "version": 1,
        }
        BASELINE_FILE.write_text(json.dumps(baseline, indent=2, default=str))
        logger.info("Baseline created at %s (hash=%s)", BASELINE_FILE, current_hash)
        return 0  # always clean on first run

    # ── Comparison path ──────────────────────────────────────────────────
    baseline = json.loads(BASELINE_FILE.read_text())
    baseline_hash = baseline.get("hash", "")

    logger.info("Baseline hash=%s  Current hash=%s", baseline_hash, current_hash)

    if current_hash == baseline_hash:
        logger.info("No drift detected — feature distributions match baseline.")
        return 0

    # ── Per-feature drift check ──────────────────────────────────────────
    drift_found = False
    for ticker in current:
        if ticker not in baseline.get("tickers", {}):
            logger.warning("New ticker %s not in baseline — skipping drift check.", ticker)
            continue

        b_feats = baseline["tickers"][ticker].get("features", {})
        c_feats = current[ticker].get("features", {})

        for feat in set(b_feats) | set(c_feats):
            b = b_feats.get(feat)
            c = c_feats.get(feat)
            if b is None or c is None:
                logger.info("Feature %s/%s: new or removed (baseline=%s, current=%s)", ticker, feat, b is not None, c is not None)
                drift_found = True
                continue

            # z-score of the current mean against baseline distribution
            if b["std"] < 1e-12:
                continue  # constant feature
            z = abs(c["mean"] - b["mean"]) / b["std"]
            if z > DRIFT_THRESHOLD:
                logger.warning(
                    "DRIFT %s/%s: z=%.2f (baseline mean=%.4f, current=%.4f)",
                    ticker, feat, z, b["mean"], c["mean"],
                )
                drift_found = True

    if drift_found:
        logger.warning("Drift detected — indicator distributions have shifted.")
        return 1

    # Hash mismatch but no per-feature drift beyond threshold → update hash
    logger.info("Minor distribution shift within threshold — updating baseline hash.")
    baseline["hash"] = current_hash
    baseline["tickers"] = current
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
