#!/usr/bin/env python3
"""
scripts/benchmark_permutation.py
─────────────────────────────────
Day-1 timing benchmark for the permutation test.

Measures how long one permutation iteration takes (walk-forward backtest
on shuffled labels) and extrapolates to N=200. Reports whether 200 is
feasible within a reasonable wall-clock window and whether caching
between permutations is necessary.

Usage:
    python scripts/benchmark_permutation.py
    python scripts/benchmark_permutation.py --ticker TCS.NS --horizon 3
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regime.backtest.engine import BacktestConfig, run_backtest
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.features.technical import add_all_features
from market_regime.models.gradient_boost import GradientBoostModel, GradientBoostConfig
from market_regime.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardCV,
    make_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("trend_", "momentum_", "volatility_", "volume_", "regime_")


@dataclass
class BenchmarkResult:
    ticker: str
    horizon_days: int
    n_permutations_target: int
    time_one_permutation: float   # seconds
    estimated_total: float         # seconds (wall-clock)
    n_folds: int
    n_test_samples: int
    feasible: bool
    recommendation: str


def run_benchmark(
    ticker: str = "RELIANCE.NS",
    horizon_months: int = 6,
    n_permutations_target: int = 200,
) -> BenchmarkResult:
    """Run one permutation iteration and extrapolate timing."""
    horizon_days = horizon_months * 21

    logger.info("Fetching data for %s...", ticker)
    df = fetch_ohlcv(ticker, lookback_years=5)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = preprocess(df)
    df = add_all_features(df)
    logger.info("Data ready: %d rows x %d cols", len(df), len(df.columns))

    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    logger.info("Using %d feature columns", len(feat_cols))

    # ── Create labels ────────────────────────────────────────────────────
    labels = make_labels(df["close"], horizon_days)
    X_all = df[feat_cols].copy()

    # ── Count folds (so we know the CV split size before timing) ─────────
    wf_config = WalkForwardConfig(
        purge_days=horizon_days,
        embargo_days=max(21, horizon_days // 6),
        min_train_years=2,
        test_months=3,
    )
    cv = WalkForwardCV(wf_config)
    splits = list(cv.split(df))
    n_folds = len(splits)
    logger.info("Walk-forward CV: %d folds", n_folds)

    # ── Time one permutation ─────────────────────────────────────────────
    # A single permutation: shuffle labels, train model per fold, collect returns.
    # We use the same model config as the CLI backtest.
    model_cfg = GradientBoostConfig(
        n_estimators=200,
        early_stopping_rounds=50,
        calibrate=False,
    )

    logger.info("Running 1 permutation iteration (shuffled labels)...")
    t0 = time.perf_counter()

    # Shuffle labels (breaks temporal relationship → null hypothesis)
    rng = np.random.default_rng(42)
    shuffled = labels.copy()
    shuffled_valid = shuffled.dropna()
    shuffled_idx = shuffled_valid.index
    shuffled.loc[shuffled_idx] = shuffled_valid.sample(frac=1, random_state=rng)

    model = GradientBoostModel(model_cfg)
    bt_config = BacktestConfig(evaluation_freq="weekly")

    result = run_backtest(
        ticker=ticker,
        model=model,
        df=df,
        feature_cols=feat_cols,
        horizon_days=horizon_days,
        wf_config=wf_config,
        bt_config=bt_config,
    )

    elapsed = time.perf_counter() - t0
    logger.info("One permutation: %.2f seconds", elapsed)

    # ── Extrapolate ──────────────────────────────────────────────────────
    estimated_total = elapsed * n_permutations_target
    n_test_samples = len(result.daily_returns) if result.daily_returns else 0

    # Feasibility: 200 permutations should finish within a reasonable time
    # (< 30 minutes wall-clock since we run this interactively or in CI)
    feasible = estimated_total < 1800  # 30 minutes

    if estimated_total < 300:
        recommendation = (
            f"Highly feasible: 200 permutations would take ~{estimated_total:.0f}s "
            f"({estimated_total/60:.1f} min). Run with full 200."
        )
    elif estimated_total < 1800:
        recommendation = (
            f"Feasible: 200 permutations would take ~{estimated_total:.0f}s "
            f"({estimated_total/60:.1f} min). Run with 200 — fits within 30 min window."
        )
    else:
        # Scale down to stay under 15 minutes
        n_feasible = int(900 / elapsed)
        recommendation = (
            f"Too slow: 200 permutations would take ~{estimated_total:.0f}s "
            f"({estimated_total/60:.1f} min). Recommend reducing to "
            f"{n_feasible} permutations (~15 min wall-clock), or enable "
            f"parallel execution."
        )

    logger.info("Recommendation: %s", recommendation)

    return BenchmarkResult(
        ticker=ticker,
        horizon_days=horizon_days,
        n_permutations_target=n_permutations_target,
        time_one_permutation=elapsed,
        estimated_total=estimated_total,
        n_folds=n_folds,
        n_test_samples=n_test_samples,
        feasible=feasible,
        recommendation=recommendation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permutation test timing benchmark"
    )
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--horizon", type=int, default=6, help="Horizon in months")
    parser.add_argument("--permutations", type=int, default=200, help="Target N")
    args = parser.parse_args()

    result = run_benchmark(
        ticker=args.ticker,
        horizon_months=args.horizon,
        n_permutations_target=args.permutations,
    )

    print("\n" + "=" * 60)
    print("  PERMUTATION TEST TIMING BENCHMARK")
    print("=" * 60)
    print(f"  Ticker:                  {result.ticker}")
    print(f"  Horizon:                 {result.horizon_days} trading days")
    print(f"  Walk-forward folds:      {result.n_folds}")
    print(f"  Test samples:            {result.n_test_samples}")
    print(f"  One permutation:         {result.time_one_permutation:.2f}s")
    print(f"  Target permutations:     {result.n_permutations_target}")
    print(f"  Estimated total:         {result.estimated_total:.0f}s "
          f"({result.estimated_total/60:.1f} min)")
    print(f"  Feasible (< 30 min):     {'YES' if result.feasible else 'NO'}")
    print(f"  Recommendation:          {result.recommendation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
