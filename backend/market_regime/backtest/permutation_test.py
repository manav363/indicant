"""
market_regime/backtest/permutation_test.py
───────────────────────────────────────────
Permutation test for walk-forward backtest significance.

Shuffles the labels N times, reruns the walk-forward backtest on each
shuffled dataset, builds a null distribution of Sharpe ratios, and
compares the actual Sharpe against this null to estimate a p-value.

Usage:
    from market_regime.backtest.engine import BacktestConfig, run_backtest
    from market_regime.backtest.permutation_test import PermutationTest

    result = PermutationTest(n_permutations=200).run(
        ticker="RELIANCE.NS",
        model=my_model,
        df=featured_df,
        feature_cols=feat_cols,
        horizon_days=126,
    )
    print(f"Actual Sharpe: {result.actual_sharpe:.2f}, p-value: {result.p_value:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from market_regime.backtest.engine import BacktestConfig, run_backtest
from market_regime.validation.walk_forward import (
    WalkForwardConfig,
    make_labels,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PermutationConfig:
    """Configuration for the permutation test."""
    n_permutations: int = 200
    random_state: int = 42
    n_jobs: int = 1               # sequential by default; parallel is future work


@dataclass
class PermutationResult:
    """Result of a permutation test against one backtest."""
    ticker: str
    horizon_days: int
    actual_sharpe: float
    null_sharpe_distribution: list[float] = field(default_factory=list)
    p_value: float = 0.0
    n_permutations: int = 0
    n_permutations_completed: int = 0
    null_mean: float = 0.0
    null_std: float = 0.0
    null_95pct: float = 0.0
    significant_at_5pct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.significant_at_5pct = self.p_value < 0.05

    def summary(self) -> dict[str, Any]:
        """Return a dict suitable for registry logging."""
        return {
            "permutation_p_value": round(self.p_value, 4),
            "n_permutations": self.n_permutations_completed,
            "null_sharpe_mean": round(self.null_mean, 4),
            "null_sharpe_std": round(self.null_std, 4),
            "null_sharpe_95pct": round(self.null_95pct, 4),
        }


class PermutationTest:
    """
    Permutation test for walk-forward backtest significance.

    The null hypothesis is that the model has no predictive power:
    the observed Sharpe ratio is drawn from the same distribution as
    Sharpe ratios obtained by training on randomly shuffled labels.

    Steps:
        1. Compute the actual (unshuffled) backtest Sharpe.
        2. For each permutation, shuffle labels and rerun backtest.
        3. Compare actual Sharpe to the null distribution.
        4. p-value = (count(null >= actual) + 1) / (N + 1)
           (+1 is the correction to avoid p=0 with finite N)
    """

    def __init__(
        self,
        config: Optional[PermutationConfig] = None,
    ) -> None:
        self.config = config or PermutationConfig()

    def run(
        self,
        ticker: str,
        model: Any,
        df: pd.DataFrame,
        feature_cols: list[str],
        horizon_days: int = 126,
        wf_config: Optional[WalkForwardConfig] = None,
        bt_config: Optional[BacktestConfig] = None,
        registry: Any = None,
        run_id: Optional[str] = None,
    ) -> PermutationResult:
        """
        Run the full permutation test.

        Parameters match ``run_backtest()`` so the caller can use the same
        config objects. The model's config is preserved across runs; a fresh
        copy is trained per fold automatically by ``run_backtest()``.

        Returns a ``PermutationResult`` with p-value and null distribution.
        """
        wf_config = wf_config or WalkForwardConfig(
            purge_days=horizon_days,
            embargo_days=max(21, horizon_days // 6),
        )
        bt_config = bt_config or BacktestConfig()
        n_perms = self.config.n_permutations
        rng = np.random.default_rng(self.config.random_state)

        logger.info(
            "Permutation test: %d permutations for %s (horizon=%dd, %d features)",
            n_perms, ticker, horizon_days, len(feature_cols),
        )

        # ── 1. Actual backtest ────────────────────────────────────────
        logger.info("Running actual (unshuffled) backtest...")
        actual_result = run_backtest(
            ticker=ticker,
            model=model,
            df=df,
            feature_cols=feature_cols,
            horizon_days=horizon_days,
            wf_config=wf_config,
            bt_config=bt_config,
            registry=registry,
            run_id=run_id,
        )
        actual_sharpe = actual_result.sharpe
        logger.info("Actual Sharpe: %.4f", actual_sharpe)

        # ── 2. Labels (shared across all permutations) ────────────────
        labels = make_labels(df["close"], horizon_days)

        # ── 3. Permutation loop ───────────────────────────────────────
        null_sharpes: list[float] = []

        for perm_idx in range(n_perms):
            shuffled = labels.copy()
            valid = shuffled.dropna()
            shuffled.loc[valid.index] = valid.sample(frac=1, random_state=rng)

            perm_model = model.__class__(model.config) if hasattr(model, "config") else model.__class__()
            # Vary the model's random_state per permutation so XGBoost's
            # internal subsample/colsample randomness differs across shuffles.
            if hasattr(perm_model.config, "random_state"):
                perm_model.config.random_state = (
                    perm_model.config.random_state + 1 + perm_idx
                )

            perm_result = run_backtest(
                ticker=ticker,
                model=perm_model,
                df=df,
                feature_cols=feature_cols,
                horizon_days=horizon_days,
                wf_config=wf_config,
                bt_config=bt_config,
                labels=shuffled,
            )

            null_sharpes.append(perm_result.sharpe)

            if (perm_idx + 1) % 50 == 0:
                logger.info(
                    "  Permutation %d/%d — current null Sharpe: %.4f",
                    perm_idx + 1, n_perms, perm_result.sharpe,
                )

        # ── 4. Compute p-value ────────────────────────────────────────
        null_array = np.array(null_sharpes, dtype=np.float64)

        # p-value with +1 correction to avoid p=0
        count_ge = int((null_array >= actual_sharpe).sum())
        p_value = (count_ge + 1) / (n_perms + 1)

        null_mean = float(np.mean(null_array))
        null_std = float(np.std(null_array, ddof=1))
        null_95pct = float(np.percentile(null_array, 95))

        logger.info(
            "Permutation test complete: p=%.4f (actual=%.4f, null=%.4f±%.4f, 95th=%.4f, %d/%d ≥ actual)",
            p_value, actual_sharpe, null_mean, null_std, null_95pct,
            count_ge, n_perms,
        )

        result = PermutationResult(
            ticker=ticker,
            horizon_days=horizon_days,
            actual_sharpe=actual_sharpe,
            null_sharpe_distribution=null_sharpes,
            p_value=p_value,
            n_permutations=n_perms,
            n_permutations_completed=n_perms,
            null_mean=null_mean,
            null_std=null_std,
            null_95pct=null_95pct,
        )

        # ── 5. Update registry with permutation results ───────────────
        if registry is not None and run_id is not None:
            from market_regime.registry.model_registry import ModelRegistry as _Reg
            if isinstance(registry, _Reg):
                registry.update_run(run_id, result.summary())
                logger.info("Updated run %s with permutation test results.", run_id)

        return result
