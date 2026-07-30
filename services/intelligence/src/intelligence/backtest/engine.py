"""
intelligence/backtest/engine.py
──────────────────────────────────
Walk-forward backtesting engine.

Runs a purged walk-forward backtest for a given model + stock, returning
out-of-sample portfolio returns and performance metrics.

Key design decisions:
    - Configurable rebalance cadence: weekly (every 5 trading days) by
      default — matching the plan's "weekly first, daily as follow-up".
      Daily mode rebalances every trading day.
    - Transaction costs: 0.1% per trade (STT + brokerage for India).
    - Position sizing: binary (+1 / -1 / 0) based on confidence thresholds.
      Kelly-based fractional sizing can be added as an extension.
    - Daily returns are always recorded for metric computation regardless
      of rebalance cadence; only the frequency of position changes differs.
    - Registry integration: results are written back via update_run(),
      including the evaluation_freq used so weekly vs daily results are
      never silently compared as equivalent.

Usage:
    from intelligence.backtest.engine import run_backtest, BacktestConfig
    from intelligence.validation.walk_forward import WalkForwardConfig

    result = run_backtest(
        ticker="RELIANCE.NS",
        model=my_model,
        df=featured_df,
        feature_cols=feat_cols,
        horizon_days=126,
    )
    print(f"Sharpe: {result.sharpe:.2f}, Max DD: {result.max_dd:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from intelligence.backtest.metrics import compute_all_metrics
from intelligence.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardCV,
    make_labels,
)

if TYPE_CHECKING:
    from intelligence.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for the backtest engine.

    Attributes
    ----------
    transaction_cost : float
        Fraction of notional lost per trade.
        0.001 = 0.1% (covers STT + brokerage for Indian equities).
    annual_trading_days : int
        Trading days per year (used for annualising metrics).
    buy_threshold : float
        P(up) at or above which we go long (+1 position).
    sell_threshold : float
        P(up) at or below which we go short (-1 position).
    evaluation_freq : str
        Rebalance cadence: 'weekly' (every 5 trading days, default) or
        'daily' (every trading day). Weekly is the default for the
        initial pass; daily is a follow-up refinement. The chosen freq
        is logged to the model registry so weekly and daily Sharpe
        ratios are never silently compared as equivalent.
    """
    transaction_cost: float = 0.001
    annual_trading_days: int = 252
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    evaluation_freq: str = "weekly"

    def __post_init__(self) -> None:
        if self.evaluation_freq not in ("weekly", "daily"):
            raise ValueError(
                f"evaluation_freq must be 'weekly' or 'daily', got '{self.evaluation_freq}'"
            )

    @property
    def rebalance_step(self) -> int:
        """Number of trading days between rebalances."""
        return 5 if self.evaluation_freq == "weekly" else 1


@dataclass
class FoldResult:
    """
    Results from a single walk-forward fold.

    Attributes
    ----------
    fold : int
        Fold number (0-indexed).
    train_start, train_end : pd.Timestamp
        Training window dates.
    test_start, test_end : pd.Timestamp
        Test window dates.
    n_train : int
        Number of training samples (after NaN alignment).
    n_test : int
        Number of test samples.
    """
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


@dataclass
class BacktestResult:
    """
    Complete backtest result with performance metrics and metadata.

    Attributes
    ----------
    ticker : str
        Stock ticker that was backtested.
    total_return : float
        Cumulative return over the full OOS period.
    cagr : float
        Compound annual growth rate.
    sharpe : float
        Annualised Sharpe ratio.
    sortino : float
        Annualised Sortino ratio.
    max_dd : float
        Maximum drawdown (negative value).
    turnover : float
        Average daily turnover (fraction of portfolio).
    cost_adjusted_sharpe : float
        Sharpe ratio on returns net of transaction costs.
    win_rate : float
        Fraction of days with positive returns.
    profit_factor : float
        Sum of gains / sum of losses.
    n_folds : int
        Number of walk-forward folds.
    num_trades : int
        Number of position changes (trades) in the backtest.
    start_date : str
        First date in the backtest period.
    end_date : str
        Last date in the backtest period.
    run_id : str | None
        Model registry run ID, if a registry was provided.
    folds : list[FoldResult]
        Per-fold details for diagnostics.
    daily_returns : list[float]
        Daily portfolio returns (net of costs) — useful for plotting.
    """
    ticker: str
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd: float = 0.0
    turnover: float = 0.0
    cost_adjusted_sharpe: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_folds: int = 0
    num_trades: int = 0
    start_date: str = ""
    end_date: str = ""
    run_id: Optional[str] = None
    folds: list[FoldResult] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)


# ── Position logic ──────────────────────────────────────────────────────


def position_from_probability(
    prob_up: float,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> float:
    """
    Map a model probability to a trading position.

    - P(up) >= buy_threshold  → +1 (long)
    - P(up) <= sell_threshold → -1 (short)
    - otherwise               →  0 (flat)

    Parameters
    ----------
    prob_up : float
        Model's predicted probability of price increase (0–1).
    buy_threshold : float
        Probability at or above which we go long.
    sell_threshold : float
        Probability at or below which we go short.

    Returns
    -------
    float
        +1.0, -1.0, or 0.0
    """
    if prob_up >= buy_threshold:
        return 1.0
    if prob_up <= sell_threshold:
        return -1.0
    return 0.0


# ── Walk-forward backtest loop ─────────────────────────────────────────


def run_backtest(
    ticker: str,
    model: BaseModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon_days: int = 126,
    wf_config: Optional[WalkForwardConfig] = None,
    bt_config: Optional[BacktestConfig] = None,
    registry: Any = None,
    run_id: Optional[str] = None,
    labels: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Run a purged walk-forward backtest for a single stock.

    The backtest:
        1. Generates walk-forward train/test splits with purging.
        2. For each fold, trains the model on expanding history.
        3. Predicts on each test window (out-of-sample).
        4. Generates daily positions based on prediction confidence.
        5. Computes daily portfolio returns (net of transaction costs).
        6. Aggregates performance metrics across all OOS predictions.

    Parameters
    ----------
    ticker : str
        Stock ticker (for metadata, e.g. "RELIANCE.NS").
    model : BaseModel
        An unfitted model instance. A new copy is trained per fold.
        Must implement fit() and predict_proba().
    df : pd.DataFrame
        Preprocessed DataFrame with features and DatetimeIndex.
        Must contain columns in `feature_cols` plus 'simple_return'
        (from preprocessor) and 'close'.
    feature_cols : list[str]
        Feature column names to use for training/prediction.
    horizon_days : int
        Prediction horizon in trading days (for label creation).
        Also sets the purge period in walk-forward CV.
    wf_config : WalkForwardConfig, optional
        Walk-forward cross-validation config.
        purge_days is set to horizon_days if not explicitly provided.
    bt_config : BacktestConfig, optional
        Backtest parameters (transaction costs, thresholds, evaluation
        cadence). evaluation_freq defaults to 'weekly' — only rebalances
        every 5th trading day. Set to 'daily' for daily rebalancing.
    registry : ModelRegistry, optional
        If provided, updates the registry with backtest metrics.
    run_id : str, optional
        Run ID to update in the registry. Required if registry is provided.

    Returns
    -------
    BacktestResult
        Aggregated backtest metrics and per-fold details.
    """
    bt_config = bt_config or BacktestConfig()

    # Sync purge_days with horizon_days (critical: label leakage prevention)
    wf_config = wf_config or WalkForwardConfig(
        purge_days=horizon_days,
        embargo_days=max(21, horizon_days // 6),
    )

    # Ensure simple_return exists (added by preprocessor)
    if "simple_return" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'simple_return' column. "
            "Run preprocess() before backtesting."
        )

    # ── Labels for supervised training ──────────────────────────────────
    if labels is None:
        labels = make_labels(df["close"], horizon_days)
    X_all = df[feature_cols].copy()

    # ── Walk-forward CV ────────────────────────────────────────────────
    cv = WalkForwardCV(wf_config)
    all_portfolio_returns: list[float] = []
    all_positions: list[float] = []
    all_dates: list[pd.Timestamp] = []
    fold_results: list[FoldResult] = []
    prev_position: float = 0.0

    for split in cv.split(df):
        fold = split.fold

        # ── Align training data ─────────────────────────────────────────
        # Drop rows where features or labels are NaN within this fold's
        # training window.
        train_X_raw = X_all.iloc[split.train_idx]
        train_y_raw = labels.iloc[split.train_idx]

        # Build aligned training set
        train_aligned = pd.concat(
            [train_X_raw, train_y_raw.rename("label")], axis=1
        ).dropna()

        if len(train_aligned) < 10:
            logger.warning(
                "Fold %d: only %d training samples after alignment — skipping.",
                fold, len(train_aligned),
            )
            continue

        X_train = train_aligned[feature_cols].values
        y_train = train_aligned["label"].values.astype(int)

        # ── Train model ─────────────────────────────────────────────────
        # Train a fresh copy for each fold (avoid state leakage)
        logger.info(
            "Fold %d: training on %d samples (purge=%dd, embargo=%dd)...",
            fold, len(X_train), wf_config.purge_days, wf_config.embargo_days,
        )
        model.fit(
            X_train, y_train,
            feature_names=feature_cols,
            registry=registry,
            metadata={"ticker": ticker, "horizon_days": horizon_days},
        )

        # ── Predict on test window ──────────────────────────────────────
        test_X = X_all.iloc[split.test_idx]
        test_dates = df.index[split.test_idx]
        probas = model.predict_proba(test_X.values)[:, 1]

        # ── Compute portfolio returns for each test day ─────────────────
        # In 'weekly' mode, we rebalance (predict + trade) every 5th
        # trading day. On non-rebalance days we hold the prior position
        # and incur no transaction cost. Daily returns are always
        # recorded for metric computation regardless of cadence.
        fold_returns: list[float] = []
        fold_positions: list[float] = []
        n_trades_this_fold = 0
        rebalance_step = bt_config.rebalance_step

        for i, test_day_idx in enumerate(split.test_idx):
            # Only rebalance on step days
            is_rebalance = (i % rebalance_step == 0)

            if is_rebalance:
                prob_up = float(probas[i]) if i < len(probas) else 0.5
                position = position_from_probability(
                    prob_up,
                    bt_config.buy_threshold,
                    bt_config.sell_threshold,
                )
                delta = abs(position - prev_position)
                cost = delta * bt_config.transaction_cost
                if delta > 0:
                    n_trades_this_fold += 1
                prev_position = position
            else:
                position = prev_position
                cost = 0.0

            # Next day's return (simple return of the underlying stock)
            next_idx = test_day_idx + 1
            if next_idx >= len(df):
                break  # last row of the DataFrame — no forward return
            forward_ret = float(df["simple_return"].iloc[next_idx])

            portfolio_return = position * forward_ret - cost
            fold_returns.append(portfolio_return)
            fold_positions.append(position)
            all_dates.append(test_dates[i])

        all_portfolio_returns.extend(fold_returns)
        all_positions.extend(fold_positions)

        fold_result = FoldResult(
            fold=fold,
            train_start=split.train_end - pd.Timedelta(days=len(split.train_idx)),
            train_end=split.train_end,
            test_start=split.test_start,
            test_end=split.test_end,
            n_train=len(X_train),
            n_test=len(fold_returns),
        )
        fold_results.append(fold_result)

        logger.info(
            "Fold %d done: test window [%s → %s], %d predictions, %d trades.",
            fold,
            split.test_start.date(),
            split.test_end.date(),
            len(fold_returns),
            n_trades_this_fold,
        )

    # ── Aggregate results ───────────────────────────────────────────────
    ret_array = np.array(all_portfolio_returns, dtype=np.float64)
    pos_array = np.array(all_positions, dtype=np.float64)

    if len(ret_array) < 5:
        logger.error("Backtest produced only %d data points — too few to evaluate.", len(ret_array))
        return BacktestResult(ticker=ticker, n_folds=len(fold_results), folds=fold_results)

    metrics = compute_all_metrics(ret_array, pos_array, bt_config.annual_trading_days)

    num_trades = int((np.abs(np.diff(pos_array, prepend=pos_array[0])) > 0).sum())

    result = BacktestResult(
        ticker=ticker,
        total_return=metrics["total_return"],
        cagr=metrics["cagr"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        max_dd=metrics["max_dd"],
        turnover=metrics["turnover"],
        cost_adjusted_sharpe=metrics["cost_adjusted_sharpe"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        n_folds=len(fold_results),
        num_trades=num_trades,
        start_date=str(all_dates[0].date()) if all_dates else "",
        end_date=str(all_dates[-1].date()) if all_dates else "",
        run_id=run_id,
        folds=fold_results,
        daily_returns=[round(r, 8) for r in all_portfolio_returns],
    )

    # ── Update registry ─────────────────────────────────────────────────
    if registry is not None and run_id is not None:
        from intelligence.registry.model_registry import ModelRegistry as _Reg

        if isinstance(registry, _Reg):
            registry.update_run(run_id, {
                "oos_sharpe": result.sharpe,
                "oos_sortino": result.sortino,
                "oos_max_dd": result.max_dd,
                "oos_turnover": result.turnover,
                "cost_adjusted_sharpe": result.cost_adjusted_sharpe,
                "evaluation_freq": bt_config.evaluation_freq,
                "status": "evaluated",
            })
            logger.info(
                "Updated run %s with backtest metrics (eval=%s).",
                run_id, bt_config.evaluation_freq,
            )

    logger.info(
        "Backtest complete: Sharpe=%.2f, Sortino=%.2f, MaxDD=%.1f%%, "
        "CAGR=%.1f%%, %d folds, %d trades.",
        result.sharpe, result.sortino, result.max_dd * 100,
        result.cagr * 100, result.n_folds, result.num_trades,
    )

    return result
