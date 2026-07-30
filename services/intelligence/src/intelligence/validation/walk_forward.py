"""
intelligence/validation/walk_forward.py
──────────────────────────────────────────
Walk-forward (time-series) cross-validation with purging.

This is the single most important file for avoiding fake results.

Why standard k-fold CV is WRONG for financial time series:
──────────────────────────────────────────────────────────
Standard k-fold randomly splits data into train/test folds.
With time series, this creates LEAKAGE:
- Test data may come BEFORE training data in time
- Features computed from rolling windows may "see" future data
- You get artificially high accuracy that collapses in live trading

The correct approach: Walk-Forward Validation
─────────────────────────────────────────────
Always train on PAST, test on FUTURE. Roll forward in time.

    |──── train ────|── gap ──|─ test ─|
    |──── train + new ────|── gap ──|─ test ─|
    |──── train + more ────|── gap ──|─ test ─|

The GAP (purge period) is critical:
- Features use rolling windows (e.g. 252-day volatility)
- Labels look forward in time (e.g. "price in 6 months")
- A sample near the train/test boundary is "contaminated":
  its LABEL overlaps with the TRAINING window
- We drop a buffer of samples around the boundary to prevent this

Purge period = prediction_horizon (e.g. 126 trading days for 6 months)

Math for embargo period:
    If we predict 6 months (126 days) ahead,
    the last 126 days of training data have labels that
    overlap with the test period.
    We add an embargo of 126 days AFTER the test period too,
    to prevent any feature leakage from the test period
    into the next training fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """
    A single train/test split from walk-forward CV.

    train_idx : indices into the original DataFrame for training
    test_idx  : indices for testing
    fold      : fold number (0-indexed)
    train_end : last date in training set
    test_start: first date in test set
    test_end  : last date in test set
    """
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward cross-validation.

    min_train_years : minimum years of data required before first test fold
        We need enough history for features (200-day SMA needs 200 days)
        and enough samples to train a model. 2 years = ~504 trading days.

    test_months : size of each test window in months
        3 months gives enough samples per fold without too many folds.

    purge_days : samples to drop at train/test boundary
        Set to prediction_horizon to fully eliminate label leakage.
        If predicting 6 months ahead, set purge_days = 126.

    embargo_days : samples to drop after test period
        Prevents feature leakage from test into next training fold.
        Typically set equal to or half of purge_days.

    n_splits : maximum number of folds
        None = use as many as the data allows.
    """
    min_train_years: int = 2
    test_months: int = 3
    purge_days: int = 126       # 6 months = 126 trading days
    embargo_days: int = 21      # 1 month buffer after test
    n_splits: int | None = None


class WalkForwardCV:
    """
    Purged walk-forward cross-validator for time series.

    Usage:
        cv = WalkForwardCV(config)
        for split in cv.split(df):
            X_train = X.iloc[split.train_idx]
            y_train = y.iloc[split.train_idx]
            X_test  = X.iloc[split.test_idx]
            y_test  = y.iloc[split.test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, df: pd.DataFrame) -> Iterator[WalkForwardSplit]:
        """
        Generate walk-forward splits for a time-indexed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a DatetimeIndex sorted in ascending order.

        Yields
        ------
        WalkForwardSplit
            One split per fold.
        """
        cfg = self.config

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted ascending.")

        n = len(df)
        dates = df.index

        # Minimum training size in rows (~252 trading days/year)
        min_train_rows = int(cfg.min_train_years * 252)
        # Test window in rows (~21 trading days/month)
        test_rows = int(cfg.test_months * 21)

        if n < min_train_rows + test_rows:
            raise ValueError(
                f"Not enough data: {n} rows available, need at least "
                f"{min_train_rows + test_rows} "
                f"({cfg.min_train_years}yr train + {cfg.test_months}mo test)."
            )

        fold = 0
        test_start_row = min_train_rows

        while test_start_row + test_rows <= n:
            if cfg.n_splits is not None and fold >= cfg.n_splits:
                break

            test_end_row = min(test_start_row + test_rows, n)

            # ── Train indices ──────────────────────────────────────────
            # All rows before the test window, minus the purge buffer.
            # The purge removes samples whose LABELS overlap with test period.
            #
            # Example with 6-month prediction horizon:
            # If test starts at row 500, a training sample at row 374
            # (500 - 126) has its label pointing to row 500 — which is
            # in the test set. So we exclude rows 374-499 from training.
            purge_start = test_start_row - cfg.purge_days
            train_end_row = max(0, purge_start)

            if train_end_row < min_train_rows // 2:
                # Not enough training data even after purging
                test_start_row += test_rows
                continue

            train_idx = np.arange(0, train_end_row)

            # ── Test indices ───────────────────────────────────────────
            test_idx = np.arange(test_start_row, test_end_row)

            split = WalkForwardSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                train_end=dates[train_end_row - 1],
                test_start=dates[test_start_row],
                test_end=dates[test_end_row - 1],
            )

            logger.debug(
                "Fold %d: train [%s → %s] (%d rows) | purge %d days | "
                "test [%s → %s] (%d rows)",
                fold,
                dates[0].date(), split.train_end.date(), len(train_idx),
                cfg.purge_days,
                split.test_start.date(), split.test_end.date(), len(test_idx),
            )

            yield split

            # Roll forward: next test window starts after current test
            # + embargo period
            test_start_row = test_end_row + cfg.embargo_days
            fold += 1

    def n_splits_available(self, df: pd.DataFrame) -> int:
        """Return how many folds would be generated for a given DataFrame."""
        return sum(1 for _ in self.split(df))


def make_labels(
    close: pd.Series,
    horizon_days: int = 126,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Create binary prediction labels from future returns.

    Label definition:
        y_t = 1  if  (P_{t+horizon} - P_t) / P_t > threshold
        y_t = 0  otherwise

    Parameters
    ----------
    close : pd.Series
        Closing price series with DatetimeIndex.
    horizon_days : int
        How many trading days ahead to look.
        126 = ~6 months, 252 = ~1 year
    threshold : float
        Minimum return to be labelled as 1 (buy).
        0.0 means any positive return = 1.
        0.05 means only label 1 if return > 5%.

    Returns
    -------
    pd.Series
        Binary labels (0 or 1). Last horizon_days rows will be NaN
        (we don't know the future for them).

    Important:
        The last `horizon_days` rows will have NaN labels because
        we can't know the future price. These rows MUST be excluded
        from training. The walk-forward splitter handles this automatically
        since test folds only include rows with known labels.
    """
    future_price = close.shift(-horizon_days)
    future_return = (future_price - close) / close
    labels = (future_return > threshold).astype(float)

    # Mark future rows as NaN
    labels.iloc[-horizon_days:] = np.nan

    pct_positive = labels.dropna().mean() * 100
    logger.info(
        "Labels created: horizon=%d days, threshold=%.1f%%, "
        "%.1f%% positive (buy) labels",
        horizon_days, threshold * 100, pct_positive
    )

    return labels


def prepare_ml_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon_days: int = 126,
    label_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare clean X, y for ML training from a featured DataFrame.

    Steps:
    1. Create forward-looking labels
    2. Select feature columns
    3. Drop rows where features OR labels are NaN
    4. Return aligned X, y

    Parameters
    ----------
    df : pd.DataFrame
        Output of add_all_features() — must have 'close' column.
    feature_cols : list[str]
        Which columns to use as features.
    horizon_days : int
        Prediction horizon (passed to make_labels).
    label_threshold : float
        Minimum return to label as 1.

    Returns
    -------
    X : pd.DataFrame  — feature matrix, no NaNs
    y : pd.Series     — binary labels, aligned with X
    """
    labels = make_labels(df["close"], horizon_days, label_threshold)

    X = df[feature_cols].copy()
    y = labels.copy()

    # Align and drop NaN rows
    combined = pd.concat([X, y.rename("label")], axis=1).dropna()

    X_clean = combined[feature_cols]
    y_clean = combined["label"].astype(int)

    logger.info(
        "Dataset prepared: %d rows, %d features. "
        "%.1f%% positive labels.",
        len(X_clean), len(feature_cols), y_clean.mean() * 100
    )

    return X_clean, y_clean
