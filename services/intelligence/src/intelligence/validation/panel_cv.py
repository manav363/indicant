"""Purged, embargoed cross-validation for a panel.

The single most dangerous file in the project.

v1's walk-forward splitter works on one symbol's series and splits by row. On a
pooled panel that is wrong in a way that produces *better* numbers: splitting by
row puts RELIANCE-on-2015-03-04 in train and TCS-on-2015-03-04 in test, so the
model sees the market's state on a day it is being tested on. Every metric
downstream is then fiction, and nothing about the code looks broken.

So: **split by date, never by row.** Every symbol on a given date goes to the
same fold.

Two more guards, both from López de Prado:

* **Purge** — drop training samples whose *label* resolves inside the test
  window. With a 126-day horizon, the last 126 days of training know the answer
  to the first part of the test.
* **Embargo** — drop training samples immediately *after* the test window.
  Features are rolling functions of the recent past, so a sample just after the
  test period was partly computed from it.

Purge without embargo is the common half-measure. It closes the leak going
forward and leaves the one coming back.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelSplit:
    """One fold. Indices are positional into the panel frame."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: tuple[date, date]
    test_dates: tuple[date, date]
    n_purged: int = 0
    n_embargoed: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"PanelSplit(fold={self.fold} "
            f"train={self.train_dates[0]}..{self.train_dates[1]} n={self.n_train:,} "
            f"test={self.test_dates[0]}..{self.test_dates[1]} n={self.n_test:,} "
            f"purged={self.n_purged:,} embargoed={self.n_embargoed:,})"
        )


class CVMode(StrEnum):
    """Which data a fold may train on. These are not interchangeable.

    EXPANDING — train only on dates BEFORE the test window (walk-forward).
        Use when simulating a strategy. A backtest that trained on the future
        is not a backtest, however well purged.

    PURGED_KFOLD — train on everything outside the purged and embargoed bands,
        including dates after the test window (López de Prado's PurgedKFold).
        Use when generating out-of-fold predictions to feed a meta-learner:
        the goal there is an unbiased generalisation estimate, and using both
        sides of the window gives every fold more data without leaking, because
        purge and embargo remove the contaminated boundary.

    Getting these backwards is silent. Training on the future in a backtest
    produces a Sharpe you cannot earn; restricting a K-fold to the past just
    wastes data. Only one of those is dangerous, which is why EXPANDING is the
    default.
    """

    EXPANDING = "expanding"
    PURGED_KFOLD = "purged_kfold"


@dataclass(frozen=True)
class PanelCVConfig:
    n_splits: int = 5
    # Must be >= the label horizon. Anything less leaves training samples whose
    # labels resolve inside the test window.
    purge_days: int = 126
    # AFML suggests ~1% of the sample; the floor matters more than the fraction
    # because feature windows are absolute, not proportional.
    embargo_days: int = 21
    min_train_dates: int = 252
    mode: CVMode = CVMode.EXPANDING


class PurgedPanelCV:
    """Purged, embargoed CV over dates. See `CVMode` for the two variants."""

    def __init__(self, config: PanelCVConfig | None = None) -> None:
        self.config = config or PanelCVConfig()

    def split(
        self,
        panel: pd.DataFrame,
        *,
        date_column: str = "date",
    ) -> Iterator[PanelSplit]:
        cfg = self.config
        if panel.empty:
            raise ValueError("cannot split an empty panel")
        if date_column not in panel.columns:
            raise ValueError(f"panel has no {date_column!r} column to split on")

        dates = np.array(sorted(pd.unique(panel[date_column])))
        n_dates = len(dates)

        if n_dates < cfg.min_train_dates + cfg.n_splits:
            raise ValueError(
                f"panel spans {n_dates} dates; need at least "
                f"{cfg.min_train_dates + cfg.n_splits} for "
                f"{cfg.n_splits} folds with a {cfg.min_train_dates}-date minimum train"
            )

        # Positional index of each row's date, so purging is a vector op rather
        # than a per-row date comparison on a million rows.
        date_pos = pd.Series(
            pd.Index(dates).get_indexer(panel[date_column]), index=panel.index
        ).to_numpy()

        testable = n_dates - cfg.min_train_dates
        fold_size = testable // cfg.n_splits
        if fold_size < 1:
            raise ValueError("not enough dates for the requested number of folds")

        for fold in range(cfg.n_splits):
            test_start = cfg.min_train_dates + fold * fold_size
            test_end = (
                n_dates - 1
                if fold == cfg.n_splits - 1
                else test_start + fold_size - 1
            )
            if test_start > test_end:
                continue

            test_mask = (date_pos >= test_start) & (date_pos <= test_end)

            # Purge: training samples within purge_days BEFORE the test window
            # have labels that resolve inside it.
            purge_start = max(0, test_start - cfg.purge_days)
            purged_mask = (date_pos >= purge_start) & (date_pos < test_start)

            # Embargo: training samples within embargo_days AFTER the test
            # window carry features computed partly from it.
            embargo_end = min(n_dates - 1, test_end + cfg.embargo_days)
            embargo_mask = (date_pos > test_end) & (date_pos <= embargo_end)

            if cfg.mode is CVMode.EXPANDING:
                # Walk-forward: the future does not exist yet. The embargo is a
                # no-op here by construction — training is already restricted to
                # dates before the test window, so there is nothing after it to
                # exclude. Reported as 0 rather than as the band's size, because
                # a non-zero count would imply rows were dropped that never
                # would have been included.
                train_mask = (date_pos < test_start) & ~purged_mask
                n_embargoed = 0
            else:
                train_mask = ~test_mask & ~purged_mask & ~embargo_mask
                n_embargoed = int(embargo_mask.sum())

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size == 0 or test_idx.size == 0:
                continue

            yield PanelSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_dates=(dates[date_pos[train_idx].min()],
                             dates[date_pos[train_idx].max()]),
                test_dates=(dates[test_start], dates[test_end]),
                n_purged=int(purged_mask.sum()),
                n_embargoed=n_embargoed,
            )

    def n_folds(self, panel: pd.DataFrame, **kwargs: object) -> int:
        return sum(1 for _ in self.split(panel, **kwargs))  # type: ignore[arg-type]


def assert_no_date_overlap(splits: Sequence[PanelSplit], panel: pd.DataFrame) -> None:
    """Fail loudly if any date appears in both train and test of one fold.

    Called by tests, and cheap enough to call before a real training run. The
    bug it catches does not raise on its own — it just makes the results better
    than they should be.
    """
    for split in splits:
        train_dates = set(panel.iloc[split.train_idx]["date"])
        test_dates = set(panel.iloc[split.test_idx]["date"])
        overlap = train_dates & test_dates
        if overlap:
            raise AssertionError(
                f"fold {split.fold}: {len(overlap)} dates in both train and test "
                f"(e.g. {sorted(overlap)[:3]}). Splitting by row instead of by "
                f"date leaks the market's state across the cross-section."
            )
