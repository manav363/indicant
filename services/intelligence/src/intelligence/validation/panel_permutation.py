"""Permutation test on the panel.

v1's strongest asset, extended from 5 tickers to the whole cross-section.

The question: could this Sharpe have come from noise? Shuffle the labels,
re-run everything, repeat N times, and see where the real number sits in the
resulting null distribution.

v1 ran this on 5 tickers at ~1,200 samples each and found nothing significant
(closest p=0.0995). That was the correct result *and* a test with almost no
power — you cannot resolve a sub-1% effect from 1,200 noisy samples. On a
panel the test finally has something to detect.

Two details carried forward from v1 because they were right:

* **Shuffle per fold, not globally.** Global shuffling can produce folds whose
  train and test share a shuffled-label pattern, which inflates the null toward
  zero and destroys the test's power.
* **+1 correction on both numerator and denominator.** With finite N the
  minimum achievable p is 1/(N+1). Without the correction you report p=0, which
  claims infinite significance from a finite experiment.

One added: labels are shuffled **within a date**, preserving the
cross-sectional structure. Shuffling across dates would destroy the market-wide
co-movement that every date shares, making the null far too easy to beat.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from intelligence.validation.panel_cv import PurgedPanelCV

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanelPermutationConfig:
    n_permutations: int = 200
    random_state: int = 42
    # Preserve the cross-sectional structure by shuffling within each date.
    # Shuffling across dates destroys market-wide co-movement and makes the
    # null distribution far too weak.
    within_date: bool = True
    # Vary the model's seed per permutation so the null is not correlated by
    # every draw sharing one subsampling pattern.
    vary_model_seed: bool = True


@dataclass
class PermutationResult:
    actual_score: float
    null_scores: np.ndarray
    n_permutations: int
    within_date: bool

    @property
    def p_value(self) -> float:
        """(count >= actual + 1) / (N + 1).

        The +1 on both sides is not a rounding convention. Without it a run
        where no null beat the actual reports p=0, which asserts infinite
        significance from a finite experiment.
        """
        valid = self.null_scores[np.isfinite(self.null_scores)]
        if valid.size == 0:
            return float("nan")
        count_ge = int(np.sum(valid >= self.actual_score))
        return (count_ge + 1) / (valid.size + 1)

    @property
    def null_mean(self) -> float:
        valid = self.null_scores[np.isfinite(self.null_scores)]
        return float(valid.mean()) if valid.size else float("nan")

    @property
    def null_std(self) -> float:
        valid = self.null_scores[np.isfinite(self.null_scores)]
        return float(valid.std(ddof=1)) if valid.size > 1 else float("nan")

    @property
    def null_95pct(self) -> float:
        valid = self.null_scores[np.isfinite(self.null_scores)]
        return float(np.percentile(valid, 95)) if valid.size else float("nan")

    @property
    def z_score(self) -> float:
        """How many null standard deviations above the null mean.

        Reported alongside p because it says *how far* rather than only
        *whether*, and a z of 0.2 and a z of 1.9 both fail at alpha=0.05 while
        meaning quite different things about where to look next.
        """
        if not np.isfinite(self.null_std) or self.null_std == 0:
            return float("nan")
        return (self.actual_score - self.null_mean) / self.null_std

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> dict[str, float | bool | int]:
        return {
            "actual_score": self.actual_score,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_95pct": self.null_95pct,
            "z_score": self.z_score,
            "n_permutations": self.n_permutations,
            "significant_at_05": self.is_significant(),
        }

    def verdict(self) -> str:
        """One plain sentence, for the /model page.

        A null result stated clearly is more credible than a hedge.
        """
        if not np.isfinite(self.p_value):
            return "The permutation test could not be evaluated."
        if self.is_significant():
            return (
                f"The result survives label shuffling (p={self.p_value:.4f}): fewer "
                f"than 1 in 20 random relabellings did this well."
            )
        return (
            f"The result does NOT survive label shuffling (p={self.p_value:.4f}). "
            f"Randomly reshuffled labels produced a score this good "
            f"{self.p_value:.1%} of the time, so this edge is not distinguishable "
            f"from chance."
        )


def shuffle_labels(
    y: pd.Series,
    dates: pd.Series,
    rng: np.random.Generator,
    *,
    within_date: bool = True,
) -> pd.Series:
    """Shuffle labels, optionally preserving cross-sectional structure.

    `within_date=True` permutes which *symbol* got which outcome on a given
    day, while leaving that day's overall distribution intact. That is the right
    null for a cross-sectional model: it asks "could this stock-picking be
    random?" rather than "could the market's direction be random?", and only the
    first is a question about the model.
    """
    shuffled = y.copy()
    if not within_date:
        values = y.to_numpy(copy=True)
        rng.shuffle(values)
        return pd.Series(values, index=y.index, name=y.name)

    for _, idx in dates.groupby(dates).groups.items():
        block = y.loc[idx].to_numpy(copy=True)
        rng.shuffle(block)
        shuffled.loc[idx] = block
    return shuffled


def run_panel_permutation(
    panel: pd.DataFrame,
    y: pd.Series,
    score_fn: Callable[[pd.DataFrame, pd.Series, int], float],
    cv: PurgedPanelCV,
    config: PanelPermutationConfig | None = None,
    *,
    progress: Callable[[int, float], None] | None = None,
) -> PermutationResult:
    """Compare the real score against a null built by shuffling labels.

    `score_fn(panel, y, seed) -> float` runs the entire pipeline — including
    cross-validation — so the null distribution reflects every source of
    optimism in the real run, not just the model fit.
    """
    cfg = config or PanelPermutationConfig()
    rng = np.random.default_rng(cfg.random_state)

    actual = score_fn(panel, y, cfg.random_state)

    nulls = np.full(cfg.n_permutations, np.nan)
    dates = panel["date"]

    for i in range(cfg.n_permutations):
        shuffled = shuffle_labels(y, dates, rng, within_date=cfg.within_date)
        seed = int(rng.integers(0, 2**31 - 1)) if cfg.vary_model_seed else cfg.random_state
        try:
            nulls[i] = score_fn(panel, shuffled, seed)
        except Exception as exc:
            # A failed permutation is recorded as NaN and excluded, never
            # silently counted as a loss for the null — that would bias the
            # p-value toward significance.
            logger.warning("permutation %s failed: %s", i, exc)
        if progress is not None:
            progress(i + 1, nulls[i])

    return PermutationResult(
        actual_score=actual,
        null_scores=nulls,
        n_permutations=cfg.n_permutations,
        within_date=cfg.within_date,
    )
