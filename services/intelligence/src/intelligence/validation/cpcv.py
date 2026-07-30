"""Combinatorial Purged Cross-Validation (López de Prado, AFML ch. 12).

Walk-forward gives you **one** Sharpe number per configuration. That is not
enough to survive scrutiny, because the obvious question is "how stable is
this?" and a single path cannot answer it.

CPCV splits the timeline into N groups and tests on every combination of k of
them, purging and embargoing around each test group. With N=6, k=2 that is 15
train/test combinations, which recombine into multiple complete backtest paths.
The output is a *distribution* of Sharpe ratios.

A strategy with mean Sharpe 0.4 and sd 0.1 and one with mean 0.4 and sd 0.8 are
completely different objects. Single-path walk-forward reports both as "0.4".
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from intelligence.validation.panel_cv import PanelSplit


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    n_test_groups: int = 2
    purge_days: int = 126
    embargo_days: int = 21
    min_train_rows: int = 500

    @property
    def n_combinations(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Complete backtest paths recoverable from the combinations.

        Each group appears in `C(N-1, k-1)` test sets, and one path needs each
        group tested exactly once.
        """
        from math import comb

        return comb(self.n_groups - 1, self.n_test_groups - 1)


class CombinatorialPurgedCV:
    """Generates every train/test combination with purge and embargo."""

    def __init__(self, config: CPCVConfig | None = None) -> None:
        self.config = config or CPCVConfig()

    def split(
        self, panel: pd.DataFrame, *, date_column: str = "date"
    ) -> Iterator[PanelSplit]:
        cfg = self.config
        if panel.empty:
            raise ValueError("cannot split an empty panel")

        dates = np.array(sorted(pd.unique(panel[date_column])))
        n_dates = len(dates)
        if n_dates < cfg.n_groups * 2:
            raise ValueError(
                f"panel spans {n_dates} dates; need at least {cfg.n_groups * 2} "
                f"for {cfg.n_groups} groups"
            )

        date_pos = pd.Index(dates).get_indexer(panel[date_column])
        bounds = np.linspace(0, n_dates, cfg.n_groups + 1).astype(int)
        groups = [(bounds[i], bounds[i + 1] - 1) for i in range(cfg.n_groups)]

        for fold, test_groups in enumerate(
            combinations(range(cfg.n_groups), cfg.n_test_groups)
        ):
            test_mask = np.zeros(len(panel), dtype=bool)
            excluded = np.zeros(len(panel), dtype=bool)

            for g in test_groups:
                lo, hi = groups[g]
                test_mask |= (date_pos >= lo) & (date_pos <= hi)
                # Purge before and embargo after EACH test group. With
                # non-contiguous test groups there are several boundaries, and
                # missing any one of them leaks.
                excluded |= (date_pos >= max(0, lo - cfg.purge_days)) & (date_pos < lo)
                excluded |= (date_pos > hi) & (
                    date_pos <= min(n_dates - 1, hi + cfg.embargo_days)
                )

            train_mask = ~test_mask & ~excluded
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            if train_idx.size < cfg.min_train_rows or test_idx.size == 0:
                continue

            yield PanelSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_dates=(dates[date_pos[train_idx].min()],
                             dates[date_pos[train_idx].max()]),
                test_dates=(dates[date_pos[test_idx].min()],
                            dates[date_pos[test_idx].max()]),
                n_purged=int(excluded.sum()),
            )


@dataclass
class SharpeDistribution:
    """The output CPCV exists to produce."""

    sharpes: np.ndarray
    n_combinations: int

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.sharpes)) if self.sharpes.size else float("nan")

    @property
    def std(self) -> float:
        return float(np.nanstd(self.sharpes, ddof=1)) if self.sharpes.size > 1 else float("nan")

    @property
    def median(self) -> float:
        return float(np.nanmedian(self.sharpes)) if self.sharpes.size else float("nan")

    def percentile(self, q: float) -> float:
        return float(np.nanpercentile(self.sharpes, q)) if self.sharpes.size else float("nan")

    @property
    def prob_negative(self) -> float:
        """Share of paths that lost money.

        Arguably the most useful single number here: a mean Sharpe of 0.4 with
        40% of paths negative is a very different proposition from the same mean
        with 5% negative.
        """
        valid = self.sharpes[np.isfinite(self.sharpes)]
        return float(np.mean(valid < 0)) if valid.size else float("nan")

    def summary(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "p05": self.percentile(5),
            "p95": self.percentile(95),
            "prob_negative": self.prob_negative,
            "n_paths": float(self.sharpes.size),
        }


def expected_max_sharpe(n_trials: int, variance_of_trial_sharpes: float) -> float:
    """Expected maximum Sharpe from `n_trials` draws of pure noise (AFML 8.1).

    The benchmark a real result must clear. Note it scales with the standard
    deviation of the trial Sharpes — the Euler-Mascheroni term alone is
    dimensionless and is NOT a Sharpe.
    """
    from scipy.stats import norm

    if n_trials < 2 or variance_of_trial_sharpes <= 0:
        return 0.0
    euler = 0.5772156649015329
    z = (1 - euler) * norm.ppf(1 - 1 / n_trials) + euler * norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    return float(np.sqrt(variance_of_trial_sharpes) * z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    variance_of_trial_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the true Sharpe exceeds the best-of-N-noise benchmark
    (Bailey & López de Prado, 2014).

    The answer to "did you just pick the best of 200 runs?" — exactly what v1's
    registry shows: 85 logged runs across 5 tickers, best reported.

    `variance_of_trial_sharpes` is REQUIRED and has no default. An earlier
    version made it optional and fell back to the bare Euler-Mascheroni term,
    which is dimensionless — using it as a Sharpe threshold implicitly assumes
    the trial Sharpes have variance 1.0. That is enormous, so the benchmark came
    out around 1.57 and *every* realistic result was deflated to ~0. A default
    that silently produces "not significant" for everything is worse than no
    function, because it looks rigorous.

    Compute it from the Sharpes you actually observed across configurations:
    `np.var([s for s in trial_sharpes], ddof=1)`.
    """
    from scipy.stats import norm

    if n_observations < 2 or n_trials < 1:
        return float("nan")
    if variance_of_trial_sharpes < 0:
        raise ValueError("variance_of_trial_sharpes must be non-negative")

    threshold = expected_max_sharpe(n_trials, variance_of_trial_sharpes)

    numerator = (observed_sharpe - threshold) * np.sqrt(n_observations - 1)
    denominator = np.sqrt(
        1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(norm.cdf(numerator / denominator))


def deflated_sharpe_from_trials(
    trial_sharpes: Sequence[float],
    *,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR computed directly from every configuration's Sharpe.

    The form to prefer: it takes the search you actually ran rather than asking
    you to supply a variance you would probably guess.
    """
    arr = np.asarray(trial_sharpes, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return deflated_sharpe_ratio(
        float(arr.max()),
        n_trials=arr.size,
        n_observations=n_observations,
        variance_of_trial_sharpes=float(arr.var(ddof=1)),
        skew=skew,
        kurtosis=kurtosis,
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark), correcting for non-normal returns.

    Financial returns are skewed and fat-tailed, which makes the naive
    t-statistic on a Sharpe ratio too generous.
    """
    from scipy.stats import norm

    if n_observations < 2:
        return float("nan")
    numerator = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1)
    denominator = np.sqrt(
        1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(norm.cdf(numerator / denominator))


def min_track_record_length(
    observed_sharpe: float,
    *,
    target_confidence: float = 0.95,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Observations needed before the Sharpe is statistically distinguishable
    from the benchmark.

    Frequently the most sobering number in a report: an observed Sharpe of 0.4
    often needs more history than the strategy has, which means "we do not know
    yet" is the honest conclusion.
    """
    from scipy.stats import norm

    if observed_sharpe <= benchmark_sharpe:
        return float("inf")
    z = norm.ppf(target_confidence)
    numerator = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    return float(1 + numerator * (z / (observed_sharpe - benchmark_sharpe)) ** 2)


# Standard deviations below this are floating-point residue, not dispersion.
# `np.std` of a constant array returns ~1e-18 rather than exactly 0, so an
# `sd == 0` guard does not fire and the Sharpe comes back as ~1e17. A flat
# equity curve — a strategy that never traded, or one position held throughout —
# would report a spectacular Sharpe and nothing would flag it.
_MIN_STD = 1e-12


def sharpe_from_returns(returns: Sequence[float], periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd < _MIN_STD:
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(periods_per_year))


def evaluate_paths(
    path_returns: Sequence[Sequence[float]],
    *,
    periods_per_year: int = 252,
) -> SharpeDistribution:
    sharpes = np.array(
        [sharpe_from_returns(r, periods_per_year) for r in path_returns],
        dtype="float64",
    )
    return SharpeDistribution(sharpes=sharpes, n_combinations=len(path_returns))
