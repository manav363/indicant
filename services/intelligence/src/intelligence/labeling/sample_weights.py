"""Sample weights by average uniqueness (López de Prado, AFML ch. 4).

This is a correctness fix, not an enhancement.

Triple-barrier labels **overlap in time**: the observation at t and the one at
t+1 can both resolve at t+40, so they describe largely the same future. v1
treated every observation as independent. That inflates the effective sample
size, and every confidence interval and p-value computed from it is
correspondingly too narrow — the model looks more certain than the data
supports.

Concretely: 1,000 observations with an average uniqueness of 0.1 carry about as
much independent information as 100. Reporting a p-value as though there were
1,000 is not a rounding error, it is the difference between significant and not.

Two corrections here:

* **Average uniqueness** down-weights redundant observations.
* **Sequential bootstrap** draws samples in proportion to how much *new*
  information they add, so bagged learners stop seeing the same future forty
  times over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(n: int, exit_index: np.ndarray) -> np.ndarray:
    """How many labels are 'live' at each bar.

    A bar where 40 labels are open is a bar whose outcome is being counted 40
    times. Computed with a difference array rather than nested loops: the naive
    version is O(n * horizon), which on a 10^6-row panel is minutes.
    """
    delta = np.zeros(n + 1, dtype="float64")
    for i, end in enumerate(exit_index):
        if end < 0:
            continue
        delta[i] += 1.0
        delta[min(int(end) + 1, n)] -= 1.0
    return np.cumsum(delta)[:n]


def average_uniqueness(exit_index: np.ndarray, n: int | None = None) -> np.ndarray:
    """Mean of 1/concurrency over each label's lifespan.

    1.0 means the label had the bar to itself. 0.05 means twenty labels shared
    every bar of its life, so it carries a twentieth of the information its
    row count implies.
    """
    n = n if n is not None else len(exit_index)
    conc = concurrency(n, exit_index)
    out = np.full(n, np.nan)

    for i, end in enumerate(exit_index):
        if end < 0:
            continue
        span = conc[i : int(end) + 1]
        live = span[span > 0]
        out[i] = float(np.mean(1.0 / live)) if live.size else np.nan
    return out


def return_attribution_weights(
    exit_index: np.ndarray,
    returns: np.ndarray,
    n: int | None = None,
) -> np.ndarray:
    """Weight by |return| attributed across concurrent labels.

    Uniqueness alone says a label is distinct; it does not say it matters. A
    label spanning a flat stretch and one spanning a crash are equally unique
    and very unequally informative. AFML 4.10.
    """
    n = n if n is not None else len(exit_index)
    conc = concurrency(n, exit_index)
    out = np.full(n, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_bar = np.where(conc > 0, returns[:n] / conc, 0.0)

    for i, end in enumerate(exit_index):
        if end < 0:
            continue
        out[i] = float(abs(np.nansum(per_bar[i : int(end) + 1])))

    total = np.nansum(out)
    if total > 0:
        # Normalise to mean 1 so weights are comparable across folds and do not
        # silently rescale a loss function.
        out = out * (np.count_nonzero(~np.isnan(out)) / total)
    return out


def time_decay(uniqueness: np.ndarray, last_weight: float = 1.0) -> np.ndarray:
    """Linearly decay weight with age. `last_weight=1.0` disables decay.

    Off by default. Decay assumes recent data is more relevant, which is a
    market view, not a fact — and it quietly shrinks the effective sample size
    the panel was built to enlarge.
    """
    valid = ~np.isnan(uniqueness)
    if not valid.any() or last_weight >= 1.0:
        return np.where(valid, 1.0, np.nan)

    cumulative = np.nancumsum(np.where(valid, uniqueness, 0.0))
    total = cumulative[-1]
    if total <= 0:
        return np.where(valid, 1.0, np.nan)

    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        # Negative last_weight zeroes out the oldest observations entirely.
        slope = 1.0 / ((last_weight + 1) * total)
    const = 1.0 - slope * total

    weights = const + slope * cumulative
    return np.where(valid, np.clip(weights, 0.0, None), np.nan)


def compute_weights(
    labelled: pd.DataFrame,
    *,
    use_return_attribution: bool = True,
    last_weight: float = 1.0,
) -> pd.Series:
    """Sample weights for a labelled panel, computed per symbol.

    Per-symbol because concurrency is only meaningful within one series — two
    symbols' labels overlapping in calendar time are not redundant, they are two
    genuinely different observations of the same day.
    """
    if labelled.empty:
        return pd.Series(dtype="float64")

    pieces: list[pd.Series] = []
    for _, group in labelled.groupby("symbol", sort=True):
        frame = group.sort_values("date")
        exits = frame["exit_index"].fillna(-1).to_numpy(dtype="int64")
        n = len(frame)
        # exit_index is positional within the symbol's own series.
        exits = np.where(exits >= n, n - 1, exits)

        uniq = average_uniqueness(exits, n)

        if use_return_attribution and "barrier_return" in frame.columns:
            rets = frame["barrier_return"].fillna(0.0).to_numpy(dtype="float64")
            weights = return_attribution_weights(exits, rets, n)
            weights = np.where(np.isnan(weights), uniq, weights)
        else:
            weights = uniq

        if last_weight < 1.0:
            weights = weights * time_decay(uniq, last_weight)

        pieces.append(pd.Series(weights, index=frame.index))

    return pd.concat(pieces).reindex(labelled.index).rename("sample_weight")


def effective_sample_size(weights: pd.Series) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    The number to quote instead of `len(df)` when reporting how much data a
    result rests on. If this is 8,000 on a 100,000-row panel, saying '100,000
    samples' is not a description, it is a claim the data does not support.
    """
    w = weights.dropna().to_numpy(dtype="float64")
    if w.size == 0 or np.sum(w**2) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w**2))


def sequential_bootstrap(
    exit_index: np.ndarray,
    n_samples: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw indices favouring observations that add new information.

    Standard bootstrap on overlapping labels draws the same future repeatedly,
    so a bagged ensemble's members end up correlated and its variance estimate
    is too small. Sequential bootstrap re-weights after every draw by how much
    each candidate still overlaps what has been drawn. AFML 4.5.

    O(n_samples * n). Deliberately not used on the full panel — it is for the
    bagged base learners, where the number of draws is bounded.
    """
    rng = rng or np.random.default_rng()
    n = len(exit_index)
    n_samples = n_samples or n
    if n == 0:
        return np.array([], dtype="int64")

    drawn: list[int] = []
    conc = np.zeros(n, dtype="float64")

    for _ in range(n_samples):
        avg_uniq = np.zeros(n, dtype="float64")
        for i, end in enumerate(exit_index):
            if end < 0:
                continue
            span = slice(i, int(end) + 1)
            # +1 counts this candidate as if it were drawn.
            avg_uniq[i] = float(np.mean(1.0 / (conc[span] + 1.0)))

        total = avg_uniq.sum()
        if total <= 0:
            break
        probabilities = avg_uniq / total
        pick = int(rng.choice(n, p=probabilities))
        drawn.append(pick)

        end = exit_index[pick]
        if end >= 0:
            conc[pick : int(end) + 1] += 1.0

    return np.array(drawn, dtype="int64")
