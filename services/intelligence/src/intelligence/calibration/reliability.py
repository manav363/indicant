"""L6 — calibration evidence.

v1 applied Platt scaling and asserted the probabilities were calibrated. This
module produces the *evidence*: a reliability curve and a Brier score, both
stored in the model card and rendered on the /model page.

That distinction is the project's whole pitch. "When we say 70%, we are right
about 70% of the time" is a claim. A reliability diagram is a demonstration, and
a reader can check it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from indicant_contracts import CalibrationBin

# Fewer than this in a bin and its observed rate is noise — 3 samples give
# rates of 0, 0.33, 0.67 or 1 regardless of the underlying probability.
MIN_BIN_COUNT = 20


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    min_count: int = MIN_BIN_COUNT,
) -> list[CalibrationBin]:
    """Bin predictions and compare mean predicted against observed frequency.

    Perfect calibration is the diagonal. Bins below `min_count` are dropped
    rather than plotted, because a bin of 3 draws a wildly misleading point on
    a chart people will read as evidence.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if y_true.size == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True so p=1.0 lands in the last bin rather than falling outside.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count < min_count:
            continue
        bins.append(
            CalibrationBin(
                bin_lower=float(edges[b]),
                bin_upper=float(edges[b + 1]),
                mean_predicted=float(y_prob[in_bin].mean()),
                observed_rate=float(y_true[in_bin].mean()),
                count=count,
            )
        )
    return bins


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is the
    score of always predicting 0.5.

    Insensitive to rare-event probabilities, so it is reported alongside the
    reliability curve rather than instead of it.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if not mask.any():
        return float("nan")
    return float(np.mean((y_prob[mask] - y_true[mask]) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier relative to always predicting the base rate.

    > 0 means the model beats a constant forecast. This is the number worth
    quoting: a raw Brier of 0.24 sounds fine until you learn that predicting the
    base rate every time scores 0.2499.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if not mask.any():
        return float("nan")
    y = y_true[mask]
    base_rate = float(y.mean())
    reference = float(np.mean((base_rate - y) ** 2))
    if reference == 0:
        return float("nan")
    return float(1.0 - brier_score(y_true, y_prob) / reference)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Count-weighted mean |predicted - observed| across bins.

    One number for "how far off are the stated probabilities". 0.05 means a
    stated 70% is really about 65% or 75%.
    """
    bins = reliability_curve(y_true, y_prob, n_bins=n_bins, min_count=1)
    if not bins:
        return float("nan")
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return float(
        sum(b.count * abs(b.mean_predicted - b.observed_rate) for b in bins) / total
    )


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, object]:
    """Everything the /model page needs to show what the model gets wrong."""
    bins = reliability_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "bins": bins,
        "brier_score": brier_score(y_true, y_prob),
        "brier_skill_score": brier_skill_score(y_true, y_prob),
        "expected_calibration_error": expected_calibration_error(
            y_true, y_prob, n_bins=n_bins
        ),
        "base_rate": float(np.nanmean(y_true)) if y_true.size else float("nan"),
        "n_samples": int(np.isfinite(y_prob).sum()),
    }


def bins_to_frame(bins: list[CalibrationBin]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bin_lower": b.bin_lower,
                "bin_upper": b.bin_upper,
                "mean_predicted": b.mean_predicted,
                "observed_rate": b.observed_rate,
                "count": b.count,
                "gap": b.observed_rate - b.mean_predicted,
            }
            for b in bins
        ]
    )
