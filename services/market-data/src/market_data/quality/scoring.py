"""Per-symbol quality scoring and universe eligibility.

This is where the quality gate meets the user-facing promise that no ticker
falls back. The mechanism is not retry logic — it is scope:

    eligible_universe = { symbols the system can actually answer for }

and the frontend offers nothing else. A thin-data symbol is not a crash and not
a fallback; it is correctly out of scope, with a reason a person can read.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from indicant_contracts import (
    EligibilityThresholds,
    QualityScore,
    UniverseSnapshot,
)

from market_data._dates import as_date

# A symbol not seen in this many trading days is treated as gone.
RECENCY_WINDOW_DAYS = 20

# History beyond this adds no further confidence, so the completeness component
# saturates rather than rewarding age indefinitely.
HISTORY_SATURATION_DAYS = 1260  # ~5 years

# Turnover at which the liquidity component saturates.
TURNOVER_SATURATION = 5e8  # Rs 50 crore


class QualityScorer:
    """Computes per-symbol scores from the lake's own contents.

    Deliberately derives everything from observed data rather than from a
    configured symbol list — a configured list is how survivorship bias gets
    reintroduced after being removed.
    """

    def __init__(self, thresholds: EligibilityThresholds | None = None) -> None:
        self.thresholds = thresholds or EligibilityThresholds()

    def score_all(
        self,
        prices: pd.DataFrame,
        *,
        as_of: date,
        expected_days: int,
        quarantine_counts: dict[str, int] | None = None,
        continuity_breaks: dict[str, int] | None = None,
    ) -> list[QualityScore]:
        """One score per symbol present in `prices`.

        `expected_days` is how many trading days the exchange had over the
        window — the denominator for completeness. Passing observed days instead
        would score every symbol as perfectly complete, which is the bug this
        parameter exists to prevent.
        """
        if prices.empty:
            return []

        quarantine_counts = quarantine_counts or {}
        continuity_breaks = continuity_breaks or {}

        grouped = prices.groupby("symbol")
        scores: list[QualityScore] = []

        for symbol, frame in grouped:
            sym = str(symbol)
            n_days = len(frame)
            last_seen = as_date(frame["date"].max())

            observed_gap = _trading_days_between(prices, last_seen, as_of)
            n_quarantined = quarantine_counts.get(sym, 0)
            n_breaks = continuity_breaks.get(sym, 0)

            scores.append(
                QualityScore(
                    symbol=sym,
                    as_of=as_of,
                    history_completeness=_saturating(
                        n_days, min(expected_days, HISTORY_SATURATION_DAYS)
                    ),
                    validity_clean_rate=_clean_rate(n_days, n_quarantined),
                    continuity_clean_rate=_clean_rate(n_days, n_breaks),
                    liquidity_adequacy=_saturating(
                        float(frame["turnover"].median()), TURNOVER_SATURATION
                    ),
                    recency=_recency(observed_gap),
                    history_days=n_days,
                    median_turnover=float(frame["turnover"].median()),
                )
            )
        return scores

    def build_universe(
        self,
        scores: list[QualityScore],
        *,
        as_of: date,
        index_name: str | None = None,
    ) -> UniverseSnapshot:
        """Partition scored symbols into eligible and excluded-with-a-reason."""
        symbols: list[str] = []
        eligible: list[str] = []
        excluded: dict[str, str] = {}

        for score in sorted(scores, key=lambda s: s.symbol):
            symbols.append(score.symbol)
            reason = self.thresholds.evaluate(score)
            if reason is None:
                eligible.append(score.symbol)
            else:
                excluded[score.symbol] = reason

        return UniverseSnapshot(
            as_of=as_of,
            index_name=index_name,
            symbols=tuple(symbols),
            eligible_symbols=tuple(eligible),
            excluded=excluded,
        )

    @staticmethod
    def to_frame(scores: list[QualityScore]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": s.symbol,
                    "as_of": s.as_of,
                    "score": s.score,
                    "history_completeness": s.history_completeness,
                    "validity_clean_rate": s.validity_clean_rate,
                    "continuity_clean_rate": s.continuity_clean_rate,
                    "liquidity_adequacy": s.liquidity_adequacy,
                    "recency": s.recency,
                    "history_days": s.history_days,
                    "median_turnover": s.median_turnover,
                }
                for s in scores
            ]
        )


def _saturating(value: float, ceiling: float) -> float:
    """Linear to `ceiling`, then flat. Clamped to [0, 1]."""
    if ceiling <= 0:
        return 0.0
    return float(min(1.0, max(0.0, value / ceiling)))


def _clean_rate(total: int, bad: int) -> float:
    if total <= 0:
        return 0.0
    return float(max(0.0, 1.0 - bad / total))


def _recency(trading_days_since_last_seen: int) -> float:
    """1.0 if seen today, decaying to 0 across the recency window.

    A delisted symbol therefore scores 0 on recency and drops out of the
    eligible universe automatically, while remaining in history — which is
    exactly the behaviour that keeps backtests survivorship-bias-free.
    """
    if trading_days_since_last_seen <= 0:
        return 1.0
    if trading_days_since_last_seen >= RECENCY_WINDOW_DAYS:
        return 0.0
    return float(1.0 - trading_days_since_last_seen / RECENCY_WINDOW_DAYS)


def _trading_days_between(prices: pd.DataFrame, start: date, end: date) -> int:
    """Count observed trading days strictly after `start`, up to `end`.

    Uses the lake's own observed dates rather than a calendar, so a market
    closure never counts against a symbol's recency.
    """
    if start >= end:
        return 0
    all_dates = pd.to_datetime(prices["date"].unique())
    mask = (all_dates > pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    return int(np.count_nonzero(mask))

