"""L1 — triple-barrier labelling (López de Prado).

v1 labelled every observation the same way: `y = 1 if P[t+126] > P[t]`. A fixed
six-month horizon with a zero threshold. Two problems that only became visible
after the permutation test failed:

1. **It assumes a fixed horizon means the same thing for every stock.** Six
   months on a low-volatility IT name and on a high-volatility Adani name are
   not the same bet, but they got the same label rule.
2. **It ignores path.** A stock that fell 40% and recovered to +1% got `y=1`,
   identical to one that rose smoothly. Nobody holds through the first path.

Triple-barrier fixes both by asking *what happened first*:

        ┌──────────────── profit-take   (+k1 * sigma_t)
        │
   P_t ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤ time barrier (t + horizon)
        │
        └──────────────── stop-loss     (-k2 * sigma_t)

Barriers scale with each symbol's own realised volatility, which is what makes
labels comparable across the cross-section — exactly what a pooled panel needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

# Volatility estimation window. Long enough to be stable, short enough to track
# a regime change.
VOL_WINDOW = 63

# Annualisation is deliberately NOT applied: barriers are compared against
# cumulative returns over the horizon, so the volatility must be in the same
# (per-period, un-annualised) units.


class BarrierTouched(IntEnum):
    STOP_LOSS = -1
    TIME = 0
    PROFIT_TAKE = 1


@dataclass(frozen=True)
class TripleBarrierConfig:
    """`pt_multiple` and `sl_multiple` are hyperparameters, and tuning them on
    the same data you evaluate on is a classic overfit — they belong inside the
    CV loop, not chosen once by eyeballing results.

    Symmetric by default. Asymmetric barriers encode a directional view, which
    should be a deliberate, argued choice rather than a default.
    """

    horizon_days: int = 126  # ~6 months
    pt_multiple: float = 2.0
    sl_multiple: float = 2.0
    vol_window: int = VOL_WINDOW
    min_vol: float = 1e-6  # floor, so a flat series cannot produce zero-width barriers
    # When True, a time-barrier exit is labelled by the sign of its return
    # rather than as flat. Useful for a binary side model; the three-class form
    # is more honest about what actually happened.
    binary_time_barrier: bool = False


def realised_volatility(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling std of log returns, past-only.

    This must never look forward. A forward-looking vol estimate leaks into the
    *labels*, which is the worst possible place for it — the model would be
    fitting a target that already knows the answer.

    `min_periods=window` is deliberate and not merely conservative. A partial
    window can produce a std from two observations, which is not a volatility
    estimate; it is a number. Feeding it into the barrier width gives absurdly
    narrow barriers at the start of every series, so the first observations get
    labelled profit-take on noise and the model learns from them first. Better
    to leave the warm-up unlabelled and say so.
    """
    log_ret = np.log(close).diff()
    return log_ret.rolling(window, min_periods=window).std()


def apply_triple_barrier(
    close: pd.Series,
    config: TripleBarrierConfig | None = None,
    *,
    volatility: pd.Series | None = None,
) -> pd.DataFrame:
    """Label one symbol's price series.

    Returns one row per observation with the barrier touched, the realised
    return at exit, and the index of the exit bar. The exit index is what the
    sample-weight machinery needs to compute label overlap.

    The last `horizon_days` observations have no complete future, so their
    labels are NaN — the same trap v1 documented for fixed-horizon labels, and
    it applies identically here.
    """
    cfg = config or TripleBarrierConfig()
    n = len(close)
    prices = close.to_numpy(dtype="float64")

    vol = (
        volatility
        if volatility is not None
        else realised_volatility(close, cfg.vol_window)
    ).to_numpy(dtype="float64")

    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    exit_idx = np.full(n, -1, dtype="int64")
    touched = np.full(n, np.nan)

    for i in range(n):
        # No complete future window: cannot label without looking past the end
        # of the data, which would be inventing the answer.
        if i + cfg.horizon_days >= n:
            break
        sigma = vol[i]
        if not np.isfinite(sigma) or sigma < cfg.min_vol:
            continue

        entry = prices[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        upper = cfg.pt_multiple * sigma
        lower = -cfg.sl_multiple * sigma

        window = prices[i + 1 : i + 1 + cfg.horizon_days]
        cum = np.log(window / entry)

        hit_pt = np.flatnonzero(cum >= upper)
        hit_sl = np.flatnonzero(cum <= lower)

        first_pt = hit_pt[0] if hit_pt.size else np.inf
        first_sl = hit_sl[0] if hit_sl.size else np.inf

        if first_pt == np.inf and first_sl == np.inf:
            j = len(window) - 1
            outcome = BarrierTouched.TIME
        elif first_pt <= first_sl:
            j = int(first_pt)
            outcome = BarrierTouched.PROFIT_TAKE
        else:
            j = int(first_sl)
            outcome = BarrierTouched.STOP_LOSS

        exit_idx[i] = i + 1 + j
        ret[i] = cum[j]
        touched[i] = int(outcome)

        if outcome is BarrierTouched.TIME and cfg.binary_time_barrier:
            label[i] = 1.0 if cum[j] > 0 else -1.0
        else:
            label[i] = float(outcome)

    return pd.DataFrame(
        {
            "label": label,
            "barrier_return": ret,
            "barrier_touched": touched,
            "exit_index": exit_idx,
            "volatility": vol,
        },
        index=close.index,
    )


def label_panel(
    panel: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
    *,
    price_column: str = "close",
) -> pd.DataFrame:
    """Label every symbol in a panel.

    Per-symbol, because the barrier walk is a function of one price series.
    Running it over a stacked frame would let one symbol's prices resolve
    another's barriers.
    """
    cfg = config or TripleBarrierConfig()
    if panel.empty:
        return panel.assign(label=pd.Series(dtype="float64"))

    out: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.sort_values("date").reset_index(drop=True)
        labels = apply_triple_barrier(frame[price_column], cfg)
        merged = pd.concat([frame, labels.reset_index(drop=True)], axis=1)
        merged["symbol"] = symbol
        out.append(merged)

    return pd.concat(out, ignore_index=True).sort_values(
        ["date", "symbol"], kind="stable"
    ).reset_index(drop=True)


def to_binary_side(labels: pd.Series) -> pd.Series:
    """Collapse {-1, 0, 1} to {0, 1} for a binary side model.

    Time-barrier exits become 0 (not up), which is the conservative reading:
    an outcome that never reached the profit target is not a win.
    """
    return (labels > 0).astype("float64").where(labels.notna())


def label_distribution(labels: pd.Series) -> dict[str, float]:
    """Share of each barrier outcome.

    Worth checking per regime: if stop-losses dominate in 2008 and profit-takes
    in 2021, the labels are describing the market rather than a skill, and any
    model trained across both will mostly learn the calendar.
    """
    valid = labels.dropna()
    if valid.empty:
        return {}
    counts = valid.value_counts(normalize=True)
    names = {1.0: "profit_take", 0.0: "time", -1.0: "stop_loss"}
    return {names.get(float(k), str(k)): float(v) for k, v in counts.items()}
