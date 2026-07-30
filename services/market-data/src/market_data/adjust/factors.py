"""Corporate-action back-adjustment.

A 1:2 split halves the price. Left unadjusted, the price series shows a -50%
return on the ex-date that never happened, and every momentum, volatility and
drawdown feature computed from it is wrong. Over a 20-year backfill there are
thousands of these.

The adjustment is *backward*: prices before an ex-date are multiplied by the
cumulative ratio of all actions on or after that date, so the most recent price
is unchanged and history is restated to be comparable with it. Choosing backward
over forward matters — forward adjustment changes today's price, which means the
number on the screen stops matching the exchange.

The correctness check is Tier 4: after adjustment, `prev_close` on day *t* must
equal `close` on day *t-1*. An unexplained break there means the factors are
wrong, and it is the single loudest signal this module has.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
from indicant_contracts import CorporateAction

from market_data._dates import as_date

# Adjustment factors below this are almost certainly a data error rather than a
# real action — a 1:1000 split does not happen. Guards against a malformed
# ratio silently annihilating a price series.
MIN_PLAUSIBLE_RATIO = 1e-4
MAX_PLAUSIBLE_RATIO = 1e4


class AdjustmentError(ValueError):
    pass


def cumulative_factors(
    actions: Sequence[CorporateAction],
    *,
    symbol: str,
    dates: Sequence[date],
) -> pd.Series:
    """Cumulative back-adjustment factor for each date.

    For date *d*, the factor is the product of every price-affecting action
    with ``ex_date > d``. Strictly greater-than: on the ex-date itself the
    exchange already reports the post-split price, so that day needs no
    adjustment.
    """
    relevant = sorted(
        (
            a
            for a in actions
            if a.symbol.upper() == symbol.upper() and a.affects_price
        ),
        key=lambda a: a.ex_date,
    )
    for action in relevant:
        if not MIN_PLAUSIBLE_RATIO <= action.ratio <= MAX_PLAUSIBLE_RATIO:
            raise AdjustmentError(
                f"{symbol}: implausible ratio {action.ratio} for "
                f"{action.action_type.value} on {action.ex_date}"
            )

    ordered = sorted(dates)
    factors: list[float] = []
    for d in ordered:
        factor = 1.0
        for action in relevant:
            if action.ex_date > d:
                factor *= action.ratio
        factors.append(factor)

    return pd.Series(factors, index=pd.Index(ordered, name="date"), name="adj_factor")


def adjust_symbol(
    prices: pd.DataFrame,
    actions: Sequence[CorporateAction],
    *,
    symbol: str,
) -> pd.DataFrame:
    """Back-adjust one symbol's OHLC and volume.

    Volume is adjusted *inversely* — a 1:2 split halves the price and doubles
    the share count, so an unadjusted volume series shows a spurious doubling.
    Turnover is left alone: rupees traded is unaffected by a split, and
    adjusting it would break the Tier-2 turnover reconciliation.
    """
    if prices.empty:
        return prices.copy()

    out = prices.sort_values("date").copy()
    factors = cumulative_factors(actions, symbol=symbol, dates=out["date"].tolist())
    out["adj_factor"] = out["date"].map(factors).astype("float64")

    for col in ("open", "high", "low", "close", "prev_close"):
        if col in out.columns:
            out[col] = out[col] * out["adj_factor"]

    if "volume" in out.columns:
        # Inverse: shares outstanding move opposite to price.
        out["volume"] = (out["volume"] / out["adj_factor"]).round().astype("int64")

    out["adj_close"] = out["close"]
    return out


def adjust_all(
    prices: pd.DataFrame,
    actions: Sequence[CorporateAction],
) -> pd.DataFrame:
    """Back-adjust every symbol in a price frame."""
    if prices.empty:
        return prices.copy()

    by_symbol = {a_symbol: [] for a_symbol in {a.symbol.upper() for a in actions}}
    for action in actions:
        by_symbol[action.symbol.upper()].append(action)

    adjusted: list[pd.DataFrame] = []
    for symbol, frame in prices.groupby("symbol"):
        sym = str(symbol).upper()
        adjusted.append(adjust_symbol(frame, by_symbol.get(sym, []), symbol=sym))

    return pd.concat(adjusted, ignore_index=True).sort_values(
        ["symbol", "date"], kind="stable"
    ).reset_index(drop=True)


def continuity_breaks(
    prices: pd.DataFrame,
    *,
    tolerance: float = 0.005,
    actions: Sequence[CorporateAction] = (),
) -> pd.DataFrame:
    """Every place where prev_close does not match the prior close.

    This is the Tier-4 sweep run across the whole lake rather than one day at a
    time. Output is the audit trail for "is the adjustment pipeline correct
    across 20 years", which is the largest correctness risk in the ingest path.

    Note the interaction with quarantine: if day *t* is held by a Tier-2 rule,
    day *t+1* legitimately shows a break, because the prior close it should
    match is not in the lake. That is a true finding, not noise — it is the lake
    telling you it has a hole — so it is reported rather than suppressed. Replay
    the quarantine and the break disappears.
    """
    if prices.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "prev_close", "prior_close", "ratio", "explained"]
        )

    explained_keys = {
        (a.symbol.upper(), a.ex_date): a.ratio for a in actions if a.affects_price
    }

    out = prices.sort_values(["symbol", "date"]).copy()
    out["prior_close"] = out.groupby("symbol")["close"].shift(1)
    checkable = out[out["prev_close"].notna() & out["prior_close"].notna()].copy()
    if checkable.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "prev_close", "prior_close", "ratio", "explained"]
        )

    checkable["ratio"] = checkable["prev_close"] / checkable["prior_close"]
    breaks = checkable[(checkable["ratio"] - 1.0).abs() > tolerance].copy()
    if breaks.empty:
        return breaks[["symbol", "date", "prev_close", "prior_close", "ratio"]].assign(
            explained=pd.Series(dtype="bool")
        )

    def _is_explained(row: pd.Series) -> bool:
        ratio = explained_keys.get((str(row["symbol"]).upper(), as_date(row["date"])))
        if ratio is None:
            return False
        return abs(row["ratio"] - ratio) <= tolerance

    breaks["explained"] = breaks.apply(_is_explained, axis=1)
    return breaks[
        ["symbol", "date", "prev_close", "prior_close", "ratio", "explained"]
    ].reset_index(drop=True)

