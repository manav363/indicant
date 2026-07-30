"""The six rule tiers.

Every rule is a pure function of a `RuleContext` returning one `RuleResult`.
Pure because a rule that reaches out to fetch something cannot be tested with a
hand-built failing fixture, and a rule without such a fixture is decoration.

Tier semantics:

* **T1 structural** — fatal, kills the file. The bytes are not a bhavcopy.
* **T2 validity** — fatal per row. `high < low` is not a market condition.
* **T3 completeness** — is anything *missing* that should be here.
* **T4 continuity** — the leak detector. An unexplained `prev_close` break
  means the adjustment pipeline is wrong, and every derived feature inherits
  that. Weighted highest in the quality score for exactly this reason.
* **T5 plausibility** — statistical. Flags, rarely rejects, because a real
  20% circuit move and a decimal-point error look identical in isolation.
* **T6 cross-source** — agreement with an independent feed.

Every failing result carries `evidence` with both sides of the comparison.
The contract enforces this; it is not a convention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from indicant_contracts import (
    CANONICAL_PRICE_COLUMNS,
    CorporateAction,
    RuleResult,
    Severity,
    Tier,
)

from market_data._dates import as_date

# NSE price band for most equities. A move beyond this without a corporate
# action is either a band-free scrip (F&O names, index constituents) or bad
# data — which is why this is a warning that names the symbol, not a rejection.
CIRCUIT_LIMIT_PCT = 20.0

# Sigma threshold for the return outlier rule. 6 sigma on a fat-tailed
# distribution is common enough that this must never be fatal.
RETURN_SIGMA_THRESHOLD = 6.0

# Volume this many times the trailing median suggests a unit error.
VOLUME_SPIKE_MULTIPLE = 50.0

# Consecutive identical closes implying a halt or a dead scrip.
STALENESS_DAYS = 5

# Row count this far from the trailing median means a truncated or padded file.
ROW_COUNT_TOLERANCE_PCT = 25.0

# turnover ~= volume * average price. Loose because average price is not in
# every era's file, so we bound it by the day's own high/low instead.
TURNOVER_TOLERANCE = 0.05

# prev_close mismatch beyond this fraction needs a corporate action to explain it.
CONTINUITY_TOLERANCE = 0.005

# Cap on rows embedded in evidence payloads. Evidence is for a human reading a
# report, so it is sampled. `affected_symbols` is NEVER sampled: the gate maps
# it back to rows to decide what gets quarantined, so truncating it would let
# bad rows past the gate silently.
MAX_EVIDENCE_ROWS = 20


@dataclass
class RuleContext:
    """Everything the rules may look at. Nothing else is reachable.

    `history` is prior-day data for continuity and statistical rules;
    `corporate_actions` explains legitimate price breaks; `expected_trading_day`
    comes from the calendar. All optional — a first ingest has no history, and
    rules degrade to passing-with-a-note rather than failing on absence.
    """

    df: pd.DataFrame
    trade_date: date
    expected_trading_day: bool | None = None
    previous_close: pd.DataFrame | None = None
    previous_symbols: frozenset[str] = frozenset()
    trailing_row_counts: Sequence[int] = ()
    history: pd.DataFrame | None = None
    corporate_actions: Sequence[CorporateAction] = ()
    cross_source: pd.DataFrame | None = None
    expected_symbols: frozenset[str] = frozenset()
    _ca_index: dict[str, CorporateAction] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._ca_index = {
            ca.symbol.upper(): ca
            for ca in self.corporate_actions
            if ca.ex_date == self.trade_date and ca.affects_price
        }

    def action_for(self, symbol: str) -> CorporateAction | None:
        return self._ca_index.get(symbol.upper())


Rule = Callable[[RuleContext], RuleResult]


def _ok(rule_id: str, tier: Tier, severity: Severity, message: str) -> RuleResult:
    return RuleResult(
        rule_id=rule_id, tier=tier, severity=severity, passed=True, message=message
    )


def _all_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    """Complete, deduplicated, sorted. Never truncated — the gate quarantines
    by this list, so dropping entries would admit bad rows.
    """
    return tuple(sorted({str(s).upper() for s in symbols}))


# ==========================================================================
# TIER 1 — STRUCTURAL. Fatal. The file is not usable at all.
# ==========================================================================


def t1_schema(ctx: RuleContext) -> RuleResult:
    rule_id = "T1.structural.schema"
    missing = [c for c in CANONICAL_PRICE_COLUMNS if c not in ctx.df.columns]
    if missing:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.STRUCTURAL,
            severity=Severity.FATAL,
            passed=False,
            message=f"canonical columns missing: {missing}",
            evidence={"missing": missing, "found": [str(c) for c in ctx.df.columns]},
        )
    return _ok(rule_id, Tier.STRUCTURAL, Severity.FATAL, "all canonical columns present")


def t1_non_empty(ctx: RuleContext) -> RuleResult:
    rule_id = "T1.structural.non_empty"
    if ctx.df.empty:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.STRUCTURAL,
            severity=Severity.FATAL,
            passed=False,
            message="file parsed to zero rows",
            evidence={"rows": 0, "trade_date": ctx.trade_date.isoformat()},
        )
    return _ok(rule_id, Tier.STRUCTURAL, Severity.FATAL, f"{len(ctx.df)} rows")


def t1_date_matches(ctx: RuleContext) -> RuleResult:
    """The date inside the file must be the date we asked for.

    Catches the failure mode where the archive serves a redirect or a stale
    file — you get 200 OK and plausible data for the wrong day, which is worse
    than a 404 because nothing downstream notices.
    """
    rule_id = "T1.structural.date_match"
    if "date" not in ctx.df.columns or ctx.df.empty:
        return _ok(rule_id, Tier.STRUCTURAL, Severity.FATAL, "no rows to check")
    found = {as_date(v) for v in ctx.df["date"].unique()}
    if found != {ctx.trade_date}:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.STRUCTURAL,
            severity=Severity.FATAL,
            passed=False,
            message=f"file contains dates {sorted(d.isoformat() for d in found)}",
            evidence={
                "requested": ctx.trade_date.isoformat(),
                "found": sorted(d.isoformat() for d in found),
            },
        )
    return _ok(rule_id, Tier.STRUCTURAL, Severity.FATAL, "date matches request")


# ==========================================================================
# TIER 2 — VALIDITY. Fatal per row. Quarantine the row, keep the file.
# ==========================================================================


def t2_high_low(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.high_low"
    bad = ctx.df[ctx.df["high"] < ctx.df["low"]]
    return _row_rule(
        rule_id,
        bad,
        message="high < low",
        cols=("symbol", "high", "low"),
    )


def t2_high_bounds(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.high_bounds"
    bad = ctx.df[ctx.df["high"] < ctx.df[["open", "close"]].max(axis=1)]
    return _row_rule(
        rule_id,
        bad,
        message="high below max(open, close)",
        cols=("symbol", "high", "open", "close"),
    )


def t2_low_bounds(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.low_bounds"
    bad = ctx.df[ctx.df["low"] > ctx.df[["open", "close"]].min(axis=1)]
    return _row_rule(
        rule_id,
        bad,
        message="low above min(open, close)",
        cols=("symbol", "low", "open", "close"),
    )


def t2_positive_prices(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.positive_prices"
    price_cols = ["open", "high", "low", "close"]
    bad = ctx.df[(ctx.df[price_cols] <= 0).any(axis=1) | ctx.df[price_cols].isna().any(axis=1)]
    return _row_rule(
        rule_id,
        bad,
        message="non-positive or missing price",
        cols=("symbol", *price_cols),
    )


def t2_non_negative_volume(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.non_negative_volume"
    bad = ctx.df[(ctx.df["volume"] < 0) | (ctx.df["turnover"] < 0)]
    return _row_rule(
        rule_id,
        bad,
        message="negative volume or turnover",
        cols=("symbol", "volume", "turnover"),
    )


def t2_delivery_bounds(ctx: RuleContext) -> RuleResult:
    rule_id = "T2.validity.delivery_bounds"
    has_delivery = ctx.df["delivery_qty"].notna()
    bad = ctx.df[has_delivery & (ctx.df["delivery_qty"] > ctx.df["volume"])]
    return _row_rule(
        rule_id,
        bad,
        message="delivery quantity exceeds traded volume",
        cols=("symbol", "delivery_qty", "volume"),
    )


def t2_turnover_reconciles(ctx: RuleContext) -> RuleResult:
    """turnover must be consistent with volume x a price inside the day's range.

    This is the rule that catches a missed lakhs-to-rupees conversion — a
    silent 100,000x error that otherwise propagates into every liquidity
    feature and every eligibility decision. Severity is ERROR rather than FATAL
    because a genuine wide-spread day can sit slightly outside the band.
    """
    rule_id = "T2.validity.turnover_reconciles"
    df = ctx.df
    traded = df[(df["volume"] > 0) & (df["turnover"] > 0)].copy()
    if traded.empty:
        return _ok(rule_id, Tier.VALIDITY, Severity.ERROR, "no traded rows to reconcile")

    implied = traded["turnover"] / traded["volume"]
    lower = traded["low"] * (1 - TURNOVER_TOLERANCE)
    upper = traded["high"] * (1 + TURNOVER_TOLERANCE)
    bad = traded[(implied < lower) | (implied > upper)].copy()

    if bad.empty:
        return _ok(
            rule_id, Tier.VALIDITY, Severity.ERROR, f"{len(traded)} rows reconcile"
        )

    bad["implied_price"] = bad["turnover"] / bad["volume"]
    bad["ratio_to_close"] = bad["implied_price"] / bad["close"]
    return RuleResult(
        rule_id=rule_id,
        tier=Tier.VALIDITY,
        severity=Severity.ERROR,
        passed=False,
        message=f"{len(bad)} rows where turnover/volume falls outside the day's range",
        affected_symbols=_all_symbols(bad["symbol"].tolist()),
        affected_rows=len(bad),
        evidence={
            "rows": len(bad),
            "median_ratio_to_close": float(bad["ratio_to_close"].median()),
            "hint": (
                "a median ratio near 1e5 or 1e-5 means a lakhs/rupees unit "
                "conversion was missed"
            ),
            "sample": bad.head(5)[
                ["symbol", "volume", "turnover", "implied_price", "close"]
            ].to_dict("records"),
        },
    )


# ==========================================================================
# TIER 3 — COMPLETENESS. Is anything missing that should be here.
# ==========================================================================


def t3_is_trading_day(ctx: RuleContext) -> RuleResult:
    """A file for a non-trading day means we fetched the wrong thing."""
    rule_id = "T3.completeness.trading_day"
    if ctx.expected_trading_day is None:
        return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "no calendar available")
    if not ctx.expected_trading_day and not ctx.df.empty:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.COMPLETENESS,
            severity=Severity.WARNING,
            passed=False,
            message=f"{ctx.trade_date} was predicted closed but a file exists",
            evidence={
                "trade_date": ctx.trade_date.isoformat(),
                "rows": len(ctx.df),
                "hint": "the holiday list for this year is wrong; curate it",
            },
        )
    return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "consistent with calendar")


def t3_row_count(ctx: RuleContext) -> RuleResult:
    rule_id = "T3.completeness.row_count"
    if not ctx.trailing_row_counts:
        return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "no trailing baseline")
    median = float(np.median(list(ctx.trailing_row_counts)))
    if median <= 0:
        return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "empty baseline")
    deviation = abs(len(ctx.df) - median) / median * 100
    if deviation > ROW_COUNT_TOLERANCE_PCT:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.COMPLETENESS,
            severity=Severity.ERROR,
            passed=False,
            message=f"row count {len(ctx.df)} deviates {deviation:.1f}% from median {median:.0f}",
            evidence={
                "rows": len(ctx.df),
                "trailing_median": median,
                "deviation_pct": round(deviation, 2),
                "tolerance_pct": ROW_COUNT_TOLERANCE_PCT,
            },
        )
    return _ok(
        rule_id,
        Tier.COMPLETENESS,
        Severity.WARNING,
        f"row count within {ROW_COUNT_TOLERANCE_PCT}% of median",
    )


def t3_missing_symbols(ctx: RuleContext) -> RuleResult:
    """Symbols present yesterday and absent today.

    Delisting and suspension are legitimate, so this cross-checks corporate
    actions before flagging. Without that check the rule fires on every real
    delisting and gets muted, which is how a genuine ingestion gap gets missed.
    """
    rule_id = "T3.completeness.missing_symbols"
    if not ctx.previous_symbols:
        return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "no prior day")

    today = frozenset(ctx.df["symbol"].astype(str).str.upper())
    vanished = ctx.previous_symbols - today
    explained = {s for s in vanished if ctx.action_for(s) is not None}
    unexplained = sorted(vanished - explained)

    if unexplained:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.COMPLETENESS,
            severity=Severity.WARNING,
            passed=False,
            message=f"{len(unexplained)} symbols present yesterday are missing today",
            affected_symbols=_all_symbols(unexplained),
            affected_rows=len(unexplained),
            evidence={
                "missing_count": len(unexplained),
                "explained_by_corporate_action": sorted(explained),
                "sample": unexplained[:MAX_EVIDENCE_ROWS],
            },
        )
    return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "no unexplained disappearances")


def t3_expected_constituents(ctx: RuleContext) -> RuleResult:
    rule_id = "T3.completeness.expected_constituents"
    if not ctx.expected_symbols:
        return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "no expected set supplied")
    today = frozenset(ctx.df["symbol"].astype(str).str.upper())
    missing = sorted(ctx.expected_symbols - today)
    if missing:
        return RuleResult(
            rule_id=rule_id,
            tier=Tier.COMPLETENESS,
            severity=Severity.ERROR,
            passed=False,
            message=f"{len(missing)} expected index constituents absent",
            affected_symbols=_all_symbols(missing),
            affected_rows=len(missing),
            evidence={"missing": missing[:MAX_EVIDENCE_ROWS], "count": len(missing)},
        )
    return _ok(rule_id, Tier.COMPLETENESS, Severity.WARNING, "all constituents present")


# ==========================================================================
# TIER 4 — CONTINUITY. The leak detector.
# ==========================================================================


def t4_prev_close_continuity(ctx: RuleContext) -> RuleResult:
    """Today's prev_close must equal yesterday's close, or a corporate action
    must explain the gap.

    An unexplained break means the adjustment pipeline is wrong. Every feature
    computed from those prices is then wrong in a way that no test on the
    feature code would ever reveal, which is why this is ERROR and loud rather
    than a quiet note.
    """
    rule_id = "T4.continuity.prev_close"
    if ctx.previous_close is None or ctx.previous_close.empty:
        return _ok(rule_id, Tier.CONTINUITY, Severity.ERROR, "no prior close available")

    prior = ctx.previous_close[["symbol", "close"]].rename(columns={"close": "yesterday_close"})
    merged = ctx.df[["symbol", "prev_close", "close"]].merge(prior, on="symbol", how="inner")
    merged = merged[merged["prev_close"].notna() & merged["yesterday_close"].notna()]
    if merged.empty:
        return _ok(rule_id, Tier.CONTINUITY, Severity.ERROR, "no overlapping symbols")

    merged["gap"] = (
        (merged["prev_close"] - merged["yesterday_close"]).abs() / merged["yesterday_close"]
    )
    breaks = merged[merged["gap"] > CONTINUITY_TOLERANCE].copy()

    if breaks.empty:
        return _ok(
            rule_id,
            Tier.CONTINUITY,
            Severity.ERROR,
            f"{len(merged)} symbols continuous",
        )

    breaks["expected_ratio"] = breaks["prev_close"] / breaks["yesterday_close"]
    breaks["action"] = [ctx.action_for(s) for s in breaks["symbol"]]

    def _explained(row: pd.Series) -> bool:
        action = row["action"]
        if action is None:
            return False
        return abs(row["expected_ratio"] - action.ratio) <= CONTINUITY_TOLERANCE

    breaks["is_explained"] = breaks.apply(_explained, axis=1)
    unexplained = breaks[~breaks["is_explained"]]

    if unexplained.empty:
        return _ok(
            rule_id,
            Tier.CONTINUITY,
            Severity.ERROR,
            f"{len(breaks)} breaks, all explained by corporate actions",
        )

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.CONTINUITY,
        severity=Severity.ERROR,
        passed=False,
        message=(
            f"{len(unexplained)} unexplained prev_close breaks — the adjustment "
            f"pipeline is wrong for these symbols"
        ),
        affected_symbols=_all_symbols(unexplained["symbol"].tolist()),
        affected_rows=len(unexplained),
        evidence={
            "unexplained_count": len(unexplained),
            "explained_count": int(breaks["is_explained"].sum()),
            "tolerance": CONTINUITY_TOLERANCE,
            "sample": unexplained.head(5)[
                ["symbol", "prev_close", "yesterday_close", "expected_ratio"]
            ].to_dict("records"),
        },
    )


# ==========================================================================
# TIER 5 — PLAUSIBILITY. Statistical. Flags, does not reject.
# ==========================================================================


def t5_circuit_breach(ctx: RuleContext) -> RuleResult:
    rule_id = "T5.plausibility.circuit_breach"
    df = ctx.df
    subset = df[df["prev_close"].notna() & (df["prev_close"] > 0)].copy()
    if subset.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no prev_close to compare")

    subset["move_pct"] = (subset["close"] - subset["prev_close"]) / subset["prev_close"] * 100
    breaches = subset[subset["move_pct"].abs() > CIRCUIT_LIMIT_PCT].copy()
    breaches = breaches[[ctx.action_for(s) is None for s in breaches["symbol"]]]

    if breaches.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no unexplained circuit breaches")

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.PLAUSIBILITY,
        severity=Severity.WARNING,
        passed=False,
        message=f"{len(breaches)} moves beyond +/-{CIRCUIT_LIMIT_PCT}% with no corporate action",
        affected_symbols=_all_symbols(breaches["symbol"].tolist()),
        affected_rows=len(breaches),
        evidence={
            "count": len(breaches),
            "limit_pct": CIRCUIT_LIMIT_PCT,
            "sample": breaches.head(5)[["symbol", "prev_close", "close", "move_pct"]].to_dict(
                "records"
            ),
        },
    )


def t5_return_outlier(ctx: RuleContext) -> RuleResult:
    rule_id = "T5.plausibility.return_sigma"
    if ctx.history is None or ctx.history.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no history for sigma")

    hist = ctx.history.sort_values(["symbol", "date"])
    stats = (
        hist.assign(log_ret=np.log(hist["close"]).groupby(hist["symbol"]).diff())
        .groupby("symbol")["log_ret"]
        .std()
        .rename("sigma")
        .reset_index()
    )

    today = ctx.df[ctx.df["prev_close"].notna() & (ctx.df["prev_close"] > 0)].copy()
    if today.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no comparable rows")

    today["log_ret"] = np.log(today["close"] / today["prev_close"])
    merged = today.merge(stats, on="symbol", how="inner")
    merged = merged[merged["sigma"].notna() & (merged["sigma"] > 0)]
    if merged.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no usable sigma")

    merged["z"] = merged["log_ret"] / merged["sigma"]
    outliers = merged[merged["z"].abs() > RETURN_SIGMA_THRESHOLD].copy()
    outliers = outliers[[ctx.action_for(s) is None for s in outliers["symbol"]]]

    if outliers.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no sigma outliers")

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.PLAUSIBILITY,
        severity=Severity.WARNING,
        passed=False,
        message=f"{len(outliers)} returns beyond {RETURN_SIGMA_THRESHOLD} sigma",
        affected_symbols=_all_symbols(outliers["symbol"].tolist()),
        affected_rows=len(outliers),
        evidence={
            "count": len(outliers),
            "threshold_sigma": RETURN_SIGMA_THRESHOLD,
            "sample": outliers.head(5)[["symbol", "log_ret", "sigma", "z"]].to_dict("records"),
        },
    )


def t5_staleness(ctx: RuleContext) -> RuleResult:
    rule_id = "T5.plausibility.staleness"
    if ctx.history is None or ctx.history.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.INFO, "no history for staleness")

    recent = ctx.history.sort_values("date").groupby("symbol").tail(STALENESS_DAYS)
    counts = recent.groupby("symbol")["close"].agg(["nunique", "count"])
    stale = counts[(counts["count"] >= STALENESS_DAYS) & (counts["nunique"] == 1)]

    if stale.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.INFO, "no stale series")

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.PLAUSIBILITY,
        severity=Severity.INFO,
        passed=False,
        message=f"{len(stale)} symbols unchanged for {STALENESS_DAYS} sessions (halt or illiquid)",
        affected_symbols=_all_symbols(stale.index.tolist()),
        affected_rows=len(stale),
        evidence={"count": len(stale), "window_days": STALENESS_DAYS},
    )


def t5_zero_volume_with_move(ctx: RuleContext) -> RuleResult:
    rule_id = "T5.plausibility.zero_volume_move"
    df = ctx.df
    bad = df[
        (df["volume"] == 0)
        & df["prev_close"].notna()
        & (df["close"] != df["prev_close"])
    ]
    if bad.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no zero-volume price moves")
    return RuleResult(
        rule_id=rule_id,
        tier=Tier.PLAUSIBILITY,
        severity=Severity.WARNING,
        passed=False,
        message=f"{len(bad)} rows moved price on zero volume",
        affected_symbols=_all_symbols(bad["symbol"].tolist()),
        affected_rows=len(bad),
        evidence={
            "count": len(bad),
            "sample": bad.head(5)[["symbol", "prev_close", "close", "volume"]].to_dict(
                "records"
            ),
        },
    )


def t5_volume_spike(ctx: RuleContext) -> RuleResult:
    rule_id = "T5.plausibility.volume_spike"
    if ctx.history is None or ctx.history.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no history for volume median")

    median = ctx.history.groupby("symbol")["volume"].median().rename("median_volume")
    merged = ctx.df.merge(median, on="symbol", how="inner")
    merged = merged[merged["median_volume"] > 0]
    if merged.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no usable volume median")

    merged["multiple"] = merged["volume"] / merged["median_volume"]
    spikes = merged[merged["multiple"] > VOLUME_SPIKE_MULTIPLE]
    if spikes.empty:
        return _ok(rule_id, Tier.PLAUSIBILITY, Severity.WARNING, "no volume spikes")

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.PLAUSIBILITY,
        severity=Severity.WARNING,
        passed=False,
        message=f"{len(spikes)} volumes above {VOLUME_SPIKE_MULTIPLE}x trailing median",
        affected_symbols=_all_symbols(spikes["symbol"].tolist()),
        affected_rows=len(spikes),
        evidence={
            "count": len(spikes),
            "multiple_threshold": VOLUME_SPIKE_MULTIPLE,
            "sample": spikes.head(5)[["symbol", "volume", "median_volume", "multiple"]].to_dict(
                "records"
            ),
        },
    )


# ==========================================================================
# TIER 6 — CROSS-SOURCE.
# ==========================================================================


def t6_cross_source_close(ctx: RuleContext) -> RuleResult:
    rule_id = "T6.cross_source.close_agreement"
    if ctx.cross_source is None or ctx.cross_source.empty:
        return _ok(rule_id, Tier.CROSS_SOURCE, Severity.INFO, "no cross-source feed")

    other = ctx.cross_source[["symbol", "close"]].rename(columns={"close": "other_close"})
    merged = ctx.df[["symbol", "close"]].merge(other, on="symbol", how="inner")
    if merged.empty:
        return _ok(rule_id, Tier.CROSS_SOURCE, Severity.INFO, "no overlapping symbols")

    merged["diff_pct"] = (merged["close"] - merged["other_close"]).abs() / merged[
        "other_close"
    ] * 100
    disagree = merged[merged["diff_pct"] > 1.0]
    if disagree.empty:
        return _ok(
            rule_id,
            Tier.CROSS_SOURCE,
            Severity.INFO,
            f"{len(merged)} symbols agree within 1%",
        )

    return RuleResult(
        rule_id=rule_id,
        tier=Tier.CROSS_SOURCE,
        severity=Severity.WARNING,
        passed=False,
        message=f"{len(disagree)} symbols disagree with the cross-source feed by over 1%",
        affected_symbols=_all_symbols(disagree["symbol"].tolist()),
        affected_rows=len(disagree),
        evidence={
            "count": len(disagree),
            "sample": disagree.head(5).to_dict("records"),
        },
    )


# ==========================================================================
# Registry
# ==========================================================================

ALL_RULES: tuple[Rule, ...] = (
    t1_schema,
    t1_non_empty,
    t1_date_matches,
    t2_high_low,
    t2_high_bounds,
    t2_low_bounds,
    t2_positive_prices,
    t2_non_negative_volume,
    t2_delivery_bounds,
    t2_turnover_reconciles,
    t3_is_trading_day,
    t3_row_count,
    t3_missing_symbols,
    t3_expected_constituents,
    t4_prev_close_continuity,
    t5_circuit_breach,
    t5_return_outlier,
    t5_staleness,
    t5_zero_volume_with_move,
    t5_volume_spike,
    t6_cross_source_close,
)

STRUCTURAL_RULES: tuple[Rule, ...] = (t1_schema, t1_non_empty, t1_date_matches)

# Rules whose failures identify specific rows to quarantine, in the order the
# gate applies them.
ROW_LEVEL_RULES: tuple[Rule, ...] = (
    t2_high_low,
    t2_high_bounds,
    t2_low_bounds,
    t2_positive_prices,
    t2_non_negative_volume,
    t2_delivery_bounds,
)


def rule_ids() -> tuple[str, ...]:
    """Every rule id, for the test that asserts each has a failing fixture."""
    return tuple(sorted(r(_EMPTY_CTX).rule_id for r in ALL_RULES))


# Reserved evidence key. Row-level rules record the exact positional indices
# they objected to, and the gate quarantines by those. Matching on symbol alone
# would over-quarantine: one symbol can have both an EQ and a BE row, and only
# one of them may be malformed.
ROW_INDEX_KEY = "row_index"


def _row_rule(
    rule_id: str,
    bad: pd.DataFrame,
    *,
    message: str,
    cols: Sequence[str],
) -> RuleResult:
    if bad.empty:
        return _ok(rule_id, Tier.VALIDITY, Severity.FATAL, f"no rows with {message}")
    present = [c for c in cols if c in bad.columns]
    return RuleResult(
        rule_id=rule_id,
        tier=Tier.VALIDITY,
        severity=Severity.FATAL,
        passed=False,
        message=f"{len(bad)} rows: {message}",
        affected_symbols=_all_symbols(bad["symbol"].tolist()),
        affected_rows=len(bad),
        evidence={
            "count": len(bad),
            "check": message,
            ROW_INDEX_KEY: [int(i) for i in bad.index],
            "sample": bad.head(MAX_EVIDENCE_ROWS)[present].to_dict("records"),
        },
    )



def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_PRICE_COLUMNS})


_EMPTY_CTX = RuleContext(df=_empty_frame(), trade_date=date(2000, 1, 1))
