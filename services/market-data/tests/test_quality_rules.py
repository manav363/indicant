"""One adversarial fixture per rule.

This file is the Phase 3 gate. A quality gate whose rules have never been shown
a failing input is decoration — so for every rule there is a hand-built frame
that must trip it, and the final test asserts no rule escapes that requirement.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from indicant_contracts import CorporateAction, CorporateActionType, Severity, Tier

from conftest import (
    TRADE_DATE,
    canonical_frame,
    canonical_row,
    history_frame,
    prior_close_frame,
)
from market_data.quality import rules as R
from market_data.quality.rules import RuleContext

# Every rule id that must have a failing fixture below. Keyed by the test that
# provides it, so the coverage assertion at the bottom can prove completeness.
COVERED: dict[str, str] = {}


def covers(rule_id: str):
    """Mark a test as the adversarial fixture for a rule."""

    def decorator(fn):
        COVERED[rule_id] = fn.__name__
        return fn

    return decorator


def ctx(df: pd.DataFrame, **kwargs) -> RuleContext:
    return RuleContext(df=df, trade_date=TRADE_DATE, **kwargs)


# ==========================================================================
# TIER 1 — STRUCTURAL
# ==========================================================================


@covers("T1.structural.schema")
def test_missing_canonical_column_is_fatal() -> None:
    broken = canonical_frame().drop(columns=["close"])
    result = R.t1_schema(ctx(broken))
    assert not result.passed
    assert result.blocks_file
    assert result.evidence["missing"] == ["close"]


def test_complete_schema_passes() -> None:
    assert R.t1_schema(ctx(canonical_frame())).passed


@covers("T1.structural.non_empty")
def test_zero_rows_is_fatal() -> None:
    result = R.t1_non_empty(ctx(canonical_frame().iloc[0:0]))
    assert not result.passed
    assert result.blocks_file
    assert result.evidence["rows"] == 0


@covers("T1.structural.date_match")
def test_wrong_date_inside_file_is_fatal() -> None:
    """A 200 OK carrying yesterday's file is worse than a 404 — nothing
    downstream would notice."""
    stale = canonical_frame()
    stale["date"] = date(2015, 3, 3)
    result = R.t1_date_matches(ctx(stale))
    assert not result.passed
    assert result.blocks_file
    assert result.evidence["requested"] == "2015-03-04"
    assert result.evidence["found"] == ["2015-03-03"]


def test_matching_date_passes() -> None:
    assert R.t1_date_matches(ctx(canonical_frame())).passed


# ==========================================================================
# TIER 2 — VALIDITY
# ==========================================================================


@covers("T2.validity.high_low")
def test_high_below_low_is_caught() -> None:
    df = canonical_frame([canonical_row(high=90.0, low=98.0, open=95.0, close=95.0)])
    result = R.t2_high_low(ctx(df))
    assert not result.passed
    assert result.affected_symbols == ("RELIANCE",)
    assert result.evidence[R.ROW_INDEX_KEY] == [0]


@covers("T2.validity.high_bounds")
def test_high_below_close_is_caught() -> None:
    df = canonical_frame([canonical_row(high=101.0, close=102.0)])
    assert not R.t2_high_bounds(ctx(df)).passed


@covers("T2.validity.low_bounds")
def test_low_above_open_is_caught() -> None:
    df = canonical_frame([canonical_row(low=101.0, open=100.0)])
    assert not R.t2_low_bounds(ctx(df)).passed


@covers("T2.validity.positive_prices")
def test_zero_close_is_caught() -> None:
    df = canonical_frame([canonical_row(close=0.0, low=0.0)])
    assert not R.t2_positive_prices(ctx(df)).passed


def test_nan_price_is_caught_by_positive_prices() -> None:
    df = canonical_frame([canonical_row(close=float("nan"))])
    assert not R.t2_positive_prices(ctx(df)).passed


@covers("T2.validity.non_negative_volume")
def test_negative_volume_is_caught() -> None:
    df = canonical_frame([canonical_row(volume=-5)])
    assert not R.t2_non_negative_volume(ctx(df)).passed


@covers("T2.validity.delivery_bounds")
def test_delivery_exceeding_volume_is_caught() -> None:
    df = canonical_frame([canonical_row(volume=1000, delivery_qty=5000)])
    result = R.t2_delivery_bounds(ctx(df))
    assert not result.passed
    assert result.affected_rows == 1


def test_missing_delivery_is_not_a_failure() -> None:
    """Non-EQ series do not report delivery. Absence is not a defect."""
    df = canonical_frame([canonical_row(delivery_qty=None, delivery_pct=None)])
    assert R.t2_delivery_bounds(ctx(df)).passed


@covers("T2.validity.turnover_reconciles")
def test_missed_lakhs_conversion_is_caught() -> None:
    """The 100,000x unit bug. turnover left in lakhs instead of rupees."""
    df = canonical_frame([canonical_row(volume=1_000_000, turnover=1020.0)])
    result = R.t2_turnover_reconciles(ctx(df))
    assert not result.passed
    assert result.evidence["median_ratio_to_close"] < 1e-3
    assert "lakhs" in str(result.evidence["hint"])


def test_correct_turnover_reconciles() -> None:
    assert R.t2_turnover_reconciles(ctx(canonical_frame())).passed


def test_zero_volume_rows_are_skipped_by_reconciliation() -> None:
    df = canonical_frame([canonical_row(volume=0, turnover=0.0)])
    assert R.t2_turnover_reconciles(ctx(df)).passed


# ==========================================================================
# TIER 3 — COMPLETENESS
# ==========================================================================


@covers("T3.completeness.trading_day")
def test_file_on_a_predicted_holiday_is_flagged() -> None:
    result = R.t3_is_trading_day(ctx(canonical_frame(), expected_trading_day=False))
    assert not result.passed
    assert "holiday list" in str(result.evidence["hint"])


def test_file_on_a_trading_day_passes() -> None:
    assert R.t3_is_trading_day(ctx(canonical_frame(), expected_trading_day=True)).passed


def test_absent_calendar_does_not_fail() -> None:
    assert R.t3_is_trading_day(ctx(canonical_frame())).passed


@covers("T3.completeness.row_count")
def test_truncated_file_is_caught_by_row_count() -> None:
    """3 rows against a 2000-row trailing median."""
    result = R.t3_row_count(ctx(canonical_frame(), trailing_row_counts=[2000] * 20))
    assert not result.passed
    assert result.severity is Severity.ERROR
    assert result.evidence["deviation_pct"] > R.ROW_COUNT_TOLERANCE_PCT


def test_row_count_within_tolerance_passes() -> None:
    assert R.t3_row_count(ctx(canonical_frame(), trailing_row_counts=[3, 3, 3])).passed


@covers("T3.completeness.missing_symbols")
def test_symbol_vanishing_without_explanation_is_flagged() -> None:
    result = R.t3_missing_symbols(
        ctx(canonical_frame(), previous_symbols=frozenset({"RELIANCE", "TCS", "INFY", "GONE"}))
    )
    assert not result.passed
    assert result.affected_symbols == ("GONE",)


def test_delisting_explains_a_vanished_symbol() -> None:
    """Without this check the rule fires on every real delisting and gets muted."""
    action = CorporateAction(
        symbol="GONE",
        action_type=CorporateActionType.RIGHTS,
        ex_date=TRADE_DATE,
        ratio=0.5,
    )
    result = R.t3_missing_symbols(
        ctx(
            canonical_frame(),
            previous_symbols=frozenset({"RELIANCE", "TCS", "INFY", "GONE"}),
            corporate_actions=[action],
        )
    )
    assert result.passed


@covers("T3.completeness.expected_constituents")
def test_absent_index_constituent_is_caught() -> None:
    result = R.t3_expected_constituents(
        ctx(canonical_frame(), expected_symbols=frozenset({"RELIANCE", "HDFCBANK"}))
    )
    assert not result.passed
    assert result.affected_symbols == ("HDFCBANK",)


# ==========================================================================
# TIER 4 — CONTINUITY (the leak detector)
# ==========================================================================


@covers("T4.continuity.prev_close")
def test_unexplained_prev_close_break_is_caught() -> None:
    """prev_close says 50, yesterday's close was 99, and no corporate action.

    This is the adjustment pipeline being wrong. Every feature derived from
    these prices inherits the error, and no test on the feature code would
    reveal it.
    """
    df = canonical_frame([canonical_row(prev_close=50.0)])
    result = R.t4_prev_close_continuity(
        ctx(df, previous_close=prior_close_frame({"RELIANCE": 99.0}))
    )
    assert not result.passed
    assert result.severity is Severity.ERROR
    assert result.evidence["unexplained_count"] == 1
    assert "adjustment pipeline" in result.message


def test_continuous_prices_pass() -> None:
    result = R.t4_prev_close_continuity(
        ctx(canonical_frame(), previous_close=prior_close_frame())
    )
    assert result.passed


def test_split_explains_a_prev_close_break() -> None:
    """A 1:2 split halves prices. prev_close 49.5 against a close of 99 is
    exactly ratio 0.5 and must not be flagged."""
    df = canonical_frame([canonical_row(prev_close=49.5, open=50.0, high=52.0,
                                        low=49.0, close=51.0, turnover=51_000_000.0)])
    action = CorporateAction(
        symbol="RELIANCE",
        action_type=CorporateActionType.SPLIT,
        ex_date=TRADE_DATE,
        ratio=0.5,
    )
    result = R.t4_prev_close_continuity(
        ctx(
            df,
            previous_close=prior_close_frame({"RELIANCE": 99.0}),
            corporate_actions=[action],
        )
    )
    assert result.passed
    assert "explained by corporate actions" in result.message


def test_wrong_ratio_does_not_explain_the_break() -> None:
    """A recorded split with the wrong ratio must still fail — otherwise a bad
    corporate-action record silently launders a bad price."""
    df = canonical_frame([canonical_row(prev_close=33.0)])
    action = CorporateAction(
        symbol="RELIANCE",
        action_type=CorporateActionType.SPLIT,
        ex_date=TRADE_DATE,
        ratio=0.5,  # would imply 49.5, not 33.0
    )
    result = R.t4_prev_close_continuity(
        ctx(
            df,
            previous_close=prior_close_frame({"RELIANCE": 99.0}),
            corporate_actions=[action],
        )
    )
    assert not result.passed


def test_no_prior_close_is_not_a_failure() -> None:
    assert R.t4_prev_close_continuity(ctx(canonical_frame())).passed


# ==========================================================================
# TIER 5 — PLAUSIBILITY
# ==========================================================================


@covers("T5.plausibility.circuit_breach")
def test_move_beyond_circuit_limit_is_flagged() -> None:
    df = canonical_frame([canonical_row(prev_close=100.0, close=140.0,
                                        high=140.0, low=99.0, open=100.0,
                                        turnover=140_000_000.0)])
    result = R.t5_circuit_breach(ctx(df))
    assert not result.passed
    assert result.severity is Severity.WARNING


def test_corporate_action_suppresses_a_circuit_flag() -> None:
    df = canonical_frame([canonical_row(prev_close=100.0, close=140.0,
                                        high=140.0, low=99.0, open=100.0,
                                        turnover=140_000_000.0)])
    action = CorporateAction(
        symbol="RELIANCE",
        action_type=CorporateActionType.BONUS,
        ex_date=TRADE_DATE,
        ratio=0.7,
    )
    assert R.t5_circuit_breach(ctx(df, corporate_actions=[action])).passed


@covers("T5.plausibility.return_sigma")
def test_extreme_sigma_move_is_flagged() -> None:
    """A calm 0.1-jitter history makes a 15% move an enormous z-score."""
    df = canonical_frame([canonical_row(prev_close=99.0, close=114.0,
                                        high=114.0, low=98.0, open=99.0,
                                        turnover=114_000_000.0)])
    result = R.t5_return_outlier(ctx(df, history=history_frame()))
    assert not result.passed
    assert abs(result.evidence["sample"][0]["z"]) > R.RETURN_SIGMA_THRESHOLD


def test_normal_move_is_not_a_sigma_outlier() -> None:
    df = canonical_frame([canonical_row(prev_close=99.0, close=99.2,
                                        high=99.5, low=98.9, open=99.0,
                                        turnover=99_200_000.0)])
    assert R.t5_return_outlier(ctx(df, history=history_frame())).passed


@covers("T5.plausibility.staleness")
def test_unchanged_close_for_a_week_is_flagged() -> None:
    stale = history_frame(n=10, jitter=False)
    result = R.t5_staleness(ctx(canonical_frame(), history=stale))
    assert not result.passed
    assert result.severity is Severity.INFO
    assert "RELIANCE" in result.affected_symbols


def test_moving_series_is_not_stale() -> None:
    assert R.t5_staleness(ctx(canonical_frame(), history=history_frame())).passed


@covers("T5.plausibility.zero_volume_move")
def test_price_move_on_zero_volume_is_flagged() -> None:
    df = canonical_frame([canonical_row(volume=0, turnover=0.0,
                                        prev_close=99.0, close=102.0)])
    result = R.t5_zero_volume_with_move(ctx(df))
    assert not result.passed


@covers("T5.plausibility.volume_spike")
def test_volume_unit_error_is_flagged() -> None:
    df = canonical_frame([canonical_row(volume=100_000_000,
                                        turnover=10_200_000_000.0)])
    result = R.t5_volume_spike(ctx(df, history=history_frame()))
    assert not result.passed
    assert result.evidence["sample"][0]["multiple"] > R.VOLUME_SPIKE_MULTIPLE


def test_normal_volume_is_not_a_spike() -> None:
    assert R.t5_volume_spike(ctx(canonical_frame(), history=history_frame())).passed


# ==========================================================================
# TIER 6 — CROSS-SOURCE
# ==========================================================================


@covers("T6.cross_source.close_agreement")
def test_cross_source_disagreement_is_flagged() -> None:
    other = pd.DataFrame([{"symbol": "RELIANCE", "close": 150.0}])
    result = R.t6_cross_source_close(ctx(canonical_frame(), cross_source=other))
    assert not result.passed


def test_cross_source_agreement_passes() -> None:
    other = pd.DataFrame([{"symbol": "RELIANCE", "close": 102.3}])
    assert R.t6_cross_source_close(ctx(canonical_frame(), cross_source=other)).passed


def test_absent_cross_source_does_not_fail() -> None:
    assert R.t6_cross_source_close(ctx(canonical_frame())).passed


# ==========================================================================
# Coverage assertion — the actual Phase 3 gate
# ==========================================================================


def test_every_rule_has_an_adversarial_fixture() -> None:
    """No rule may ship without a hand-built input that trips it.

    If this fails, a rule was added without proving it can fail — which means
    it has never been shown to work.
    """
    declared = set(R.rule_ids())
    covered = set(COVERED)
    missing = sorted(declared - covered)
    stale = sorted(covered - declared)
    assert not missing, f"rules with no failing fixture: {missing}"
    assert not stale, f"fixtures for rules that no longer exist: {stale}"


def test_all_rules_return_the_expected_tier() -> None:
    """A rule's id prefix must match the tier it reports, or the report groups
    findings under the wrong heading."""
    empty = ctx(canonical_frame())
    for rule in R.ALL_RULES:
        result = rule(empty)
        expected_tier = int(result.rule_id[1])
        assert int(result.tier) == expected_tier, result.rule_id


@pytest.mark.parametrize("rule", R.ALL_RULES, ids=lambda r: r.__name__)
def test_rules_tolerate_an_empty_frame(rule) -> None:
    """A rule must not raise on an empty frame — the gate runs every tier even
    when validity quarantined everything."""
    result = rule(ctx(canonical_frame().iloc[0:0]))
    assert result.rule_id
    assert isinstance(result.tier, Tier)
