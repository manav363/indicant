"""Contract round-trip and invariant tests.

These are cheap and they are the reason a contract change cannot silently
diverge between two services.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from indicant_contracts import (
    CANONICAL_PRICE_COLUMNS,
    Dataset,
    Direction,
    EligibilityThresholds,
    ErrorEnvelope,
    ExplanationFact,
    LakePaths,
    ModelCard,
    OHLCVBar,
    Prediction,
    QualityReport,
    QualityScore,
    RuleResult,
    Series,
    Severity,
    Signal,
    Strength,
    SymbolMeta,
    Tier,
    TradingCalendar,
    UniverseSnapshot,
    Verdict,
)


def _bar(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "date": date(2026, 7, 30),
        "symbol": "RELIANCE",
        "series": Series.EQ,
        "open": 1300.0,
        "high": 1330.0,
        "low": 1295.0,
        "close": 1321.2,
        "prev_close": 1298.0,
        "volume": 5_000_000,
        "turnover": 6.6e9,
        "trades": 120_000,
        "delivery_qty": 2_000_000,
        "delivery_pct": 40.0,
        "isin": "INE002A01018",
    }
    return base | overrides


# --------------------------------------------------------------------------
# OHLCVBar invariants
# --------------------------------------------------------------------------


class TestOHLCVBar:
    def test_valid_bar_round_trips(self) -> None:
        bar = OHLCVBar(**_bar())
        assert OHLCVBar.model_validate(bar.model_dump()) == bar

    def test_symbol_is_normalised_to_upper(self) -> None:
        assert OHLCVBar(**_bar(symbol=" reliance ")).symbol == "RELIANCE"

    def test_high_below_low_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            OHLCVBar(**_bar(high=1200.0, low=1295.0))

    def test_high_below_close_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max"):
            OHLCVBar(**_bar(high=1310.0, close=1321.2))

    def test_low_above_open_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min"):
            OHLCVBar(**_bar(low=1305.0, open=1300.0))

    def test_delivery_exceeding_volume_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="delivery_qty"):
            OHLCVBar(**_bar(volume=1000, delivery_qty=2000))

    def test_zero_close_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVBar(**_bar(close=0.0))

    def test_negative_volume_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVBar(**_bar(volume=-1))

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVBar(**_bar(vwap=1310.0))

    def test_bar_is_frozen(self) -> None:
        bar = OHLCVBar(**_bar())
        with pytest.raises(ValidationError):
            bar.close = 1400.0  # type: ignore[misc]

    def test_model_fields_match_canonical_columns(self) -> None:
        """The bar model and the lake schema must not drift apart."""
        assert set(OHLCVBar.model_fields) == set(CANONICAL_PRICE_COLUMNS)


# --------------------------------------------------------------------------
# Lake layout
# --------------------------------------------------------------------------


class TestLakePaths:
    @pytest.fixture
    def paths(self) -> LakePaths:
        return LakePaths(root=Path("/lake"))

    def test_prices_partition_by_year(self, paths: LakePaths) -> None:
        f = paths.file(Dataset.PRICES, when=date(2015, 3, 4))
        assert f == Path("/lake/prices/year=2015/prices_2015.parquet")

    def test_quarantine_partitions_by_date(self, paths: LakePaths) -> None:
        f = paths.file(Dataset.QUARANTINE, when=date(2015, 3, 4))
        assert f == Path("/lake/quality/quarantine/date=2015-03-04/data.parquet")

    def test_glob_is_recursive(self, paths: LakePaths) -> None:
        assert paths.glob(Dataset.PRICES) == "/lake/prices/**/*.parquet"

    def test_catalog_and_state_live_at_root(self, paths: LakePaths) -> None:
        assert paths.catalog_db == Path("/lake/catalog.duckdb")
        assert paths.ingest_state == Path("/lake/_ingest_state.json")

    def test_every_dataset_resolves(self, paths: LakePaths) -> None:
        """A new Dataset member must not silently produce a broken path."""
        for ds in Dataset:
            assert paths.file(ds, when=date(2020, 1, 1)).suffix == ".parquet"


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------


class TestQuality:
    def test_failing_rule_without_evidence_is_rejected(self) -> None:
        """A rule that cannot say why it failed is not actionable."""
        with pytest.raises(ValidationError, match="without evidence"):
            RuleResult(
                rule_id="T2.validity.high_low",
                tier=Tier.VALIDITY,
                severity=Severity.FATAL,
                passed=False,
                message="high < low",
            )

    def test_passing_rule_needs_no_evidence(self) -> None:
        r = RuleResult(
            rule_id="T2.validity.high_low",
            tier=Tier.VALIDITY,
            severity=Severity.FATAL,
            passed=True,
            message="ok",
        )
        assert r.passed

    def test_malformed_rule_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RuleResult(
                rule_id="validity-high-low",
                tier=Tier.VALIDITY,
                severity=Severity.FATAL,
                passed=True,
                message="ok",
            )

    def test_structural_fatal_blocks_the_file(self) -> None:
        r = RuleResult(
            rule_id="T1.structural.schema",
            tier=Tier.STRUCTURAL,
            severity=Severity.FATAL,
            passed=False,
            message="missing columns",
            evidence={"missing": ["close"]},
        )
        assert r.blocks_file and r.blocks_rows

    def test_plausibility_warning_blocks_nothing(self) -> None:
        r = RuleResult(
            rule_id="T5.plausibility.return_sigma",
            tier=Tier.PLAUSIBILITY,
            severity=Severity.WARNING,
            passed=False,
            message="7 sigma move",
            evidence={"sigma": 7.1},
        )
        assert not r.blocks_file and not r.blocks_rows

    def test_quarantined_is_usable_but_rejected_is_not(self) -> None:
        assert Verdict.QUARANTINED.is_usable
        assert Verdict.PASS.is_usable
        assert Verdict.PASS_WITH_WARNINGS.is_usable
        assert not Verdict.REJECTED.is_usable

    def test_report_row_counts_must_reconcile(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError, match="exceeds rows_in"):
            QualityReport(
                run_id="r1",
                trade_date=date(2026, 7, 30),
                started_at=now,
                finished_at=now,
                verdict=Verdict.PASS,
                rows_in=10,
                rows_accepted=8,
                rows_quarantined=5,
                results=(),
            )

    def test_quality_score_weights_sum_to_one(self) -> None:
        assert sum(QualityScore.WEIGHTS.values()) == pytest.approx(1.0)

    def test_perfect_score_is_one(self) -> None:
        s = QualityScore(
            symbol="RELIANCE",
            as_of=date(2026, 7, 30),
            history_completeness=1.0,
            validity_clean_rate=1.0,
            continuity_clean_rate=1.0,
            liquidity_adequacy=1.0,
            recency=1.0,
            history_days=4700,
            median_turnover=1e9,
        )
        assert s.score == pytest.approx(1.0)


class TestEligibility:
    @pytest.fixture
    def thresholds(self) -> EligibilityThresholds:
        return EligibilityThresholds()

    def _score(self, **overrides: object) -> QualityScore:
        base: dict[str, object] = {
            "symbol": "RELIANCE",
            "as_of": date(2026, 7, 30),
            "history_completeness": 1.0,
            "validity_clean_rate": 1.0,
            "continuity_clean_rate": 1.0,
            "liquidity_adequacy": 1.0,
            "recency": 1.0,
            "history_days": 4700,
            "median_turnover": 1e9,
        }
        return QualityScore(**(base | overrides))  # type: ignore[arg-type]

    def test_clean_symbol_is_eligible(self, thresholds: EligibilityThresholds) -> None:
        assert thresholds.evaluate(self._score()) is None

    def test_short_history_excluded_with_a_readable_reason(
        self, thresholds: EligibilityThresholds
    ) -> None:
        reason = thresholds.evaluate(self._score(history_days=100))
        assert reason is not None
        assert "100 trading days" in reason

    def test_illiquid_symbol_excluded(self, thresholds: EligibilityThresholds) -> None:
        reason = thresholds.evaluate(self._score(median_turnover=1000.0))
        assert reason is not None
        assert "liquidity floor" in reason

    def test_low_quality_excluded(self, thresholds: EligibilityThresholds) -> None:
        reason = thresholds.evaluate(self._score(continuity_clean_rate=0.1))
        assert reason is not None
        assert "quality score" in reason

    def test_history_check_precedes_quality_check(
        self, thresholds: EligibilityThresholds
    ) -> None:
        """A brand-new listing should be told it is new, not that its data is bad."""
        reason = thresholds.evaluate(self._score(history_days=10, continuity_clean_rate=0.0))
        assert reason is not None
        assert "trading days" in reason


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------


class TestUniverseSnapshot:
    def test_eligible_must_be_a_subset(self) -> None:
        with pytest.raises(ValidationError, match="not in universe"):
            UniverseSnapshot(
                as_of=date(2026, 7, 30),
                symbols=("RELIANCE",),
                eligible_symbols=("RELIANCE", "TCS"),
            )

    def test_counts(self) -> None:
        u = UniverseSnapshot(
            as_of=date(2026, 7, 30),
            symbols=("RELIANCE", "TCS", "NEWLISTING"),
            eligible_symbols=("RELIANCE", "TCS"),
            excluded={"NEWLISTING": "only 40 trading days of history"},
        )
        assert (u.total, u.eligible_count) == (3, 2)


class TestSymbolMeta:
    def test_delisted_symbol_is_not_active(self) -> None:
        meta = SymbolMeta(
            symbol="DEADCO",
            first_seen=date(2008, 1, 1),
            last_seen=date(2019, 6, 3),
            status="delisted",  # type: ignore[arg-type]
            delisted_on=date(2019, 6, 3),
        )
        assert not meta.is_active


class TestTradingCalendar:
    def test_membership(self) -> None:
        cal = TradingCalendar(
            start=date(2026, 7, 27),
            end=date(2026, 7, 31),
            trading_days=(date(2026, 7, 27), date(2026, 7, 28)),
        )
        assert cal.is_trading_day(date(2026, 7, 27))
        assert not cal.is_trading_day(date(2026, 7, 29))
        assert cal.count == 2


# --------------------------------------------------------------------------
# Prediction consistency
# --------------------------------------------------------------------------


class TestPrediction:
    def _pred(self, **overrides: object) -> Prediction:
        base: dict[str, object] = {
            "symbol": "RELIANCE",
            "as_of": date(2026, 7, 30),
            "horizon_months": 6,
            "signal": Signal.BUY,
            "probability_up": 0.66,
            "confidence": 0.66,
            "strength": Strength.MODERATE,
            "current_price": 1321.2,
            "suggested_position_pct": 4.5,
        }
        return Prediction(**(base | overrides))  # type: ignore[arg-type]

    def test_buy_confidence_is_p_up(self) -> None:
        assert self._pred().confidence == pytest.approx(0.66)

    def test_sell_confidence_is_one_minus_p_up(self) -> None:
        p = self._pred(signal=Signal.SELL, probability_up=0.30, confidence=0.70)
        assert p.confidence == pytest.approx(0.70)

    def test_sell_with_buy_shaped_confidence_is_rejected(self) -> None:
        """Guards the class of bug where a strong SELL renders as 30% confident."""
        with pytest.raises(ValidationError, match="inconsistent"):
            self._pred(signal=Signal.SELL, probability_up=0.30, confidence=0.30)

    def test_facts_carry_direction_and_rank(self) -> None:
        fact = ExplanationFact(
            feature="momentum_roc_6m",
            display_name="6-month price change",
            value=0.182,
            display_value="+18.2%",
            shap=0.08,
            direction=Direction.SUPPORTS_UP,
            rank=1,
        )
        assert self._pred(facts=(fact,)).facts[0].display_name == "6-month price change"

    def test_conviction_defaults_to_none_before_meta_labeller_exists(self) -> None:
        assert self._pred().conviction is None


# --------------------------------------------------------------------------
# Model card — a null result must be first-class
# --------------------------------------------------------------------------


class TestModelCard:
    def _card(self, **overrides: object) -> ModelCard:
        base: dict[str, object] = {
            "run_id": "r1",
            "trained_at": datetime.now(UTC),
            "model_type": "gradient_boost",
            "n_train_samples": 1_000_000,
            "n_features": 92,
            "universe_size": 1700,
            "train_start": date(2006, 1, 2),
            "train_end": date(2024, 12, 31),
        }
        return ModelCard(**(base | overrides))  # type: ignore[arg-type]

    def test_untested_significance_is_none_not_false(self) -> None:
        """'Never tested' must not render as 'tested and failed'."""
        assert self._card().is_significant() is None

    def test_insignificant_result_is_reported_as_false(self) -> None:
        assert self._card(permutation_p_value=0.6816).is_significant() is False

    def test_significant_result(self) -> None:
        assert self._card(permutation_p_value=0.01).is_significant() is True

    def test_alpha_is_adjustable(self) -> None:
        assert self._card(permutation_p_value=0.08).is_significant(alpha=0.10) is True

    def test_beats_baseline_is_none_without_a_baseline(self) -> None:
        assert self._card(oos_sharpe=0.4).beats_baseline is None

    def test_baseline_comparison(self) -> None:
        assert self._card(oos_sharpe=0.4, baseline_oos_sharpe=0.5).beats_baseline is False
        assert self._card(oos_sharpe=0.6, baseline_oos_sharpe=0.5).beats_baseline is True


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TestErrorEnvelope:
    def test_ineligible_symbol_gets_a_human_reason(self) -> None:
        env = ErrorEnvelope.not_eligible("NEWCO", "only 40 trading days of history")
        assert "NEWCO" in env.user_message
        assert "40 trading days" in env.user_message
        assert env.detail["symbol"] == "NEWCO"

    def test_engineer_and_user_messages_are_separate(self) -> None:
        env = ErrorEnvelope.not_eligible("NEWCO", "reason")
        assert env.message != env.user_message
