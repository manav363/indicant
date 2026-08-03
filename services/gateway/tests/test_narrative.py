"""Narrative and chart-payload tests.

Every sentence the product will ever show, tested against a fixed input with no
model in the loop. That is the point of putting the narrative in the gateway:
copy is verifiable on its own, and a wording change cannot move a number.
"""

from __future__ import annotations

from datetime import date

import pytest
from indicant_contracts import (
    Direction,
    ExplanationFact,
    Prediction,
    PrimaryRegime,
    Signal,
    Strength,
)

from gateway.charts.payloads import (
    BarDirection,
    candlestick_payload,
    direction_of,
    reliability_payload,
    return_bars_payload,
    verdict_bar_payload,
    volume_payload,
)
from gateway.narrative import copy
from gateway.narrative.templates import (
    conviction_sentence,
    probability_sentence,
    regime_sentence,
    render,
    render_ineligible,
)


def fact(
    name: str = "momentum_roc_6m",
    display: str = "6-month price change",
    value: float = 0.182,
    shown: str = "+18.2%",
    shap: float = 0.08,
    up: bool = True,
    rank: int = 1,
) -> ExplanationFact:
    return ExplanationFact(
        feature=name,
        display_name=display,
        value=value,
        display_value=shown,
        shap=shap,
        direction=Direction.SUPPORTS_UP if up else Direction.SUPPORTS_DOWN,
        rank=rank,
    )


def prediction(**overrides: object) -> Prediction:
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
        "facts": (
            fact(),
            fact("regime_adx", "trend strength", 31.4, "31", 0.05, True, 2),
            fact("volatility_atr_21", "typical daily range", 28.0, "28.00",
                 -0.03, False, 3),
        ),
    }
    return Prediction(**(base | overrides))  # type: ignore[arg-type]


# ==========================================================================
# The rule the narrative layer exists to enforce
# ==========================================================================


class TestProbabilitySentence:
    def test_states_the_probability_and_what_it_means_when_wrong(self) -> None:
        """THE requirement. '66%' invites the reader to hear 'this will go up';
        '66%, and about 34 of every 100 such calls went the other way' does not.
        """
        text = probability_sentence(prediction())
        assert "66%" in text
        assert "34" in text
        assert "not a promise" in text

    def test_failure_count_is_the_complement(self) -> None:
        for p_up, expect_miss in [(0.70, "30"), (0.85, "15"), (0.55, "45")]:
            text = probability_sentence(
                prediction(probability_up=p_up, confidence=p_up)
            )
            assert expect_miss in text

    def test_sell_signals_also_state_the_upside_probability(self) -> None:
        """A SELL at p_up=0.30 still gets told 30% — the number is about the
        stock, not about the call."""
        text = probability_sentence(
            prediction(signal=Signal.SELL, probability_up=0.30, confidence=0.70)
        )
        assert "30%" in text
        assert "70" in text

    def test_a_near_even_call_is_named_as_a_coin_flip(self) -> None:
        """Saying so is more useful than manufacturing a view."""
        text = probability_sentence(
            prediction(signal=Signal.HOLD, probability_up=0.51, confidence=0.51)
        )
        assert "coin flip" in text
        assert "%" not in text

    def test_no_probability_is_ever_stated_bare(self) -> None:
        """Sweep the space: every non-coin-flip sentence must carry both the
        probability and its complement."""
        for pct in range(10, 91, 5):
            p = pct / 100
            if abs(p - 0.5) < 0.04:
                continue
            text = probability_sentence(
                prediction(
                    signal=Signal.BUY if p > 0.5 else Signal.SELL,
                    probability_up=p,
                    confidence=p if p > 0.5 else 1 - p,
                )
            )
            assert f"{round(p * 100)}%" in text
            assert str(100 - round(p * 100)) in text


# ==========================================================================
# Headlines, drivers, regime, conviction
# ==========================================================================


class TestHeadline:
    @pytest.mark.parametrize("signal", list(Signal))
    @pytest.mark.parametrize("strength", list(Strength))
    def test_every_signal_and_strength_has_copy(self, signal, strength) -> None:
        """A missing combination would KeyError in production on a real call."""
        p_up = {Signal.BUY: 0.7, Signal.SELL: 0.3, Signal.HOLD: 0.5}[signal]
        conf = {Signal.BUY: 0.7, Signal.SELL: 0.7, Signal.HOLD: 0.5}[signal]
        narrative = render(
            prediction(signal=signal, strength=strength,
                       probability_up=p_up, confidence=conf)
        )
        assert "RELIANCE" in narrative.headline
        assert "{" not in narrative.headline  # no unfilled placeholder

    def test_weak_signals_are_hedged_in_the_copy(self) -> None:
        n = render(prediction(strength=Strength.WEAK))
        assert "slightly" in n.headline


class TestDrivers:
    def test_supporting_and_opposing_are_separated(self) -> None:
        """'What's pushing it up' and 'what's holding it back' is how a person
        reads a recommendation."""
        n = render(prediction())
        assert len(n.supports) == 2
        assert len(n.opposes) == 1

    def test_driver_lines_carry_a_glyph_and_a_value(self) -> None:
        n = render(prediction())
        assert n.supports[0].startswith("▲")
        assert "+18.2%" in n.supports[0]
        assert n.opposes[0].startswith("▼")

    def test_no_facts_yields_no_driver_lines(self) -> None:
        n = render(prediction(facts=()))
        assert n.supports == () and n.opposes == ()
        assert copy.NO_DRIVERS in n.as_text()


class TestRegime:
    def test_aligned_regime_is_noted_as_consistent(self) -> None:
        text = regime_sentence(prediction(regime=PrimaryRegime.BULL))
        assert text is not None and "consistent" in text

    def test_buy_in_a_bear_market_is_flagged_as_tension(self) -> None:
        text = regime_sentence(
            prediction(signal=Signal.BUY, regime=PrimaryRegime.BEAR)
        )
        assert text is not None
        assert "tension" in text
        assert "fight the market" in text

    def test_sell_in_a_bull_market_is_flagged(self) -> None:
        text = regime_sentence(
            prediction(signal=Signal.SELL, probability_up=0.3, confidence=0.7,
                       regime=PrimaryRegime.BULL)
        )
        assert text is not None and "tension" in text

    def test_hold_is_never_described_as_fighting_the_market(self) -> None:
        text = regime_sentence(
            prediction(signal=Signal.HOLD, probability_up=0.5, confidence=0.5,
                       regime=PrimaryRegime.BEAR)
        )
        assert text is not None and "tension" not in text

    def test_absent_regime_yields_nothing(self) -> None:
        assert regime_sentence(prediction()) is None


class TestConviction:
    def test_untrained_meta_labeller_says_nothing(self) -> None:
        """Inventing a confidence statement for a model that does not exist is
        exactly what this project refuses."""
        assert conviction_sentence(prediction()) is None

    def test_low_conviction_advises_sizing_down(self) -> None:
        text = conviction_sentence(prediction(conviction=0.30))
        assert text is not None and "size it small or skip" in text

    def test_high_conviction_is_stated(self) -> None:
        text = conviction_sentence(prediction(conviction=0.75))
        assert text is not None and "above average" in text

    def test_middling_conviction_stays_silent(self) -> None:
        assert conviction_sentence(prediction(conviction=0.52)) is None


class TestCaveats:
    def test_an_insignificant_model_says_so_plainly(self) -> None:
        """This is the project's actual position, and stating it is what makes
        the rest credible."""
        n = render(prediction(), is_significant=False)
        joined = " ".join(n.caveats)
        assert "NOT statistically distinguishable from chance" in joined

    def test_an_untested_model_is_not_described_as_failing(self) -> None:
        n = render(prediction(), is_significant=None)
        joined = " ".join(n.caveats)
        assert "never formally tested" in joined
        assert "NOT statistically" not in joined

    def test_not_advice_is_always_present(self) -> None:
        for sig in (True, False, None):
            n = render(prediction(), is_significant=sig)
            assert any("not investment advice" in c for c in n.caveats)


class TestFullRender:
    def test_text_rendering_has_no_unfilled_placeholders(self) -> None:
        text = render(prediction(regime=PrimaryRegime.BULL, conviction=0.7),
                      is_significant=False).as_text()
        assert "{" not in text and "}" not in text

    def test_reads_as_connected_prose(self) -> None:
        text = render(prediction(regime=PrimaryRegime.BULL),
                      is_significant=False).as_text()
        assert "RELIANCE" in text
        assert copy.SUPPORTS_HEADER in text
        assert copy.OPPOSES_HEADER in text

    def test_ineligible_symbol_gets_a_reason_not_an_error(self) -> None:
        """The system declining to guess, with the reason — a better answer
        than a low-confidence number."""
        text = render_ineligible("NEWCO", "only 40 trading days of history")
        assert "NEWCO" in text
        assert "40 trading days" in text
        assert "rather say so" in text


# ==========================================================================
# Chart payloads — direction encoded four ways
# ==========================================================================


class TestDirectionEncoding:
    def test_up_down_flat(self) -> None:
        assert direction_of(1.0) is BarDirection.UP
        assert direction_of(-1.0) is BarDirection.DOWN
        assert direction_of(0.0) is BarDirection.FLAT

    def test_nan_is_flat_not_a_crash(self) -> None:
        assert direction_of(float("nan")) is BarDirection.FLAT

    def test_each_direction_has_a_distinct_glyph_and_label(self) -> None:
        """The non-colour encodings must actually differ, or the redundancy is
        cosmetic."""
        glyphs = {d.glyph for d in BarDirection}
        labels = {d.label for d in BarDirection}
        assert len(glyphs) == 3
        assert len(labels) == 3

    def test_payloads_ship_a_token_not_a_hex_colour(self) -> None:
        """The frontend owns the palette, including the colour-vision-safe
        alternate. Hex on the wire would freeze one theme."""
        for d in BarDirection:
            assert d.css_var.startswith("--dir-")
            assert "#" not in d.css_var


def ohlc(n: int = 3, *, key: str = "time") -> list[dict[str, object]]:
    """Bars in the shape market-data actually returns: a list of dicts.

    These were DataFrames, which is why the gateway imported pandas at all.
    `key` is parameterised because the two producers disagree on the field
    name — market-data says `time`, the lake says `date` — and both must work.
    """
    rows = [
        (date(2026, 7, 27 + i), o, h, low, c, v)
        for i, (o, h, low, c, v) in enumerate(
            zip(
                [100.0, 105.0, 103.0][:n],
                [106.0, 107.0, 104.0][:n],
                [99.0, 102.0, 100.0][:n],
                [105.0, 103.0, 103.0][:n],
                [1_000_000, 2_000_000, 1_500_000][:n],
            )
        )
    ]
    return [
        {key: d, "open": o, "high": h, "low": low, "close": c, "volume": v}
        for d, o, h, low, c, v in rows
    ]


class TestCandlestickPayload:
    def test_body_direction_is_close_vs_open(self) -> None:
        """Not close-vs-previous-close: a candle body is green when the session
        closed above where it OPENED, which is what the shape depicts."""
        bars = candlestick_payload(ohlc())
        assert bars[0]["direction"] == "up"     # 100 -> 105
        assert bars[1]["direction"] == "down"   # 105 -> 103
        assert bars[2]["direction"] == "flat"   # 103 -> 103

    def test_every_bar_carries_all_four_encodings(self) -> None:
        """Strip the colour and the chart must still read."""
        for bar in candlestick_payload(ohlc()):
            assert {"direction", "glyph", "label", "colorVar"} <= set(bar)

    def test_empty_input_is_an_empty_list(self) -> None:
        assert candlestick_payload([]) == []
        assert volume_payload([]) == []

    def test_either_date_field_name_is_accepted(self) -> None:
        """market-data says `time`, the lake says `date`.

        The gateway used to rename one to the other at the call site, and when
        the upstream name changed the rename was simply absent — a KeyError on
        every chart request. Reading both here means the seam cannot drift.
        """
        by_time = candlestick_payload(ohlc(key="time"))
        by_date = candlestick_payload(ohlc(key="date"))
        assert by_time == by_date
        assert by_time[0]["time"] == "2026-07-27"

    def test_a_bar_with_neither_name_is_an_error_not_a_silent_gap(self) -> None:
        with pytest.raises(KeyError):
            candlestick_payload([{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}])


class TestVolumePayload:
    def test_bars_match_the_candles_above_them(self) -> None:
        candles = candlestick_payload(ohlc())
        volumes = volume_payload(ohlc())
        assert [c["direction"] for c in candles] == [v["direction"] for v in volumes]


class TestReturnBars:
    def test_position_encodes_direction_independently_of_colour(self) -> None:
        bars = return_bars_payload(
            [{"symbol": "A", "return_pct": 5.0}, {"symbol": "B", "return_pct": -3.0}]
        )
        assert bars[0]["value"] > 0 and bars[0]["direction"] == "up"
        assert bars[1]["value"] < 0 and bars[1]["direction"] == "down"

    def test_display_value_carries_an_explicit_sign(self) -> None:
        bars = return_bars_payload([{"symbol": "A", "return_pct": 5.0}])
        assert bars[0]["displayValue"] == "+5.00%"


class TestVerdictBar:
    def test_an_even_call_renders_as_zero_length(self) -> None:
        """The bar's length encodes conviction, and an even call has none."""
        p = verdict_bar_payload(probability_up=0.5, signal="HOLD", strength="weak")
        assert p["magnitude"] == 0.0
        assert p["direction"] == "flat"

    def test_a_strong_call_fills_the_bar(self) -> None:
        p = verdict_bar_payload(probability_up=0.95, signal="BUY", strength="strong")
        assert p["magnitude"] > 0.85
        assert p["extendsRight"] is True

    def test_a_bearish_call_extends_left(self) -> None:
        p = verdict_bar_payload(probability_up=0.2, signal="SELL", strength="strong")
        assert p["extendsRight"] is False
        assert p["direction"] == "down"

    def test_magnitude_is_capped_at_one(self) -> None:
        p = verdict_bar_payload(probability_up=1.0, signal="BUY", strength="strong")
        assert p["magnitude"] == 1.0


class TestReliabilityPayload:
    def test_ships_the_reference_diagonal(self) -> None:
        """The chart cannot be drawn without the line that gives it meaning."""
        out = reliability_payload(
            [{"mean_predicted": 0.7, "observed_rate": 0.65, "count": 100}]
        )
        assert out["reference"] == [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
        assert out["referenceLabel"] == "perfect calibration"

    def test_overconfident_bins_are_marked_down(self) -> None:
        out = reliability_payload(
            [{"mean_predicted": 0.80, "observed_rate": 0.55, "count": 100}]
        )
        assert out["bins"][0]["direction"] == "down"
        assert out["bins"][0]["gap"] < 0
