"""SHAP fact tests.

The boundary this module enforces: intelligence emits numbers, never prose. If
a sentence ever appears in an ExplanationFact, the gateway's narrative layer has
lost its monopoly on wording and a copy change can start moving figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from indicant_contracts import Direction, ExplanationFact

from intelligence.explain.shap_facts import (
    MIN_ABS_SHAP,
    ShapExplainer,
    describe,
    facts_from_shap,
    split_by_direction,
)


class TestDescribe:
    def test_known_features_get_a_human_label(self) -> None:
        label, _ = describe("momentum_roc_6m")
        assert label == "6-month price change"

    def test_cross_sectional_suffix_is_explained_not_shown_raw(self) -> None:
        """A user should never see 'momentum_roc_6m_xs'."""
        label, _ = describe("momentum_roc_6m_xs")
        assert label == "6-month price change (vs the rest of the market)"

    def test_sector_neutral_suffix_is_explained(self) -> None:
        label, _ = describe("momentum_roc_6m_sect")
        assert "vs its sector" in label

    def test_unknown_features_fall_back_to_a_readable_label(self) -> None:
        """Blunter than a curated label, but never a raw slug."""
        label, _ = describe("momentum_some_new_thing")
        assert label == "some new thing"
        assert "_" not in label

    def test_fallback_strips_the_category_prefix(self) -> None:
        for prefix in ("trend_", "volatility_", "volume_", "regime_"):
            label, _ = describe(f"{prefix}mystery_metric")
            assert not label.startswith(prefix)

    def test_percentage_features_are_formatted_with_a_sign(self) -> None:
        # ROC arrives ALREADY in percent — the formula in technical.py ends in
        # "* 100" — so it is passed through at scale 1.0. These assertions used
        # to expect a second x100, which is exactly the bug that rendered a
        # -11.9% move as "-1193.0%".
        _, display = describe("momentum_roc_6m")
        assert display.format(18.2) == "+18.2%"

    def test_negative_percentages_keep_their_sign(self) -> None:
        _, display = describe("momentum_roc_6m")
        assert display.format(-7.5) == "-7.5%"

    def test_ratio_features_are_scaled_to_percent_but_roc_is_not(self) -> None:
        """The two families must not be confused: one is a ratio, one is already %."""
        _, ratio = describe("trend_price_vs_sma200")   # a ratio, needs x100
        _, already = describe("momentum_roc_6m")       # already percent
        assert ratio.format(0.05) == "+5.0%"
        assert already.format(0.05) == "+0.1%"

    def test_non_finite_values_say_unavailable(self) -> None:
        _, display = describe("momentum_roc_6m")
        assert display.format(float("nan")) == "unavailable"


class TestFactsFromShap:
    def test_ranked_by_absolute_influence(self) -> None:
        """Ranking by signed value would list every bullish driver first and
        bury the reason a call is weak — usually the more useful half."""
        facts = facts_from_shap(
            np.array([0.01, -0.20, 0.05]),
            ["a", "b", "c"],
            np.array([1.0, 2.0, 3.0]),
            top_n=3,
        )
        assert [f.feature for f in facts] == ["b", "c", "a"]

    def test_ranks_are_sequential_from_one(self) -> None:
        facts = facts_from_shap(
            np.array([0.3, -0.2, 0.1]), ["a", "b", "c"], np.array([1.0, 2.0, 3.0])
        )
        assert [f.rank for f in facts] == [1, 2, 3]

    def test_top_n_is_respected(self) -> None:
        facts = facts_from_shap(
            np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
            list("abcde"),
            np.arange(5, dtype="float64"),
            top_n=3,
        )
        assert len(facts) == 3

    def test_direction_follows_the_sign(self) -> None:
        facts = facts_from_shap(
            np.array([0.3, -0.3]), ["up", "down"], np.array([1.0, 2.0])
        )
        assert facts[0].direction is Direction.SUPPORTS_UP
        assert facts[1].direction is Direction.SUPPORTS_DOWN

    def test_negligible_contributions_are_dropped(self) -> None:
        """Below the threshold a contribution is indistinguishable from
        rounding, and showing it as a driver is false precision."""
        facts = facts_from_shap(
            np.array([0.5, MIN_ABS_SHAP / 10]),
            ["real", "noise"],
            np.array([1.0, 2.0]),
        )
        assert [f.feature for f in facts] == ["real"]

    def test_mismatched_lengths_raise_with_both_counts(self) -> None:
        with pytest.raises(ValueError, match="3 SHAP values for 2 features"):
            facts_from_shap(np.array([0.1, 0.2, 0.3]), ["a", "b"], np.array([1.0, 2.0]))

    def test_facts_contain_no_prose(self) -> None:
        """THE boundary test. intelligence emits numbers and labels; the
        gateway owns every sentence."""
        facts = facts_from_shap(
            np.array([0.3, -0.2]),
            ["momentum_roc_6m", "volatility_atr_21"],
            np.array([0.18, 28.0]),
        )
        for f in facts:
            # A label is a noun phrase; a sentence has a verb and punctuation.
            assert not f.display_name.endswith(".")
            assert len(f.display_name.split()) < 10

    def test_all_zero_shap_yields_no_facts(self) -> None:
        assert facts_from_shap(np.zeros(3), list("abc"), np.zeros(3)) == ()


class TestSplitByDirection:
    def test_partitions_into_supporting_and_opposing(self) -> None:
        facts = facts_from_shap(
            np.array([0.3, -0.2, 0.1]), list("abc"), np.array([1.0, 2.0, 3.0])
        )
        up, down = split_by_direction(facts)
        assert {f.feature for f in up} == {"a", "c"}
        assert {f.feature for f in down} == {"b"}

    def test_empty_input(self) -> None:
        assert split_by_direction(()) == ((), ())


class TestShapExplainer:
    def test_falls_back_transparently_for_a_linear_model(self) -> None:
        """The fallback reports its method rather than passing linear
        attribution off as SHAP."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["f0", "f1", "f2"])
        y = (X["f0"] > 0).astype(int)
        model = LogisticRegression().fit(X, y)

        explainer = ShapExplainer(model, list(X.columns))
        assert explainer.method in {"tree_shap", "linear_fallback"}
        facts = explainer.explain_row(X, 0)
        assert all(isinstance(f, ExplanationFact) for f in facts)

    def test_a_model_with_no_coefficients_yields_zero_attribution(self) -> None:
        class Opaque:
            pass

        X = pd.DataFrame(np.ones((5, 2)), columns=["a", "b"])
        explainer = ShapExplainer(Opaque(), ["a", "b"])
        assert explainer.method == "linear_fallback"
        assert (explainer.shap_values(X) == 0).all()

    def test_method_is_always_reported(self) -> None:
        """Nothing downstream may present a fallback as SHAP."""
        class Opaque:
            pass

        assert ShapExplainer(Opaque(), ["a"]).method != "unavailable"
