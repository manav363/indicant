"""CPCV, Deflated Sharpe and panel permutation tests.

These cover the machinery whose entire purpose is to stop the project fooling
itself. The most important tests here assert that a *known-random* input is
correctly reported as insignificant — a validation layer that cannot detect
noise is worse than none, because it launders it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelligence.validation.cpcv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    SharpeDistribution,
    deflated_sharpe_from_trials,
    deflated_sharpe_ratio,
    evaluate_paths,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_from_returns,
)
from intelligence.validation.panel_cv import assert_no_date_overlap
from intelligence.validation.panel_permutation import (
    PanelPermutationConfig,
    PermutationResult,
    run_panel_permutation,
    shuffle_labels,
)


def make_panel(n_symbols: int = 10, n_dates: int = 600, signal: float = 0.0,
               seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """`signal=0` gives labels that are pure noise — the case the validation
    layer must correctly call insignificant."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates).date
    rows = []
    for s in range(n_symbols):
        f = rng.normal(size=(n_dates, 3))
        logit = signal * f[:, 0] + rng.normal(0, 1.0, n_dates)
        rows.append(
            pd.DataFrame({
                "date": dates, "symbol": f"S{s:02d}",
                "f0": f[:, 0], "f1": f[:, 1], "f2": f[:, 2],
                "label": (logit > 0).astype("float64"),
            })
        )
    panel = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)
    return panel, panel["label"]


# ==========================================================================
# CPCV
# ==========================================================================


class TestCombinatorialPurgedCV:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        return make_panel()[0]

    def test_combination_count_matches_the_formula(self) -> None:
        # C(6, 2) = 15
        assert CPCVConfig(n_groups=6, n_test_groups=2).n_combinations == 15

    def test_path_count_matches_the_formula(self) -> None:
        # C(5, 1) = 5 complete backtest paths
        assert CPCVConfig(n_groups=6, n_test_groups=2).n_paths == 5

    def test_generates_every_combination(self, panel) -> None:
        cv = CombinatorialPurgedCV(
            CPCVConfig(n_groups=6, n_test_groups=2, purge_days=10,
                       embargo_days=5, min_train_rows=100)
        )
        assert len(list(cv.split(panel))) == 15

    def test_no_date_appears_in_both_train_and_test(self, panel) -> None:
        """Same leak as the walk-forward splitter, and it matters more here:
        with non-contiguous test groups there are several boundaries, and
        missing any one of them leaks."""
        cv = CombinatorialPurgedCV(
            CPCVConfig(n_groups=6, n_test_groups=2, purge_days=10,
                       embargo_days=5, min_train_rows=100)
        )
        assert_no_date_overlap(list(cv.split(panel)), panel)

    def test_purge_and_embargo_are_applied_at_every_boundary(self, panel) -> None:
        cv = CombinatorialPurgedCV(
            CPCVConfig(n_groups=6, n_test_groups=2, purge_days=20,
                       embargo_days=20, min_train_rows=100)
        )
        for split in cv.split(panel):
            assert split.n_purged > 0

    def test_more_purging_leaves_less_training_data(self, panel) -> None:
        common = dict(n_groups=6, n_test_groups=2, min_train_rows=100)
        light = sum(s.n_train for s in CombinatorialPurgedCV(
            CPCVConfig(**common, purge_days=5, embargo_days=5)).split(panel))
        heavy = sum(s.n_train for s in CombinatorialPurgedCV(
            CPCVConfig(**common, purge_days=40, embargo_days=40)).split(panel))
        assert heavy < light

    def test_empty_panel_raises(self) -> None:
        with pytest.raises(ValueError, match="empty panel"):
            list(CombinatorialPurgedCV().split(pd.DataFrame()))

    def test_short_panel_raises_with_the_numbers(self) -> None:
        short = make_panel(n_dates=8)[0]
        with pytest.raises(ValueError, match="need at least"):
            list(CombinatorialPurgedCV(CPCVConfig(n_groups=6)).split(short))


class TestSharpeDistribution:
    def test_distinguishes_stable_from_volatile_at_the_same_mean(self) -> None:
        """The reason CPCV exists. Single-path walk-forward reports both of
        these as '0.4'."""
        stable = SharpeDistribution(np.array([0.38, 0.40, 0.42, 0.39, 0.41]), 5)
        volatile = SharpeDistribution(np.array([-0.5, 1.3, 0.1, 0.9, 0.2]), 5)
        assert stable.mean == pytest.approx(volatile.mean, abs=0.1)
        assert stable.std < volatile.std

    def test_prob_negative_counts_losing_paths(self) -> None:
        dist = SharpeDistribution(np.array([-0.2, 0.3, -0.1, 0.5, 0.4]), 5)
        assert dist.prob_negative == pytest.approx(0.4)

    def test_summary_has_every_field_the_model_page_needs(self) -> None:
        dist = SharpeDistribution(np.array([0.1, 0.2, 0.3]), 3)
        assert set(dist.summary()) == {
            "mean", "std", "median", "p05", "p95", "prob_negative", "n_paths"
        }

    def test_empty_distribution_is_nan_not_zero(self) -> None:
        """Zero would read as 'measured, and it is zero'."""
        assert np.isnan(SharpeDistribution(np.array([]), 0).mean)

    def test_evaluate_paths_computes_sharpes(self) -> None:
        rng = np.random.default_rng(0)
        paths = [rng.normal(0.001, 0.01, 252) for _ in range(5)]
        assert evaluate_paths(paths).sharpes.size == 5


class TestSharpeFromReturns:
    def test_positive_drift_gives_positive_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        assert sharpe_from_returns(rng.normal(0.002, 0.01, 1000)) > 0

    def test_constant_returns_are_nan_not_astronomical(self) -> None:
        """Regression. np.std of a constant array returns ~1e-18 rather than
        exactly 0, so an `sd == 0` guard never fires and the Sharpe came back
        as ~1e17. A flat equity curve — a strategy that never traded, or one
        position held throughout — would have reported a spectacular Sharpe with
        nothing to flag it.
        """
        assert np.isnan(sharpe_from_returns([0.01] * 100))
        assert np.isnan(sharpe_from_returns([0.0] * 100))

    def test_too_few_observations_is_nan(self) -> None:
        assert np.isnan(sharpe_from_returns([0.01]))


# ==========================================================================
# Deflated Sharpe
# ==========================================================================


class TestDeflatedSharpe:
    VAR = 0.04  # trial Sharpes with sd 0.2 — a realistic spread across configs

    def test_more_trials_lowers_the_deflated_sharpe(self) -> None:
        """The answer to 'did you just pick the best of 200 runs?' — which is
        exactly what v1's registry shows: 85 logged runs, best reported."""
        few = deflated_sharpe_ratio(0.6, n_trials=2, n_observations=1000,
                                    variance_of_trial_sharpes=self.VAR)
        many = deflated_sharpe_ratio(0.6, n_trials=500, n_observations=1000,
                                     variance_of_trial_sharpes=self.VAR)
        assert many < few

    def test_wider_search_dispersion_raises_the_benchmark(self) -> None:
        """A search whose configurations produced wildly different Sharpes had
        more opportunity to get lucky."""
        tight = deflated_sharpe_ratio(0.6, n_trials=100, n_observations=1000,
                                      variance_of_trial_sharpes=0.01)
        wide = deflated_sharpe_ratio(0.6, n_trials=100, n_observations=1000,
                                     variance_of_trial_sharpes=0.25)
        assert wide < tight

    def test_a_mediocre_sharpe_found_by_wide_search_is_not_credible(self) -> None:
        assert deflated_sharpe_ratio(0.2, n_trials=1000, n_observations=500,
                                     variance_of_trial_sharpes=0.09) < 0.5

    def test_a_strong_sharpe_from_a_narrow_search_survives(self) -> None:
        assert deflated_sharpe_ratio(2.5, n_trials=2, n_observations=2000,
                                     variance_of_trial_sharpes=self.VAR) > 0.9

    def test_longer_history_raises_confidence(self) -> None:
        common = dict(n_trials=10, variance_of_trial_sharpes=self.VAR)
        short = deflated_sharpe_ratio(0.5, n_observations=60, **common)
        long = deflated_sharpe_ratio(0.5, n_observations=5000, **common)
        assert long > short

    def test_variance_has_no_default(self) -> None:
        """Regression. An optional variance defaulted to the bare
        Euler-Mascheroni term, which is dimensionless — using it as a Sharpe
        threshold assumes trial variance of 1.0, deflating EVERY realistic
        result to ~0. A function that always says 'not significant' while
        looking rigorous is worse than no function.
        """
        with pytest.raises(TypeError):
            deflated_sharpe_ratio(1.0, n_trials=10, n_observations=100)  # type: ignore[call-arg]

    def test_negative_variance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            deflated_sharpe_ratio(1.0, n_trials=10, n_observations=100,
                                  variance_of_trial_sharpes=-1.0)

    def test_degenerate_inputs_are_nan(self) -> None:
        common = dict(variance_of_trial_sharpes=self.VAR)
        assert np.isnan(deflated_sharpe_ratio(1.0, n_trials=0,
                                              n_observations=100, **common))
        assert np.isnan(deflated_sharpe_ratio(1.0, n_trials=10,
                                              n_observations=1, **common))

    def test_expected_max_grows_with_trials(self) -> None:
        assert expected_max_sharpe(100, 0.04) > expected_max_sharpe(5, 0.04)

    def test_expected_max_is_zero_without_dispersion(self) -> None:
        """No spread across configurations means no opportunity to get lucky."""
        assert expected_max_sharpe(100, 0.0) == 0.0

    def test_from_trials_uses_the_search_you_actually_ran(self) -> None:
        rng = np.random.default_rng(0)
        # 85 runs of noise, mirroring v1's registry.
        noise_trials = rng.normal(0.0, 0.2, 85)
        assert deflated_sharpe_from_trials(noise_trials, n_observations=1000) < 0.95

    def test_from_trials_needs_at_least_two(self) -> None:
        assert np.isnan(deflated_sharpe_from_trials([0.5], n_observations=1000))


class TestProbabilisticSharpe:
    def test_higher_sharpe_gives_higher_probability(self) -> None:
        low = probabilistic_sharpe_ratio(0.2, n_observations=500)
        high = probabilistic_sharpe_ratio(1.5, n_observations=500)
        assert high > low

    def test_negative_skew_reduces_confidence(self) -> None:
        """Financial returns are skewed and fat-tailed, which makes the naive
        t-statistic on a Sharpe too generous.

        Uses a small Sharpe and short history: PSR saturates at 1.0 once the
        t-statistic is large, so the correction is only visible in the region
        where the answer is actually in doubt.
        """
        normal = probabilistic_sharpe_ratio(0.08, n_observations=60, skew=0.0)
        skewed = probabilistic_sharpe_ratio(0.08, n_observations=60, skew=-1.5)
        assert skewed < normal

    def test_fat_tails_reduce_confidence(self) -> None:
        thin = probabilistic_sharpe_ratio(0.08, n_observations=60, kurtosis=3.0)
        fat = probabilistic_sharpe_ratio(0.08, n_observations=60, kurtosis=12.0)
        assert fat < thin

    def test_saturates_for_an_overwhelming_track_record(self) -> None:
        """Documented, not a defect: a Sharpe of 1.0 over 500 observations is
        beyond doubt, and PSR correctly reports ~1."""
        assert probabilistic_sharpe_ratio(1.0, n_observations=500) > 0.999


class TestMinTrackRecordLength:
    def test_a_weak_sharpe_needs_a_long_record(self) -> None:
        """Often the most sobering number in a report."""
        assert min_track_record_length(0.3) > min_track_record_length(1.5)

    def test_a_sharpe_at_the_benchmark_is_never_distinguishable(self) -> None:
        assert min_track_record_length(0.0) == float("inf")


# ==========================================================================
# Panel permutation — must correctly report noise as noise
# ==========================================================================


class TestShuffleLabels:
    def test_within_date_preserves_each_day_distribution(self) -> None:
        """The right null for a cross-sectional model: it asks 'could this
        stock-picking be random?', not 'could the market's direction be
        random?'. Only the first is a question about the model."""
        panel, y = make_panel(n_symbols=10, n_dates=50)
        rng = np.random.default_rng(0)
        shuffled = shuffle_labels(y, panel["date"], rng, within_date=True)

        before = y.groupby(panel["date"]).sum()
        after = shuffled.groupby(panel["date"]).sum()
        pd.testing.assert_series_equal(before, after, check_names=False)

    def test_within_date_actually_moves_labels(self) -> None:
        panel, y = make_panel(n_symbols=10, n_dates=50)
        shuffled = shuffle_labels(y, panel["date"], np.random.default_rng(0),
                                  within_date=True)
        assert (shuffled.to_numpy() != y.to_numpy()).any()

    def test_global_shuffle_does_not_preserve_daily_totals(self) -> None:
        panel, y = make_panel(n_symbols=10, n_dates=50)
        shuffled = shuffle_labels(y, panel["date"], np.random.default_rng(0),
                                  within_date=False)
        before = y.groupby(panel["date"]).sum()
        after = shuffled.groupby(panel["date"]).sum()
        assert not before.equals(after)

    def test_shuffling_preserves_the_overall_label_count(self) -> None:
        panel, y = make_panel()
        for within in (True, False):
            shuffled = shuffle_labels(y, panel["date"], np.random.default_rng(1),
                                      within_date=within)
            assert shuffled.sum() == y.sum()


class TestPermutationResult:
    def test_p_value_never_reaches_zero(self) -> None:
        """Without the +1 correction a clean sweep reports p=0, claiming
        infinite significance from a finite experiment."""
        result = PermutationResult(
            actual_score=10.0, null_scores=np.zeros(100),
            n_permutations=100, within_date=True,
        )
        assert result.p_value == pytest.approx(1 / 101)
        assert result.p_value > 0

    def test_p_value_is_one_when_every_null_matches(self) -> None:
        result = PermutationResult(
            actual_score=0.0, null_scores=np.zeros(100),
            n_permutations=100, within_date=True,
        )
        assert result.p_value == pytest.approx(1.0)

    def test_z_score_says_how_far_not_just_whether(self) -> None:
        rng = np.random.default_rng(0)
        result = PermutationResult(
            actual_score=2.0, null_scores=rng.normal(0, 1, 500),
            n_permutations=500, within_date=True,
        )
        assert 1.5 < result.z_score < 2.5

    def test_failed_permutations_are_excluded_not_counted_as_losses(self) -> None:
        """Counting a crashed permutation as a loss for the null would bias the
        p-value toward significance."""
        nulls = np.array([0.1, 0.2, np.nan, np.nan, 0.3])
        result = PermutationResult(actual_score=0.25, null_scores=nulls,
                                   n_permutations=5, within_date=True)
        # Only 3 valid nulls, 1 of which beats 0.25.
        assert result.p_value == pytest.approx(2 / 4)

    def test_verdict_states_a_null_result_plainly(self) -> None:
        result = PermutationResult(
            actual_score=0.0, null_scores=np.random.default_rng(0).normal(0, 1, 200),
            n_permutations=200, within_date=True,
        )
        verdict = result.verdict()
        assert "does NOT survive" in verdict
        assert "not distinguishable from chance" in verdict

    def test_verdict_states_a_positive_result_plainly(self) -> None:
        result = PermutationResult(
            actual_score=100.0, null_scores=np.zeros(200),
            n_permutations=200, within_date=True,
        )
        assert "survives label shuffling" in result.verdict()

    def test_summary_has_every_registry_field(self) -> None:
        result = PermutationResult(actual_score=0.1, null_scores=np.zeros(10),
                                   n_permutations=10, within_date=True)
        assert set(result.summary()) == {
            "actual_score", "p_value", "null_mean", "null_std", "null_95pct",
            "z_score", "n_permutations", "significant_at_05",
        }


class TestRunPanelPermutation:
    def test_pure_noise_is_correctly_reported_as_insignificant(self) -> None:
        """THE test for this module.

        A validation layer that cannot detect noise is worse than none — it
        launders it. Labels here are independent of the features by
        construction, so the p-value must not be significant.
        """
        panel, y = make_panel(n_symbols=8, n_dates=300, signal=0.0, seed=1)

        def score(p: pd.DataFrame, labels: pd.Series, seed: int) -> float:
            # Correlation between a feature and the labels. With no real
            # relationship this is noise around zero, exactly like the null.
            return float(abs(np.corrcoef(p["f0"], labels)[0, 1]))

        from intelligence.validation.panel_cv import PanelCVConfig, PurgedPanelCV

        result = run_panel_permutation(
            panel, y, score,
            PurgedPanelCV(PanelCVConfig(n_splits=2, min_train_dates=100)),
            PanelPermutationConfig(n_permutations=60, random_state=0),
        )
        assert not result.is_significant()

    def test_a_planted_signal_is_detected(self) -> None:
        """The complement: if the test cannot detect a signal that IS there,
        a null result would mean nothing."""
        panel, y = make_panel(n_symbols=8, n_dates=300, signal=3.0, seed=2)

        def score(p: pd.DataFrame, labels: pd.Series, seed: int) -> float:
            return float(abs(np.corrcoef(p["f0"], labels)[0, 1]))

        from intelligence.validation.panel_cv import PanelCVConfig, PurgedPanelCV

        result = run_panel_permutation(
            panel, y, score,
            PurgedPanelCV(PanelCVConfig(n_splits=2, min_train_dates=100)),
            PanelPermutationConfig(n_permutations=60, random_state=0),
        )
        assert result.is_significant()

    def test_a_crashing_score_function_yields_nan_nulls(self) -> None:
        panel, y = make_panel(n_symbols=4, n_dates=200)
        calls = {"n": 0}

        def flaky(p: pd.DataFrame, labels: pd.Series, seed: int) -> float:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("boom")
            return 0.5

        from intelligence.validation.panel_cv import PanelCVConfig, PurgedPanelCV

        result = run_panel_permutation(
            panel, y, flaky,
            PurgedPanelCV(PanelCVConfig(n_splits=2, min_train_dates=100)),
            PanelPermutationConfig(n_permutations=5, random_state=0),
        )
        assert np.isnan(result.null_scores).all()
        assert np.isnan(result.p_value)
