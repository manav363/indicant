"""Panel CV, stacking and calibration tests.

The leakage tests are the point. A stack trained on unpurged out-of-fold
predictions reports excellent numbers and nothing in the code looks broken, so
the only defence is a test that constructs the leak deliberately and proves the
splitter refuses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelligence.calibration.reliability import (
    brier_score,
    brier_skill_score,
    calibration_report,
    expected_calibration_error,
    reliability_curve,
)
from intelligence.models.stack import (
    BaseLearnerSpec,
    baseline_comparison,
    default_base_learners,
    fit_meta_labeller,
    fit_meta_learner,
    generate_oof,
)
from intelligence.validation.panel_cv import (
    CVMode,
    PanelCVConfig,
    PurgedPanelCV,
    assert_no_date_overlap,
)


def make_panel(n_symbols: int = 12, n_dates: int = 800, seed: int = 0) -> pd.DataFrame:
    """A panel with a weak, real signal: feature f0 mildly predicts the label."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates).date
    rows = []
    for s in range(n_symbols):
        f = rng.normal(size=(n_dates, 5))
        # Deliberately weak — a strong signal would hide leakage by making
        # every configuration look good.
        logit = 0.45 * f[:, 0] + rng.normal(0, 1.0, n_dates)
        y = (logit > 0).astype("float64")
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": f"SYM{s:02d}",
                    "f0": f[:, 0], "f1": f[:, 1], "f2": f[:, 2],
                    "f3": f[:, 3], "f4": f[:, 4],
                    "label": y,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


FEATURES = ["f0", "f1", "f2", "f3", "f4"]


# ==========================================================================
# Purged panel CV — the leakage guard
# ==========================================================================


class TestPurgedPanelCV:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        return make_panel()

    @pytest.fixture
    def cv(self) -> PurgedPanelCV:
        return PurgedPanelCV(
            PanelCVConfig(n_splits=4, purge_days=20, embargo_days=10, min_train_dates=300)
        )

    def test_produces_the_requested_folds(self, cv, panel) -> None:
        assert len(list(cv.split(panel))) == 4

    def test_splits_by_date_not_by_row(self, cv, panel) -> None:
        """THE test.

        Splitting by row puts RELIANCE-on-2015-03-04 in train and
        TCS-on-2015-03-04 in test, so the model sees the market's state on a day
        it is being scored on. Every downstream metric is then fiction and
        nothing looks broken.
        """
        assert_no_date_overlap(list(cv.split(panel)), panel)

    def test_all_symbols_on_a_date_land_in_the_same_fold(self, cv, panel) -> None:
        for split in cv.split(panel):
            test_dates = panel.iloc[split.test_idx].groupby("date")["symbol"].nunique()
            full = panel.groupby("date")["symbol"].nunique()
            for d, n in test_dates.items():
                assert n == full[d], f"date {d} split across folds"

    def test_expanding_mode_never_trains_on_the_future(self, cv, panel) -> None:
        """The default mode is walk-forward. A backtest that trained on data
        after its test window is not a backtest, however well purged."""
        for split in cv.split(panel):
            assert split.train_dates[1] < split.test_dates[0]

    def test_purged_kfold_mode_uses_both_sides_of_the_window(self, panel) -> None:
        """The other legitimate mode, for generating OOF predictions where the
        goal is an unbiased generalisation estimate rather than a simulated
        strategy. Purge and embargo remove the contaminated boundary."""
        kfold = PurgedPanelCV(
            PanelCVConfig(n_splits=4, purge_days=20, embargo_days=10,
                          min_train_dates=300, mode=CVMode.PURGED_KFOLD)
        )
        splits = list(kfold.split(panel))
        # At least one early fold has training data after its test window.
        assert any(s.train_dates[1] > s.test_dates[1] for s in splits)
        # And it is still leak-free.
        assert_no_date_overlap(splits, panel)

    def test_kfold_trains_on_more_data_than_expanding(self, panel) -> None:
        common = dict(n_splits=4, purge_days=20, embargo_days=10, min_train_dates=300)
        expanding = sum(
            s.n_train
            for s in PurgedPanelCV(PanelCVConfig(**common, mode=CVMode.EXPANDING)).split(panel)
        )
        kfold = sum(
            s.n_train
            for s in PurgedPanelCV(
                PanelCVConfig(**common, mode=CVMode.PURGED_KFOLD)
            ).split(panel)
        )
        assert kfold > expanding

    def test_purge_removes_rows_before_the_test_window(self, cv, panel) -> None:
        """Training rows whose labels resolve inside the test window."""
        for split in cv.split(panel):
            assert split.n_purged > 0

    def test_embargo_removes_rows_after_the_test_window_in_kfold(self, panel) -> None:
        """The half-measure to avoid: purge closes the leak going forward and
        leaves the one coming back. Features are rolling functions of the recent
        past, so a row just after the test window was partly computed from it.

        Only meaningful in PURGED_KFOLD — see the expanding-mode test below.
        """
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=15,
                          min_train_dates=300, mode=CVMode.PURGED_KFOLD)
        )
        assert any(s.n_embargoed > 0 for s in cv.split(panel))

    def test_embargo_is_a_no_op_in_expanding_mode(self, panel) -> None:
        """Walk-forward already excludes everything after the test window, so
        the embargo has nothing left to remove. Reporting a non-zero count would
        imply rows were dropped that were never eligible."""
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=30,
                          min_train_dates=300, mode=CVMode.EXPANDING)
        )
        assert all(s.n_embargoed == 0 for s in cv.split(panel))

    def test_zero_embargo_admits_more_training_data_in_kfold(self, panel) -> None:
        common = dict(n_splits=3, purge_days=20, min_train_dates=300,
                      mode=CVMode.PURGED_KFOLD)
        n_with = sum(
            s.n_train
            for s in PurgedPanelCV(PanelCVConfig(**common, embargo_days=30)).split(panel)
        )
        n_without = sum(
            s.n_train
            for s in PurgedPanelCV(PanelCVConfig(**common, embargo_days=0)).split(panel)
        )
        assert n_with < n_without

    def test_purged_rows_are_in_neither_train_nor_test(self, cv, panel) -> None:
        for split in cv.split(panel):
            assert not set(split.train_idx) & set(split.test_idx)

    def test_empty_panel_raises(self, cv) -> None:
        with pytest.raises(ValueError, match="empty panel"):
            list(cv.split(pd.DataFrame()))

    def test_too_short_panel_raises_with_the_numbers(self, cv) -> None:
        with pytest.raises(ValueError, match="need at least"):
            list(cv.split(make_panel(n_dates=100)))

    def test_missing_date_column_raises(self, cv, panel) -> None:
        with pytest.raises(ValueError, match="no 'date' column"):
            list(cv.split(panel.drop(columns=["date"])))

    def test_overlap_assertion_catches_a_deliberate_leak(self, panel) -> None:
        """Prove the guard works by handing it a leak."""
        from intelligence.validation.panel_cv import PanelSplit

        bad = PanelSplit(
            fold=0,
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([1, 2, 3]),  # overlaps train
            train_dates=(panel["date"].iloc[0], panel["date"].iloc[2]),
            test_dates=(panel["date"].iloc[1], panel["date"].iloc[3]),
        )
        with pytest.raises(AssertionError, match="both train and test"):
            assert_no_date_overlap([bad], panel)


# ==========================================================================
# L2 out-of-fold generation
# ==========================================================================


def _fast_learners() -> list[BaseLearnerSpec]:
    """Two cheap learners so the suite stays fast; the roster is tested
    separately."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    return [
        BaseLearnerSpec("elasticnet", lambda: LogisticRegression(max_iter=500),
                        needs_scaling=True),
        BaseLearnerSpec("tree", lambda: DecisionTreeClassifier(max_depth=4,
                                                              random_state=0)),
    ]


class TestOutOfFold:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        return make_panel()

    @pytest.fixture
    def cv(self) -> PurgedPanelCV:
        return PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=10, min_train_dates=300)
        )

    def test_produces_a_column_per_learner(self, panel, cv) -> None:
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        assert list(oof.predictions.columns) == ["elasticnet", "tree"]

    def test_predictions_are_probabilities(self, panel, cv) -> None:
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        valid = oof.predictions.dropna()
        assert (valid >= 0).all().all() and (valid <= 1).all().all()

    def test_rows_outside_every_test_fold_stay_null(self, panel, cv) -> None:
        """Early training-only rows have no OOF prediction. Filling them with
        0.5 would train the meta-learner on fabricated inputs."""
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        assert oof.coverage < 1.0
        assert oof.predictions.iloc[0].isna().all()

    def test_every_covered_row_belongs_to_exactly_one_fold(self, panel, cv) -> None:
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        covered = oof.covered_mask
        assert (oof.fold_of_row[covered] >= 0).all()

    def test_oof_predictions_are_genuinely_out_of_sample(self, panel, cv) -> None:
        """A model that had seen its own test rows would score far better than
        this. Weak signal in, weak-but-real AUC out."""
        from sklearn.metrics import roc_auc_score

        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        mask = oof.covered_mask
        auc = roc_auc_score(panel["label"][mask], oof.predictions["elasticnet"][mask])
        assert 0.55 < auc < 0.85, f"AUC {auc:.3f} — too low to be signal, too high to be honest"

    def test_too_short_panel_raises_before_training_anything(self, panel) -> None:
        cv = PurgedPanelCV(PanelCVConfig(n_splits=5, min_train_dates=5000))
        with pytest.raises(ValueError, match="need at least"):
            generate_oof(panel[FEATURES], panel["label"], cv, panel,
                         learners=_fast_learners())

    def test_default_roster_includes_the_baseline(self) -> None:
        names = {s.name for s in default_base_learners()}
        assert "elasticnet" in names
        assert len(names) >= 3


# ==========================================================================
# L3 meta-learner and L5 meta-labeller
# ==========================================================================


class TestMetaLearner:
    @pytest.fixture
    def fitted(self):
        panel = make_panel()
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=10, min_train_dates=300)
        )
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        return panel, oof, fit_meta_learner(oof, panel["label"])

    def test_meta_oof_is_a_probability(self, fitted) -> None:
        _, _, stack = fitted
        valid = stack.meta_oof.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_meta_learner_only_sees_covered_rows(self, fitted) -> None:
        _, oof, stack = fitted
        assert stack.n_train == int(oof.covered_mask.sum())

    def test_uncovered_rows_have_no_meta_prediction(self, fitted) -> None:
        _, oof, stack = fitted
        assert stack.meta_oof[~oof.covered_mask].isna().all()

    def test_refuses_to_fit_on_too_few_covered_rows(self) -> None:
        """Rather than fitting a meta-learner on 40 rows and reporting a
        number."""
        panel = make_panel(n_symbols=1, n_dates=800)
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=2, purge_days=20, embargo_days=10, min_train_dates=700)
        )
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        with pytest.raises(ValueError, match="need"):
            fit_meta_learner(oof, panel["label"])

    def test_regime_can_be_added_as_a_feature(self, fitted) -> None:
        panel, oof, _ = fitted
        regime = pd.Series(np.random.default_rng(0).integers(0, 3, len(panel)),
                           index=panel.index)
        stack = fit_meta_learner(oof, panel["label"], regime=regime)
        assert "regime" in stack.learner_names


class TestMetaLabeller:
    @pytest.fixture
    def setup(self):
        panel = make_panel()
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=10, min_train_dates=300)
        )
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        stack = fit_meta_learner(oof, panel["label"])
        return panel, stack

    def test_trains_only_on_rows_where_the_primary_fired(self, setup) -> None:
        """Training on every row would make it a second side model rather than
        a filter."""
        panel, stack = setup
        result = fit_meta_labeller(stack.meta_oof, panel["label"], panel[FEATURES])
        n_fired = int((stack.meta_oof >= 0.5).sum())
        assert result.n_train <= n_fired

    def test_base_rate_is_the_primary_accuracy_on_its_own_calls(self, setup) -> None:
        panel, stack = setup
        result = fit_meta_labeller(stack.meta_oof, panel["label"], panel[FEATURES])
        assert 0.3 < result.base_rate < 0.9

    def test_refuses_when_the_primary_barely_fired(self, setup) -> None:
        panel, stack = setup
        with pytest.raises(ValueError, match="fired on only"):
            fit_meta_labeller(stack.meta_oof, panel["label"], panel[FEATURES],
                              threshold=0.999)

    def test_refuses_when_there_is_nothing_to_learn(self, setup) -> None:
        """If the primary was uniformly right, a meta-labeller has no signal."""
        panel, stack = setup
        always_right = pd.Series(1.0, index=panel.index)
        with pytest.raises(ValueError, match="uniformly right or wrong"):
            fit_meta_labeller(stack.meta_oof, always_right, panel[FEATURES])


class TestBaselineComparison:
    def test_reports_both_sides(self) -> None:
        """Reported, never asserted. If the stack cannot beat a regularised
        linear model, that is the finding."""
        panel = make_panel()
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=3, purge_days=20, embargo_days=10, min_train_dates=300)
        )
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        stack = fit_meta_learner(oof, panel["label"])
        cmp = baseline_comparison(oof, stack.meta_oof, panel["label"])

        assert cmp["baseline_auc"] is not None
        assert cmp["stack_auc"] is not None
        assert cmp["improvement"] == pytest.approx(
            cmp["stack_auc"] - cmp["baseline_auc"]  # type: ignore[operator]
        )

    def test_returns_nulls_rather_than_a_fake_number(self) -> None:
        panel = make_panel(n_symbols=1, n_dates=400)
        cv = PurgedPanelCV(
            PanelCVConfig(n_splits=2, purge_days=10, embargo_days=5, min_train_dates=300)
        )
        oof = generate_oof(panel[FEATURES], panel["label"], cv, panel,
                           learners=_fast_learners())
        cmp = baseline_comparison(oof, pd.Series(np.nan, index=panel.index), panel["label"])
        assert cmp["stack_auc"] is None


# ==========================================================================
# L6 calibration evidence
# ==========================================================================


class TestCalibration:
    def test_perfectly_calibrated_model_sits_on_the_diagonal(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20_000)
        y = (rng.uniform(0, 1, 20_000) < p).astype("float64")
        for b in reliability_curve(y, p):
            assert abs(b.mean_predicted - b.observed_rate) < 0.05

    def test_overconfident_model_is_visibly_off_the_diagonal(self) -> None:
        """Raw boosted-tree output pushed toward 0 and 1 — the thing Platt
        scaling exists to fix."""
        rng = np.random.default_rng(1)
        true_p = rng.uniform(0.3, 0.7, 20_000)
        y = (rng.uniform(0, 1, 20_000) < true_p).astype("float64")
        overconfident = np.clip((true_p - 0.5) * 3 + 0.5, 0.01, 0.99)
        assert expected_calibration_error(y, overconfident) > 0.05

    def test_thin_bins_are_dropped(self) -> None:
        """A bin of 3 draws a wildly misleading point on a chart people read as
        evidence."""
        y = np.array([1.0, 0.0, 1.0])
        p = np.array([0.95, 0.96, 0.97])
        assert reliability_curve(y, p) == []

    def test_brier_of_a_coin_flip(self) -> None:
        y = np.array([1.0, 0.0] * 500)
        assert brier_score(y, np.full(1000, 0.5)) == pytest.approx(0.25)

    def test_brier_skill_beats_the_base_rate_when_the_model_is_informative(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype("float64")
        assert brier_skill_score(y, p) > 0

    def test_brier_skill_is_negative_for_an_anti_informative_model(self) -> None:
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype("float64")
        assert brier_skill_score(y, 1.0 - p) < 0

    def test_report_carries_everything_the_model_page_needs(self) -> None:
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype("float64")
        report = calibration_report(y, p)
        assert set(report) == {
            "bins", "brier_score", "brier_skill_score",
            "expected_calibration_error", "base_rate", "n_samples",
        }

    def test_empty_input_does_not_raise(self) -> None:
        assert reliability_curve(np.array([]), np.array([])) == []
        assert np.isnan(brier_score(np.array([]), np.array([])))
