"""L0 panel and L1 labelling tests.

The no-lookahead tests here are the ones that matter. Every other test in this
file checks that a number is right; these check that a number could have been
known at the time it is used, which is the property that decides whether the
whole project is measuring anything real.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intelligence.labeling.sample_weights import (
    average_uniqueness,
    compute_weights,
    concurrency,
    effective_sample_size,
    return_attribution_weights,
    sequential_bootstrap,
    time_decay,
)
from intelligence.labeling.triple_barrier import (
    BarrierTouched,
    TripleBarrierConfig,
    apply_triple_barrier,
    label_distribution,
    label_panel,
    realised_volatility,
    to_binary_side,
)
from intelligence.panel.builder import (
    MIN_SYMBOLS_FOR_RANK,
    PanelBuilder,
    PanelConfig,
    panel_coverage,
)


def price_frame(
    symbol: str = "AAA",
    n: int = 400,
    start: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """A synthetic OHLCV frame in canonical lake shape."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n).date
    steps = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "series": "EQ",
            "open": close * 0.999,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "prev_close": np.r_[close[0], close[:-1]],
            "volume": rng.integers(500_000, 2_000_000, n),
            "turnover": close * 1_000_000,
            "trades": 5_000,
            "delivery_qty": 400_000,
            "delivery_pct": 40.0,
            "isin": None,
        }
    )


def multi_symbol_frame(n_symbols: int = 25, n: int = 400) -> pd.DataFrame:
    return pd.concat(
        [price_frame(f"SYM{i:02d}", n=n, start=100.0 + i, seed=i) for i in range(n_symbols)],
        ignore_index=True,
    )


# ==========================================================================
# L0 — panel construction
# ==========================================================================


class TestPanelBuilder:
    @pytest.fixture
    def builder(self) -> PanelBuilder:
        return PanelBuilder(client=None)  # type: ignore[arg-type]

    def test_builds_a_date_symbol_panel(self, builder: PanelBuilder) -> None:
        result = builder.build_from_prices(multi_symbol_frame(n_symbols=25))
        assert result.n_symbols == 25
        assert result.n_rows > 0
        assert len(result.feature_columns) > 40

    def test_short_history_symbols_are_skipped_with_a_reason(
        self, builder: PanelBuilder
    ) -> None:
        prices = pd.concat(
            [price_frame("LONG", n=400), price_frame("SHORT", n=50, seed=9)],
            ignore_index=True,
        )
        result = builder.build_from_prices(prices)
        assert result.n_symbols == 1
        assert "SHORT" in result.skipped
        assert "50 rows" in result.skipped["SHORT"]

    def test_raw_price_columns_are_not_features(self, builder: PanelBuilder) -> None:
        """A Rs 50 stock and a Rs 5,000 stock are not 100x different in any way
        a model should learn."""
        result = builder.build_from_prices(multi_symbol_frame())
        for raw in ("close", "open", "volume", "turnover"):
            assert raw not in result.feature_columns

    def test_identity_columns_are_not_features(self, builder: PanelBuilder) -> None:
        result = builder.build_from_prices(multi_symbol_frame())
        for ident in ("symbol", "date", "series"):
            assert ident not in result.feature_columns

    def test_indicators_do_not_roll_across_the_symbol_boundary(
        self, builder: PanelBuilder
    ) -> None:
        """The single most dangerous panel bug: computing a rolling mean over a
        stacked frame mixes one symbol's prices into another's indicator.

        AAA alone and AAA-inside-a-panel must produce identical features.
        """
        alone = builder.build_from_prices(price_frame("AAA", n=400))
        together = builder.build_from_prices(
            pd.concat([price_frame("AAA", n=400), price_frame("BBB", n=400, seed=5)],
                      ignore_index=True)
        )
        a_only = alone.frame[alone.frame.symbol == "AAA"].reset_index(drop=True)
        a_in_panel = together.frame[together.frame.symbol == "AAA"].reset_index(drop=True)

        assert len(a_only) == len(a_in_panel)
        for col in ("trend_sma_50", "momentum_rsi_14"):
            if col in a_only.columns:
                pd.testing.assert_series_equal(
                    a_only[col], a_in_panel[col], check_names=False
                )

    def test_empty_prices_yield_an_empty_panel(self, builder: PanelBuilder) -> None:
        assert builder.build_from_prices(pd.DataFrame()).n_rows == 0

    def test_coverage_reports_symbols_per_date(self, builder: PanelBuilder) -> None:
        result = builder.build_from_prices(multi_symbol_frame(n_symbols=25))
        cov = panel_coverage(result.frame)
        assert set(cov.columns) == {"date", "n_symbols"}
        assert cov["n_symbols"].max() == 25


class TestCrossSectionalRanks:
    @pytest.fixture
    def builder(self) -> PanelBuilder:
        return PanelBuilder(client=None)  # type: ignore[arg-type]

    def test_ranks_are_added(self, builder: PanelBuilder) -> None:
        result = builder.build_from_prices(multi_symbol_frame(n_symbols=25))
        assert any(c.endswith("_xs") for c in result.frame.columns)

    def test_ranks_are_computed_within_a_date(self, builder: PanelBuilder) -> None:
        """THE leakage test for L0.

        Ranking the whole panel at once leaks the future distribution into every
        historical row. A per-date rank must span [0, 1] on each date
        independently, and the values on one date must not depend on any other.
        """
        result = builder.build_from_prices(multi_symbol_frame(n_symbols=25))
        col = next(c for c in result.frame.columns if c.endswith("_xs"))

        for _, group in result.frame.groupby("date"):
            values = group[col].dropna()
            if len(values) < MIN_SYMBOLS_FOR_RANK:
                continue
            assert values.min() >= 0.0
            assert values.max() <= 1.0
            # A full percentile rank over k symbols has mean ~0.5 regardless of
            # the underlying values — that is what makes it cross-sectional.
            assert abs(values.mean() - 0.5) < 0.15

    def test_truncating_history_does_not_change_earlier_ranks(
        self, builder: PanelBuilder
    ) -> None:
        """Direct no-lookahead check: ranks for 2018 must be identical whether
        or not 2019 exists in the frame."""
        full = multi_symbol_frame(n_symbols=25, n=400)
        cutoff = sorted(full["date"].unique())[300]
        truncated = full[full["date"] <= cutoff]

        r_full = builder.build_from_prices(full)
        r_trunc = builder.build_from_prices(truncated)
        col = next(c for c in r_full.frame.columns if c.endswith("_xs"))

        merged = r_full.frame.merge(
            r_trunc.frame, on=["date", "symbol"], suffixes=("_full", "_trunc")
        )
        assert not merged.empty
        pd.testing.assert_series_equal(
            merged[f"{col}_full"], merged[f"{col}_trunc"], check_names=False
        )

    def test_ranks_are_null_when_too_few_symbols(self, builder: PanelBuilder) -> None:
        """With 3 symbols the percentiles are 0, 0.5, 1 regardless of value —
        an artefact of the count, not a market fact."""
        result = builder.build_from_prices(multi_symbol_frame(n_symbols=3))
        xs_cols = [c for c in result.frame.columns if c.endswith("_xs")]
        assert xs_cols
        assert result.frame[xs_cols].isna().all().all()

    def test_ranks_can_be_disabled(self, builder: PanelBuilder) -> None:
        result = builder.build_from_prices(
            multi_symbol_frame(), PanelConfig(add_cross_sectional=False)
        )
        assert not any(c.endswith("_xs") for c in result.frame.columns)


# ==========================================================================
# L1 — triple-barrier labelling
# ==========================================================================


class TestRealisedVolatility:
    def test_past_only(self) -> None:
        """Vol at t must not move when data after t changes.

        A forward-looking vol estimate leaks into the *labels*, which is the
        worst possible place — the model fits a target that already knows.
        """
        close = pd.Series(price_frame(n=300)["close"].to_numpy())
        vol_full = realised_volatility(close)
        vol_trunc = realised_volatility(close.iloc[:200])
        pd.testing.assert_series_equal(
            vol_full.iloc[:200], vol_trunc, check_names=False
        )

    def test_flat_series_has_zero_volatility(self) -> None:
        assert realised_volatility(pd.Series([100.0] * 100)).dropna().eq(0).all()


class TestTripleBarrier:
    def _rising(self, n: int = 200, seed: int = 0) -> pd.Series:
        """Strong upward drift with realistic noise.

        A *constant* drift would have zero return volatility, which makes
        vol-scaled barriers zero-width — the labeller correctly refuses those,
        so a noiseless fixture would test nothing.
        """
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.01, 0.002, n)
        return pd.Series(100 * np.exp(np.cumsum(steps)))

    def _falling(self, n: int = 200, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        steps = rng.normal(-0.01, 0.002, n)
        return pd.Series(100 * np.exp(np.cumsum(steps)))

    def test_steady_rise_hits_the_profit_take(self) -> None:
        out = apply_triple_barrier(
            self._rising(), TripleBarrierConfig(horizon_days=20, vol_window=10)
        )
        labels = out["label"].dropna()
        assert (labels == BarrierTouched.PROFIT_TAKE).mean() > 0.9

    def test_steady_fall_hits_the_stop_loss(self) -> None:
        out = apply_triple_barrier(
            self._falling(), TripleBarrierConfig(horizon_days=20, vol_window=10)
        )
        labels = out["label"].dropna()
        assert (labels == BarrierTouched.STOP_LOSS).mean() > 0.9

    def test_unreachable_barriers_force_time_exits(self) -> None:
        """With barriers far beyond any move in the window, every exit is a
        time exit."""
        out = apply_triple_barrier(
            self._rising(),
            TripleBarrierConfig(horizon_days=5, vol_window=10,
                                pt_multiple=500.0, sl_multiple=500.0),
        )
        labels = out["label"].dropna()
        assert len(labels) > 100
        assert (labels == BarrierTouched.TIME).all()

    def test_barriers_are_scaled_to_LOCAL_volatility(self) -> None:
        """The property vol-scaling exists for, stated directly.

        The same drift that sits inside the barriers during a normal stretch
        breaches them during a calm one, because the barriers narrow with local
        volatility. This is intended: a 5% move is unremarkable in a volatile
        name and a large event in a quiet one, and a fixed-width barrier cannot
        express that difference.
        """
        cfg = TripleBarrierConfig(horizon_days=5, vol_window=10,
                                  pt_multiple=50.0, sl_multiple=50.0)
        series = self._rising()
        out = apply_triple_barrier(series, cfg)

        labelled = out.dropna(subset=["label"])
        hit = labelled[labelled["label"] != BarrierTouched.TIME]
        timed_out = labelled[labelled["label"] == BarrierTouched.TIME]

        assert not hit.empty, "fixture produced no barrier hits to compare"
        # Rows that hit a barrier sat in measurably calmer conditions.
        assert hit["volatility"].median() < timed_out["volatility"].median()

    def test_last_horizon_rows_are_unlabelled(self) -> None:
        """No complete future window. Labelling them would be inventing the
        answer — the same trap v1 documented for fixed-horizon labels."""
        n, horizon = 200, 20
        out = apply_triple_barrier(
            self._rising(n), TripleBarrierConfig(horizon_days=horizon, vol_window=10)
        )
        assert out["label"].iloc[-horizon:].isna().all()

    def test_barriers_scale_with_volatility(self) -> None:
        """The property that makes labels comparable across the cross-section.

        A calm series and a wild one with the same drift should not both be
        labelled the same way — the wild one's barriers are wider.
        """
        cfg = TripleBarrierConfig(horizon_days=30, vol_window=20)
        calm = apply_triple_barrier(
            pd.Series(price_frame(n=300, vol=0.002, seed=1)["close"].to_numpy()), cfg
        )
        wild = apply_triple_barrier(
            pd.Series(price_frame(n=300, vol=0.05, seed=1)["close"].to_numpy()), cfg
        )
        assert wild["volatility"].mean() > calm["volatility"].mean() * 5

    def test_exit_index_is_after_entry(self) -> None:
        out = apply_triple_barrier(
            self._rising(), TripleBarrierConfig(horizon_days=20, vol_window=10)
        )
        labelled = out[out["exit_index"] >= 0]
        assert (labelled["exit_index"].to_numpy() > np.arange(len(out))[
            out["exit_index"] >= 0
        ]).all()

    def test_exit_never_exceeds_the_horizon(self) -> None:
        horizon = 20
        out = apply_triple_barrier(
            self._rising(), TripleBarrierConfig(horizon_days=horizon, vol_window=10)
        )
        mask = out["exit_index"] >= 0
        assert mask.any(), "fixture produced no labels at all"
        offsets = out.loc[mask, "exit_index"].to_numpy() - np.flatnonzero(mask)
        assert offsets.max() <= horizon

    def test_zero_volatility_rows_are_left_unlabelled(self) -> None:
        """A flat series has zero-width barriers; labelling it would produce a
        profit-take on a price that never moved."""
        out = apply_triple_barrier(
            pd.Series([100.0] * 100), TripleBarrierConfig(horizon_days=10, vol_window=5)
        )
        assert out["label"].isna().all()

    def test_binary_time_barrier_uses_the_sign(self) -> None:
        cfg = TripleBarrierConfig(horizon_days=5, vol_window=10,
                                  pt_multiple=50.0, sl_multiple=50.0,
                                  binary_time_barrier=True)
        out = apply_triple_barrier(self._rising(), cfg)
        assert (out["label"].dropna() == 1.0).all()

    def test_to_binary_side_treats_time_exits_as_not_up(self) -> None:
        """An outcome that never reached the profit target is not a win."""
        labels = pd.Series([1.0, 0.0, -1.0, np.nan])
        assert to_binary_side(labels).tolist()[:3] == [1.0, 0.0, 0.0]

    def test_distribution_sums_to_one(self) -> None:
        out = apply_triple_barrier(
            pd.Series(price_frame(n=400)["close"].to_numpy()),
            TripleBarrierConfig(horizon_days=20, vol_window=20),
        )
        dist = label_distribution(out["label"])
        assert dist
        assert abs(sum(dist.values()) - 1.0) < 1e-9


class TestLabelPanel:
    def test_labels_every_symbol(self) -> None:
        panel = multi_symbol_frame(n_symbols=5, n=300)
        out = label_panel(panel, TripleBarrierConfig(horizon_days=20, vol_window=20))
        assert out["symbol"].nunique() == 5
        assert out["label"].notna().any()

    def test_barrier_walk_does_not_cross_symbols(self) -> None:
        """One symbol's prices must not resolve another's barriers."""
        cfg = TripleBarrierConfig(horizon_days=20, vol_window=20)
        both = label_panel(
            pd.concat([price_frame("AAA", n=300), price_frame("BBB", n=300, seed=3)],
                      ignore_index=True),
            cfg,
        )
        alone = label_panel(price_frame("AAA", n=300), cfg)

        a_both = both[both.symbol == "AAA"].sort_values("date")["label"].reset_index(drop=True)
        a_alone = alone.sort_values("date")["label"].reset_index(drop=True)
        pd.testing.assert_series_equal(a_both, a_alone, check_names=False)


# ==========================================================================
# Sample weights — the correctness fix v1 was missing
# ==========================================================================


class TestConcurrency:
    def test_counts_live_labels(self) -> None:
        # Labels 0..2, each spanning 3 bars.
        exits = np.array([2, 3, 4, -1, -1])
        conc = concurrency(5, exits)
        assert conc[2] == 3.0  # all three live at bar 2

    def test_no_labels_means_zero_concurrency(self) -> None:
        assert concurrency(5, np.array([-1] * 5)).sum() == 0.0


class TestAverageUniqueness:
    def test_isolated_label_is_fully_unique(self) -> None:
        uniq = average_uniqueness(np.array([2, -1, -1, -1]), 4)
        assert uniq[0] == pytest.approx(1.0)

    def test_overlap_reduces_uniqueness_in_proportion_to_sharing(self) -> None:
        """Labels 0 and 1 both end at bar 2, but label 0 starts at bar 0 and so
        has that bar to itself.

        Label 0 spans bars 0,1,2 with concurrency 1,2,2 -> mean(1, .5, .5) = 2/3
        Label 1 spans bars   1,2 with concurrency   2,2 -> mean(.5, .5)   = 1/2

        The later, fully-shared label is correctly worth less.
        """
        uniq = average_uniqueness(np.array([2, 2, -1]), 3)
        assert uniq[0] == pytest.approx(2 / 3)
        assert uniq[1] == pytest.approx(0.5)
        assert uniq[1] < uniq[0]

    def test_uniqueness_is_bounded(self) -> None:
        exits = np.array([5, 6, 7, 8, 9, -1, -1, -1, -1, -1])
        uniq = average_uniqueness(exits, 10)
        valid = uniq[~np.isnan(uniq)]
        assert (valid > 0).all() and (valid <= 1.0).all()


class TestEffectiveSampleSize:
    def test_equal_weights_give_full_n(self) -> None:
        assert effective_sample_size(pd.Series([1.0] * 100)) == pytest.approx(100.0)

    def test_concentrated_weights_shrink_it(self) -> None:
        """The number to quote instead of len(df). If this is 8,000 on a
        100,000-row panel, '100,000 samples' is a claim the data cannot support.
        """
        w = pd.Series([100.0] + [0.01] * 99)
        assert effective_sample_size(w) < 5.0

    def test_empty_is_zero(self) -> None:
        assert effective_sample_size(pd.Series(dtype="float64")) == 0.0


class TestReturnAttribution:
    def test_flat_stretch_weighs_less_than_a_crash(self) -> None:
        """Uniqueness says a label is distinct; it does not say it matters."""
        exits = np.array([2, 5, -1, -1, -1, -1])
        returns = np.array([0.0, 0.0, 0.0, -0.5, -0.5, 0.0])
        w = return_attribution_weights(exits, returns, 6)
        assert w[1] > w[0]


class TestTimeDecay:
    def test_disabled_by_default(self) -> None:
        uniq = np.array([1.0, 1.0, 1.0])
        assert np.allclose(time_decay(uniq, last_weight=1.0), 1.0)

    def test_older_observations_weigh_less(self) -> None:
        uniq = np.array([1.0, 1.0, 1.0, 1.0])
        decayed = time_decay(uniq, last_weight=0.5)
        assert decayed[0] < decayed[-1]


class TestComputeWeights:
    def test_weights_align_with_the_frame(self) -> None:
        panel = multi_symbol_frame(n_symbols=3, n=300)
        labelled = label_panel(panel, TripleBarrierConfig(horizon_days=20, vol_window=20))
        weights = compute_weights(labelled)
        assert len(weights) == len(labelled)
        assert weights.index.equals(labelled.index)

    def test_overlapping_labels_reduce_effective_sample_size(self) -> None:
        """The whole point. 1,000 overlapping observations do not carry 1,000
        observations' worth of evidence, and a p-value computed as though they
        did is too narrow."""
        panel = price_frame("AAA", n=400)
        labelled = label_panel(panel, TripleBarrierConfig(horizon_days=60, vol_window=20))
        weights = compute_weights(labelled).dropna()
        assert effective_sample_size(weights) < len(weights)

    def test_concurrency_is_computed_per_symbol(self) -> None:
        """Two symbols' labels overlapping in calendar time are not redundant —
        they are two genuinely different observations of the same day."""
        cfg = TripleBarrierConfig(horizon_days=20, vol_window=20)
        one = compute_weights(label_panel(price_frame("AAA", n=300), cfg)).dropna()
        two = compute_weights(
            label_panel(
                pd.concat([price_frame("AAA", n=300), price_frame("BBB", n=300, seed=4)],
                          ignore_index=True),
                cfg,
            )
        ).dropna()
        # Adding a second symbol must not dilute the first symbol's weights.
        assert abs(one.mean() - two.mean()) < 0.35

    def test_empty_frame_yields_empty_weights(self) -> None:
        assert compute_weights(pd.DataFrame()).empty


class TestSequentialBootstrap:
    def test_returns_the_requested_count(self) -> None:
        exits = np.array([2, 3, 4, 5, 6, -1, -1])
        drawn = sequential_bootstrap(exits, n_samples=5,
                                     rng=np.random.default_rng(0))
        assert len(drawn) == 5

    def test_favours_less_overlapping_observations(self) -> None:
        """Standard bootstrap draws the same future repeatedly, correlating the
        ensemble's members and shrinking its variance estimate."""
        # Index 0 is isolated; 1-4 all overlap heavily.
        exits = np.array([0, 5, 5, 5, 5, -1])
        rng = np.random.default_rng(7)
        drawn = sequential_bootstrap(exits, n_samples=200, rng=rng)
        share_isolated = np.mean(drawn == 0)
        assert share_isolated > 1 / len(exits)

    def test_empty_input(self) -> None:
        assert sequential_bootstrap(np.array([], dtype="int64")).size == 0
