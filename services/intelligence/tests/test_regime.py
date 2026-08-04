"""
tests/test_regime.py
────────────────────
Tests for the regime detection module — shared classifier, market-wide
aggregator, and historical spot-checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from intelligence.regime.classifier import RegimeClassifier, RegimeResult


def _dummy_frame(n: int = 300):
    """A calm synthetic OHLCV frame, long enough for ADX and the regime history."""
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 + np.arange(n) * 0.05, index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

N_TRADING_DAYS = 500  # ~2 years of daily data


def _make_price_series(
    start_price: float = 100.0,
    n_days: int = N_TRADING_DAYS,
    trend: str = "bull",
    volatility: float = 0.01,
) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame with a known trend for testing.

    Parameters
    ----------
    trend : str
        "bull" → upward drift
        "bear" → downward drift
        "range" → no drift (random walk)
    volatility : float
        Daily return standard deviation.
    """
    np.random.seed(42)

    if trend == "bull":
        drift = 0.0008  # ~20% annualised
    elif trend == "bear":
        drift = -0.0008
    else:
        drift = 0.0

    log_returns = np.random.normal(drift, volatility, n_days)
    prices = start_price * np.exp(np.cumsum(log_returns))
    closes = np.maximum(prices, 1.0)  # floor at 1

    high = closes * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low = closes * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = low + np.random.uniform(0, 1, n_days) * (high - low)

    dates = [
        datetime(2022, 1, 1) + timedelta(days=i)
        for i in range(n_days)
    ]

    df = pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": np.random.randint(1_000_000, 10_000_000, n_days),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()
    return df


def _make_df_with_regime_features(
    adx: float,
    plus_di: float,
    minus_di: float,
    drawdown: float = -0.05,
    trend_consistency: float = 0.55,
    rv_63: float = 0.20,
    n_days: int = N_TRADING_DAYS,
) -> pd.DataFrame:
    """Build a minimal DataFrame with regime feature columns."""
    df = _make_price_series(trend="range", n_days=n_days)
    df["regime_adx"] = adx
    df["regime_plus_di"] = plus_di
    df["regime_minus_di"] = minus_di
    df["regime_drawdown"] = drawdown
    df["regime_trend_consistency_63"] = trend_consistency
    df["rv_63"] = rv_63
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Classifier — primary regime
# ══════════════════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    """Test the core classification rules."""

    def test_adx_below_20_is_range_bound(self):
        """ADX < 20 → RangeBound regardless of DI values."""
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=30.0, minus_di=20.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "RangeBound"
        assert result.trend_direction == "up"

    def test_adx_above_20_bull(self):
        """ADX ≥ 20 + +DI > -DI → Bull."""
        df = _make_df_with_regime_features(
            adx=30.0, plus_di=35.0, minus_di=18.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "Bull"
        assert result.trend_direction == "up"

    def test_adx_above_20_bear(self):
        """ADX ≥ 20 + -DI > +DI → Bear."""
        df = _make_df_with_regime_features(
            adx=28.0, plus_di=15.0, minus_di=32.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "Bear"
        assert result.trend_direction == "down"

    def test_adx_strong_bull(self):
        """ADX > 40 → Bull with high confidence."""
        df = _make_df_with_regime_features(
            adx=45.0, plus_di=40.0, minus_di=12.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "Bull"
        assert result.regime_score > 0.70

    def test_adx_edge_case_equal_di(self):
        """ADX ≥ 20 but +DI == -DI (rare) → RangeBound."""
        df = _make_df_with_regime_features(
            adx=25.0, plus_di=25.0, minus_di=25.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "RangeBound"

    def test_adx_nan(self):
        """NaN ADX → RangeBound with zero confidence."""
        df = _make_df_with_regime_features(
            adx=np.nan, plus_di=30.0, minus_di=20.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "RangeBound"
        assert result.regime_score == 0.0
        assert result.adx is None

    def test_missing_required_columns_raises(self):
        """Missing regime columns → ValueError."""
        df = pd.DataFrame({"close": [100.0]})
        classifier = RegimeClassifier()
        with pytest.raises(ValueError, match="missing required regime columns"):
            classifier.classify(df, ticker="TEST.NS")

    def test_range_bound_confidence_high_at_low_adx(self):
        """RangeBound confidence near 1.0 when ADX is near 0."""
        df = _make_df_with_regime_features(
            adx=2.0, plus_di=10.0, minus_di=8.0
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.primary_regime == "RangeBound"
        assert result.regime_score > 0.85
        assert result.regime_score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Volatility regime
# ══════════════════════════════════════════════════════════════════════════════


class TestVolatilityRegime:
    """Test volatility regime classification."""

    def test_high_volatility(self):
        """Latest rv_63 above 80th percentile → high vol."""
        df = _make_price_series(trend="range", n_days=N_TRADING_DAYS)
        df["regime_adx"] = 15.0
        df["regime_plus_di"] = 20.0
        df["regime_minus_di"] = 18.0
        # Most of the series at 0.15, last few at 0.40 (well above 80th pctile)
        rv = np.full(N_TRADING_DAYS, 0.15)
        rv[-5:] = 0.40
        df["rv_63"] = rv

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.volatility_regime == "high"

    def test_low_volatility(self):
        """Latest rv_63 below 20th percentile → low vol."""
        df = _make_price_series(trend="range", n_days=N_TRADING_DAYS)
        df["regime_adx"] = 15.0
        df["regime_plus_di"] = 20.0
        df["regime_minus_di"] = 18.0
        # Most of the series at 0.30, last few at 0.05 (well below 20th pctile)
        rv = np.full(N_TRADING_DAYS, 0.30)
        rv[-5:] = 0.05
        df["rv_63"] = rv

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.volatility_regime == "low"

    def test_normal_volatility(self):
        """Latest rv_63 in mid-range → normal vol."""
        df = _make_price_series(trend="range", n_days=N_TRADING_DAYS)
        df["regime_adx"] = 15.0
        df["regime_plus_di"] = 20.0
        df["regime_minus_di"] = 18.0
        # rv_63 has a wide spread; latest at 0.12 is mid-range
        rv = np.concatenate([
            np.full(200, 0.05),
            np.full(200, 0.20),
            np.full(99, 0.10),
            [0.12],
        ])
        df["rv_63"] = rv

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.volatility_regime == "normal"

    def test_volatility_fallback_no_rv63(self):
        """No rv_63 column → normal vol."""
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0
        )
        df = df.drop(columns=["rv_63"])
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.volatility_regime == "normal"


# ══════════════════════════════════════════════════════════════════════════════
# Drawdown regime
# ══════════════════════════════════════════════════════════════════════════════


class TestDrawdownRegime:
    """Test drawdown regime classification."""

    @pytest.mark.parametrize(
        "drawdown,expected",
        [
            (-0.02, "peak"),
            (-0.10, "normal"),
            (-0.20, "correction"),
            (-0.35, "bear"),
            (-0.05, "peak"),       # boundary: -5% → peak (>= threshold)
            (-0.1501, "correction"),  # just past -15% → correction
        ],
    )
    def test_drawdown_classification(self, drawdown: float, expected: str):
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0,
            drawdown=drawdown,
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.drawdown_regime == expected

    def test_drawdown_missing(self):
        """No drawdown column → normal drawdown regime."""
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0,
        )
        df = df.drop(columns=["regime_drawdown"])
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.drawdown_regime == "normal"


# ══════════════════════════════════════════════════════════════════════════════
# Composite signal
# ══════════════════════════════════════════════════════════════════════════════


class TestCompositeSignal:
    """Test the risk_on/risk_off/neutral composite signal."""

    def test_risk_on(self):
        """Bull + normal vol + peak drawdown → risk_on."""
        df = _make_df_with_regime_features(
            adx=30.0, plus_di=35.0, minus_di=15.0,
            drawdown=-0.02,
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.composite_signal == "risk_on"

    def test_risk_off(self):
        """Bear + high vol + bear drawdown → risk_off."""
        # High vol: create series where latest is near extreme
        df = _make_price_series(trend="range", n_days=N_TRADING_DAYS)
        df["regime_adx"] = 30.0
        df["regime_plus_di"] = 15.0
        df["regime_minus_di"] = 35.0
        df["regime_drawdown"] = -0.35
        rv = np.full(N_TRADING_DAYS, 0.10)
        rv[-5:] = 0.50  # well above 80th pctile
        df["rv_63"] = rv

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.composite_signal == "risk_off"

    def test_neutral_mixed_signals(self):
        """Bull + high vol → neutral."""
        df = _make_price_series(trend="range", n_days=N_TRADING_DAYS)
        df["regime_adx"] = 30.0
        df["regime_plus_di"] = 35.0
        df["regime_minus_di"] = 15.0
        df["regime_drawdown"] = -0.02
        rv = np.full(N_TRADING_DAYS, 0.10)
        rv[-5:] = 0.50
        df["rv_63"] = rv

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert result.composite_signal == "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# Regime history
# ══════════════════════════════════════════════════════════════════════════════


class TestRegimeHistory:
    """Test regime history building."""

    def test_history_length(self):
        """History should contain up to REGIME_HISTORY_DAYS entries."""
        n_days = 300
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0,
            n_days=n_days,
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        # With 300 days, history should be 252 (config limit)
        assert len(result.regime_history) == 252

    def test_history_small_df(self):
        """Less than REGIME_HISTORY_DAYS rows → all rows in history."""
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0,
            n_days=50,
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        assert len(result.regime_history) == 50

    def test_history_has_dates_and_regimes(self):
        """Each history entry should have date and regime keys."""
        df = _make_df_with_regime_features(
            adx=15.0, plus_di=20.0, minus_di=18.0,
            n_days=100,
        )
        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="TEST.NS")
        entry = result.regime_history[0]
        assert "date" in entry
        assert "regime" in entry
        assert entry["regime"] in ("Bull", "Bear", "RangeBound")


# ══════════════════════════════════════════════════════════════════════════════
# Historical spot-checks (2020 crash, 2021 bull, 2022 correction)
# ══════════════════════════════════════════════════════════════════════════════


class TestHistoricalSpotChecks:
    """
    Verify the classifier produces expected regimes for known market periods.

    These use synthetic data designed to mimic the statistical properties
    of each period — not actual historical data (which would require
    a network fetch), but controlled scenarios that match the regime
    characteristics of the period.
    """

    def test_2020_crash_scenario(self):
        """
        March 2020 COVID crash simulation.

        Characteristics:
        - Sharp downtrend (high negative drift)
        - ADX spiking above 40 (very strong trend)
        - minus_di > plus_di
        - Realised volatility extremely high
        - Drawdown > 20%
        """
        df = _make_price_series(
            start_price=120.0,
            n_days=300,
            trend="bear",
            volatility=0.025,  # elevated vol
        )
        df["regime_adx"] = 55.0
        df["regime_plus_di"] = 10.0
        df["regime_minus_di"] = 45.0
        prices = df["close"]
        rolling_peak = prices.cummax()
        df["regime_drawdown"] = (prices - rolling_peak) / rolling_peak
        # rv_63: low vol in early 2020, spiking at the crash
        rv = np.full(300, 0.12)
        rv[-20:] = np.linspace(0.12, 0.45, 20)  # vol ramps up
        df["rv_63"] = rv
        df["regime_trend_consistency_63"] = 0.30

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="SPOT")

        assert result.primary_regime == "Bear"
        assert result.drawdown_regime in ("correction", "bear")
        assert result.volatility_regime == "high"
        assert result.trend_direction == "down"
        assert result.regime_score > 0.70

    def test_2021_bull_scenario(self):
        """
        Mid-2021 recovery bull market simulation.

        Characteristics:
        - Sustained uptrend
        - ADX 25-35 (developing/strong trend)
        - plus_di > minus_di
        - Low volatility
        - Drawdown near peak
        """
        df = _make_price_series(
            start_price=100.0,
            n_days=300,
            trend="bull",
            volatility=0.008,  # low vol
        )
        df["regime_adx"] = 30.0
        df["regime_plus_di"] = 32.0
        df["regime_minus_di"] = 16.0
        df["regime_drawdown"] = -0.02
        df["rv_63"] = 0.10
        df["regime_trend_consistency_63"] = 0.65

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="SPOT")

        assert result.primary_regime == "Bull"
        assert result.drawdown_regime == "peak"
        assert result.composite_signal == "risk_on"
        assert result.trend_direction == "up"

    def test_2022_correction_scenario(self):
        """
        2022 rate-hike correction simulation.

        Characteristics:
        - Moderate downtrend / choppy
        - ADX 20-30 (developing trend)
        - Mixed DI (sometimes bear dominant, sometimes sideways)
        - Moderate-high volatility
        - Drawdown -15% to -25%
        """
        df = _make_price_series(
            start_price=115.0,
            n_days=300,
            trend="bear",
            volatility=0.018,
        )
        df["regime_adx"] = 25.0
        df["regime_plus_di"] = 20.0
        df["regime_minus_di"] = 28.0
        prices = df["close"]
        rolling_peak = prices.cummax()
        df["regime_drawdown"] = (prices - rolling_peak) / rolling_peak
        df["rv_63"] = 0.30
        df["regime_trend_consistency_63"] = 0.40

        classifier = RegimeClassifier()
        result = classifier.classify(df, ticker="SPOT")

        # 2022 was a correction — bearish with elevated vol but not crash territory.
        # The classifier produces neutral here because realised volatility
        # (constant rv_63) doesn't spike enough to tip into full risk_off.
        # This conservative stance during an ambiguous correction is
        # preferable to overconfidently calling risk-off.
        assert result.primary_regime == "Bear"
        assert result.trend_direction == "down"
        assert result.composite_signal == "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# Market-wide aggregator (unit-level, no network)
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketAggregatorUnit:
    """Unit tests for MarketRegimeResult aggregation logic.

    These tests avoid network calls by passing pre-classified results
    directly to the aggregation helpers.
    """

    def test_aggregate_signal_risk_on_majority(self):
        """More risk_on than risk_off/neutral → risk_on."""
        from intelligence.regime.market import RegimeAggregator

        results = [
            RegimeResult(ticker="A.NS", primary_regime="Bull", regime_score=0.8,
                         adx=30.0, trend_direction="up", volatility_regime="normal",
                         drawdown_regime="peak", composite_signal="risk_on"),
            RegimeResult(ticker="B.NS", primary_regime="Bull", regime_score=0.8,
                         adx=30.0, trend_direction="up", volatility_regime="normal",
                         drawdown_regime="peak", composite_signal="risk_on"),
            RegimeResult(ticker="C.NS", primary_regime="Bear", regime_score=0.8,
                         adx=30.0, trend_direction="down", volatility_regime="high",
                         drawdown_regime="bear", composite_signal="risk_off"),
        ]
        signal = RegimeAggregator._aggregate_signal(results)
        assert signal == "risk_on"

    def test_aggregate_signal_risk_off_majority(self):
        """More risk_off than risk_on/neutral → risk_off."""
        from intelligence.regime.market import RegimeAggregator

        results = [
            RegimeResult(ticker="A.NS", primary_regime="Bear", regime_score=0.8,
                         adx=30.0, trend_direction="down", volatility_regime="high",
                         drawdown_regime="bear", composite_signal="risk_off"),
            RegimeResult(ticker="B.NS", primary_regime="Bear", regime_score=0.8,
                         adx=30.0, trend_direction="down", volatility_regime="high",
                         drawdown_regime="bear", composite_signal="risk_off"),
            RegimeResult(ticker="C.NS", primary_regime="Bull", regime_score=0.8,
                         adx=30.0, trend_direction="up", volatility_regime="normal",
                         drawdown_regime="peak", composite_signal="risk_on"),
        ]
        signal = RegimeAggregator._aggregate_signal(results)
        assert signal == "risk_off"

    def test_aggregate_signal_neutral(self):
        """Even split of risk_on/risk_off → neutral."""
        from intelligence.regime.market import RegimeAggregator

        results = [
            RegimeResult(ticker="A.NS", primary_regime="Bull", regime_score=0.8,
                         adx=30.0, trend_direction="up", volatility_regime="normal",
                         drawdown_regime="peak", composite_signal="risk_on"),
            RegimeResult(ticker="B.NS", primary_regime="Bear", regime_score=0.8,
                         adx=30.0, trend_direction="down", volatility_regime="high",
                         drawdown_regime="bear", composite_signal="risk_off"),
        ]
        signal = RegimeAggregator._aggregate_signal(results)
        assert signal == "neutral"

    def test_constituents_come_from_the_source_not_a_constant(self):
        """v1 hardcoded a NIFTY 50 list, which is survivorship bias in constant
        form: it describes today's index and silently applies it to every
        historical date. Constituents now come from the point-in-time universe
        via the injected source."""
        from intelligence.regime.market import RegimeAggregator
        from intelligence.regime.source import FrameRegimeSource

        source = FrameRegimeSource({"AAA": _dummy_frame(), "BBB": _dummy_frame()})
        agg = RegimeAggregator(source=source)
        assert agg.tickers == ["AAA", "BBB"]

    def test_source_is_required(self):
        """No default source: reaching for data must be an explicit decision,
        not something that happens inside a request path."""
        from intelligence.regime.market import RegimeAggregator

        with pytest.raises(TypeError):
            RegimeAggregator()  # type: ignore[call-arg]

    def test_reporting_ratio(self):
        """MarketRegimeResult.reporting_ratio reflects data quality."""
        from intelligence.regime.market import MarketRegimeResult

        result = MarketRegimeResult(
            analysis_date="2026-06-17",
            total_constituents=50,
            constituents_reporting=42,
            primary_regime="Bull",
            regime_distribution={"Bull": 42},
            market_adx=25.0,
            composite_signal="risk_on",
        )
        assert result.reporting_ratio == 42 / 50
        # Edge: full reporting
        result_full = MarketRegimeResult(
            analysis_date="2026-06-17",
            total_constituents=50,
            constituents_reporting=50,
            primary_regime="Bull",
            regime_distribution={"Bull": 50},
            market_adx=25.0,
            composite_signal="risk_on",
        )
        assert result_full.reporting_ratio == 1.0
        # Edge: no reporting
        result_empty = MarketRegimeResult(
            analysis_date="2026-06-17",
            total_constituents=50,
            constituents_reporting=0,
            primary_regime="Unknown",
            regime_distribution={},
            market_adx=None,
            composite_signal="neutral",
        )
        assert result_empty.reporting_ratio == 0.0


class TestMarketAggregatorCache:
    """Tests for the in-memory TTL cache on RegimeAggregator.analyse()."""

    def test_second_call_within_ttl_uses_cache(self, monkeypatch):
        """
        Calling analyse() twice within the TTL window should not re-fetch
        any constituents — the second call returns the cached result.
        """
        from intelligence.regime.market import RegimeAggregator
        from intelligence.regime.source import FrameRegimeSource

        call_count = 0
        original = RegimeAggregator._classify_single

        def counting_classify(self_, ticker):
            nonlocal call_count
            call_count += 1
            return original(self_, ticker)

        monkeypatch.setattr(
            RegimeAggregator, "_classify_single", counting_classify
        )

        # Use a minimal ticker list so the test is fast
        source = FrameRegimeSource(
            {"RELIANCE": _dummy_frame(), "TCS": _dummy_frame()}
        )
        agg = RegimeAggregator(source=source, tickers=["RELIANCE", "TCS"])

        # First call — should fetch both tickers
        result1 = agg.analyse()
        first_count = call_count
        assert first_count == 2, f"Expected 2 fetches, got {first_count}"

        # Second call — should hit cache, zero additional fetches
        result2 = agg.analyse()
        assert call_count == first_count, (
            f"Expected no additional fetches (cached), got {call_count - first_count}"
        )

        # Both results should be the same object (identity, not just equality)
        assert result1 is result2

    def test_cache_key_format(self):
        """Cache key should be YYYY-MM-DD-HH."""
        from intelligence.regime.market import RegimeAggregator
        from intelligence.regime.source import FrameRegimeSource

        agg = RegimeAggregator(source=FrameRegimeSource({}), tickers=[])
        key = agg._cache_key()
        parts = key.split("-")
        assert len(parts) == 4  # year-month-day-hour
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day
        assert len(parts[3]) == 2  # hour

    def test_cache_ttl_minutes_in_result(self):
        """MarketRegimeResult should advertise the cache TTL."""
        from intelligence.regime.market import MarketRegimeResult

        result = MarketRegimeResult(
            analysis_date="2026-06-17",
            total_constituents=50,
            constituents_reporting=50,
            primary_regime="Bull",
            regime_distribution={"Bull": 50},
            market_adx=25.0,
            composite_signal="risk_on",
        )
        assert result.cache_ttl_minutes == 15


class TestRegimeSubsystemReachability:
    """Pin which regime implementation the API actually uses.

    This project's recurring failure is code that is written, tested, and never
    reached from production — `adjust_all` was one, the missing `--registry`
    CLI flag was another, and 208 green tests hid both. This class does not
    test regime maths; it tests the WIRING, which is the part a unit test
    normally cannot see.
    """

    def test_the_served_regime_comes_from_serving_not_from_this_module(self) -> None:
        """If someone wires RegimeClassifier into the API, this test should be
        updated deliberately — not discovered months later from a support
        ticket about two different regimes for the same stock."""
        import inspect

        from intelligence.api import main as api_main

        source = inspect.getsource(api_main)
        assert "RegimeClassifier" not in source, (
            "RegimeClassifier now appears in the API. That may be correct, but "
            "the module docstring in intelligence/regime/__init__.py states it "
            "is unreachable — update it in the same change."
        )
        assert "_service().predict" in source or "svc.predict" in source

    def test_serving_owns_the_regime_rule_that_ships(self) -> None:
        """The two-condition rule is the one users see, so its inputs are
        load-bearing and must exist in the panel the builder emits."""
        from intelligence.serving import (
            ADX_TRENDING_THRESHOLD,
            REGIME_ADX_FEATURE,
            TREND_FEATURE,
        )

        assert REGIME_ADX_FEATURE == "regime_adx"
        assert TREND_FEATURE == "trend_price_vs_sma200"
        assert 0 < ADX_TRENDING_THRESHOLD < 100
