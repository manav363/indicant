"""Normalizer, store, calendar, adjustment and pipeline tests.

Nothing here touches the network. The HTTP fetcher runs against a stub session;
everything else runs on hand-built frames and a tmp_path lake.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest
from indicant_contracts import (
    CANONICAL_PRICE_COLUMNS,
    CorporateAction,
    CorporateActionType,
    Dataset,
    SymbolChange,
    Verdict,
)

from conftest import canonical_frame, canonical_row
from market_data._dates import as_date
from market_data.adjust.factors import (
    AdjustmentError,
    adjust_symbol,
    continuity_breaks,
    cumulative_factors,
)
from market_data.adjust.symbol_map import SymbolMap
from market_data.ingest import corporate_actions as ca
from market_data.ingest.bhavcopy import (
    BhavcopyNotAvailable,
    HttpBhavcopyFetcher,
    LocalBhavcopyFetcher,
    SourceKind,
    candidate_sources,
    legacy_equity_url,
    udiff_equity_url,
)
from market_data.ingest.calendar import TradingCalendarService
from market_data.normalize.canonical import Era, era_for, normalise
from market_data.normalize.legacy import LAKHS_TO_RUPEES
from market_data.pipeline import IngestPipeline
from market_data.quality.gate import QualityGate
from market_data.quality.quarantine import QuarantineStore
from market_data.quality.scoring import QualityScorer
from market_data.store.catalog import Catalog
from market_data.store.lake import Lake, group_by_year

# ==========================================================================
# Era dispatch — the cutover that decides which normalizer runs
# ==========================================================================


class TestEraDispatch:
    def test_2006_is_legacy(self) -> None:
        assert era_for(date(2006, 1, 2)) is Era.LEGACY

    def test_day_before_cutover_is_legacy(self) -> None:
        assert era_for(date(2024, 7, 5)) is Era.LEGACY

    def test_cutover_day_is_udiff(self) -> None:
        assert era_for(date(2024, 7, 8)) is Era.UDIFF

    def test_2026_is_udiff(self) -> None:
        assert era_for(date(2026, 7, 30)) is Era.UDIFF

    def test_legacy_covers_the_bulk_of_the_backfill(self) -> None:
        """Sanity-check the claim that legacy is the primary path, not the edge
        case — it drives which module gets the careful treatment."""
        years = range(2006, 2027)
        legacy_years = sum(1 for y in years if era_for(date(y, 1, 2)) is Era.LEGACY)
        assert legacy_years >= 18


# ==========================================================================
# Legacy normalizer
# ==========================================================================


class TestLegacyNormalizer:
    def test_equity_bhavcopy_maps_to_canonical(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "SYMBOL": "reliance",
                    "SERIES": "EQ",
                    "OPEN": "100.0",
                    "HIGH": "105.0",
                    "LOW": "98.0",
                    "CLOSE": "102.0",
                    "LAST": "102.0",
                    "PREVCLOSE": "99.0",
                    "TOTTRDQTY": "1000000",
                    "TOTTRDVAL": "102000000.0",
                    "TIMESTAMP": "04-MAR-2015",
                    "TOTALTRADES": "5000",
                    "ISIN": "INE002A01018",
                }
            ]
        )
        out = normalise(raw, trade_date=date(2015, 3, 4))
        assert list(out.columns) == list(CANONICAL_PRICE_COLUMNS)
        assert out.loc[0, "symbol"] == "RELIANCE"
        assert out.loc[0, "turnover"] == pytest.approx(102_000_000.0)
        assert out.loc[0, "date"] == date(2015, 3, 4)

    def test_delivery_bhavcopy_converts_lakhs_to_rupees(self) -> None:
        """The 100,000x bug. TURNOVER_LACS is in lakhs, not rupees."""
        raw = pd.DataFrame(
            [
                {
                    " SYMBOL": "RELIANCE",
                    " SERIES": "EQ",
                    " OPEN_PRICE": "100.0",
                    " HIGH_PRICE": "105.0",
                    " LOW_PRICE": "98.0",
                    " CLOSE_PRICE": "102.0",
                    " PREV_CLOSE": "99.0",
                    " TTL_TRD_QNTY": "1000000",
                    " TURNOVER_LACS": "1020.0",
                    " NO_OF_TRADES": "5000",
                    " DELIV_QTY": "400000",
                    " DELIV_PER": "40.0",
                }
            ]
        )
        out = normalise(raw, trade_date=date(2015, 3, 4))
        assert out.loc[0, "turnover"] == pytest.approx(1020.0 * LAKHS_TO_RUPEES)
        # And the result reconciles: turnover/volume lands inside the day's range.
        assert 98.0 <= out.loc[0, "turnover"] / out.loc[0, "volume"] <= 105.0

    def test_leading_spaces_in_headers_are_stripped(self) -> None:
        """NSE ships ' SYMBOL' with a leading space in the delivery file."""
        raw = pd.DataFrame([{" SYMBOL": "X", " SERIES": "EQ", " DELIV_QTY": "1"}])
        with pytest.raises(Exception) as exc:
            normalise(raw, trade_date=date(2015, 3, 4))
        # Fails on missing price columns, not on the space-prefixed names.
        assert "OPEN_PRICE" in str(exc.value)

    def test_dash_delivery_becomes_null_not_zero(self) -> None:
        """Non-EQ series report '-'. Treating that as 0% delivery would be a lie."""
        raw = pd.DataFrame(
            [
                {
                    "SYMBOL": "SOMEBOND",
                    "SERIES": "N1",
                    "OPEN_PRICE": "100.0",
                    "HIGH_PRICE": "100.0",
                    "LOW_PRICE": "100.0",
                    "CLOSE_PRICE": "100.0",
                    "PREV_CLOSE": "100.0",
                    "TTL_TRD_QNTY": "10",
                    "TURNOVER_LACS": "0.01",
                    "NO_OF_TRADES": "1",
                    "DELIV_QTY": "-",
                    "DELIV_PER": "-",
                }
            ]
        )
        out = normalise(raw, trade_date=date(2015, 3, 4))
        assert pd.isna(out.loc[0, "delivery_qty"])
        assert pd.isna(out.loc[0, "delivery_pct"])

    def test_missing_column_raises_with_both_sides(self) -> None:
        raw = pd.DataFrame([{"SYMBOL": "X", "SERIES": "EQ"}])
        with pytest.raises(Exception) as exc:
            normalise(raw, trade_date=date(2015, 3, 4))
        assert "missing" in str(exc.value).lower()


# ==========================================================================
# UDiFF normalizer
# ==========================================================================


class TestUdiffNormalizer:
    def _row(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "TradDt": "2025-01-02",
            "Sgmt": "CM",
            "Src": "NSE",
            "FinInstrmTp": "STK",
            "ISIN": "INE002A01018",
            "TckrSymb": "RELIANCE",
            "SctySrs": "EQ",
            "XpryDt": "",
            "StrkPric": "",
            "OptnTp": "",
            "OpnPric": "1300.0",
            "HghPric": "1330.0",
            "LwPric": "1295.0",
            "ClsPric": "1321.2",
            "PrvsClsgPric": "1298.0",
            "TtlTradgVol": "5000000",
            "TtlTrfVal": "6600000000.0",
            "TtlNbOfTxsExctd": "120000",
            "SsnId": "F1",
        }
        return base | overrides

    def test_cash_equity_row_maps_to_canonical(self) -> None:
        out = normalise(pd.DataFrame([self._row()]), trade_date=date(2025, 1, 2))
        assert list(out.columns) == list(CANONICAL_PRICE_COLUMNS)
        assert out.loc[0, "symbol"] == "RELIANCE"
        assert out.loc[0, "turnover"] == pytest.approx(6.6e9)

    def test_turnover_is_already_rupees(self) -> None:
        """No lakhs conversion in UDiFF — applying one would be a 100,000x error
        in the other direction."""
        out = normalise(pd.DataFrame([self._row()]), trade_date=date(2025, 1, 2))
        implied = out.loc[0, "turnover"] / out.loc[0, "volume"]
        assert 1295.0 <= implied <= 1330.0

    def test_option_rows_are_filtered_out(self) -> None:
        rows = [
            self._row(),
            self._row(TckrSymb="RELIANCE", XpryDt="2025-01-30", OptnTp="CE",
                      StrkPric="1300", FinInstrmTp="OPTSTK"),
        ]
        out = normalise(pd.DataFrame(rows), trade_date=date(2025, 1, 2))
        assert len(out) == 1

    def test_future_rows_are_filtered_out(self) -> None:
        rows = [self._row(), self._row(XpryDt="2025-01-30", FinInstrmTp="FUTSTK")]
        out = normalise(pd.DataFrame(rows), trade_date=date(2025, 1, 2))
        assert len(out) == 1

    def test_non_cash_segment_is_filtered_out(self) -> None:
        rows = [self._row(), self._row(Sgmt="FO", TckrSymb="NIFTY")]
        out = normalise(pd.DataFrame(rows), trade_date=date(2025, 1, 2))
        assert len(out) == 1

    def test_missing_filter_column_widens_rather_than_empties(self) -> None:
        """A future UDiFF revision dropping a column must not silently yield
        zero rows — Tier 3 would then be the only thing that noticed."""
        row = self._row()
        del row["Sgmt"]
        out = normalise(pd.DataFrame([row]), trade_date=date(2025, 1, 2))
        assert len(out) == 1


# ==========================================================================
# URL construction and fetching
# ==========================================================================


class TestUrls:
    def test_legacy_url_shape(self) -> None:
        assert legacy_equity_url(date(2024, 7, 9)).endswith(
            "/content/historical/EQUITIES/2024/JUL/cm09JUL2024bhav.csv.zip"
        )

    def test_udiff_url_shape(self) -> None:
        assert udiff_equity_url(date(2025, 1, 2)).endswith(
            "/content/cm/BhavCopy_NSE_CM_0_0_0_20250102_F_0000.csv.zip"
        )

    def test_legacy_era_prefers_the_delivery_report(self) -> None:
        """Delivery is strictly richer — it carries delivery_qty/pct."""
        assert candidate_sources(date(2015, 3, 4))[0][0] is SourceKind.LEGACY_DELIVERY

    def test_udiff_era_prefers_udiff(self) -> None:
        assert candidate_sources(date(2025, 1, 2))[0][0] is SourceKind.UDIFF_EQUITY


class _StubResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class _StubSession:
    def __init__(self, responses: dict[str, _StubResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> _StubResponse:
        self.calls.append(url)
        return self._responses.get(url, _StubResponse(404))


class TestHttpFetcher:
    def test_404_on_all_sources_raises_not_available(self) -> None:
        """A holiday. Must be distinguishable from a transport failure."""
        session = _StubSession({})
        fetcher = HttpBhavcopyFetcher(
            session=session, rate_limit_seconds=0, sleep=lambda _: None
        )
        with pytest.raises(BhavcopyNotAvailable) as exc:
            fetcher.fetch(date(2015, 8, 15))
        assert len(exc.value.attempted) == 2

    def test_falls_back_to_the_second_source(self) -> None:
        url = legacy_equity_url(date(2015, 3, 4))
        session = _StubSession({url: _StubResponse(200, b"SYMBOL,SERIES\nX,EQ\n")})
        fetcher = HttpBhavcopyFetcher(
            session=session, rate_limit_seconds=0, sleep=lambda _: None
        )
        result = fetcher.fetch(date(2015, 3, 4))
        assert result.kind is SourceKind.LEGACY_EQUITY

    def test_retries_then_raises_on_persistent_500(self) -> None:
        url = legacy_equity_url(date(2015, 3, 4))
        session = _StubSession({url: _StubResponse(500)})
        fetcher = HttpBhavcopyFetcher(
            session=session, max_retries=3, rate_limit_seconds=0, sleep=lambda _: None
        )
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            fetcher.fetch(date(2015, 3, 4))

    def test_zip_payload_is_decoded(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("cm04MAR2015bhav.csv", "SYMBOL,SERIES\nRELIANCE,EQ\n")
        url = legacy_equity_url(date(2015, 3, 4))
        session = _StubSession({url: _StubResponse(200, buf.getvalue())})
        fetcher = HttpBhavcopyFetcher(
            session=session, rate_limit_seconds=0, sleep=lambda _: None
        )
        frame = fetcher.fetch(date(2015, 3, 4)).to_frame()
        assert frame.loc[0, "SYMBOL"] == "RELIANCE"


class TestLocalFetcher:
    def test_reads_a_local_delivery_file(self, tmp_path) -> None:
        path = tmp_path / "sec_bhavdata_full_20150304.csv"
        path.write_text("SYMBOL,SERIES\nRELIANCE,EQ\n")
        result = LocalBhavcopyFetcher(tmp_path).fetch(date(2015, 3, 4))
        assert result.kind is SourceKind.LEGACY_DELIVERY

    def test_missing_file_raises_not_available(self, tmp_path) -> None:
        with pytest.raises(BhavcopyNotAvailable):
            LocalBhavcopyFetcher(tmp_path).fetch(date(2015, 3, 4))


# ==========================================================================
# Calendar
# ==========================================================================


class TestCalendar:
    def test_weekends_are_not_trading_days(self) -> None:
        cal = TradingCalendarService()
        assert not cal.is_expected_trading_day(date(2026, 8, 1))  # Saturday
        assert not cal.is_expected_trading_day(date(2026, 8, 2))  # Sunday

    def test_fixed_national_holidays_are_known(self) -> None:
        cal = TradingCalendarService()
        assert cal.is_known_holiday(date(2026, 1, 26))
        assert cal.is_known_holiday(date(2026, 8, 15))
        assert cal.is_known_holiday(date(2026, 10, 2))

    def test_uncurated_year_is_reported_as_such(self) -> None:
        """A festival holiday in an uncurated year will produce a false
        suspicion, and the caller must be able to know that."""
        cal = TradingCalendarService()
        assert not cal.has_curated_data(2015)

    def test_curated_holiday_is_respected(self) -> None:
        cal = TradingCalendarService({2015: {date(2015, 3, 6)}})
        assert cal.has_curated_data(2015)
        assert not cal.is_expected_trading_day(date(2015, 3, 6))

    def test_reconciliation_reports_missing_days(self) -> None:
        cal = TradingCalendarService()
        rec = cal.reconcile(
            start=date(2015, 3, 2),
            end=date(2015, 3, 6),
            observed=[date(2015, 3, 2), date(2015, 3, 3)],
        )
        assert set(rec.missing_trading_days) == {
            date(2015, 3, 4),
            date(2015, 3, 5),
            date(2015, 3, 6),
        }
        assert rec.coverage == pytest.approx(2 / 5)
        assert not rec.is_clean

    def test_reconciliation_reports_unexpected_days(self) -> None:
        """A file on a predicted holiday means the holiday list is wrong."""
        cal = TradingCalendarService({2015: {date(2015, 3, 4)}})
        rec = cal.reconcile(
            start=date(2015, 3, 4),
            end=date(2015, 3, 4),
            observed=[date(2015, 3, 4)],
        )
        assert rec.unexpected_trading_days == (date(2015, 3, 4),)

    def test_holidays_can_be_derived_from_observed_days(self) -> None:
        """Once a year is ingested, its weekday gaps *are* the holidays. This
        turns the curated list into something the system maintains itself."""
        cal = TradingCalendarService()
        observed = [
            d
            for d in pd.date_range("2015-03-02", "2015-03-13", freq="D").date
            if d.weekday() < 5 and d != date(2015, 3, 6)
        ]
        derived = cal.learn_from_observed(observed, year=2015)
        assert date(2015, 3, 6) in derived
        assert cal.has_curated_data(2015)

    def test_partial_year_does_not_mark_the_future_as_holidays(self) -> None:
        cal = TradingCalendarService()
        derived = cal.learn_from_observed([date(2015, 3, 2), date(2015, 3, 3)], year=2015)
        assert not any(d > date(2015, 3, 3) for d in derived)

    def test_deriving_from_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match="no observed"):
            TradingCalendarService().learn_from_observed([], year=2015)

    def test_round_trips_through_a_file(self, tmp_path) -> None:
        cal = TradingCalendarService({2015: {date(2015, 3, 6)}})
        path = cal.save(tmp_path / "h.json")
        assert TradingCalendarService.from_file(path).is_known_holiday(date(2015, 3, 6))


# ==========================================================================
# Lake
# ==========================================================================


class TestLake:
    def test_write_then_read_round_trips(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        lake.write_year([canonical_frame()], year=2015)
        out = lake.read_prices()
        assert len(out) == 3
        assert set(out["symbol"]) == {"RELIANCE", "TCS", "INFY"}

    def test_reingesting_a_date_replaces_rather_than_duplicates(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        lake.write_year([canonical_frame()], year=2015)
        revised = canonical_frame([canonical_row("RELIANCE", close=999.0, high=999.0,
                                                 turnover=999_000_000.0)])
        lake.append_day(revised, trade_date=date(2015, 3, 4))
        out = lake.read_prices(symbols=["RELIANCE"])
        assert len(out) == 1
        assert out.loc[0, "close"] == pytest.approx(999.0)

    def test_empty_lake_reads_as_an_empty_frame_not_an_error(self, lake_paths) -> None:
        """A first run is a valid state."""
        out = Lake(lake_paths).read_prices()
        assert out.empty
        assert list(out.columns) == list(CANONICAL_PRICE_COLUMNS)

    def test_refuses_to_write_an_empty_year(self, lake_paths) -> None:
        with pytest.raises(ValueError, match="empty"):
            Lake(lake_paths).write_year([canonical_frame().iloc[0:0]], year=2015)

    def test_observed_trading_days_is_derived_from_data(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        lake.write_year([canonical_frame()], year=2015)
        assert lake.observed_trading_days() == [date(2015, 3, 4)]

    def test_symbol_span_reports_first_and_last_seen(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        early = canonical_frame([canonical_row("DEADCO")])
        late = canonical_frame([canonical_row("RELIANCE")])
        late["date"] = date(2015, 6, 1)
        lake.write_year([early, late], year=2015)
        span = lake.symbol_span().set_index("symbol")
        assert span.loc["DEADCO", "n_days"] == 1
        # DuckDB returns Timestamps; coerce before comparing.
        assert as_date(span.loc["RELIANCE", "last_seen"]) == date(2015, 6, 1)

    def test_predicate_pushdown_filters_by_symbol_and_date(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        lake.write_year([canonical_frame()], year=2015)
        out = lake.read_prices(symbols=["TCS"], start=date(2015, 1, 1), end=date(2015, 12, 31))
        assert list(out["symbol"]) == ["TCS"]

    def test_non_eq_series_is_excluded_by_default(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        mixed = canonical_frame(
            [canonical_row("RELIANCE"), canonical_row("SOMEBOND", series="N1")]
        )
        lake.write_year([mixed], year=2015)
        assert set(lake.read_prices()["symbol"]) == {"RELIANCE"}

    def test_group_by_year_buckets_frames(self) -> None:
        a = canonical_frame()
        b = canonical_frame()
        b["date"] = date(2016, 3, 4)
        buckets = group_by_year([a, b])
        assert sorted(buckets) == [2015, 2016]


# ==========================================================================
# Adjustment
# ==========================================================================


class TestAdjustment:
    def _split(self, ex_date: date, ratio: float = 0.5) -> CorporateAction:
        return CorporateAction(
            symbol="RELIANCE",
            action_type=CorporateActionType.SPLIT,
            ex_date=ex_date,
            ratio=ratio,
        )

    def test_dates_before_ex_date_are_scaled(self) -> None:
        factors = cumulative_factors(
            [self._split(date(2015, 3, 4))],
            symbol="RELIANCE",
            dates=[date(2015, 3, 3), date(2015, 3, 4), date(2015, 3, 5)],
        )
        assert factors[date(2015, 3, 3)] == pytest.approx(0.5)
        # On the ex-date the exchange already reports the post-split price.
        assert factors[date(2015, 3, 4)] == pytest.approx(1.0)
        assert factors[date(2015, 3, 5)] == pytest.approx(1.0)

    def test_multiple_actions_compound(self) -> None:
        factors = cumulative_factors(
            [self._split(date(2015, 3, 4)), self._split(date(2016, 3, 4))],
            symbol="RELIANCE",
            dates=[date(2015, 3, 3)],
        )
        assert factors[date(2015, 3, 3)] == pytest.approx(0.25)

    def test_dividends_do_not_adjust_price(self) -> None:
        div = CorporateAction(
            symbol="RELIANCE",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=date(2015, 3, 4),
            ratio=1.0,
        )
        factors = cumulative_factors([div], symbol="RELIANCE", dates=[date(2015, 3, 3)])
        assert factors[date(2015, 3, 3)] == pytest.approx(1.0)

    def test_implausible_ratio_raises(self) -> None:
        """Guards against a malformed ratio annihilating a price series."""
        with pytest.raises(AdjustmentError, match="implausible"):
            cumulative_factors(
                [self._split(date(2015, 3, 4), ratio=1e-9)],
                symbol="RELIANCE",
                dates=[date(2015, 3, 3)],
            )

    def test_volume_is_adjusted_inversely_to_price(self) -> None:
        """A 1:2 split halves price and doubles share count."""
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "RELIANCE", "close": 100.0,
                 "open": 100.0, "high": 100.0, "low": 100.0, "prev_close": 100.0,
                 "volume": 1_000_000, "turnover": 1e8},
            ]
        )
        out = adjust_symbol(prices, [self._split(date(2015, 3, 4))], symbol="RELIANCE")
        assert out.loc[0, "close"] == pytest.approx(50.0)
        assert out.loc[0, "volume"] == 2_000_000

    def test_turnover_is_not_adjusted(self) -> None:
        """Rupees traded is unaffected by a split, and adjusting it would break
        the Tier-2 turnover reconciliation."""
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "RELIANCE", "close": 100.0,
                 "open": 100.0, "high": 100.0, "low": 100.0, "prev_close": 100.0,
                 "volume": 1_000_000, "turnover": 1e8},
            ]
        )
        out = adjust_symbol(prices, [self._split(date(2015, 3, 4))], symbol="RELIANCE")
        assert out.loc[0, "turnover"] == pytest.approx(1e8)

    def test_continuity_sweep_finds_unexplained_breaks(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "X", "close": 100.0, "prev_close": 99.0},
                {"date": date(2015, 3, 4), "symbol": "X", "close": 51.0, "prev_close": 50.0},
            ]
        )
        breaks = continuity_breaks(prices)
        assert len(breaks) == 1
        assert not bool(breaks.loc[0, "explained"])

    def test_continuity_sweep_marks_explained_breaks(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "RELIANCE", "close": 100.0,
                 "prev_close": 99.0},
                {"date": date(2015, 3, 4), "symbol": "RELIANCE", "close": 51.0,
                 "prev_close": 50.0},
            ]
        )
        breaks = continuity_breaks(prices, actions=[self._split(date(2015, 3, 4))])
        assert bool(breaks.loc[0, "explained"])

    def test_prev_close_uses_the_prior_days_factor(self) -> None:
        """Regression. prev_close belongs to the PREVIOUS trading day, so it
        takes the previous day's factor — not this row's.

        Using this row's factor leaves prev_close unadjusted while close is
        adjusted, producing an artificial continuity break on every corporate
        action. The Tier-4 rule caught this on real data.
        """
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "RELIANCE", "close": 100.0,
                 "open": 100.0, "high": 100.0, "low": 100.0, "prev_close": 99.0,
                 "volume": 1_000_000, "turnover": 1e8},
                # Ex-date: exchange reports the post-split price, and prev_close
                # is the pre-split 100.0.
                {"date": date(2015, 3, 4), "symbol": "RELIANCE", "close": 51.0,
                 "open": 50.0, "high": 52.0, "low": 49.0, "prev_close": 100.0,
                 "volume": 2_000_000, "turnover": 1.02e8},
            ]
        )
        out = adjust_symbol(prices, [self._split(date(2015, 3, 4))], symbol="RELIANCE")

        # Day 1 fully scaled by 0.5.
        assert out.loc[0, "close"] == pytest.approx(50.0)
        # Day 2 is on/after the ex-date, so its own prices are untouched...
        assert out.loc[1, "close"] == pytest.approx(51.0)
        # ...but its prev_close takes day 1's factor and becomes comparable.
        assert out.loc[1, "prev_close"] == pytest.approx(50.0)

    def test_adjustment_leaves_no_continuity_breaks(self) -> None:
        """The end-to-end invariant: after adjustment, prev_close(t) must equal
        close(t-1) with no corporate action needed to explain it."""
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "RELIANCE", "close": 100.0,
                 "open": 100.0, "high": 100.0, "low": 100.0, "prev_close": 100.0,
                 "volume": 1_000_000, "turnover": 1e8},
                {"date": date(2015, 3, 4), "symbol": "RELIANCE", "close": 50.0,
                 "open": 50.0, "high": 50.0, "low": 50.0, "prev_close": 100.0,
                 "volume": 2_000_000, "turnover": 1e8},
            ]
        )
        out = adjust_symbol(prices, [self._split(date(2015, 3, 4))], symbol="RELIANCE")
        assert continuity_breaks(out).empty

    def test_clean_series_has_no_breaks(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": date(2015, 3, 3), "symbol": "X", "close": 100.0, "prev_close": 99.0},
                {"date": date(2015, 3, 4), "symbol": "X", "close": 102.0, "prev_close": 100.0},
            ]
        )
        assert continuity_breaks(prices).empty


# ==========================================================================
# Corporate action parsing
# ==========================================================================


class TestCorporateActionParsing:
    def test_face_value_split_ratio(self) -> None:
        parsed = ca.parse_purpose("FACE VALUE SPLIT FROM RS 10/- TO RS 2/-")
        assert parsed.action_type is CorporateActionType.SPLIT
        assert parsed.ratio == pytest.approx(0.2)
        assert parsed.confident

    def test_bonus_one_for_one(self) -> None:
        parsed = ca.parse_purpose("BONUS 1:1")
        assert parsed.action_type is CorporateActionType.BONUS
        assert parsed.ratio == pytest.approx(0.5)
        assert parsed.confident

    def test_bonus_three_for_five(self) -> None:
        parsed = ca.parse_purpose("BONUS ISSUE 3:5")
        assert parsed.ratio == pytest.approx(5 / 8)

    def test_rights_is_parsed_but_not_confident(self) -> None:
        """The true ratio needs the subscription price, so a rights issue must
        never silently explain a Tier-4 break."""
        parsed = ca.parse_purpose("RIGHTS 1:4 @ PREMIUM RS 90/-")
        assert parsed.action_type is CorporateActionType.RIGHTS
        assert not parsed.confident

    def test_dividend_does_not_change_price(self) -> None:
        parsed = ca.parse_purpose("ANNUAL GENERAL MEETING / DIVIDEND RS 5.50 PER SHARE")
        assert parsed.action_type is CorporateActionType.DIVIDEND
        assert parsed.ratio == pytest.approx(1.0)

    def test_unrecognised_text_is_ratio_one_and_not_confident(self) -> None:
        """A wrong ratio launders a bad price; a missing one gets investigated."""
        parsed = ca.parse_purpose("SOMETHING ENTIRELY UNEXPECTED")
        assert parsed.action_type is CorporateActionType.OTHER
        assert parsed.ratio == pytest.approx(1.0)
        assert not parsed.confident

    def test_empty_text_is_handled(self) -> None:
        assert ca.parse_purpose("").ratio == pytest.approx(1.0)

    def test_confident_only_drops_rights(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "A", "ex_date": date(2015, 3, 4), "purpose": "BONUS 1:1"},
                {"symbol": "B", "ex_date": date(2015, 3, 4), "purpose": "RIGHTS 1:4"},
            ]
        )
        assert len(ca.from_frame(df, confident_only=True)) == 1
        assert len(ca.from_frame(df, confident_only=False)) == 2

    def test_frame_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            ca.from_frame(pd.DataFrame([{"symbol": "A"}]))


# ==========================================================================
# Symbol map
# ==========================================================================


class TestSymbolMap:
    def test_follows_a_rename(self) -> None:
        m = SymbolMap([SymbolChange(old_symbol="OLDCO", new_symbol="NEWCO",
                                   effective_date=date(2015, 3, 4))])
        assert m.current_symbol("OLDCO") == "NEWCO"

    def test_point_in_time_resolution(self) -> None:
        """Asking what a company was called in 2014 is a different question from
        what it is called now."""
        m = SymbolMap([SymbolChange(old_symbol="OLDCO", new_symbol="NEWCO",
                                   effective_date=date(2015, 3, 4))])
        assert m.current_symbol("OLDCO", as_of=date(2014, 1, 1)) == "OLDCO"
        assert m.current_symbol("OLDCO", as_of=date(2016, 1, 1)) == "NEWCO"

    def test_chained_renames_resolve_to_the_latest(self) -> None:
        m = SymbolMap(
            [
                SymbolChange(old_symbol="A", new_symbol="B", effective_date=date(2010, 1, 1)),
                SymbolChange(old_symbol="B", new_symbol="C", effective_date=date(2015, 1, 1)),
            ]
        )
        assert m.current_symbol("A") == "C"

    def test_history_chain_stitches_a_full_history(self) -> None:
        m = SymbolMap(
            [
                SymbolChange(old_symbol="A", new_symbol="B", effective_date=date(2010, 1, 1)),
                SymbolChange(old_symbol="B", new_symbol="C", effective_date=date(2015, 1, 1)),
            ]
        )
        assert m.history_chain("C") == ("A", "B", "C")

    def test_cyclic_rename_does_not_hang(self) -> None:
        """A data error creating A->B->A would otherwise loop forever, and a
        hang inside an ingest loop looks exactly like a slow network."""
        m = SymbolMap(
            [
                SymbolChange(old_symbol="A", new_symbol="B", effective_date=date(2010, 1, 1)),
                SymbolChange(old_symbol="B", new_symbol="A", effective_date=date(2011, 1, 1)),
            ]
        )
        assert m.current_symbol("A") in {"A", "B"}

    def test_apply_preserves_the_original_symbol(self) -> None:
        """Losing the name a row actually traded under makes any later
        disagreement with the exchange unreconcilable."""
        m = SymbolMap([SymbolChange(old_symbol="RELIANCE", new_symbol="RELNEW",
                                   effective_date=date(2015, 1, 1))])
        out = m.apply(canonical_frame())
        assert "RELNEW" in set(out["symbol"])
        assert "RELIANCE" in set(out["original_symbol"])

    def test_unknown_symbol_is_returned_unchanged(self) -> None:
        assert SymbolMap().current_symbol("WHATEVER") == "WHATEVER"


# ==========================================================================
# Gate orchestration
# ==========================================================================


class TestGateOrchestration:
    def test_clean_batch_passes(self) -> None:
        from market_data.quality.rules import RuleContext

        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(), trade_date=date(2015, 3, 4))
        )
        assert outcome.verdict in {Verdict.PASS, Verdict.PASS_WITH_WARNINGS}
        assert len(outcome.accepted) == 3
        assert outcome.quarantined.empty

    def test_structural_failure_rejects_the_whole_file(self) -> None:
        from market_data.quality.rules import RuleContext

        broken = canonical_frame().drop(columns=["close"])
        outcome = QualityGate().run(RuleContext(df=broken, trade_date=date(2015, 3, 4)))
        assert outcome.verdict is Verdict.REJECTED
        assert not outcome.is_usable
        assert outcome.accepted.empty

    def test_bad_row_is_quarantined_and_good_rows_survive(self) -> None:
        from market_data.quality.rules import RuleContext

        mixed = canonical_frame(
            [
                canonical_row("GOOD"),
                canonical_row("BAD", high=90.0, low=98.0),
            ]
        )
        outcome = QualityGate().run(RuleContext(df=mixed, trade_date=date(2015, 3, 4)))
        assert outcome.verdict is Verdict.QUARANTINED
        assert outcome.is_usable
        assert set(outcome.accepted["symbol"]) == {"GOOD"}
        assert set(outcome.quarantined["symbol"]) == {"BAD"}

    def test_quarantine_records_the_rule_that_held_the_row(self) -> None:
        from market_data.quality.rules import RuleContext

        mixed = canonical_frame(
            [
                canonical_row("GOOD"),
                canonical_row("BAD", volume=-1, delivery_qty=None, delivery_pct=None),
            ]
        )
        outcome = QualityGate().run(RuleContext(df=mixed, trade_date=date(2015, 3, 4)))
        assert outcome.quarantined.loc[:, "quarantine_rule_id"].iloc[0].startswith("T2.")

    def test_more_than_twenty_bad_rows_are_all_quarantined(self) -> None:
        """Regression: evidence sampling must never truncate the quarantine set,
        or bad rows enter the lake silently."""
        from market_data.quality.rules import RuleContext

        rows = [canonical_row(f"BAD{i}", high=90.0, low=98.0) for i in range(50)]
        rows.append(canonical_row("GOOD"))
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        assert len(outcome.quarantined) == 50
        assert set(outcome.accepted["symbol"]) == {"GOOD"}

    def test_row_counts_reconcile(self) -> None:
        from market_data.quality.rules import RuleContext

        rows = [canonical_row("GOOD"), canonical_row("BAD", volume=-1, delivery_qty=None,
                                      delivery_pct=None)]
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        r = outcome.report
        assert r.rows_accepted + r.rows_quarantined == r.rows_in

    def test_only_one_series_row_is_quarantined_not_both(self) -> None:
        """Symbol-based matching would over-quarantine here: same symbol, two
        series, only one malformed."""
        from market_data.quality.rules import RuleContext

        rows = [
            canonical_row("DUAL", series="EQ"),
            canonical_row("DUAL", series="BE", high=90.0, low=98.0),
        ]
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        assert len(outcome.quarantined) == 1
        assert outcome.quarantined.iloc[0]["series"] == "BE"
        assert len(outcome.accepted) == 1


# ==========================================================================
# Quarantine and replay
# ==========================================================================


class TestQuarantineReplay:
    def test_held_rows_are_persisted_with_their_rule(self, lake_paths) -> None:
        from market_data.quality.rules import RuleContext

        lake = Lake(lake_paths)
        store = QuarantineStore(lake)
        rows = [canonical_row("GOOD"), canonical_row("BAD", volume=-1, delivery_qty=None,
                                      delivery_pct=None)]
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        store.write(outcome)
        held = store.read(date(2015, 3, 4))
        assert len(held) == 1
        assert held.iloc[0]["quarantine_rule_id"] == "T2.validity.non_negative_volume"

    def test_nothing_written_when_nothing_held(self, lake_paths) -> None:
        from market_data.quality.rules import RuleContext

        lake = Lake(lake_paths)
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(), trade_date=date(2015, 3, 4))
        )
        assert QuarantineStore(lake).write(outcome) is None

    def test_report_is_persisted_with_evidence(self, lake_paths) -> None:
        from market_data.quality.rules import RuleContext

        lake = Lake(lake_paths)
        store = QuarantineStore(lake)
        broken = canonical_frame([canonical_row("BAD", high=90.0, low=98.0)])
        outcome = QualityGate().run(RuleContext(df=broken, trade_date=date(2015, 3, 4)))
        store.write_report(outcome.report)
        reports = store.read_reports(date(2015, 3, 4))
        failing = reports[~reports["passed"]]
        assert not failing.empty
        assert failing.iloc[0]["evidence"]  # JSON string, non-empty

    def test_rule_ids_held_summarises_by_rule(self, lake_paths) -> None:
        from market_data.quality.rules import RuleContext

        lake = Lake(lake_paths)
        store = QuarantineStore(lake)
        rows = [canonical_row(f"BAD{i}", volume=-1, delivery_qty=None,
                              delivery_pct=None) for i in range(3)]
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        store.write(outcome)
        assert store.rule_ids_held(date(2015, 3, 4)) == {
            "T2.validity.non_negative_volume": 3
        }

    def test_replay_after_excluding_a_false_positive_rule(self, lake_paths) -> None:
        """The whole point of quarantine over deletion: retire a wrong rule and
        recover the rows without re-downloading anything."""
        from market_data.quality.rules import RuleContext, t2_non_negative_volume

        lake = Lake(lake_paths)
        store = QuarantineStore(lake)
        rows = [canonical_row("GOOD"), canonical_row("HELD", volume=-1, delivery_qty=None,
                                       delivery_pct=None)]
        outcome = QualityGate().run(
            RuleContext(df=canonical_frame(rows), trade_date=date(2015, 3, 4))
        )
        store.write(outcome)
        assert len(store.read(date(2015, 3, 4))) == 1

        # Retire the rule, replay.
        relaxed = QualityGate(
            rules=[r for r in QualityGate()._rules if r is not t2_non_negative_volume]
        )
        replayed = store.replay(trade_date=date(2015, 3, 4), gate=relaxed)
        assert replayed is not None
        assert len(replayed.accepted) == 1
        assert set(replayed.accepted["symbol"]) == {"HELD"}

    def test_replay_returns_none_when_nothing_is_held(self, lake_paths) -> None:
        store = QuarantineStore(Lake(lake_paths))
        assert store.replay(trade_date=date(2015, 3, 4), gate=QualityGate()) is None


# ==========================================================================
# Scoring, eligibility and the PIT universe
# ==========================================================================


def _price_history(symbol: str, *, n_days: int, turnover: float,
                   start: date = date(2015, 1, 1)) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days).date
    return pd.DataFrame(
        [
            {
                "date": d,
                "symbol": symbol,
                "series": "EQ",
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "prev_close": 100.0, "volume": 1_000_000, "turnover": turnover,
                "trades": 100, "delivery_qty": 400_000, "delivery_pct": 40.0,
                "isin": None,
            }
            for d in dates
        ]
    )


class TestScoringAndUniverse:
    def test_liquid_long_history_symbol_is_eligible(self) -> None:
        prices = _price_history("RELIANCE", n_days=800, turnover=1e9)
        as_of = max(prices["date"])
        scorer = QualityScorer()
        scores = scorer.score_all(prices, as_of=as_of, expected_days=800)
        universe = scorer.build_universe(scores, as_of=as_of)
        assert "RELIANCE" in universe.eligible_symbols

    def test_short_history_symbol_is_excluded_with_a_readable_reason(self) -> None:
        prices = _price_history("NEWCO", n_days=40, turnover=1e9)
        as_of = max(prices["date"])
        scorer = QualityScorer()
        scores = scorer.score_all(prices, as_of=as_of, expected_days=40)
        universe = scorer.build_universe(scores, as_of=as_of)
        assert "NEWCO" not in universe.eligible_symbols
        assert "trading days" in universe.excluded["NEWCO"]

    def test_illiquid_symbol_is_excluded(self) -> None:
        prices = _price_history("PENNY", n_days=800, turnover=1000.0)
        as_of = max(prices["date"])
        scorer = QualityScorer()
        scores = scorer.score_all(prices, as_of=as_of, expected_days=800)
        universe = scorer.build_universe(scores, as_of=as_of)
        assert "PENNY" in universe.excluded
        assert "liquidity" in universe.excluded["PENNY"]

    def test_delisted_symbol_stays_in_history_but_leaves_the_universe(self) -> None:
        """The core survivorship-bias property, asserted directly."""
        alive = _price_history("ALIVE", n_days=800, turnover=1e9)
        dead = _price_history("DEADCO", n_days=800, turnover=1e9,
                              start=date(2010, 1, 1))
        prices = pd.concat([alive, dead], ignore_index=True)
        as_of = max(alive["date"])

        scorer = QualityScorer()
        scores = scorer.score_all(prices, as_of=as_of, expected_days=800)
        universe = scorer.build_universe(scores, as_of=as_of)

        assert "DEADCO" in universe.symbols       # still in history
        assert "DEADCO" not in universe.eligible_symbols  # not tradeable now
        assert "ALIVE" in universe.eligible_symbols

    def test_every_excluded_symbol_has_a_reason(self) -> None:
        """No symbol may be silently dropped — the UI needs a sentence."""
        prices = pd.concat(
            [
                _price_history("GOOD", n_days=800, turnover=1e9),
                _price_history("SHORT", n_days=10, turnover=1e9),
                _price_history("THIN", n_days=800, turnover=100.0),
            ],
            ignore_index=True,
        )
        as_of = max(prices["date"])
        scorer = QualityScorer()
        scores = scorer.score_all(prices, as_of=as_of, expected_days=800)
        universe = scorer.build_universe(scores, as_of=as_of)
        ineligible = set(universe.symbols) - set(universe.eligible_symbols)
        assert ineligible == set(universe.excluded)
        assert all(universe.excluded[s] for s in ineligible)

    def test_catalog_marks_a_silent_symbol_as_delisted(self, lake_paths) -> None:
        lake = Lake(lake_paths)
        recent = _price_history("ALIVE", n_days=5, start=date(2015, 6, 1), turnover=1e9)
        old = _price_history("DEADCO", n_days=5, start=date(2015, 1, 1), turnover=1e9)
        lake.write_year([recent, old], year=2015)

        catalog = Catalog(lake)
        assert catalog.delisted_symbols(as_of=date(2015, 6, 5)) == ["DEADCO"]

    def test_pit_universe_excludes_future_information(self, lake_paths) -> None:
        """Reading the whole history then filtering would let a symbol's future
        liquidity decide whether it was eligible in the past."""
        lake = Lake(lake_paths)
        prices = _price_history("RELIANCE", n_days=400, start=date(2015, 1, 1),
                                turnover=1e9)
        lake.write_year([prices[prices["date"] < date(2016, 1, 1)]], year=2015)
        lake.write_year([prices[prices["date"] >= date(2016, 1, 1)]], year=2016)

        early = Catalog(lake).universe_as_of(date(2015, 6, 1))
        assert "RELIANCE" in early.symbols
        # Only ~100 trading days available by then, so not yet eligible.
        assert "RELIANCE" not in early.eligible_symbols

    def test_empty_lake_yields_an_empty_universe(self, lake_paths) -> None:
        universe = Catalog(Lake(lake_paths)).universe_as_of(date(2015, 3, 4))
        assert universe.total == 0


# ==========================================================================
# Pipeline
# ==========================================================================


def _write_local_delivery(root, d: date, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / f"sec_bhavdata_full_{d.strftime('%Y%m%d')}.csv",
                              index=False)


def _delivery_row(symbol: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "SYMBOL": symbol,
        "SERIES": "EQ",
        "OPEN_PRICE": 100.0,
        "HIGH_PRICE": 105.0,
        "LOW_PRICE": 98.0,
        "CLOSE_PRICE": 102.0,
        "PREV_CLOSE": 99.0,
        "TTL_TRD_QNTY": 1_000_000,
        "TURNOVER_LACS": 1020.0,
        "NO_OF_TRADES": 5000,
        "DELIV_QTY": 400_000,
        "DELIV_PER": 40.0,
    }
    return base | overrides


class TestPipeline:
    def test_ingests_a_day_end_to_end(self, lake_paths, tmp_path) -> None:
        src = tmp_path / "raw"
        _write_local_delivery(src, date(2015, 3, 4), [_delivery_row("RELIANCE")])

        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake,
            fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        result = pipeline.ingest_day(date(2015, 3, 4))

        assert result.ingested
        assert result.rows_accepted == 1
        out = lake.read_prices()
        assert out.loc[0, "symbol"] == "RELIANCE"
        assert out.loc[0, "turnover"] == pytest.approx(1020.0 * 100_000)

    def test_missing_file_is_a_holiday_not_an_error(self, lake_paths, tmp_path) -> None:
        pipeline = IngestPipeline(
            lake=Lake(lake_paths),
            fetcher=LocalBhavcopyFetcher(tmp_path / "empty"),
            calendar=TradingCalendarService(),
        )
        result = pipeline.ingest_day(date(2015, 8, 15))
        assert result.is_holiday
        assert result.error is None
        assert not result.ingested

    def test_bad_row_is_quarantined_and_not_written(self, lake_paths, tmp_path) -> None:
        src = tmp_path / "raw"
        _write_local_delivery(
            src,
            date(2015, 3, 4),
            [
                _delivery_row("GOOD"),
                _delivery_row("BAD", HIGH_PRICE=90.0, LOW_PRICE=98.0),
            ],
        )
        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake, fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        result = pipeline.ingest_day(date(2015, 3, 4))

        assert result.rows_quarantined == 1
        assert set(lake.read_prices()["symbol"]) == {"GOOD"}
        assert len(QuarantineStore(lake).read(date(2015, 3, 4))) == 1

    def test_backfill_writes_one_file_per_year(self, lake_paths, tmp_path) -> None:
        src = tmp_path / "raw"
        for d in pd.bdate_range("2015-03-02", "2015-03-06").date:
            _write_local_delivery(src, d, [_delivery_row("RELIANCE")])

        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake, fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        result = pipeline.backfill(start=date(2015, 3, 2), end=date(2015, 3, 6))

        assert result.ingested_days == 5
        year_files = list(lake_paths.dataset_dir(Dataset.PRICES).rglob("*.parquet"))
        assert len(year_files) == 1
        assert len(lake.read_prices()) == 5

    def test_backfill_skips_weekends_without_a_request(self, lake_paths, tmp_path) -> None:
        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake, fetcher=LocalBhavcopyFetcher(tmp_path / "empty"),
            calendar=TradingCalendarService(),
        )
        # 2015-03-07 and 03-08 are a weekend.
        result = pipeline.backfill(start=date(2015, 3, 7), end=date(2015, 3, 8))
        assert result.days == []

    def test_backfill_is_resumable(self, lake_paths, tmp_path) -> None:
        """A 4,700-request run will be interrupted. Restarting from 2006 every
        time makes the backfill impossible to finish."""
        src = tmp_path / "raw"
        for d in pd.bdate_range("2015-03-02", "2015-03-06").date:
            _write_local_delivery(src, d, [_delivery_row("RELIANCE")])

        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake, fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        pipeline.backfill(start=date(2015, 3, 2), end=date(2015, 3, 4))
        second = pipeline.backfill(start=date(2015, 3, 2), end=date(2015, 3, 6))

        # Only the two not-yet-done days are re-attempted.
        assert {d.trade_date for d in second.days} == {date(2015, 3, 5), date(2015, 3, 6)}

    def test_continuity_is_checked_within_an_unflushed_year(
        self, lake_paths, tmp_path
    ) -> None:
        """Buffered days must still feed the Tier-4 check, or a whole year of
        continuity goes unverified."""
        src = tmp_path / "raw"
        _write_local_delivery(src, date(2015, 3, 3),
                              [_delivery_row("RELIANCE", CLOSE_PRICE=100.0)])
        # Day 2's prev_close disagrees with day 1's close, unexplained.
        _write_local_delivery(src, date(2015, 3, 4),
                              [_delivery_row("RELIANCE", PREV_CLOSE=50.0)])

        lake = Lake(lake_paths)
        pipeline = IngestPipeline(
            lake=lake, fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        pipeline.backfill(start=date(2015, 3, 3), end=date(2015, 3, 4))

        reports = QuarantineStore(lake).read_reports(date(2015, 3, 4))
        continuity = reports[reports["rule_id"] == "T4.continuity.prev_close"]
        assert not continuity.empty
        assert not bool(continuity.iloc[0]["passed"])

    def test_corrupt_resume_state_does_not_block_a_restart(
        self, lake_paths, tmp_path
    ) -> None:
        lake_paths.ingest_state.parent.mkdir(parents=True, exist_ok=True)
        lake_paths.ingest_state.write_text("{not json")
        src = tmp_path / "raw"
        _write_local_delivery(src, date(2015, 3, 4), [_delivery_row("RELIANCE")])

        pipeline = IngestPipeline(
            lake=Lake(lake_paths), fetcher=LocalBhavcopyFetcher(src),
            calendar=TradingCalendarService(),
        )
        result = pipeline.backfill(start=date(2015, 3, 4), end=date(2015, 3, 4))
        assert result.ingested_days == 1
