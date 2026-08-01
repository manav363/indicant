"""market-data control-plane API.

Deliberately does NOT serve bulk OHLCV. `intelligence` reads parquet paths
directly, because moving a million rows as JSON to train a model is minutes of
serialization for zero benefit. This API answers small questions: what is the
universe, is this symbol any good, was this a trading day, what did the gate say.

Not publicly routable. nginx exposes only the gateway.
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query
from indicant_contracts import (
    Dataset,
    ErrorCode,
    ErrorEnvelope,
    QualityScore,
    SymbolMeta,
    TradingCalendar,
    UniverseSnapshot,
)

from market_data.ingest.calendar import TradingCalendarService
from market_data.quality.quarantine import QuarantineStore
from market_data.quality.scoring import QualityScorer
from market_data.settings import get_settings
from market_data.store.catalog import Catalog
from market_data.store.lake import Lake

app = FastAPI(
    title="Indicant market-data",
    version="2.0.0",
    description="Ingest, validate, adjust and serve NSE market data. Single writer to the lake.",
)


def _lake() -> Lake:
    return Lake(get_settings().paths)


def _catalog() -> Catalog:
    settings = get_settings()
    return Catalog(Lake(settings.paths), settings.eligibility)


@app.get("/health")
def health() -> dict[str, object]:
    """Reports lake state, not just process liveness.

    A service that answers 200 with an empty lake is up but useless, and the
    difference matters to whatever is deciding whether to start a training run.
    """
    lake = _lake()
    days = lake.observed_trading_days()
    return {
        "status": "ok",
        "service": "market-data",
        "lake_root": str(lake.paths.root),
        "has_data": bool(days),
        "trading_days": len(days),
        "first_date": days[0].isoformat() if days else None,
        "last_date": days[-1].isoformat() if days else None,
    }


@app.get("/universe", response_model=UniverseSnapshot)
def universe(
    as_of: date = Query(default_factory=date.today),
    index: str | None = None,
) -> UniverseSnapshot:
    """Point-in-time universe. `as_of` is required in spirit — defaulting to
    today is a convenience, but asking without a date is how survivorship bias
    gets in.
    """
    return _catalog().universe_as_of(as_of, index_name=index)


@app.get("/symbols/{symbol}/meta", response_model=SymbolMeta)
def symbol_meta(symbol: str) -> SymbolMeta:
    registry = {m.symbol: m for m in _catalog().build_symbol_registry()}
    meta = registry.get(symbol.upper())
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                code=ErrorCode.SYMBOL_NOT_FOUND,
                message=f"{symbol} is not present in the lake",
                user_message=f"We have no record of {symbol.upper()} on the NSE.",
                detail={"symbol": symbol.upper()},
            ).model_dump(),
        )
    return meta


@app.get("/symbols/{symbol}/quality", response_model=QualityScore)
def symbol_quality(
    symbol: str,
    as_of: date = Query(default_factory=date.today),
) -> QualityScore:
    settings = get_settings()
    lake = Lake(settings.paths)
    prices = lake.read_prices(symbols=[symbol.upper()], end=as_of)
    if prices.empty:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                code=ErrorCode.DATA_UNAVAILABLE,
                message=f"no prices for {symbol} up to {as_of}",
                user_message=f"We have no price history for {symbol.upper()} yet.",
                detail={"symbol": symbol.upper(), "as_of": as_of.isoformat()},
            ).model_dump(),
        )
    scorer = QualityScorer(settings.eligibility)
    expected = len([d for d in lake.observed_trading_days() if d <= as_of])
    scores = scorer.score_all(prices, as_of=as_of, expected_days=expected)
    return scores[0]


@app.get("/calendar/trading-days", response_model=TradingCalendar)
def trading_days(start: date, end: date) -> TradingCalendar:
    return TradingCalendarService.from_file().build(start, end)


# Bars beyond this and it stops being a chart request and starts being a bulk
# read, which belongs on the data plane.
MAX_HISTORY_BARS = 2000


@app.get("/symbols/{symbol}/history")
def symbol_history(
    symbol: str,
    start: date,
    end: date,
    adjusted: bool = True,
) -> list[dict[str, object]]:
    """Bounded OHLCV for ONE symbol, for drawing a chart.

    This does not contradict the "no bulk OHLCV over JSON" rule. That rule
    exists because a training panel is ~10^6 rows and serialising it is minutes
    of waste — `intelligence` reads parquet directly for exactly that reason.

    A chart is a different object: one symbol, a year, ~250 rows, ~25KB of JSON.
    That is a control-plane-sized answer to a control-plane-sized question, and
    the alternative — mounting the lake into the gateway — would give a
    public-facing service filesystem access to the whole data plane to save
    25KB. The row cap is what keeps the distinction from eroding.
    """
    settings = get_settings()
    lake = Lake(settings.paths)

    # Adjusted prices are the default: an unadjusted chart shows a fake -50%
    # cliff on every split date, which a user reads as a crash.
    dataset = Dataset.ADJUSTED if adjusted else Dataset.PRICES
    if adjusted and not lake.has_data(Dataset.ADJUSTED):
        dataset = Dataset.PRICES

    frame = lake.read_prices(
        symbols=[symbol.upper()], start=start, end=end, dataset=dataset
    )
    if frame.empty:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                code=ErrorCode.DATA_UNAVAILABLE,
                message=f"no rows for {symbol} in {start}..{end}",
                user_message=f"We have no price history for {symbol.upper()} in that period.",
                detail={"symbol": symbol.upper()},
            ).model_dump(),
        )

    frame = frame.sort_values("date").tail(MAX_HISTORY_BARS)
    return [
        {
            "time": str(row["date"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        }
        for _, row in frame.iterrows()
    ]


@app.get("/internal/quality/quarantine")
def quarantine_summary(trade_date: date | None = None) -> dict[str, object]:
    store = QuarantineStore(_lake())
    return {"held_by_rule": store.rule_ids_held(trade_date)}


@app.get("/internal/quality/runs")
def quality_runs(trade_date: date | None = None, limit: int = 200) -> dict[str, object]:
    store = QuarantineStore(_lake())
    frame = store.read_reports(trade_date)
    if frame.empty:
        return {"runs": []}
    failing = frame[~frame["passed"]].head(limit)
    return {
        "total_rule_results": len(frame),
        "failures": failing.to_dict("records"),
    }


@app.get("/internal/quality/coverage")
def coverage() -> dict[str, object]:
    """Predicted vs observed trading days.

    The one endpoint that answers "is the lake missing anything", which is the
    question a nightly job should be alerting on.
    """
    lake = _lake()
    observed = lake.observed_trading_days()
    if not observed:
        return {"coverage": 0.0, "observed": 0, "missing": []}
    calendar = TradingCalendarService.from_file()
    rec = calendar.reconcile(start=observed[0], end=observed[-1], observed=observed)
    return {
        "coverage": rec.coverage,
        "observed": len(observed),
        "missing": [d.isoformat() for d in rec.missing_trading_days[:50]],
        "unexpected": [d.isoformat() for d in rec.unexpected_trading_days[:50]],
        "uncurated_years": list(rec.uncurated_years),
    }
