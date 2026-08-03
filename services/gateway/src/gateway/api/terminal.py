"""Terminal API — one composed endpoint per screen.

The editorial-era API had the frontend make several calls and stitch the results
itself. A trading terminal has a different shape: a screen is a screen, and the
browser should make ONE request for it. That is not just convenience — every
extra round trip is latency the user watches, and stitching in the browser means
the loading states multiply.

So:

    GET /api/search?q=REL          type-ahead over the eligible universe
    GET /api/stock/{symbol}        everything the stock screen renders
    GET /api/screen                the ranked table
    GET /api/market                the market-pulse header

`/api/stock` is the important one. It fans out to market-data and intelligence
in parallel and returns candles, prediction, narrative, indicators and regime in
a single response — because that is one screen.

Search is served from market-data's eligible universe rather than a full symbol
list. A terminal that autocompletes a symbol it cannot then answer for is worse
than one that never offered it.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from indicant_contracts import ErrorCode, ErrorEnvelope

from gateway.cache import TradingDayCache
from gateway.charts.payloads import (
    candlestick_payload,
    return_bars_payload,
    verdict_bar_payload,
    volume_payload,
)
from gateway.composition.client import (
    UpstreamClient,
    first_failure,
    gather_upstreams,
)

MARKET_DATA_URL = os.environ.get("INDICANT_MARKET_DATA_URL", "http://market-data:8000")
INTELLIGENCE_URL = os.environ.get("INDICANT_INTELLIGENCE_URL", "http://intelligence:8000")

router = APIRouter(prefix="/api", tags=["terminal"])

_market_data = UpstreamClient(MARKET_DATA_URL)
_intelligence = UpstreamClient(INTELLIGENCE_URL)

# Universe changes once a day; a terminal's search box hits this on every
# keystroke. Caching it is the difference between a snappy box and a slow one.
_universe_cache = TradingDayCache(ttl_seconds=900, max_entries=64)

# Type-ahead caps. A dropdown longer than this is not a dropdown.
SEARCH_LIMIT = 12
SCREEN_LIMIT = 50


def _today() -> date:
    return date.today()


async def _eligible_universe(client: httpx.AsyncClient) -> dict[str, Any]:
    """The eligible universe, cached by trading day."""
    key = "universe"
    cached = _universe_cache.get(key, trading_day=_today())
    if cached is not None:
        return cached

    result = await _market_data.get(client, "market-data", "/universe")
    if not result.ok:
        raise HTTPException(
            status_code=503,
            detail=(result.error or ErrorEnvelope(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="universe unavailable",
                user_message="We could not load the list of tradeable stocks.",
            )).model_dump(),
        )
    _universe_cache.set(key, result.data, trading_day=_today())
    return dict(result.data)


def _rank_matches(query: str, symbols: list[str]) -> list[str]:
    """Prefix matches first, then substring.

    A trader typing "REL" wants RELIANCE at the top, not RELAXO because it also
    contains those letters somewhere. Ranking by match position is the whole
    difference between a search box that feels fast and one that feels wrong.
    """
    q = query.strip().upper()
    if not q:
        return []
    prefix = [s for s in symbols if s.startswith(q)]
    contains = [s for s in symbols if q in s and not s.startswith(q)]
    return (sorted(prefix) + sorted(contains))[:SEARCH_LIMIT]


@router.get("/search")
async def search(q: str = Query(min_length=1, max_length=32)) -> dict[str, Any]:
    """Type-ahead over the ELIGIBLE universe only.

    Deliberately not over every symbol NSE lists. Autocompleting a ticker the
    system then refuses to answer for is a worse experience than never offering
    it — the refusal arrives after the user has committed to a click.
    """
    async with httpx.AsyncClient() as client:
        universe = await _eligible_universe(client)

    eligible = list(universe.get("eligible_symbols", []))
    excluded = dict(universe.get("excluded", {}))
    matches = _rank_matches(q, eligible)

    # Surface near-misses so a user who types a real but ineligible ticker is
    # told why, instead of getting an empty dropdown that looks like a typo.
    ineligible = _rank_matches(q, list(excluded))[:3]

    return {
        "query": q.upper(),
        "results": [{"symbol": s, "eligible": True} for s in matches],
        "ineligible": [
            {"symbol": s, "eligible": False, "reason": excluded[s]} for s in ineligible
        ],
        "asOf": universe.get("as_of"),
    }


@router.get("/stock/{symbol}")
async def stock(
    symbol: str,
    horizon_months: int = Query(default=6, ge=1, le=24),
    lookback_days: int = Query(default=365, ge=30, le=2000),
) -> dict[str, Any]:
    """Everything the stock screen renders, in one call.

    Four upstream calls, issued in parallel. Sequential would make the screen's
    latency their sum; this makes it their max.

    Only the price history is essential. A terminal with a chart and no
    prediction is degraded but useful; one with a prediction and no chart is
    not a terminal.
    """
    sym = symbol.strip().upper()
    end = _today()
    start = end - timedelta(days=lookback_days)

    async with httpx.AsyncClient() as client:
        results = await gather_upstreams(
            [
                (
                    _market_data, "history",
                    f"/symbols/{sym}/history?start={start}&end={end}", None,
                ),
                (_market_data, "meta", f"/symbols/{sym}/meta", None),
                (
                    _intelligence, "prediction",
                    f"/predict?symbol={sym}&horizon_months={horizon_months}", None,
                ),
                (_intelligence, "regime", f"/regime/{sym}", None),
            ],
            client=client,
        )

    failed = first_failure(results, essential=["history"])
    if failed is not None:
        envelope = failed.error
        raise HTTPException(
            status_code=failed.status_code or 503,
            detail=(envelope or ErrorEnvelope(
                code=ErrorCode.INTERNAL,
                message=f"{sym}: history unavailable",
                user_message="We could not load price history for this stock.",
            )).model_dump(),
        )

    bars = results["history"].data or []

    payload: dict[str, Any] = {
        "symbol": sym,
        "asOf": end.isoformat(),
        "horizonMonths": horizon_months,
        "candles": candlestick_payload(bars),
        "volume": volume_payload(bars),
        "meta": results["meta"].data if results["meta"].ok else None,
        "prediction": None,
        "verdictBar": None,
        "regime": results["regime"].data if results["regime"].ok else None,
        "degraded": [name for name, r in results.items() if not r.ok],
    }

    # The prediction is optional by design. When there is no trained model the
    # chart still renders, and the UI says the prediction is unavailable rather
    # than showing a fabricated one.
    if results["prediction"].ok and results["prediction"].data:
        pred = dict(results["prediction"].data)
        payload["prediction"] = pred
        payload["verdictBar"] = verdict_bar_payload(
            probability_up=pred["probability_up"],
            signal=pred["signal"],
            strength=pred["strength"],
        )
    elif results["prediction"].error is not None:
        # Carry WHY, so the UI can distinguish "no model yet" from "broken".
        payload["predictionUnavailable"] = results["prediction"].error.model_dump()

    return payload


@router.get("/screen")
async def screen(
    horizon_months: int = Query(default=6, ge=1, le=24),
    limit: int = Query(default=25, ge=1, le=SCREEN_LIMIT),
    sort: str = Query(default="probability", pattern="^(probability|change|symbol)$"),
) -> dict[str, Any]:
    """The ranked table."""
    async with httpx.AsyncClient() as client:
        result = await _intelligence.get(
            client, "intelligence",
            f"/screen?horizon_months={horizon_months}&top={limit}&sort={sort}",
        )

    if not result.ok:
        return {
            "rows": [],
            "bars": [],
            "unavailable": (result.error or ErrorEnvelope(
                code=ErrorCode.MODEL_NOT_TRAINED,
                message="screen unavailable",
                user_message="The screener needs a trained model, and there is not one yet.",
            )).model_dump(),
        }

    rows = list(result.data or [])
    return {
        "rows": rows,
        "bars": return_bars_payload(
            [
                {
                    "symbol": r["symbol"],
                    "return_pct": (r.get("probability_up", 0.5) - 0.5) * 200,
                }
                for r in rows
            ]
        ),
        "sort": sort,
        "horizonMonths": horizon_months,
    }


@router.get("/market")
async def market() -> dict[str, Any]:
    """The market-pulse header: regime, breadth, and lake freshness.

    Lake freshness is included deliberately. A terminal showing yesterday's
    close as though it were live is the single most misleading thing this UI
    could do, so the header always carries the as-of date it is really showing.
    """
    async with httpx.AsyncClient() as client:
        results = await gather_upstreams(
            [
                (_intelligence, "regime", "/regime/market", None),
                (_market_data, "health", "/health", None),
                (_intelligence, "model", "/model/current", None),
            ],
            client=client,
        )

    health = results["health"].data if results["health"].ok else {}
    model = results["model"].data if results["model"].ok else None

    p = (model or {}).get("permutation_p_value")
    return {
        "regime": results["regime"].data if results["regime"].ok else None,
        "lake": {
            "lastDate": health.get("last_date"),
            "tradingDays": health.get("trading_days", 0),
            "hasData": health.get("has_data", False),
        },
        "model": {
            "trained": model is not None,
            "runId": (model or {}).get("run_id"),
            "pValue": p,
            # Precomputed so no client re-derives the significance rule and
            # drifts from the backend's definition of it.
            "isSignificant": None if p is None else bool(p < 0.05),
        },
        "degraded": [name for name, r in results.items() if not r.ok],
    }
