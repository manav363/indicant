"""Gateway — the only publicly exposed service.

Phase 1 scaffold. What is real already: the upstream health fan-out, and the
composition boundary itself. What is not: prediction routes, because there is
no trained model, and returning a plausible-looking number would be the exact
dishonesty this project is built to avoid.

The narrative layer lives here rather than in `intelligence` on purpose.
`intelligence` emits structured `ExplanationFact`s; the gateway turns them into
English. That means user-facing wording changes without redeploying a model, and
the copy is unit-testable against fixed facts with no model in the loop.
"""

from __future__ import annotations

import asyncio
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from indicant_contracts import Prediction

from gateway.api.terminal import router as terminal_router
from gateway.api.terminal import warm_universe_cache
from gateway.charts.payloads import verdict_bar_payload
from gateway.composition.client import (
    UpstreamClient,
    first_failure,
    gather_upstreams,
)
from gateway.narrative.templates import render

MARKET_DATA_URL = os.environ.get("INDICANT_MARKET_DATA_URL", "http://market-data:8000")
INTELLIGENCE_URL = os.environ.get("INDICANT_INTELLIGENCE_URL", "http://intelligence:8000")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("INDICANT_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

app = FastAPI(
    title="Indicant gateway",
    version="2.0.0",
    description="Composes market-data and intelligence. The only public surface.",
)

app.include_router(terminal_router)


# asyncio only holds a WEAK reference to a running task. Without a strong one
# the warm-up could be garbage-collected mid-flight, and the symptom would be
# the thing it exists to prevent — a 15s first request — appearing at random.
_background: set[asyncio.Task[None]] = set()


@app.on_event("startup")
async def _warm() -> None:
    """Fire and forget — startup must not block on an upstream."""
    task = asyncio.create_task(warm_universe_cache())
    _background.add(task)
    task.add_done_callback(_background.discard)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def _probe(client: httpx.AsyncClient, name: str, url: str) -> dict[str, object]:
    try:
        resp = await client.get(f"{url}/health", timeout=5.0)
        resp.raise_for_status()
        return {"name": name, "reachable": True, "detail": resp.json()}
    except Exception as exc:
        # Report the failure rather than raising: a degraded upstream should be
        # visible in the health payload, not turn the gateway itself into a 500.
        return {"name": name, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/health")
async def health() -> dict[str, object]:
    """Fans out to both upstreams in parallel.

    Sequential probes would make gateway health latency the sum of every
    upstream timeout, which is the wrong signal when one is slow.
    """
    async with httpx.AsyncClient() as client:
        upstreams = await asyncio.gather(
            _probe(client, "market-data", MARKET_DATA_URL),
            _probe(client, "intelligence", INTELLIGENCE_URL),
        )
    return {
        "status": "ok" if all(u["reachable"] for u in upstreams) else "degraded",
        "service": "gateway",
        "upstreams": list(upstreams),
    }


@app.get("/api/universe")
async def universe(as_of: str | None = None, index: str | None = None) -> dict[str, object]:
    """Proxies the point-in-time universe.

    Real because the frontend needs it to know which symbols it may offer — and
    offering only what the system can answer for is what "no stock falls back"
    actually means.
    """
    params: dict[str, str] = {}
    if as_of:
        params["as_of"] = as_of
    if index:
        params["index"] = index
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{MARKET_DATA_URL}/universe", params=params, timeout=30.0)
        resp.raise_for_status()
        return dict(resp.json())


@app.get("/api/predict/{symbol}")
async def predict(symbol: str, horizon_months: int = 6) -> dict[str, object]:
    """Composed stock view: prediction + regime, rendered to plain English.

    Fans out in parallel — the two upstream calls have no dependency on each
    other, so sequential would make page latency their sum.

    The prediction is essential; the regime is not. A missing regime read
    degrades the page to one without market context, which is still useful. A
    missing prediction has nothing to show, so it surfaces the upstream's own
    error envelope rather than the gateway inventing one.
    """
    intel = UpstreamClient(INTELLIGENCE_URL)
    results = await gather_upstreams(
        [
            (intel, "prediction", f"/predict?symbol={symbol.upper()}"
                                  f"&horizon_months={horizon_months}", None),
            (intel, "regime", f"/regime/{symbol.upper()}", None),
            (intel, "model", "/model/current", None),
        ]
    )

    failed = first_failure(results, essential=["prediction"])
    if failed is not None:
        envelope = failed.error
        raise HTTPException(
            status_code=failed.status_code or 503,
            detail=envelope.model_dump() if envelope else {"code": "internal"},
        )

    payload = results["prediction"].data
    prediction = Prediction.model_validate(payload)

    model_card = results["model"].data if results["model"].ok else None
    is_significant = None
    if isinstance(model_card, dict):
        p = model_card.get("permutation_p_value")
        is_significant = None if p is None else bool(p < 0.05)

    narrative = render(prediction, is_significant=is_significant)

    return {
        "prediction": payload,
        "narrative": {
            "headline": narrative.headline,
            "probability": narrative.probability,
            "supports": list(narrative.supports),
            "opposes": list(narrative.opposes),
            "regime": narrative.regime,
            "conviction": narrative.conviction,
            "caveats": list(narrative.caveats),
        },
        "verdictBar": verdict_bar_payload(
            probability_up=prediction.probability_up,
            signal=prediction.signal.value,
            strength=prediction.strength.value,
        ),
        # Reported rather than hidden: a page built from a partial fan-out is
        # not the same object as one built from a complete fan-out.
        "degraded": [name for name, r in results.items() if not r.ok],
    }
