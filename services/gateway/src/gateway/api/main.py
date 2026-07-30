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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
