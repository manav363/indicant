"""intelligence control-plane API.

Phase 1 scaffold. The prediction endpoints deliberately return 503 with
`MODEL_NOT_TRAINED` rather than a placeholder number — a fake prediction that
looks real is exactly the failure mode this project exists to avoid, and a
stubbed 0.5 probability would flow straight into the gateway's plain-language
layer and be rendered as a genuine call.

Not publicly routable. nginx exposes only the gateway.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from indicant_contracts import ErrorCode, ErrorEnvelope, LakePaths

from intelligence.data.lake_client import LakeClient

app = FastAPI(
    title="Indicant intelligence",
    version="2.0.0",
    description="Features, labelling, the layered model stack, calibration and validation.",
)


def _client() -> LakeClient:
    root = Path(os.environ.get("INDICANT_LAKE_ROOT", "./data")).expanduser().resolve()
    return LakeClient(LakePaths(root=root))


def _not_trained(detail: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=ErrorEnvelope(
            code=ErrorCode.MODEL_NOT_TRAINED,
            message=detail,
            user_message=(
                "The model has not been trained yet, so there is nothing honest to "
                "report for this stock."
            ),
        ).model_dump(),
    )


@app.get("/health")
def health() -> dict[str, object]:
    client = _client()
    days = client.trading_days()
    return {
        "status": "ok",
        "service": "intelligence",
        "lake_ready": client.is_ready,
        "trading_days": len(days),
        "panel_rows": client.row_count(adjusted=False),
        # Explicit, so nothing downstream mistakes a scaffold for a trained system.
        "model_trained": False,
        "phase": "scaffold — L0/L1/L2/L3/L5 land in phases 6-7",
    }


@app.get("/model/current")
def current_model() -> dict[str, object]:
    raise _not_trained("no model has been trained on this lake yet")


@app.post("/predict")
def predict(symbol: str, horizon_months: int = 6) -> dict[str, object]:
    raise _not_trained(f"cannot predict {symbol} at {horizon_months}m: no trained model")


@app.get("/universe/eligible")
def eligible(as_of: date) -> dict[str, object]:
    """Reads the point-in-time universe market-data wrote.

    Real now, because training must filter to this rather than to whatever
    symbols happen to appear in the price data.
    """
    symbols = _client().eligible_symbols(as_of)
    return {"as_of": as_of.isoformat(), "count": len(symbols), "symbols": symbols}
