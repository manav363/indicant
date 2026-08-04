"""intelligence HTTP API.

Serves a trained artifact. The model is loaded ONCE at startup, not per request:
a stack is tens of megabytes of fitted trees and reloading it per call would put
that on the latency of every prediction.

The refusal path is the important one. With no artifact every prediction
endpoint returns 503 `model_not_trained` rather than a neutral 0.5 — a 0.5 would
reach the gateway's narrative layer and be rendered as a genuine call about a
real company.

Not publicly routable. nginx exposes only the gateway.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from indicant_contracts import ErrorCode, ErrorEnvelope, LakePaths, Prediction

from intelligence.data.lake_client import (
    LakeClient,
    LakeNotAdjusted,
    UniverseNotComputed,
)
from intelligence.serving import ModelNotTrained, PredictionService, load_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indicant intelligence",
    version="2.0.0",
    description="Features, labelling, the layered model stack, calibration and validation.",
)

_state: dict[str, Any] = {"client": None, "service": None, "model": None, "error": None}


def _lake_root() -> Path:
    return Path(os.environ.get("INDICANT_LAKE_ROOT", "/data")).expanduser().resolve()


@app.on_event("startup")
def _load() -> None:
    """Load the lake client and, if present, the model.

    A missing artifact is NOT a startup failure — the service comes up and
    reports `model_trained: false`. Refusing to start would mean the whole stack
    cannot boot until someone has trained, which makes the first run impossible.
    """
    root = _lake_root()
    client = LakeClient(LakePaths(root=root))
    _state["client"] = client

    try:
        model = load_model(root / "models" / "current.pkl")
        _state["model"] = model
        _state["service"] = PredictionService(client, model)
        logger.info("loaded model %s (%d features)", model.run_id, len(model.feature_names))
    except ModelNotTrained as exc:
        _state["error"] = str(exc)
        logger.warning("no model available: %s", exc)
    except Exception as exc:
        # An artifact that EXISTS but will not load is a different failure from
        # one that was never trained, and it used to take the whole service
        # down in a crash-loop — which reaches the user as a blank page with no
        # explanation anywhere. Come up, serve /health, and carry the real
        # reason so it is diagnosable from outside the container. (The first
        # instance of this was a missing libgomp1 for LightGBM.)
        _state["error"] = f"model artifact present but failed to load: {exc}"
        _state["load_failed"] = True
        logger.exception("model failed to load")


def _service() -> PredictionService:
    svc = _state.get("service")
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorEnvelope(
                code=ErrorCode.MODEL_NOT_TRAINED,
                message=_state.get("error") or "no model loaded",
                user_message=(
                    "The model has not been trained yet, so there is nothing "
                    "honest to report for this stock."
                ),
            ).model_dump(),
        )
    return svc


@app.get("/health")
def health() -> dict[str, Any]:
    client: LakeClient | None = _state.get("client")
    model = _state.get("model")
    days = client.trading_days() if client else []
    return {
        "status": "ok",
        "service": "intelligence",
        "lake_ready": bool(client and client.is_ready),
        "trading_days": len(days),
        "last_date": days[-1].isoformat() if days else None,
        # Explicit, so nothing downstream mistakes an untrained service for a
        # trained one that happens to be quiet.
        "model_trained": model is not None,
        "model_run_id": getattr(model, "run_id", None),
        "p_value": getattr(model, "p_value", None),
        # Distinguishes "never trained" from "trained but broken". Without it
        # both look identical from the outside, and only one is a bug.
        "model_load_failed": bool(_state.get("load_failed")),
        "model_error": _state.get("error"),
    }


@app.get("/model/current")
def current_model() -> dict[str, Any]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorEnvelope(
                code=ErrorCode.MODEL_NOT_TRAINED,
                message=_state.get("error") or "no model trained",
                user_message="No model has been trained on this data yet.",
            ).model_dump(),
        )
    return {
        "run_id": model.run_id,
        "trained_at": model.trained_at,
        "model_type": "stacked",
        "n_features": len(model.feature_names),
        "universe_size": len(model.universe),
        "permutation_p_value": model.p_value,
        "is_significant": model.is_significant,
    }


@app.get("/predict", response_model=Prediction)
def predict(
    symbol: str = Query(min_length=1, max_length=32),
    horizon_months: int = Query(default=6, ge=1, le=24),
) -> Prediction:
    try:
        return _service().predict(symbol, horizon_months=horizon_months)
    except HTTPException:
        raise
    except (KeyError, LakeNotAdjusted, UniverseNotComputed) as exc:
        # A symbol we cannot serve is a scope refusal, not a crash — and the
        # reason travels to the user rather than being flattened into a 500.
        raise HTTPException(
            status_code=422,
            detail=ErrorEnvelope(
                code=ErrorCode.INSUFFICIENT_HISTORY,
                message=str(exc),
                user_message=(
                    f"We do not have enough reliable history for "
                    f"{symbol.upper()} to give an honest read."
                ),
                detail={"symbol": symbol.upper()},
            ).model_dump(),
        ) from exc


# Declared BEFORE /regime/{symbol}: FastAPI matches in order, and the
# parameterised route would otherwise capture "market" as a ticker and
# return a confusing 422 about insufficient history for a stock that does
# not exist.
@app.get("/regime/market")
def market_regime() -> dict[str, Any]:
    """Market-wide regime, aggregated over the model's own universe.

    `constituents_reporting` is carried so a call built from 12 of 30 symbols is
    visibly different from one built from all 30, rather than averaged into
    false confidence.
    """
    svc = _service()
    model = _state["model"]
    client: LakeClient = _state["client"]

    days = client.trading_days()
    if not days:
        raise HTTPException(status_code=503, detail={"code": "data_unavailable"})

    counts: dict[str, int] = {}
    reporting = 0
    universe = model.universe[:30]
    for sym in universe:
        try:
            p = svc.predict(sym, horizon_months=6)
        except Exception:
            continue
        reporting += 1
        if p.regime:
            counts[p.regime.value] = counts.get(p.regime.value, 0) + 1

    if not counts:
        return {
            "as_of": days[-1].isoformat(),
            "majority_regime": None,
            "constituents_reporting": reporting,
            "total_constituents": len(universe),
        }

    majority = max(counts, key=lambda k: counts[k])
    return {
        "as_of": days[-1].isoformat(),
        "majority_regime": majority,
        "regime_distribution": counts,
        "constituents_reporting": reporting,
        "total_constituents": len(universe),
        "reporting_ratio": reporting / max(1, len(universe)),
    }


@app.get("/regime/{symbol}")
def regime(symbol: str) -> dict[str, Any]:
    """Per-stock regime, from the same features a prediction uses.

    Computed from the prediction rather than a second pass, so the regime shown
    beside a call can never disagree with the one its features encoded.
    """
    try:
        p = _service().predict(symbol, horizon_months=6)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorEnvelope(
                code=ErrorCode.INSUFFICIENT_HISTORY,
                message=str(exc),
                user_message=f"No regime read available for {symbol.upper()}.",
            ).model_dump(),
        ) from exc
    return {
        "symbol": p.symbol,
        "as_of": p.as_of.isoformat(),
        "primary_regime": p.regime.value if p.regime else None,
    }


@app.get("/screen")
def screen(
    horizon_months: int = Query(default=6, ge=1, le=24),
    top: int = Query(default=25, ge=1, le=100),
    sort: str = Query(default="probability", pattern="^(probability|change|symbol)$"),
) -> list[dict[str, Any]]:
    """Rank the model's universe.

    Ranks the TRAINING universe rather than every eligible symbol: a
    cross-sectionally-trained model can only score symbols whose cross-section it
    can reproduce, and quietly scoring others would be train/serve skew wearing a
    table.
    """
    svc = _service()
    model = _state["model"]
    return svc.screen(
        list(model.universe), horizon_months=horizon_months, top=top, sort=sort
    )


@app.get("/universe/eligible")
def eligible(as_of: date) -> dict[str, Any]:
    client: LakeClient = _state["client"]
    try:
        symbols = client.eligible_symbols(as_of)
    except UniverseNotComputed as exc:
        raise HTTPException(
            status_code=503,
            detail=ErrorEnvelope(
                code=ErrorCode.DATA_UNAVAILABLE,
                message=str(exc),
                user_message="The tradeable-stock list has not been built yet.",
            ).model_dump(),
        ) from exc
    return {"as_of": as_of.isoformat(), "count": len(symbols), "symbols": symbols}
