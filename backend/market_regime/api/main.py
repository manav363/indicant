"""
market_regime/api/main.py
──────────────────────────
FastAPI application entrypoint.

To run locally:
    uvicorn market_regime.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

# ── Load .env FIRST before anything else reads env vars ───────────────────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from market_regime.api.routes import prediction, stocks, universe
from market_regime.api.schemas import HealthResponse

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Indicant API starting up...")
    logger.info("Universe cache will be loaded on first request.")
    yield
    logger.info("Indicant API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indicant API",
    description=(
        "ML-powered long-term stock prediction for the Indian market (NSE/BSE). "
        "Provides technical analysis, regime detection, and BUY/HOLD/SELL signals."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.debug(
        "%s %s → %d (%.0fms)",
        request.method, request.url.path, response.status_code, elapsed_ms
    )
    return response

# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(stocks.router)
app.include_router(prediction.router)
app.include_router(universe.router)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(status="ok", version="0.1.0", model_loaded=True)

@app.get("/", tags=["system"])
async def root():
    return {"name": "Indicant API", "version": "0.1.0", "docs": "/docs"}
