"""
market_regime/pipeline.py
──────────────────────────
End-to-end ML pipeline orchestration.

This is the CLI entry point (indicant CLI via pyproject.toml)
and the programmatic interface for running the full pipeline
on one or many stocks.

Usage:
    indicant --ticker RELIANCE --horizon 6
    indicant --universe NIFTY50 --horizon 6 --top 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import get_tickers, load_universe
from market_regime.features.technical import add_all_features
from market_regime.models.gradient_boost import GradientBoostModel
from market_regime.models.logistic import LogisticRegressionScratch
from market_regime.validation.walk_forward import prepare_ml_dataset

logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("trend_", "momentum_", "volatility_", "volume_", "regime_")


@dataclass
class PipelineResult:
    ticker: str
    signal: str
    confidence: float
    probability_up: float
    current_price: float
    horizon_months: int
    model: str
    top_feature: str


def run_single(
    ticker: str,
    horizon_months: int = 6,
    model_name: str = "gradient_boost",
) -> PipelineResult:
    """
    Run the full ML pipeline for a single stock.

    Steps:
    1. Fetch OHLCV
    2. Preprocess
    3. Compute 46 features
    4. Create labels + dataset
    5. Train model
    6. Predict on latest row
    """
    horizon_days = horizon_months * 21

    df = fetch_ohlcv(ticker)
    df = preprocess(df)
    df = add_all_features(df)

    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    X, y = prepare_ml_dataset(df, feat_cols, horizon_days=horizon_days)

    if model_name == "logistic":
        model = LogisticRegressionScratch()
        model.fit(X.values, y.values, feat_cols)
    else:
        model = GradientBoostModel()
        model.fit(X.values, y.values, feat_cols)

    latest = df[feat_cols].dropna().iloc[[-1]]
    proba = model.predict_proba(latest.values)
    p_up = float(proba[0, 1])

    if p_up >= 0.55:
        signal = "BUY"
    elif p_up <= 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    imp = model.feature_importance(top_n=1)
    top_feature = imp.iloc[0]["feature"] if len(imp) else "unknown"

    return PipelineResult(
        ticker=ticker,
        signal=signal,
        confidence=round(p_up if p_up >= 0.5 else 1 - p_up, 4),
        probability_up=round(p_up, 4),
        current_price=round(float(df["close"].iloc[-1]), 2),
        horizon_months=horizon_months,
        model=model_name,
        top_feature=top_feature,
    )


def run_universe(
    index: str = "NIFTY50",
    horizon_months: int = 6,
    top_n: int = 10,
) -> list[PipelineResult]:
    """Run pipeline across an NSE index and return top N BUY signals."""
    tickers = get_tickers([index])
    results = []

    for ticker in tickers:
        try:
            result = run_single(ticker, horizon_months)
            results.append(result)
            logger.info("%s → %s (%.0f%%)", ticker, result.signal, result.confidence * 100)
        except Exception as e:
            logger.warning("Skipping %s: %s", ticker, e)

    buys = [r for r in results if r.signal == "BUY"]
    buys.sort(key=lambda r: r.confidence, reverse=True)
    return buys[:top_n]


def _print_result(r: PipelineResult) -> None:
    signal_color = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"}.get(r.signal, "")
    reset = "\033[0m"
    print(f"\n{'─'*50}")
    print(f"  {r.ticker:<20} ₹{r.current_price:>10,.2f}")
    print(f"  Signal:     {signal_color}{r.signal}{reset}")
    print(f"  Confidence: {r.confidence*100:.1f}%  (P_up={r.probability_up*100:.1f}%)")
    print(f"  Horizon:    {r.horizon_months} months")
    print(f"  Top driver: {r.top_feature}")
    print(f"  Model:      {r.model}")
    print(f"{'─'*50}\n")


def main() -> None:
    """CLI entry point — registered as 'indicant' in pyproject.toml."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="indicant",
        description="Indian Market Intelligence — ML stock prediction CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── indicant predict ────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Predict a single stock")
    predict_parser.add_argument("ticker", type=str, help="NSE ticker, e.g. RELIANCE")
    predict_parser.add_argument("--horizon", type=int, default=6, help="Months ahead (default: 6)")
    predict_parser.add_argument("--model", choices=["gradient_boost", "logistic"],
                                 default="gradient_boost")

    # ── indicant screen ─────────────────────────────────────────────────────
    screen_parser = subparsers.add_parser("screen", help="Screen an NSE index")
    screen_parser.add_argument("--index", default="NIFTY50",
                                choices=["NIFTY50", "NIFTY100", "NIFTY500"])
    screen_parser.add_argument("--horizon", type=int, default=6)
    screen_parser.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.command == "predict":
        ticker = args.ticker.upper()
        if "." not in ticker:
            ticker = f"{ticker}.NS"
        print(f"\nRunning ML pipeline for {ticker} ({args.horizon}-month horizon)...")
        result = run_single(ticker, args.horizon, args.model)
        _print_result(result)

    elif args.command == "screen":
        print(f"\nScreening {args.index} ({args.horizon}-month horizon, top {args.top})...")
        results = run_universe(args.index, args.horizon, args.top)
        print(f"\nTop {len(results)} BUY signals in {args.index}:")
        for r in results:
            _print_result(r)


if __name__ == "__main__":
    main()
