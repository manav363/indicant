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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from market_regime.backtest.engine import BacktestConfig, run_backtest
from market_regime.backtest.permutation_test import PermutationConfig, PermutationTest
from market_regime.data.fetcher import fetch_ohlcv
from market_regime.data.preprocessor import preprocess
from market_regime.data.universe import get_tickers
from market_regime.features.technical import add_all_features
from market_regime.models.gradient_boost import GradientBoostModel
from market_regime.models.logistic import LogisticRegressionScratch
from market_regime.validation.walk_forward import WalkForwardConfig, prepare_ml_dataset

if TYPE_CHECKING:
    from market_regime.registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("trend_", "momentum_", "volatility_", "volume_", "regime_")

# Default registry path (configurable via env or param)
_DEFAULT_REGISTRY_PATH = "model_registry.db"


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
    registry: Optional[ModelRegistry] = None,
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

    Parameters
    ----------
    ticker : str
        NSE ticker, e.g. "RELIANCE.NS"
    horizon_months : int
        Prediction horizon in months.
    model_name : str
        "gradient_boost" or "logistic"
    registry : ModelRegistry, optional
        If provided, logs the training run to the model registry.
    """
    horizon_days = horizon_months * 21

    df = fetch_ohlcv(ticker)
    df = preprocess(df)
    df = add_all_features(df)

    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    X, y = prepare_ml_dataset(df, feat_cols, horizon_days=horizon_days)

    # Build metadata for model registry (available dates from the original df)
    data_start = str(df.index[0].date()) if isinstance(df.index, pd.DatetimeIndex) else ""
    data_end = str(df.index[-1].date()) if isinstance(df.index, pd.DatetimeIndex) else ""
    meta: dict[str, Any] = {
        "ticker": ticker,
        "data_start": data_start,
        "data_end": data_end,
        "horizon_days": horizon_days,
        "label_threshold": 0.0,
    }

    if model_name == "logistic":
        model = LogisticRegressionScratch()
        model.fit(X.values, y.values, feat_cols, registry=registry, metadata=meta)
    else:
        model = GradientBoostModel()
        model.fit(X.values, y.values, feat_cols, registry=registry, metadata=meta)

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


def _print_backtest_result(result: object) -> None:
    """Pretty-print a BacktestResult to the terminal."""
    from market_regime.backtest.engine import BacktestResult

    bt = result
    assert isinstance(bt, BacktestResult)
    print(f"\n{'─'*60}")
    print(f"  Walk-Forward Backtest:  {bt.ticker} ({bt.n_folds} folds)")
    print(f"  Period:                 {bt.start_date}  →  {bt.end_date}")
    print(f"{'─'*60}")
    print(f"  Total Return            {bt.total_return:>+9.2%}")
    print(f"  CAGR                    {bt.cagr:>+9.2%}")
    print(f"  Sharpe                  {bt.sharpe:>9.2f}")
    print(f"  Sortino                 {bt.sortino:>9.2f}")
    print(f"  Max Drawdown            {bt.max_dd:>+9.2%}")
    print(f"  Win Rate                {bt.win_rate:>9.1%}")
    print(f"  Profit Factor           {bt.profit_factor:>9.2f}")
    print(f"  Turnover (daily avg)    {bt.turnover:>9.1%}")
    print(f"  Trades                  {bt.num_trades:>9d}")
    print(f"{'─'*60}\n")


def run_backtest_cli(
    ticker: str,
    horizon_months: int = 6,
    model_name: str = "gradient_boost",
    eval_freq: str = "weekly",
    registry_path: Optional[str] = None,
    n_permutations: int = 0,
) -> object:
    """Run backtest from CLI: fetch → feature → walk-forward → print."""
    horizon_days = horizon_months * 21

    df = fetch_ohlcv(ticker)
    df = preprocess(df)
    df = add_all_features(df)

    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    n_features = len(feat_cols)

    if model_name == "logistic":
        from market_regime.models.logistic import LogisticRegressionScratch
        model = LogisticRegressionScratch()
    else:
        model = GradientBoostModel()
        model.config.n_estimators = 200

    wf_config = WalkForwardConfig(
        purge_days=horizon_days,
        embargo_days=max(21, horizon_days // 6),
        min_train_years=2,
        test_months=3,
    )
    bt_config = BacktestConfig(evaluation_freq=eval_freq)

    registry: Any = None
    run_id: Optional[str] = None
    if registry_path is not None:
        from market_regime.registry.model_registry import ModelRegistry
        registry = ModelRegistry(registry_path)
        registry.create_tables()
        data_start = str(df.index[0].date()) if isinstance(df.index, pd.DatetimeIndex) else ""
        data_end = str(df.index[-1].date()) if isinstance(df.index, pd.DatetimeIndex) else ""
        run_id = registry.log_run({
            "ticker": ticker,
            "model_type": model_name,
            "model_config": model.config,
            "data_start": data_start,
            "data_end": data_end,
            "n_samples": len(df),
            "n_features": n_features,
            "horizon_days": horizon_days,
            "label_threshold": 0.0,
            "feature_list": feat_cols,
        })

    result = run_backtest(
        ticker=ticker,
        model=model,
        df=df,
        feature_cols=feat_cols,
        horizon_days=horizon_days,
        wf_config=wf_config,
        bt_config=bt_config,
        registry=registry,
        run_id=run_id,
    )

    if n_permutations > 0:
        print(f"\nRunning permutation test ({n_permutations} permutations)...")
        perm_test = PermutationTest(PermutationConfig(n_permutations=n_permutations))
        perm_result = perm_test.run(
            ticker=ticker,
            model=model,
            df=df,
            feature_cols=feat_cols,
            horizon_days=horizon_days,
            wf_config=wf_config,
            bt_config=bt_config,
            registry=registry,
            run_id=run_id,
        )
        print(f"\n  Permutation test: p={perm_result.p_value:.4f} "
              f"(actual Sharpe={perm_result.actual_sharpe:.4f}, "
              f"null={perm_result.null_mean:.4f}±{perm_result.null_std:.4f}, "
              f"95th={perm_result.null_95pct:.4f})")
        sig = "SIGNIFICANT" if perm_result.significant_at_5pct else "NOT significant"
        print(f"  → {sig} at 5% level")

    return result


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

    # ── indicant backtest ────────────────────────────────────────────────────
    bt_parser = subparsers.add_parser("backtest", help="Walk-forward backtest evaluation")
    bt_parser.add_argument("ticker", type=str, help="NSE ticker, e.g. RELIANCE")
    bt_parser.add_argument("--horizon", type=int, default=6, help="Months ahead (default: 6)")
    bt_parser.add_argument("--model", choices=["gradient_boost", "logistic"],
                           default="gradient_boost")
    bt_parser.add_argument("--eval-freq", choices=["weekly", "daily"], default="weekly",
                           help="Rebalance cadence: weekly (every 5 days) or daily (default: weekly)")
    bt_parser.add_argument("--registry", type=str, default=None,
                           help="Path to model_registry.db to persist run (e.g. model_registry.db)")
    bt_parser.add_argument("--permutations", type=int, default=0,
                           help="Run permutation test with N shuffles (e.g. 200)")

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

    elif args.command == "backtest":
        ticker = args.ticker.upper()
        if "." not in ticker:
            ticker = f"{ticker}.NS"
        print(f"\nRunning walk-forward backtest for {ticker} ({args.horizon}-month horizon, {args.eval_freq})...")
        result = run_backtest_cli(
            ticker, args.horizon, args.model, args.eval_freq,
            args.registry, args.permutations,
        )
        _print_backtest_result(result)


if __name__ == "__main__":
    main()
