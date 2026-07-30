# Indicant — Architecture Reference

## Overview

Indicant is an ML-powered long-term stock prediction system for the Indian equity market. It is organised as a single Python package (`market_regime`) with a FastAPI HTTP layer and a standalone React frontend.

The package follows a layered architecture: Data → Features → Models → Backtest → Regime → API, with cross-cutting services for experiment tracking (Model Registry) and statistical validation (Permutation Test).

```
┌──────────────────────────────────────────────────────────┐
│                        Frontend                          │
│  React 18 · Vite · Recharts · Tailwind · Zustand · Axios │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP (nginx proxy in prod)
┌────────────────────────▼─────────────────────────────────┐
│                      FastAPI                             │
│  POST /api/predict · GET /api/regime/* · /api/stocks/*   │
│  Pydantic v2 · CORS · timing middleware                  │
└──────┬──────────┬──────────┬──────────┬──────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  Data Layer  Features    Models    Regime
  ──────────  ──────────  ────────  ──────
  yfinance    46 tech     XGBoost   Per-stock
  NSE idxs    indicators  Logistic  Market-wide
  parquet     NumPy only  RandForest(cache 15m TTL)
  outlier fwd
       │          │          │          │
       └──────────┴──────────┼──────────┘
                            ▼
                  Validation Layer
              ┌─────────────────────┐
              │ Walk-forward CV     │
              │ (purge + embargo)   │
              │ 5+ folds            │
              └────────┬────────────┘
                       │
              ┌────────▼────────────┐
              │ Backtest Engine     │
              │ Sharpe · Sortino    │
              │ Max DD · CAGR       │
              │ Turnover · Win Rate │
              │ Permutation Test    │
              └────────┬────────────┘
                       │
              ┌────────▼────────────┐
              │ Model Registry      │
              │ SQLite (WAL)        │
              │ log/update/get runs │
              │ Hyperparams (JSON)  │
              │ Permutation results │
              └─────────────────────┘
```

---

## Module Reference

### `market_regime/data/` — Data Layer

| File | Responsibility |
|------|---------------|
| `fetcher.py` | OHLCV fetching via yfinance with retries, parquet cache, ticker alias map (`.NS` ↔ `.BO`) |
| `preprocessor.py` | Missing-value forward-fill, log return computation, outlier capping, `simple_return` column added |
| `universe.py` | NSE index universe manager (NIFTY 50/100/500); parquet-cached constituent lists |

**Key contracts:**
- `fetch_ohlcv(ticker, lookback_years)` → `pd.DataFrame` with columns `[open, high, low, close, volume, ticker]`
- `preprocess(df, min_rows)` → `pd.DataFrame` with added `[log_return, simple_return]`

### `market_regime/features/` — Feature Engineering

| File | Responsibility |
|------|---------------|
| `technical.py` | 46 indicators from scratch in NumPy across five categories: Trend, Momentum, Volatility, Volume, Regime |
| `fundamental.py` | (Future) Fundamental data integration |
| `macro.py` | (Future) Macro-economic indicators |

**Key functions:**
- `add_all_features(df)` → `df` with 46+ columns added; all rolling computations use only past data
- `add_regime_features(df)` → subset (ADX, DI, drawdown, 52-week position)

### `market_regime/models/` — ML Models

| File | Responsibility |
|------|---------------|
| `base.py` | Abstract `BaseModel` interface: `fit()`, `predict_proba()`, `is_fitted` |
| `gradient_boost.py` | XGBoost with Platt calibration; walk-forward compatible; model-registry integration |
| `logistic.py` | Logistic regression from scratch (gradient descent, L2, early stopping) |
| `random_forest.py` | (Future) Random Forest implementation |

**GradientBoostConfig defaults:**
- `n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`
- `early_stopping_rounds=50`, `calibrate=True`, `random_state=42`

**Validation split:** Uses `train_test_split(stratify=y)` for early stopping holdout (not sequential), preventing single-class validation folds.

### `market_regime/validation/` — Cross-Validation

| File | Responsibility |
|------|---------------|
| `walk_forward.py` | Purged walk-forward CV splitter, label construction |

**Key concepts:**
- `WalkForwardCV(n_splits=5+)` — creates temporally ordered folds
- **Purge period:** removes `horizon_days` samples at fold boundary to prevent label leakage
- **Embargo period:** removes `max(21, horizon//6)` samples at start of next training fold
- `make_labels(df['close'], horizon_days)` → binary labels: 1 if future return > 0, else 0

### `market_regime/backtest/` — Backtesting & Validation

| File | Responsibility |
|------|---------------|
| `engine.py` | `run_backtest()` — walk-forward backtest with configurable cadence, transaction costs, and position sizing |
| `metrics.py` | Pure metric functions: Sharpe, Sortino, Max DD, CAGR, turnover, win rate, profit factor |
| `permutation_test.py` | `PermutationTest` — label-shuffling significance test with configurable permutations |

**BacktestConfig:**
- `transaction_cost=0.001`, `buy_threshold=0.55`, `sell_threshold=0.45`
- `rebalance_step=5` (weekly default); `evaluation_freq='weekly'`

**PermutationTest:**
- Shuffles labels per-fold (not global) to prevent future-leakage from inflating null toward zero
- +1 correction on p-value numerator and denominator to avoid `p=0` with finite N
- Varies model `random_state` per permutation to decorrelate XGBoost's internal subsampling
- 200 permutations recommended (~76s per ticker with XGBoost)

### `market_regime/regime/` — Market Regime Detection

| File | Responsibility |
|------|---------------|
| `classifier.py` | `RegimeClassifier` — shared single-source-of-truth for all regime rules |
| `market.py` | `RegimeAggregator` — runs classifier on NIFTY 50 constituents, aggregates results |
| `config.py` | All regime threshold constants in one place |

**RegimeClassifier output (`RegimeResult`):**
- `primary_regime`: Bull / Bear / Ranging
- `composite_signal`: risk_on / risk_off / neutral
- `trend_direction`: uptrend / downtrend / ranging
- `volatility_regime`: low / normal / high
- `drawdown_regime`: peak / normal / correction / bear
- `regime_score`: 0..1
- `regime_history`: 252-day array of regime states

**RegimeAggregator (`MarketRegimeResult`):**
- `majority_regime`, `regime_distribution`, `median_adx`
- `constituents_reporting` — tracks valid-data count from day one
- `reporting_ratio` = `constituents_reporting / total_constituents`
- `cache_ttl_minutes` — advertised in response

**Cache:** In-memory TTL cache (15 min) on `RegimeAggregator.analyse()` — keyed by date+hour, auto-evicts stale entries. First request after deploy pays full constituent-fetch cost.

### `market_regime/registry/` — Model Registry

| File | Responsibility |
|------|---------------|
| `model_registry.py` | SQLite-backed `ModelRegistry` with WAL mode |

**Schema fields:** `run_id, ticker, model_type, model_config (JSON), hyperparameters, created_at, updated_at, status, oos_sharpe, oos_sortino, oos_max_dd, oos_turnover, cost_adjusted_sharpe, accuracy, precision, recall, evaluation_freq, model_artifact, permutation_p_value, n_permutations, null_sharpe_mean, null_sharpe_std, null_sharpe_95pct`

**Key methods:** `log_run()`, `update_run()` (whitelisted field updates), `get_run()`, `get_best_run()`, `list_runs()`, `delete_run()`

### `market_regime/signals/` — Signal Generation

| File | Responsibility |
|------|---------------|
| `generator.py` | BUY/HOLD/SELL signal from probabilities, ensemble voting |

### `market_regime/risk/` — Position Sizing

| File | Responsibility |
|------|---------------|
| `sizing.py` | Half-Kelly Criterion position sizing with volatility scaling and 10% max allocation |

### `market_regime/api/` — HTTP API

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app factory, CORS, middleware, environment config |
| `schemas.py` | Pydantic v2 request/response schemas |
| `routes/prediction.py` | `POST /api/predict` — full ML pipeline |
| `routes/stocks.py` | `GET /api/stocks/search`, `/api/stocks/{ticker}/history`, `GET /api/stocks/{ticker}/indicators` |
| `routes/universe.py` | `GET /api/universe` — market screener |
| `routes/regime.py` | `GET /api/regime/{ticker}` — per-stock regime; `GET /api/regime/market/summary` — market-wide regime |

### `pipeline.py` — CLI Orchestration

Single entry point for `indicant predict`, `indicant screen`, and `indicant backtest` subcommands. Wires data → features → model → registry into end-to-end runs.

---

## Design Decisions

### Why SQLite for the model registry?
- Portable (single file, no server)
- Diffable with standard tools (`sqlite3`, `diff`)
- WAL mode for safe concurrent reads
- Simple schema: runs + metrics + artifacts in one table with JSON metadata

### Why per-fold label shuffling in the permutation test?
Global shuffling (shuffle all labels once, then re-split) can create walk-forward folds where train and test contain overlapping shuffled-label patterns, inflating the null toward zero and reducing test power. Per-fold shuffling ensures each fold sees an independent randomisation.

### Why ±1 p-value correction?
With finite permutations, the minimum achievable p-value is `1 / (n_perms + 1)`. Without the +1 correction, p=0 would be reported when no null value exceeds the actual — implying infinite significance. The +1 on both numerator and denominator gives `p = (count_ge + 1) / (n + 1)`, ensuring `p > 0` always.

### Why weekly rebalancing by default?
Daily rebalancing multiplies transaction costs and can degrade Sharpe ratios for medium-frequency signals. Weekly (5 trading days) is the default; daily is available via `BacktestConfig(rebalance_step=1)`.

### Why calibrated probabilities?
Raw XGBoost probabilities are overconfident (tend toward 0 or 1). Platt scaling maps them to well-calibrated probabilities via a logistic regression on held-out validation scores.

### Why stratified validation split (not sequential)?
The sequential 80/20 split can produce validation sets concentrated in a single market regime, containing only one label class. This breaks XGBoost's early stopping. A stratified random split preserves class distribution.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Tests** | 208 passing |
| **Coverage** | 92.43% |
| **Technical indicators** | 46 (from scratch in NumPy) |
| **Model types** | XGBoost, Logistic Regression |
| **Permutation test** | 200 permutations, per-fold shuffling, 5-ticker run (none significant at p < 0.05) |
| **NIFTY coverage** | 500/504 tickers (99.2%) |
| **Backend lines** | ~3800 Python |
| **Frontend components** | 7 pages/components |
