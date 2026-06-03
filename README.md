# Indicant

ML-powered long-term stock prediction for the Indian equity market. Input a ticker, get a calibrated BUY/HOLD/SELL signal with probability, feature importances, and risk metrics — backed by a full quantitative research pipeline built from first principles.

![CI](https://github.com/manav363/indicant/actions/workflows/ci.yml/badge.svg)
![Frontend CI](https://github.com/manav363/indicant/actions/workflows/frontend.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-42%25-yellow)

---

## What it does

You type `RELIANCE`. The system:

1. Fetches 5 years of OHLCV data from NSE via yfinance (cached as parquet)
2. Computes 46 technical indicators — all implemented from scratch in NumPy
3. Creates forward-looking binary labels (will price be higher in N months?)
4. Trains a walk-forward validated XGBoost model with Platt probability calibration
5. Predicts on the most recent data point and returns a structured signal

The result: a BUY/HOLD/SELL signal with a calibrated confidence score, the top feature drivers, and a full technical indicator panel — rendered in a React dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         React + Vite                            │
│   Home (search) · StockDetail (analysis) · Universe (screener) │
│   Recharts · Tailwind · Zustand · Axios                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (nginx proxy in prod)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI (uvicorn)                           │
│   POST /api/predict · GET /api/stocks/* · GET /api/universe     │
│   Pydantic v2 schemas · CORS · timing middleware                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
    Data Layer        Feature Engine       ML Pipeline
    ──────────        ──────────────       ───────────
    yfinance          46 indicators        Walk-forward CV
    NSE universe      from scratch         (purged + embargoed)
    parquet cache     in NumPy             XGBoost + calibration
    outlier detect    5 categories         Signal generation
                                           Kelly sizing
```

---

## ML Design — the parts that matter

### Why walk-forward validation (not k-fold)

Standard k-fold cross-validation is **wrong** for financial time series. It randomly splits data, meaning test samples can appear before training samples in time — leaking future information into the model and producing artificially high accuracy that collapses in live trading.

Walk-forward validation enforces temporal ordering:

```
|─── train ───────────────|─ purge ─|─ test ─|
|─── train (extended) ────────────|─ purge ─|─ test ─|
|─── train (extended further) ──────────────|─ purge ─|─ test ─|
```

The **purge period** removes samples at the boundary whose labels overlap with the test window. For a 6-month prediction horizon, the last 126 trading days of training are purged — they have labels that point into the test period. Without purging, even a correctly ordered train/test split leaks information.

The **embargo period** removes samples at the start of the next training fold that might contain feature information derived from the test period.

### Label construction

```python
y_t = 1  if  (P_{t+horizon} - P_t) / P_t > 0
y_t = 0  otherwise
```

The last `horizon_days` rows always have NaN labels — we can't know the future price. The walk-forward splitter excludes these automatically. This is easy to get wrong; most tutorials don't handle it.

### Feature engineering from scratch

Every indicator is implemented in NumPy before being validated against library output. This matters for two reasons: it forces a precise understanding of each formula, and it means the math is auditable rather than a black box.

**Trend** — SMA, EMA, MACD, MACD Signal/Histogram, price-vs-MA distance, golden/death cross

```
EMA_t = α · P_t + (1-α) · EMA_{t-1}     where α = 2/(n+1)
MACD  = EMA(12) - EMA(26)
Signal = EMA(9) of MACD
```

**Momentum** — RSI (14, 28), Stochastic %K/%D, ROC (1/3/6/12 months), composite momentum score

```
gain_t = max(ΔP_t, 0)
loss_t = max(-ΔP_t, 0)
avg_gain = RMA(gain, 14)     # Wilder's MA: α = 1/n
RS  = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
```

**Volatility** — Bollinger Bands (%B, width), ATR, realised volatility (annualised)

```
TR_t  = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
ATR   = RMA(TR, 14)
RV_63 = std(log_returns, 63) × √252     # annualised
```

**Volume** — OBV, rolling VWAP, volume ratio, money flow ratio

```
OBV_t = OBV_{t-1} + sign(ΔP_t) × V_t
VWAP  = Σ(TP_i × V_i) / Σ(V_i)         # rolling window
```

**Regime** — ADX (+DI, -DI), trend consistency, drawdown from peak, 52-week position

```
+DM, -DM → smoothed with RMA(14) / ATR(14) → +DI, -DI
DX  = 100 × |+DI - -DI| / (+DI + -DI)
ADX = RMA(DX, 14)
```

All rolling computations use only past data. The z-score normalisation is rolling (not global) so it's safe to use as a feature without lookahead bias.

### Model stack

**Logistic regression (NumPy)** — implemented from scratch as the mathematical foundation. Gradient descent with L2 regularisation, Xavier initialisation, numerically stable sigmoid, early stopping. Used to understand what every more complex model is doing under the hood.

```
z   = Xw + b
ŷ   = σ(z) = 1 / (1 + e^{-z})
L   = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m) ||w||²
∂L/∂w = (1/m) Xᵀ(ŷ - y) + (λ/m)w
w ← w - α · ∂L/∂w
```

**XGBoost** — gradient boosting on decision trees. Each tree fits the pseudo-residuals (negative gradient of log-loss) of the previous ensemble. Regularised with L1 + L2 on leaf weights, column subsampling, row subsampling. Class imbalance handled via `scale_pos_weight`.

**Platt calibration** — raw XGBoost probabilities are overconfident. Platt scaling fits a logistic regression on top of the raw scores using held-out validation data, mapping them to calibrated probabilities. After calibration, when the model outputs P=0.70, it should be correct approximately 70% of the time.

### Signal generation

```
P(up) ≥ 0.55 → BUY
P(up) ≤ 0.45 → SELL
otherwise    → HOLD

confidence  = P(up) for BUY, 1-P(up) for SELL
strength    = strong (≥70%) | moderate (≥60%) | weak (<60%)
```

Signals are additionally checked against regime indicators (ADX, trend consistency) to flag misalignment — a BUY signal in a weak/ranging market (ADX < 20) is noted.

### Position sizing — Kelly Criterion

```
f* = (b·p - q) / b

where:
    b = win_return / loss_return    (reward-to-risk ratio)
    p = P(win) from model
    q = 1 - p

recommended = f* × 0.5             # half-Kelly (standard practice)
            × min(1, σ_target/σ)   # volatility scaling
            capped at 10%          # max per-stock allocation
```

Full Kelly is theoretically optimal but in practice too aggressive. Half-Kelly halves the position size in exchange for significantly lower variance. Volatility scaling further reduces size for high-volatility stocks.

---

## Data coverage

Tested against the full NIFTY 500 universe (504 tickers):

| Result | Count | % |
|--------|-------|---|
| Data available | 500 | 99.2% |
| Placeholder tickers (DUMMYVEDL*) | 4 | 0.8% |
| Genuine data failures | 0 | 0.0% |

Known ticker aliases handled automatically:
- `TATAMOTORS.NS` → falls back to `TATAMOTORS.BO`
- `ZOMATO.NS` → falls back to `ZOMATO.BO`

Placeholder tickers from NSE corporate restructuring events are blocklisted and excluded from the universe.

---

## Project structure

```
indicant/
├── backend/
│   ├── market_regime/
│   │   ├── data/
│   │   │   ├── fetcher.py          # yfinance with retries, parquet cache, alias map
│   │   │   ├── universe.py         # NSE index universe (NIFTY 50/100/500)
│   │   │   └── preprocessor.py     # cleaning, log returns, outlier detection
│   │   ├── features/
│   │   │   └── technical.py        # 46 indicators from scratch in NumPy (640 lines)
│   │   ├── models/
│   │   │   ├── base.py             # abstract BaseModel interface
│   │   │   ├── logistic.py         # gradient descent from scratch
│   │   │   └── gradient_boost.py   # XGBoost + Platt calibration
│   │   ├── validation/
│   │   │   └── walk_forward.py     # purged walk-forward CV + label maker
│   │   ├── signals/
│   │   │   └── generator.py        # BUY/HOLD/SELL + ensemble voting
│   │   ├── risk/
│   │   │   └── sizing.py           # Kelly Criterion position sizing
│   │   ├── api/
│   │   │   ├── main.py             # FastAPI app, CORS, middleware, dotenv
│   │   │   ├── schemas.py          # Pydantic v2 request/response models
│   │   │   └── routes/
│   │   │       ├── prediction.py   # POST /api/predict — full ML pipeline
│   │   │       ├── stocks.py       # GET /api/stocks/search + /history
│   │   │       └── universe.py     # GET /api/universe — market screener
│   │   └── pipeline.py             # CLI orchestration (indicant predict/screen)
│   ├── tests/
│   │   ├── test_data.py            # fetcher, preprocessor, walk-forward (26 tests)
│   │   ├── test_features.py        # indicator math + no-lookahead checks (17 tests)
│   │   └── test_models.py          # logistic, sigmoid, Kelly, signals (30 tests)
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx            # search + feature overview
│   │   │   ├── StockDetail.jsx     # full analysis — prediction, chart, indicators
│   │   │   └── Universe.jsx        # market screener table
│   │   ├── components/
│   │   │   ├── PredictionCard.jsx  # signal, confidence meter, P(up), horizon
│   │   │   ├── PriceChart.jsx      # price + volume (Recharts ComposedChart)
│   │   │   ├── FeaturePanel.jsx    # 46 indicators in 4 sections with gauges
│   │   │   ├── RiskMetrics.jsx     # RSI, drawdown, volatility, 52W position
│   │   │   └── StockSearch.jsx     # debounced autocomplete against NSE universe
│   │   ├── api/client.js           # axios with 120s timeout (ML pipeline time)
│   │   └── store/index.js          # Zustand global state
│   ├── nginx.conf                  # SPA routing + /api/ proxy to backend
│   ├── Dockerfile                  # multi-stage: node build → nginx serve
│   └── package.json
│
├── .github/workflows/
│   ├── ci.yml                      # backend: ruff lint + mypy + pytest (blocks merge)
│   └── frontend.yml                # frontend: npm ci + vite build + artifact upload
│
└── docker-compose.yml              # backend + frontend + volumes + health checks
```

---

## Running locally

### Prerequisites

- Python 3.11+
- Node 20+
- Docker + Docker Compose (optional)

### Dev mode

```bash
# Terminal 1 — Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn market_regime.api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### Docker (production)

```bash
cd frontend && npm install && cd ..   # generates package-lock.json
docker-compose up --build
```

Open `http://localhost`

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Full ML pipeline for a ticker |
| `GET` | `/api/stocks/search?q=RELIANCE` | Autocomplete search |
| `GET` | `/api/stocks/{ticker}/history` | OHLCV price history |
| `GET` | `/api/stocks/{ticker}/indicators` | Latest technical indicators |
| `GET` | `/api/universe?index=NIFTY50` | Ranked market screener |
| `GET` | `/health` | Health check |

Swagger UI at `http://localhost:8000/docs`

### Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE", "horizon_months": 6, "model": "gradient_boost"}'
```

```json
{
  "ticker": "RELIANCE.NS",
  "company_name": "Reliance Industries Ltd.",
  "signal": "BUY",
  "confidence": 0.6575,
  "probability_up": 0.6575,
  "horizon_months": 6,
  "current_price": 1321.2,
  "top_features": [
    {"feature": "volatility_atr_21", "importance": 41.35, "direction": "bullish"},
    {"feature": "trend_ema_50",      "importance": 38.24, "direction": "bullish"},
    {"feature": "momentum_roc_12m",  "importance": 36.24, "direction": "bearish"}
  ]
}
```

---

## Tests

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v --tb=short
```

```
73 passed in 0.92s
```

Tests cover: ticker normalisation, OHLCV validation, log return math, rolling z-score,
preprocessor pipeline, walk-forward split correctness (no overlap, temporal ordering),
label construction, all indicator math (SMA, EMA, RSI, ATR, OBV), no-lookahead SMA check,
logistic regression convergence, sigmoid bounds, gradient descent loss decrease, Kelly
Criterion bounds, signal generation, ensemble voting.

---

## CLI

```bash
# Single stock prediction
indicant predict RELIANCE --horizon 6 --model gradient_boost

# Screen an index for top BUY signals
indicant screen --index NIFTY50 --horizon 6 --top 10
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Data | yfinance, pandas, NumPy, pyarrow |
| ML | XGBoost, scikit-learn, SciPy |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS v3, Recharts, Zustand, Axios |
| DevOps | Docker, nginx, GitHub Actions, Ruff, Mypy, Pytest |

---

## What I implemented from scratch

The point of reimplementing these before reaching for libraries was to be able to answer any question about them precisely — not just call a function.

- **Sigmoid function** with numerical stability for large negative inputs
- **Binary cross-entropy loss** with L2 regularisation
- **Gradient descent** with mini-batch support, early stopping, Xavier initialisation
- **SMA, EMA** (with `adjust=False` to match TradingView behaviour)
- **Wilder's Moving Average (RMA)** used by RSI and ATR
- **RSI** with correct gain/loss separation and edge case handling (all gains → RSI=100)
- **Stochastic Oscillator** %K and %D
- **MACD** line, signal, histogram
- **Bollinger Bands** with %B and bandwidth
- **True Range** and **ATR** accounting for overnight gaps
- **OBV** with sign-based volume accumulation
- **Rolling VWAP** using typical price weighting
- **ADX** with +DI/-DI directional movement computation
- **Realised volatility** annualised with √252
- **Purged walk-forward cross-validator** with configurable purge and embargo periods
- **Forward-looking label construction** with NaN handling
- **Kelly Criterion** with half-Kelly and volatility scaling
- **Platt scaling** (logistic regression on held-out scores for calibration)

---

## Author

**Manav Garg**