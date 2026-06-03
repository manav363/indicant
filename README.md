# Indicant — Indian Market Intelligence

> ML-powered long-term stock prediction for the NSE/BSE universe.
> Walk-forward validated · No lookahead bias · Full DevOps pipeline.

![CI](https://github.com/YOUR_USERNAME/indicant/actions/workflows/ci.yml/badge.svg)
![Frontend CI](https://github.com/YOUR_USERNAME/indicant/actions/workflows/frontend.yml/badge.svg)

---

## What This Is

Indicant is a quantitative research system that takes a stock ticker as input and outputs a **BUY / HOLD / SELL signal with calibrated probability** for a user-defined time horizon (1–24 months).

It covers the **full NSE universe** (NIFTY 50/100/500, ~1800 stocks) and is built from first principles — every technical indicator is implemented in NumPy before being enhanced with production libraries.

---

## Architecture

```
Input: "RELIANCE"
        ↓
Data Layer          yfinance → OHLCV (5yr history, cached as parquet)
        ↓
Feature Engine      46 indicators across 5 categories (NumPy from scratch)
        ↓
ML Pipeline         Walk-forward validated XGBoost + Platt calibration
        ↓
Signal Layer        BUY/HOLD/SELL + confidence score + feature importance
        ↓
REST API            FastAPI · Pydantic v2 · structured JSON
        ↓
React Frontend      Dark dashboard · Recharts · Tailwind · Zustand
```

---

## ML Design Decisions

### Why Walk-Forward Validation (not k-fold)?

Standard k-fold cross-validation is **wrong** for financial time series. It randomly splits data, meaning test samples can precede training samples in time — creating data leakage that produces artificially high accuracy.

Walk-forward validation trains only on past data and tests on future data, rolling forward in time:

```
|── train ──|── purge ──|─ test ─|
|── train + new ──|── purge ──|─ test ─|
```

The **purge period** (126 days for 6-month horizon) removes samples at the boundary whose labels overlap with the test period — eliminating lookahead bias entirely.

### Feature Engineering (from scratch)

All indicators are implemented in NumPy first, then validated against library implementations:

| Category | Indicators | Key Math |
|---|---|---|
| Trend | SMA(20/50/200), EMA(12/26/50), MACD | EMA: `αPt + (1-α)EMAt-1` |
| Momentum | RSI(14/28), Stochastic %K/%D, ROC(1/3/6/12m) | RSI: `100 - 100/(1+RS)` |
| Volatility | Bollinger Bands, ATR, Realised Vol (annualised) | ATR: `RMA(TrueRange, 14)` |
| Volume | OBV, Rolling VWAP, Volume Ratio, Money Flow | VWAP: `Σ(TP·V)/Σ(V)` |
| Regime | ADX, +DI/-DI, Trend Consistency, Drawdown, 52W Position | ADX: `RMA(DX, 14)` |

### Model Stack

1. **Logistic Regression (NumPy)** — implemented from scratch with gradient descent, L2 regularisation, and Xavier initialisation. Used to understand the math.

2. **XGBoost** — gradient boosting with Platt scaling for probability calibration. Production model.

3. **Probability calibration** — raw XGBoost scores are overconfident. Platt scaling (logistic regression on top of raw scores) maps them to honest probabilities. When the model says 70%, it's right ~70% of the time.

### Label Construction

```
y_t = 1  if  (P_{t+horizon} - P_t) / P_t > 0%
y_t = 0  otherwise
```

The last `horizon_days` rows are always NaN (future unknown). The walk-forward splitter excludes these automatically.

---

## Project Structure

```
indicant/
├── backend/
│   ├── market_regime/
│   │   ├── data/
│   │   │   ├── fetcher.py          # yfinance OHLCV with caching + retries
│   │   │   ├── universe.py         # NSE index universe loader
│   │   │   └── preprocessor.py     # cleaning, log returns, outlier detection
│   │   ├── features/
│   │   │   └── technical.py        # 46 indicators from scratch in NumPy
│   │   ├── models/
│   │   │   ├── base.py             # abstract BaseModel interface
│   │   │   ├── logistic.py         # logistic regression from scratch
│   │   │   └── gradient_boost.py   # XGBoost + Platt calibration
│   │   ├── validation/
│   │   │   └── walk_forward.py     # purged walk-forward CV + label maker
│   │   ├── signals/
│   │   │   └── generator.py        # BUY/HOLD/SELL signal logic
│   │   ├── risk/
│   │   │   └── sizing.py           # Kelly Criterion position sizing
│   │   └── api/
│   │       ├── main.py             # FastAPI app + CORS + middleware
│   │       ├── schemas.py          # Pydantic v2 request/response models
│   │       └── routes/
│   │           ├── prediction.py   # POST /api/predict
│   │           ├── stocks.py       # GET /api/stocks/search + history
│   │           └── universe.py     # GET /api/universe
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx            # landing + search
│   │   │   ├── StockDetail.jsx     # full analysis page
│   │   │   └── Universe.jsx        # market screener
│   │   ├── components/
│   │   │   ├── PredictionCard.jsx  # BUY/HOLD/SELL signal card
│   │   │   ├── PriceChart.jsx      # price + volume chart (Recharts)
│   │   │   ├── FeaturePanel.jsx    # 46 indicator grid
│   │   │   ├── RiskMetrics.jsx     # risk snapshot panel
│   │   │   └── StockSearch.jsx     # autocomplete search
│   │   ├── api/client.js           # axios API client
│   │   └── store/index.js          # Zustand global state
│   ├── Dockerfile
│   └── package.json
│
├── .github/workflows/
│   ├── ci.yml                      # backend: ruff + mypy + pytest
│   └── frontend.yml                # frontend: lint + vite build
│
└── docker-compose.yml              # spin up everything with one command
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node 20+
- Docker + Docker Compose (for containerised run)

### Dev mode (two terminals)

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

### Docker (production mode)

```bash
docker-compose up --build
```

Open `http://localhost`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Run full ML prediction for a ticker |
| `GET` | `/api/stocks/search?q=RELIANCE` | Search NSE universe |
| `GET` | `/api/stocks/{ticker}/history` | OHLCV price history |
| `GET` | `/api/stocks/{ticker}/indicators` | Latest technical indicators |
| `GET` | `/api/universe?index=NIFTY50` | Screener — ranked stock universe |
| `GET` | `/health` | Health check |

Full docs at `http://localhost:8000/docs` (Swagger UI).

### Example prediction request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE", "horizon_months": 6, "model": "gradient_boost"}'
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance, pandas, NumPy |
| ML | XGBoost, scikit-learn, SciPy |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS, Recharts, Zustand |
| DevOps | Docker, GitHub Actions, Ruff, Mypy, Pytest |

---

## What I Built From Scratch

- Logistic regression: sigmoid, binary cross-entropy, gradient descent, L2 regularisation
- All 46 technical indicators: SMA, EMA, RSI (Wilder's RMA), Stochastic, MACD, Bollinger Bands, ATR, OBV, VWAP, ADX
- Purged walk-forward cross-validator with configurable purge + embargo periods
- Label construction with forward-looking returns
- Rolling z-score normalisation (no lookahead)

---

## Author

**Manav Garg** — Built as a quantitative ML portfolio project covering the Indian equity market.
