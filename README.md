# Indicant

ML-powered long-term prediction for Indian (NSE) equities. Type a ticker, get a
calibrated probability that the stock is higher in N months, the drivers behind
it, and — prominently — whether the model's edge is distinguishable from chance.

It currently is not. That is stated on every screen. See [Honest results](#honest-results).

![Python](https://img.shields.io/badge/python-3.13-blue)
![Tests](https://img.shields.io/badge/tests-665%20passing-brightgreen)

---

## What it actually does

You type `RELIANCE`. The system:

1. Reads OHLCV from a local parquet lake built from **NSE bhavcopy archives**
   — the exchange's own end-of-day files, not a third-party API.
2. Rebuilds the point-in-time universe for the date in question, including
   companies that have since delisted.
3. Computes a cross-sectional feature panel — each stock's indicators *and* its
   rank against every other stock that day.
4. Runs a stacked ensemble whose out-of-fold predictions were generated under
   purged, embargoed cross-validation.
5. Returns a probability, the SHAP drivers behind it, and the regime context.

The chart, the verdict, the drivers and the screener are one HTTP request.

---

## Honest results

The model was trained on the real lake described below. These are its actual
numbers, not a target:

| Metric | Value | Reading |
|---|---|---|
| ROC AUC | **0.5470** | Barely above coin-flip, which is normal for this problem |
| ElasticNet baseline AUC | 0.5256 | The stack beats a linear baseline by +0.0214 |
| Brier skill score | +0.0027 | Calibration is marginally better than the base rate |
| **Permutation test** | **p = 0.0597** (200 shuffles) | **NOT significant at 0.05** |

**What that means:** across 200 runs on shuffled labels, about 12 did as well as
the real model. The edge is real enough to beat a linear baseline and not strong
enough to rule out luck. Gu, Kelly and Xiu found sub-1% monthly R² is state of
the art in empirical asset pricing — an AUC of 0.547 is roughly that world.

The UI never hides this. The status bar carries `p=0.0597 NOT significant`
permanently, and the footer says the edge is not distinguishable from chance.

Two design consequences follow:

- With no trained model, every prediction endpoint returns **503**, never a
  neutral 0.5. A 0.5 would reach the narrative layer and be rendered as a
  genuine call about a real company.
- A symbol the model cannot score is **absent** from the screener rather than
  present with a placeholder.

---

## The data

Built by walking the NSE bhavcopy archive. Two format eras are handled: the
legacy layout and the UDiFF layout NSE cut over to on 2024-07-08.

| | |
|---|---|
| Rows | **3,152,890** |
| Symbols | 5,885 |
| Trading days | 1,166 |
| Range | 2022-01-03 → 2026-07-30 |
| Corporate actions | 9,925 (668 price-affecting) |
| Eligible symbols (latest PIT) | 897 |
| Delisted names retained | 1,731 |

**Survivorship bias is the reason for that last row.** Screening today's listed
companies and backtesting them through 2022 silently asks "how did the survivors
do?" — and the answer is always flattering. The universe is rebuilt per date
from what was actually trading then.

A sanity check that validated the whole pipeline: reconstructed from raw
bhavcopy, INFY closed 2022 at −20.6% and TCS at −14.7% while RELIANCE was +6.0%
and HDFCBANK +7.1% — the actual 2022 Indian IT selloff, which nothing in the
code knows about.

### The quality gate

Six tiers, run before anything enters the lake: structural, validity,
completeness, continuity, plausibility, cross-source. Failing rows are
quarantined with the rule that rejected them, not dropped.

Tier 4 (continuity) earned its keep during development: it caught a bug where
`prev_close` was being scaled by its *own* row's adjustment factor instead of
the previous day's, fabricating a continuity break on every corporate action.

---

## Architecture

```
web/                    React 18 + TS terminal (nginx, the only public ingress)
  └── /api → gateway

services/
  gateway/              Composition + plain-language narrative. Public.
  market-data/          Ingest, validate, adjust, serve. SINGLE WRITER to lake.
  intelligence/         Panel, labels, stack, validation, serving. READ-ONLY lake.
  worker/               Scheduled jobs (off by default)

packages/contracts/     Shared Pydantic schemas + a schema-freeze snapshot

data/lake/              Parquet. Gitignored.
infra/                  docker-compose.yml, nginx.conf
```

Two planes, deliberately:

- **Control plane (HTTP):** small questions — what is the universe, is this
  symbol eligible, what did the gate say.
- **Data plane (shared parquet):** `intelligence` reads parquet paths directly.
  Moving a million training rows as JSON is minutes of serialization for zero
  benefit.

`market-data` mounts the lake read-write; `intelligence` mounts it **read-only**.
That boundary is enforced by the mount, not by a code review comment.

### The model stack

| Layer | What |
|---|---|
| L0 | Pooled cross-sectional panel — 94 features incl. `_xs` market ranks |
| L1 | Triple-barrier labelling (López de Prado) + average-uniqueness weights |
| L2 | Heterogeneous base learners |
| L3 | Stacking meta-learner over purged/embargoed OOF predictions |
| L5 | Meta-labelling (not trained in the current run) |
| L6 | Platt calibration |

Validation is purged + embargoed CV **split by date, not by row** — splitting a
panel by row leaks tomorrow's cross-section into today's fold. Also implemented:
CPCV, Deflated Sharpe Ratio, and a permutation test with per-fold and
within-date shuffling.

---

## Run it

```bash
cd infra && docker compose up -d --build
```

Then open **http://localhost:8080**.

The compose file bind-mounts `data/lake`, so the stack comes up on the real lake
and trained model if you have them. A named volume would start empty, and an
empty lake means a terminal that honestly reports "no lake / untrained" — running,
but useless as a demo.

Only `web` is published. `gateway`, `market-data` and `intelligence` are
reachable only on the internal network, and nginx returns 404 for `/internal/`.

### Tests

```bash
./.venv-v2/bin/python -m pytest packages services --no-cov -q
```

653 Python tests, plus 12 in `web/` via `npm test`.

---

## Design decisions worth knowing

**Colour is never the only carrier of direction.** The obvious terminal
green/red measures a CVD ΔE of 3.8 against this surface; GitHub's own dark-mode
pair measures 3.5. Both are below the ΔE 6 floor — under deuteranopia they are
effectively one colour, and most trading screens ship something in that range.
The shipped pair (`#007928` / `#d2736c`) measures **9.7**, and every directional
element still carries a glyph, a text label and a position. Set the palette
selector to `monochrome` to check: with all colour removed the screen still reads.

**Plain language is a separate layer.** `intelligence` emits facts with numbers;
`gateway` turns facts into English. So copy changes without redeploying a model,
the narrative is unit-testable with no model in the loop, and a wording change
cannot silently alter a number.

**Probabilities are never stated bare.** Not "66% chance" but "66%, and about 34
of every 100 calls like this went the other way."

---

## Status

Research project. **Not investment advice**, and by its own permutation test not
yet a demonstrated edge.
