# How Indicant works: one request, one training run

A guided tour of the two paths through the system, with the file behind each step. The [README](../README.md) explains *what* the project is and why it reports its own uncertainty; this page explains *how the code gets there*.

There are two paths:

1. **Serving**: a person types a ticker and gets a call. This happens in milliseconds, from a saved model.
2. **Training**: the model is built from the data lake and tested against chance. This takes minutes and runs offline.

They are kept in separate modules on purpose (`serving.py` vs `train.py`), so a prediction endpoint never ends up rebuilding features for the whole market to answer one question.

## Path 1: a stock screen request

`GET /api/stock/{symbol}` is handled by the **gateway** (`services/gateway/src/gateway/api/terminal.py`). The browser only ever talks to nginx, and nginx only forwards to the gateway.

1. **Fan out.** The gateway makes four upstream calls *in parallel* with `gather_upstreams` (`composition/client.py`): price history and symbol metadata from **market-data**, and the prediction and regime from **intelligence**. In parallel, page latency is the slowest call, not the sum of all four.
2. **Degrade, don't crash.** Each upstream returns an `UpstreamResult` with an `ok` flag instead of raising. Only the price history is essential. If the prediction fails, the page still renders the chart and says why the prediction is missing (`predictionUnavailable`).
3. **Shape the payload.** `charts/payloads.py` turns bars into candlestick and volume series. The prose a user reads is produced in `narrative/`, from structured facts. No sentence is ever produced by the model service.

### Inside the prediction (`services/intelligence/src/intelligence/serving.py`)

`PredictionService.predict(symbol)` does this, in order:

1. **Refuse without a model.** No trained artifact raises `ModelNotTrained`, which the API turns into a 503. It never returns a neutral 0.5, because a 0.5 would be rendered as a real call.
2. **Rebuild the cross-section.** The model uses `_xs` features ("this stock's RSI relative to the rest of the market today"). Those only exist relative to a set of symbols, so serving rebuilds the panel for the whole training universe. The panel is memoised per trading day and bounded to two entries.
3. **Take the symbol's latest row** and check that every feature the model expects exists. If not, it raises: the model and the feature code have drifted apart.
4. **Score it.** NaNs are filled with 0.0 (training does the same in `train.py`, so the two sides agree) and `predict_proba` gives the probability of "up".
5. **Turn probability into a call.** At or above 0.55 is BUY, at or below 0.45 is SELL, anything between is HOLD. Confidence is the probability of the direction actually called, so a SELL at p_up = 0.30 is 70% confident. Strength is STRONG at 0.70+, MODERATE at 0.60+.
6. **Explain it.** SHAP produces structured driver facts. If explaining fails, the answer is an empty list, never an invented explanation.
7. **Add the regime.** From features already computed: ADX under 20 is RANGING, otherwise BULL or BEAR depending on price versus the 200-day average. A missing feature gives `None`, not a default.

## Path 2: a training run (`train.py`, `Trainer.run`)

`indicant-ml train` walks these layers. Every layer name here matches the README's model table.

| Step | What happens | Why it matters |
|---|---|---|
| L0 panel | Pooled cross-section of symbols from the point-in-time universe, 94 features | One model sees the whole market, not one stock at a time |
| L1 labels | Triple-barrier labelling plus average-uniqueness sample weights | Labels reflect a profit target, a stop and a time limit; overlapping labels do not get counted as independent evidence |
| L2 out-of-fold | Base learners predict only on folds they did not train on, with `purge_days` set to the label horizon and a 21-day embargo | Stops information from the future leaking across the train/test boundary |
| L3 stack | A meta-learner combines those out-of-fold predictions | The ensemble is judged only on honest predictions |
| L6 calibration | Reliability curve and Brier score | Checks that "66%" means roughly 66% |
| L8 permutation test | The *entire* fold loop is re-run on shuffled labels, many times | Answers "could luck alone score this well?" |

The artifact that gets saved carries the run id, the feature names, the **training universe** (needed to rebuild the cross-section at serve time) and the p-value. `TrainConfig` defaults to 50 permutations; the README's run used 200. The smallest p-value a test with N shuffles can report is 1/(N+1), and the verdict text says so when the result lands on that floor.

## Path 0: where the data comes from (`services/market-data`)

Market-data is the **only writer** to the parquet lake; intelligence mounts it read-only, enforced by Docker rather than by convention. Incoming rows go through a six-tier quality gate (`quality/gate.py`). Rows that fail are returned *separately* as `quarantined`, with the rule that rejected them, so a caller cannot accidentally write them to the lake.

## What to be careful claiming

- The headline result is **not statistically significant** (AUC 0.547, p = 0.0597 over 200 shuffles). The project's point is that it measures this honestly.
- The model artifact is a Python **pickle**. Only load artifacts you trained yourself.
- The training universe is 30 symbols, which caps what the screener can serve.
- Position sizing (`_position_size`) is a capped heuristic, `min(10, edge × confidence × 10)` percent and zero for HOLD. Its docstring calls it half-Kelly, but it is not the Kelly formula, so describe it as a heuristic.
- L5 meta-labelling is implemented and tested but not trained in the current run.

## Questions this code should let you answer

- Why does an untrained model return a 503 instead of a 0.5?
- Why is cross-validation split by *date* with purging and an embargo, instead of by row?
- What does the triple-barrier label say that "price went up" does not?
- Why does serving one symbol require building the panel for thirty?
- Why does the gateway write the English and the model service only emit facts?
