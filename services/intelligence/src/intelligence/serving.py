"""Serving path — a trained model answering questions about today.

Separate from the training code on purpose. Training runs for minutes over a
panel; serving must answer in milliseconds from an artifact. Mixing them is how
a prediction endpoint ends up rebuilding features for the whole universe to
answer one question about one stock.

The refusal behaviour is the load-bearing part. When no model is registered this
module raises rather than returning a neutral-looking 0.5, because a 0.5 would
flow through the gateway's narrative layer and be rendered as a real call. There
is no honest prediction to make without a model, and saying so is the answer.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from indicant_contracts import (
    Direction,
    ExplanationFact,
    Prediction,
    PrimaryRegime,
    Signal,
    Strength,
)

from intelligence.data.lake_client import LakeClient
from intelligence.explain.shap_facts import ShapExplainer
from intelligence.panel.builder import PanelBuilder, PanelConfig

logger = logging.getLogger(__name__)

# Signal thresholds. Deliberately not symmetric around a hair's breadth of 0.5 —
# a model that is 50.4% confident has not said anything, and dressing that as a
# BUY is how a screener fills up with noise.
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

STRONG = 0.70
MODERATE = 0.60

# Feature computation needs this much history before its output is meaningful.
SERVING_LOOKBACK_DAYS = 500


class ModelNotTrained(RuntimeError):
    """No usable artifact. Raised rather than returning a neutral prediction.

    A 0.5 would reach the narrative layer and be rendered as a genuine call
    about a real company. Refusing is the only honest option.
    """


@dataclass
class ServedModel:
    """A loaded artifact plus the metadata needed to explain its output.

    `universe` is not optional metadata — it is required to serve correctly.
    Cross-sectional features ("this stock's RSI vs the rest of the market
    today") are defined RELATIVE to a set of symbols, so reproducing them at
    serve time means rebuilding that same set. Serving a single symbol against
    a model trained with cross-sectional features is train/serve skew, and it
    is silent: the features simply come out missing or wrong.
    """

    run_id: str
    model: Any
    feature_names: list[str]
    trained_at: str
    p_value: float | None = None
    universe: list[str] = field(default_factory=list)

    @property
    def needs_cross_section(self) -> bool:
        return any(f.endswith(("_xs", "_sect")) for f in self.feature_names)

    @property
    def is_significant(self) -> bool | None:
        if self.p_value is None:
            return None
        return self.p_value < 0.05


def load_model(artifact_path: Path) -> ServedModel:
    if not artifact_path.exists():
        raise ModelNotTrained(f"no artifact at {artifact_path}")
    with artifact_path.open("rb") as fh:
        payload = pickle.load(fh)
    return ServedModel(
        run_id=payload["run_id"],
        model=payload["model"],
        feature_names=payload["feature_names"],
        trained_at=payload["trained_at"],
        p_value=payload.get("p_value"),
        universe=list(payload.get("universe", [])),
    )


def _strength(confidence: float) -> Strength:
    if confidence >= STRONG:
        return Strength.STRONG
    if confidence >= MODERATE:
        return Strength.MODERATE
    return Strength.WEAK


def _signal(probability_up: float) -> tuple[Signal, float]:
    """Signal and the confidence in THAT signal.

    Confidence is the probability of the direction actually called, so a SELL at
    p_up=0.30 is 70% confident. The contract enforces this consistency; getting
    it backwards renders a strong SELL as 30% confident and nobody notices for
    months.
    """
    if probability_up >= BUY_THRESHOLD:
        return Signal.BUY, probability_up
    if probability_up <= SELL_THRESHOLD:
        return Signal.SELL, 1.0 - probability_up
    return Signal.HOLD, max(probability_up, 1.0 - probability_up)


class PredictionService:
    def __init__(self, client: LakeClient, model: ServedModel | None = None) -> None:
        self._client = client
        self._model = model
        self._builder = PanelBuilder(client)

    @property
    def model(self) -> ServedModel:
        if self._model is None:
            raise ModelNotTrained(
                "no model has been trained on this lake yet; run 'indicant-ml train'"
            )
        return self._model

    def predict(
        self,
        symbol: str,
        *,
        horizon_months: int = 6,
        as_of: date | None = None,
    ) -> Prediction:
        """One symbol, one horizon, from the latest row of its feature panel."""
        model = self.model  # raises ModelNotTrained before any work happens
        sym = symbol.upper()

        as_of = as_of or (self._client.trading_days() or [date.today()])[-1]
        start = as_of - timedelta(days=SERVING_LOOKBACK_DAYS)

        # A cross-sectional feature is defined relative to a SET of symbols, so
        # reproducing it means rebuilding that set. Serving one symbol alone
        # against a cross-sectionally-trained model is train/serve skew: the
        # `_xs` features come out missing, and filling them with zeros would
        # hand the model a market in which every stock is exactly average.
        needs_xs = self.model.needs_cross_section
        universe = self.model.universe or [sym]
        symbols = sorted(set(universe) | {sym}) if needs_xs else [sym]

        prices = self._client.read_panel(symbols=symbols, start=start, end=as_of)
        if prices.empty:
            raise KeyError(f"{sym}: no rows in the lake for {start}..{as_of}")
        if sym not in set(prices["symbol"]):
            raise KeyError(f"{sym}: not present in the lake for {start}..{as_of}")

        result = self._builder.build_from_prices(
            prices,
            PanelConfig(add_cross_sectional=needs_xs, drop_warmup=False),
        )
        if result.frame.empty:
            raise KeyError(f"{sym}: not enough history to compute features")

        own = result.frame[result.frame["symbol"] == sym]
        if own.empty:
            raise KeyError(
                f"{sym}: dropped during feature construction "
                f"(usually too little history in {start}..{as_of})"
            )
        latest = own.sort_values("date").iloc[[-1]]
        missing = [f for f in model.feature_names if f not in latest.columns]
        if missing:
            raise ModelNotTrained(
                f"artifact expects {len(missing)} features this panel does not have "
                f"(e.g. {missing[:3]}); the model and the feature code have drifted apart"
            )

        features = latest[model.feature_names].astype("float64")
        features = features.fillna(0.0)

        proba = float(model.model.predict_proba(features.to_numpy())[0, 1])
        signal, confidence = _signal(proba)

        facts = self._explain(model, features)

        return Prediction(
            symbol=sym,
            as_of=as_of,
            horizon_months=horizon_months,
            signal=signal,
            probability_up=proba,
            confidence=confidence,
            strength=_strength(confidence),
            conviction=None,  # L5 meta-labeller is not trained in this run
            current_price=float(latest["close"].iloc[0]),
            suggested_position_pct=self._position_size(proba, confidence),
            regime=self._regime(result.frame),
            facts=facts,
            model_run_id=model.run_id,
        )

    def _explain(
        self, model: ServedModel, features: pd.DataFrame
    ) -> tuple[ExplanationFact, ...]:
        """SHAP facts, or nothing.

        An explanation failure must not take down a prediction — but a FAKE
        explanation would be worse than none, so the fallback is an empty tuple
        and the UI shows no drivers rather than invented ones.
        """
        try:
            explainer = ShapExplainer(model.model, model.feature_names)
            return explainer.explain_row(features, 0)
        except Exception as exc:
            logger.warning("explanation failed for %s: %s", model.run_id, exc)
            return ()

    def _position_size(self, proba: float, confidence: float) -> float:
        """Half-Kelly, capped. Zero for a HOLD.

        A signal the model is not confident about should size to nothing rather
        than to a small number — "a little bit of a coin flip" is not a position.
        """
        if BUY_THRESHOLD > proba > SELL_THRESHOLD:
            return 0.0
        edge = abs(proba - 0.5) * 2
        return round(min(10.0, edge * confidence * 10.0), 2)

    def _regime(self, frame: pd.DataFrame) -> PrimaryRegime | None:
        """Regime from the already-computed features, not a second pass.

        Recomputing here would mean the regime shown beside a prediction could
        differ from the one the features encoded.
        """
        if "regime_adx" not in frame.columns:
            return None
        latest = frame.sort_values("date").iloc[-1]
        adx = latest.get("regime_adx")
        if adx is None or not np.isfinite(adx):
            return None
        if adx < 20:
            return PrimaryRegime.RANGING
        trend = latest.get("trend_price_vs_sma_200", 0.0)
        if not np.isfinite(trend):
            return PrimaryRegime.RANGING
        return PrimaryRegime.BULL if trend > 0 else PrimaryRegime.BEAR

    def screen(
        self,
        symbols: list[str],
        *,
        horizon_months: int = 6,
        top: int = 25,
        sort: str = "probability",
    ) -> list[dict[str, Any]]:
        """Rank a universe. Failures are skipped, not faked.

        A symbol that cannot be predicted is absent from the screen. Including
        it with a placeholder probability would put an invented number in a
        ranked table, which is exactly where a reader trusts numbers most.
        """
        rows: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                p = self.predict(sym, horizon_months=horizon_months)
            except Exception as exc:
                logger.debug("screen: skipping %s (%s)", sym, exc)
                continue
            rows.append(
                {
                    "symbol": p.symbol,
                    "signal": p.signal.value,
                    "probability_up": p.probability_up,
                    "confidence": p.confidence,
                    "strength": p.strength.value,
                    "current_price": p.current_price,
                    "regime": p.regime.value if p.regime else None,
                }
            )

        key = {
            "probability": lambda r: -r["probability_up"],
            "change": lambda r: -abs(r["probability_up"] - 0.5),
            "symbol": lambda r: r["symbol"],
        }[sort]
        return sorted(rows, key=key)[:top]


def fact_from_row(name: str, value: float, shap: float, rank: int) -> ExplanationFact:
    """Used by tests and by the fixture path."""
    from intelligence.explain.shap_facts import describe

    label, display = describe(name)
    return ExplanationFact(
        feature=name,
        display_name=label,
        value=value,
        display_value=display.format(value),
        shap=shap,
        direction=Direction.SUPPORTS_UP if shap > 0 else Direction.SUPPORTS_DOWN,
        rank=rank,
    )
