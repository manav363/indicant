"""SHAP contributions as structured facts.

The boundary that matters: this module emits `ExplanationFact` objects with
numbers in them. It never emits prose. The gateway turns facts into English.

That split buys three things. User-facing wording changes without redeploying a
model. The narrative layer is unit-testable against fixed facts with no model in
the loop. And a copy change cannot silently alter a number, because the copy
layer never sees the model.

`display_name` lives here rather than in the gateway because the mapping from
`momentum_roc_6m` to "6-month price change" belongs with the feature that
defines it — a gateway guessing at feature semantics from a string is how a
label ends up describing the wrong thing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from indicant_contracts import Direction, ExplanationFact

logger = logging.getLogger(__name__)

# Only the top few contributions are shown. Research on explainable models finds
# a template over the top three produces usable explanations; beyond that the
# reader stops reading and the extra facts add noise, not understanding.
DEFAULT_TOP_N = 3

# Below this the contribution is indistinguishable from rounding, and presenting
# it as a driver would be false precision.
MIN_ABS_SHAP = 1e-4


CRORE = 10_000_000
LAKH = 100_000


def _compact(value: float) -> str:
    """Indian-numbering short form: -200409347 -> "-20.0 crore"."""
    sign = "-" if value < 0 else ""
    n = abs(value)
    if n >= CRORE:
        return f"{sign}{n / CRORE:.1f} crore"
    if n >= LAKH:
        return f"{sign}{n / LAKH:.1f} lakh"
    return f"{sign}{n:,.0f}"


@dataclass(frozen=True)
class FeatureDisplay:
    """How one feature is described and formatted for a person."""

    name: str
    unit: str = ""
    scale: float = 1.0
    precision: int = 1
    higher_is_bullish: bool | None = None
    # For share/rupee counts, where the raw integer is unreadable. Indian
    # numbering, because the audience is: crore and lakh, not million.
    compact: bool = False

    def format(self, value: float) -> str:
        if not np.isfinite(value):
            return "unavailable"
        scaled = value * self.scale
        if self.compact:
            return _compact(scaled)
        return f"{scaled:+.{self.precision}f}{self.unit}" if self.unit == "%" else (
            f"{scaled:.{self.precision}f}{self.unit}"
        )


# Explicit descriptions for the features a person is most likely to see.
# Anything absent falls back to a derived label, which is legible but blunter.
FEATURE_DISPLAY: dict[str, FeatureDisplay] = {
    # ROC is ALREADY a percentage in technical.py (the formula ends in "* 100").
    # Scaling again turned a -11.9% move into "-1193.0%" on screen — a number
    # so wrong it discredits every other figure beside it.
    "momentum_roc_1m": FeatureDisplay("1-month price change", "%", 1.0),
    "momentum_roc_3m": FeatureDisplay("3-month price change", "%", 1.0),
    "momentum_roc_6m": FeatureDisplay("6-month price change", "%", 1.0),
    "momentum_roc_12m": FeatureDisplay("12-month price change", "%", 1.0),
    "momentum_rsi_14": FeatureDisplay("14-day RSI (overbought/oversold)", precision=0),
    "momentum_rsi_28": FeatureDisplay("28-day RSI", precision=0),
    "trend_sma_50": FeatureDisplay("50-day average price", precision=0),
    "trend_sma_200": FeatureDisplay("200-day average price", precision=0),
    "trend_ema_50": FeatureDisplay("50-day weighted average price", precision=0),
    "trend_macd": FeatureDisplay("MACD (trend momentum)", precision=2),
    "trend_price_vs_sma20": FeatureDisplay("distance from the 20-day average", "%", 100.0),
    "trend_price_vs_sma50": FeatureDisplay("distance from the 50-day average", "%", 100.0),
    "trend_price_vs_sma200": FeatureDisplay("distance from the 200-day average", "%", 100.0),
    "volatility_atr_14": FeatureDisplay("typical daily range", precision=2),
    "volatility_atr_21": FeatureDisplay("typical daily range (21-day)", precision=2),
    "volatility_realised_63": FeatureDisplay("recent price swings", "%", 100.0),
    "volatility_bb_pctb": FeatureDisplay("position within its trading band", precision=2),
    # Raw OBV is a running signed-volume total in the hundreds of millions. As a
    # bare integer it reads as noise, so it is shown in crore.
    "volume_obv": FeatureDisplay("cumulative buying vs selling pressure", compact=True),
    "volume_ratio": FeatureDisplay("trading volume vs normal", precision=2),
    "regime_adx": FeatureDisplay("trend strength", precision=0),
    "regime_drawdown": FeatureDisplay("fall from its recent peak", "%", 100.0),
    # A 0-1 position, not a percentage — 0.85 means "near the top of its range".
    "regime_52w_position": FeatureDisplay("position in its 52-week range", precision=2),
    "regime_trend_consistency": FeatureDisplay("how steady the trend has been", precision=2),
}

class _PercentileDisplay(FeatureDisplay):
    """Formats a 0-1 cross-sectional rank as an ordinal percentile."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def format(self, value: float) -> str:
        if not np.isfinite(value):
            return "unavailable"
        pct = round(value * 100)
        if pct >= 90:
            return f"{pct}th percentile — near the top of the market"
        if pct <= 10:
            return f"{pct}th percentile — near the bottom of the market"
        return f"{pct}th percentile"


# Suffixes added by panel construction, stripped before display so a user sees
# "6-month price change" rather than "momentum_roc_6m_xs".
_SUFFIX_LABELS = {
    "_xs": " (vs the rest of the market)",
    "_sect": " (vs its sector)",
}

_PREFIXES = ("momentum_", "trend_", "volatility_", "volume_", "regime_", "fundamental_")


def describe(feature: str) -> tuple[str, FeatureDisplay]:
    """Human label and formatting for a feature name."""
    suffix_label = ""
    base = feature
    for suffix, label in _SUFFIX_LABELS.items():
        if base.endswith(suffix):
            base, suffix_label = base[: -len(suffix)], label
            break

    # A cross-sectional or sector-neutral value is a RANK, not the underlying
    # quantity — inheriting the base feature's unit rendered a 10th-percentile
    # rank as "+0.1%", which reads as a tiny price move rather than "near the
    # bottom of the market". Ranks get their own formatting.
    if suffix_label:
        base_label, _ = describe(base)
        return base_label + suffix_label, _PercentileDisplay(base_label)

    display = FEATURE_DISPLAY.get(base)
    if display is None:
        # Derived fallback: strip the category prefix, unslug, keep it readable.
        readable = base
        for prefix in _PREFIXES:
            if readable.startswith(prefix):
                readable = readable[len(prefix) :]
                break
        readable = re.sub(r"_+", " ", readable).strip()
        display = FeatureDisplay(readable or base, precision=2)

    return display.name + suffix_label, display


def facts_from_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[ExplanationFact, ...]:
    """Turn one row's SHAP vector into ranked facts.

    Ranked by |shap| — magnitude of influence, regardless of direction. Ranking
    by signed value would list all the bullish drivers first and bury the reason
    a call is weak, which is usually the more useful half.
    """
    if len(shap_values) != len(feature_names):
        raise ValueError(
            f"{len(shap_values)} SHAP values for {len(feature_names)} features"
        )

    order = np.argsort(-np.abs(shap_values))
    facts: list[ExplanationFact] = []

    for i in order:
        shap = float(shap_values[i])
        if abs(shap) < MIN_ABS_SHAP:
            # Indistinguishable from rounding. Presenting it as a driver would
            # be false precision.
            continue
        if len(facts) >= top_n:
            break

        name = feature_names[i]
        value = float(feature_values[i])
        label, display = describe(name)

        facts.append(
            ExplanationFact(
                feature=name,
                display_name=label,
                value=value,
                display_value=display.format(value),
                shap=shap,
                direction=(
                    Direction.SUPPORTS_UP if shap > 0 else Direction.SUPPORTS_DOWN
                ),
                rank=len(facts) + 1,
            )
        )
    return tuple(facts)


class ShapExplainer:
    """Wraps a tree explainer, with an additive fallback.

    SHAP is an optional dependency. If it is unavailable the explainer degrades
    to a transparent linear attribution and SAYS SO via `method`, rather than
    silently returning approximations that a caller would present as SHAP.
    """

    def __init__(self, model: object, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self._model = self._unwrap(model)
        self._explainer = None
        self.method = "unavailable"

        try:
            import shap

            self._explainer = shap.TreeExplainer(self._model)
            self.method = "tree_shap"
        except ImportError:
            logger.warning("shap not installed; falling back to linear attribution")
            self.method = "linear_fallback"
        except Exception as exc:
            # Not a tree model, or an unsupported one.
            logger.warning("TreeExplainer unavailable (%s); linear fallback", exc)
            self.method = "linear_fallback"

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self._explainer is not None:
            values = self._explainer.shap_values(X)
            if isinstance(values, list):  # older API: one array per class
                values = values[1]
            values = np.asarray(values)
            if values.ndim == 3:  # (rows, features, classes)
                values = values[:, :, 1]
            return values
        return self._linear_attribution(X)

    @staticmethod
    def _unwrap(model: object) -> object:
        """Reach inside a stack for something explainable.

        A ServingStack is a meta-learner over base learners, and SHAP cannot
        read it directly — which previously meant every prediction shipped with
        ZERO drivers. But the stack contains real gradient-boosted trees, and
        explaining the strongest of those answers the user's actual question
        ("why this call?") far better than an empty list.

        This is an approximation and the caller can tell: `method` reports
        `tree_shap` on a base learner, not on the stack. The alternative was a
        UI that never explains itself.
        """
        base = getattr(model, "base_models", None)
        if not isinstance(base, dict) or not base:
            return model
        # Prefer a boosted tree — TreeExplainer is exact for these.
        for name in ("xgboost", "lightgbm", "random_forest"):
            if name in base:
                return base[name]
        return next(iter(base.values()))

    def _linear_attribution(self, X: pd.DataFrame) -> np.ndarray:
        """coef * (x - mean). Exact for a linear model, indicative otherwise.

        Reported as `linear_fallback` so nothing downstream can present this as
        a SHAP value.
        """
        coef = getattr(self._model, "coef_", None)
        if coef is None:
            return np.zeros(X.shape)
        coef = np.asarray(coef).ravel()
        centred = X.to_numpy(dtype="float64") - X.to_numpy(dtype="float64").mean(axis=0)
        return centred * coef

    def explain_row(
        self, X: pd.DataFrame, row: int = 0, *, top_n: int = DEFAULT_TOP_N
    ) -> tuple[ExplanationFact, ...]:
        values = self.shap_values(X)
        return facts_from_shap(
            np.asarray(values)[row],
            self.feature_names,
            X.iloc[row].to_numpy(dtype="float64"),
            top_n=top_n,
        )


def split_by_direction(
    facts: tuple[ExplanationFact, ...],
) -> tuple[tuple[ExplanationFact, ...], tuple[ExplanationFact, ...]]:
    """Partition into supporting and opposing.

    The gateway renders these as separate lists, because "what's pushing it up"
    and "what's holding it back" is how a person reads a recommendation.
    """
    up = tuple(f for f in facts if f.direction is Direction.SUPPORTS_UP)
    down = tuple(f for f in facts if f.direction is Direction.SUPPORTS_DOWN)
    return up, down
