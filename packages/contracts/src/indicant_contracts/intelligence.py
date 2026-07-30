"""Intelligence service contracts.

The important boundary here: `ExplanationFact` is structured and numeric.
The intelligence service never emits prose. The gateway turns facts into
English, so user-facing wording can change without redeploying a model.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Signal(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Strength(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class Direction(StrEnum):
    SUPPORTS_UP = "supports_up"
    SUPPORTS_DOWN = "supports_down"
    NEUTRAL = "neutral"


class PrimaryRegime(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    RANGING = "ranging"


class CompositeSignal(StrEnum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


class ExplanationFact(BaseModel):
    """One SHAP contribution, ready for templating. No prose.

    `display_name` and `display_value` exist so the gateway never has to know
    that `momentum_roc_6m` means "6-month price change" — that mapping lives
    with the feature that defines it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    display_name: str
    value: float
    display_value: str
    shap: float
    direction: Direction
    rank: int = Field(ge=1)


class Prediction(BaseModel):
    """A calibrated call plus everything needed to justify and size it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    as_of: date
    horizon_months: int = Field(ge=1, le=24)
    signal: Signal
    probability_up: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    strength: Strength
    conviction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Meta-labeller P(this call is correct). None until L5 is trained.",
    )
    current_price: float = Field(gt=0)
    suggested_position_pct: float = Field(ge=0, le=100)
    regime: PrimaryRegime | None = None
    regime_aligned: bool | None = None
    facts: tuple[ExplanationFact, ...] = ()
    model_run_id: str | None = None

    @model_validator(mode="after")
    def _confidence_matches_signal(self) -> Prediction:
        """Confidence is the probability of the direction actually called.

        Getting this backwards produces a UI that shows 30% confidence on a
        strong SELL, which is the kind of bug nobody notices for months.
        """
        expected = {
            Signal.BUY: self.probability_up,
            Signal.SELL: 1.0 - self.probability_up,
            Signal.HOLD: max(self.probability_up, 1.0 - self.probability_up),
        }[self.signal]
        if abs(self.confidence - expected) > 1e-6:
            raise ValueError(
                f"confidence {self.confidence} inconsistent with {self.signal} "
                f"at p_up={self.probability_up} (expected {expected})"
            )
        return self


class RegimeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    as_of: date
    primary_regime: PrimaryRegime
    composite_signal: CompositeSignal
    trend_direction: str
    volatility_regime: str
    drawdown_regime: str
    regime_score: float = Field(ge=0, le=1)
    adx: float | None = None


class MarketRegimeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: date
    majority_regime: PrimaryRegime
    composite_signal: CompositeSignal
    regime_distribution: dict[str, float]
    median_adx: float | None = None
    constituents_reporting: int = Field(ge=0)
    total_constituents: int = Field(ge=0)
    cache_ttl_minutes: int = Field(ge=0)

    @property
    def reporting_ratio(self) -> float:
        """Guards against a partial constituent fetch being averaged away
        into a confident-looking market call.
        """
        if self.total_constituents == 0:
            return 0.0
        return self.constituents_reporting / self.total_constituents


class CalibrationBin(BaseModel):
    """One bucket of a reliability diagram. Perfect calibration is the
    diagonal: mean_predicted == observed_rate in every bin.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bin_lower: float = Field(ge=0, le=1)
    bin_upper: float = Field(ge=0, le=1)
    mean_predicted: float = Field(ge=0, le=1)
    observed_rate: float = Field(ge=0, le=1)
    count: int = Field(ge=0)


class ModelCard(BaseModel):
    """What the /model page renders. Every field here is evidence, not a claim.

    `permutation_p_value` and `deflated_sharpe` are present specifically so a
    null result is a first-class, publishable output rather than something
    buried.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trained_at: datetime
    model_type: str
    n_train_samples: int = Field(ge=0)
    n_features: int = Field(ge=0)
    universe_size: int = Field(ge=0)
    train_start: date
    train_end: date

    oos_sharpe: float | None = None
    oos_sortino: float | None = None
    oos_max_drawdown: float | None = None
    cost_adjusted_sharpe: float | None = None
    accuracy: float | None = None
    brier_score: float | None = None

    permutation_p_value: float | None = Field(default=None, ge=0, le=1)
    n_permutations: int | None = Field(default=None, ge=0)
    deflated_sharpe: float | None = None
    cpcv_sharpe_mean: float | None = None
    cpcv_sharpe_std: float | None = None

    baseline_model: str | None = None
    baseline_oos_sharpe: float | None = None

    calibration: tuple[CalibrationBin, ...] = ()

    @property
    def beats_baseline(self) -> bool | None:
        """None when there is nothing to compare against — deliberately not
        False, so 'untested' never reads as 'failed'.
        """
        if self.oos_sharpe is None or self.baseline_oos_sharpe is None:
            return None
        return self.oos_sharpe > self.baseline_oos_sharpe

    def is_significant(self, alpha: float = 0.05) -> bool | None:
        """None when no permutation test has been run. Not False — 'untested'
        must never render as 'tested and failed'.
        """
        if self.permutation_p_value is None:
            return None
        return self.permutation_p_value < alpha
