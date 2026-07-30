"""Facts in, English out.

Pure functions over `Prediction` objects. No model, no network, no state — so
every sentence the product will ever show can be tested against a fixed input,
and a copy change cannot alter a number because this layer never computes one.
"""

from __future__ import annotations

from dataclasses import dataclass

from indicant_contracts import (
    Direction,
    ExplanationFact,
    Prediction,
    PrimaryRegime,
    Signal,
)

from gateway.narrative import copy

# Below this distance from even, the model is not separating the stock from a
# coin flip and the copy says so instead of dressing it up.
COIN_FLIP_BAND = 0.04

# Meta-labeller conviction below this is a reason to size small or skip.
LOW_CONVICTION = 0.45
HIGH_CONVICTION = 0.60


@dataclass(frozen=True)
class Narrative:
    """The rendered explanation, in parts the UI can lay out independently."""

    headline: str
    probability: str
    supports: tuple[str, ...]
    opposes: tuple[str, ...]
    regime: str | None
    conviction: str | None
    caveats: tuple[str, ...]

    def as_text(self) -> str:
        """Flat rendering, for a CLI or a plain-text export."""
        parts = [self.headline, "", self.probability, ""]
        if self.supports:
            parts.append(copy.SUPPORTS_HEADER)
            parts.extend(f"  {s}" for s in self.supports)
            parts.append("")
        if self.opposes:
            parts.append(copy.OPPOSES_HEADER)
            parts.extend(f"  {s}" for s in self.opposes)
            parts.append("")
        if not self.supports and not self.opposes:
            parts.extend([copy.NO_DRIVERS, ""])
        if self.regime:
            parts.extend([self.regime, ""])
        if self.conviction:
            parts.extend([self.conviction, ""])
        parts.extend(self.caveats)
        return "\n".join(parts).strip()


def _driver_line(fact: ExplanationFact) -> str:
    return copy.DRIVER_LINE.format(
        arrow=copy.arrow(fact.direction is Direction.SUPPORTS_UP),
        name=fact.display_name,
        value=fact.display_value,
    )


def probability_sentence(prediction: Prediction) -> str:
    """The rule the whole narrative layer exists to enforce.

    A probability is never stated without what it means when it is wrong. "66%"
    invites the reader to hear "this will go up"; "66%, and about 34 of every
    100 such calls went the other way" does not.
    """
    p_up = prediction.probability_up
    if abs(p_up - 0.5) < COIN_FLIP_BAND:
        return copy.LOW_CONFIDENCE.format(symbol=prediction.symbol)

    pct = round(p_up * 100)
    return copy.PROBABILITY.format(
        symbol=prediction.symbol,
        horizon=copy.horizon_text(prediction.horizon_months),
        pct=f"{pct}%",
        misses=100 - pct,
    )


def regime_sentence(prediction: Prediction) -> str | None:
    if prediction.regime is None:
        return None

    directional = prediction.signal in {Signal.BUY, Signal.SELL}
    conflicting = (
        (prediction.signal is Signal.BUY and prediction.regime is PrimaryRegime.BEAR)
        or (prediction.signal is Signal.SELL and prediction.regime is PrimaryRegime.BULL)
    )
    if directional and conflicting:
        return copy.REGIME_CONFLICT.format(
            signal=prediction.signal.value, regime=prediction.regime.value
        )
    return copy.REGIME.get(prediction.regime)


def conviction_sentence(prediction: Prediction) -> str | None:
    """L5 output, rendered only when the meta-labeller has actually run.

    None when untrained — inventing a confidence statement for a model that
    does not exist is exactly the class of thing this project refuses.
    """
    if prediction.conviction is None:
        return None
    if prediction.conviction >= HIGH_CONVICTION:
        return copy.CONVICTION_HIGH
    if prediction.conviction <= LOW_CONVICTION:
        return copy.CONVICTION_LOW
    return None


def caveat_sentences(*, is_significant: bool | None) -> tuple[str, ...]:
    """Always includes the not-advice line and the model's honest standing.

    `is_significant=False` is stated plainly rather than omitted. A model whose
    edge is not distinguishable from chance and says so is more trustworthy than
    one that stays quiet, and this is the project's actual position.
    """
    return (
        copy.MODEL_HONESTY.format(
            significance=copy.SIGNIFICANCE_PHRASE[is_significant]
        ),
        copy.NOT_ADVICE,
    )


def render(
    prediction: Prediction,
    *,
    is_significant: bool | None = None,
) -> Narrative:
    """Build the full narrative for one prediction."""
    key = (prediction.signal, prediction.strength)
    headline = copy.HEADLINE[key].format(
        symbol=prediction.symbol,
        horizon=copy.horizon_text(prediction.horizon_months),
    )

    supports = tuple(
        _driver_line(f)
        for f in prediction.facts
        if f.direction is Direction.SUPPORTS_UP
    )
    opposes = tuple(
        _driver_line(f)
        for f in prediction.facts
        if f.direction is Direction.SUPPORTS_DOWN
    )

    return Narrative(
        headline=headline,
        probability=probability_sentence(prediction),
        supports=supports,
        opposes=opposes,
        regime=regime_sentence(prediction),
        conviction=conviction_sentence(prediction),
        caveats=caveat_sentences(is_significant=is_significant),
    )


def render_ineligible(symbol: str, reason: str) -> str:
    """Copy for a symbol below the quality bar.

    Not an error message. The system is declining to guess, and the user is
    told exactly why — which is a better answer than a low-confidence number.
    """
    return copy.INSUFFICIENT_DATA.format(symbol=symbol.upper(), reason=reason)
