"""Every user-facing sentence, in one place.

Template-based, not LLM-generated. Deterministic, testable, zero latency, and
it cannot hallucinate a number — the right trade for a project whose entire
pitch is that its outputs are honest. An LLM can be layered on later to rephrase
these, with the figures pinned so it can only change wording.

The governing rule for all of it:

    **Never state a probability without stating what it means in failure
    terms.**

"66% chance of being higher" invites the reader to hear "this will go up".
"66% — out of every 100 similar calls, about 34 went the other way" does not.
The second sentence is not a disclaimer bolted on; it is the honest content of
the first.
"""

from __future__ import annotations

from indicant_contracts import PrimaryRegime, Signal, Strength

# ---------------------------------------------------------------- headlines

HEADLINE = {
    (Signal.BUY, Strength.STRONG): "{symbol} looks clearly positive over the next {horizon}.",
    (Signal.BUY, Strength.MODERATE): "{symbol} looks moderately positive over the next {horizon}.",
    (Signal.BUY, Strength.WEAK): "{symbol} leans slightly positive over the next {horizon}, but only slightly.",
    (Signal.SELL, Strength.STRONG): "{symbol} looks clearly negative over the next {horizon}.",
    (Signal.SELL, Strength.MODERATE): "{symbol} looks moderately negative over the next {horizon}.",
    (Signal.SELL, Strength.WEAK): "{symbol} leans slightly negative over the next {horizon}, but only slightly.",
    (Signal.HOLD, Strength.STRONG): "{symbol} looks genuinely balanced over the next {horizon}.",
    (Signal.HOLD, Strength.MODERATE): "{symbol} looks balanced over the next {horizon} — no clear direction either way.",
    (Signal.HOLD, Strength.WEAK): "We do not have a clear read on {symbol} over the next {horizon}.",
}

# ------------------------------------------------------- probability framing

PROBABILITY = (
    "The model puts the chance of {symbol} being higher in {horizon} at {pct}. "
    "That is not a promise: out of every 100 calls like this one, roughly "
    "{misses} went the other way."
)

LOW_CONFIDENCE = (
    "This is close to a coin flip. The model is not finding much to separate "
    "{symbol} from an even chance right now, and saying so is more useful than "
    "manufacturing a view."
)

# --------------------------------------------------------------- driver lists

SUPPORTS_HEADER = "What's pushing it up:"
OPPOSES_HEADER = "What's holding it back:"
NO_DRIVERS = "No single factor stands out as a driver here."

DRIVER_LINE = "{arrow} {name} is {value}"

# ------------------------------------------------------------------- regime

REGIME = {
    PrimaryRegime.BULL: "The wider market is in an uptrend, which is consistent with this signal.",
    PrimaryRegime.BEAR: "The wider market is in a downtrend, which works against this signal.",
    PrimaryRegime.RANGING: "The wider market is directionless right now, which makes any single-stock call less reliable.",
}

REGIME_CONFLICT = (
    "Note the tension: this is a {signal} signal while the wider market is in a "
    "{regime} phase. Signals that fight the market resolve less often."
)

# ------------------------------------------------------------- conviction (L5)

CONVICTION_HIGH = (
    "A second model, which only judges whether calls like this one tend to be "
    "correct, rates this one above average."
)
CONVICTION_LOW = (
    "A second model, which only judges whether calls like this one tend to be "
    "correct, is not convinced by this one — which is a reason to size it small "
    "or skip it."
)

# ------------------------------------------------------------------ caveats

NOT_ADVICE = (
    "This is a research output, not investment advice, and not a recommendation "
    "to buy or sell anything."
)

MODEL_HONESTY = (
    "Across its whole test history this model's edge was {significance}. "
    "See the model page for the full evidence."
)

SIGNIFICANCE_PHRASE = {
    True: "statistically distinguishable from chance",
    False: "NOT statistically distinguishable from chance",
    None: "never formally tested",
}

INSUFFICIENT_DATA = (
    "We do not have enough reliable data on {symbol} to give an honest read — "
    "{reason}. Rather than guess, we would rather say so."
)

# --------------------------------------------------------------- horizon text

HORIZON = {
    1: "month",
    3: "3 months",
    6: "6 months",
    12: "a year",
    24: "two years",
}


def horizon_text(months: int) -> str:
    return HORIZON.get(months, f"{months} months")


def arrow(supports_up: bool) -> str:
    """Direction glyph.

    Paired with colour AND a text label in the UI, never used alone. Red/green
    is the most common colour-blindness pair, so direction is encoded three
    ways and the charts must still read with all hue removed.
    """
    return "▲" if supports_up else "▼"
