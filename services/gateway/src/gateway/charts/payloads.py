"""Chart-ready payloads.

Shaped here rather than in the browser so the frontend receives exactly what it
draws. The direction encoding is the load-bearing part.

Red/green is the most common colour-vision deficiency pair — roughly 1 in 12
men. A chart where the only difference between "up" and "down" is hue is broken
for those readers. So every directional element carries FOUR encodings:

    colour   up | down | flat
    glyph    up-arrow | down-arrow | dash
    label    "Up" | "Down" | "Flat"
    sign     +/- on the value, and bars extend either side of a centre line

The acceptance test is simple: strip every colour and the chart must still read.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class BarDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"

    @property
    def glyph(self) -> str:
        return {"up": "\u25b2", "down": "\u25bc", "flat": "\u2013"}[self.value]

    @property
    def label(self) -> str:
        return {"up": "Up", "down": "Down", "flat": "Flat"}[self.value]

    @property
    def css_var(self) -> str:
        """Token name, not a hex value.

        The frontend owns the palette, including the colour-vision-safe
        alternate, so shipping hex from the API would freeze one theme into the
        wire format.
        """
        return f"--dir-{self.value}"


# Moves smaller than this are visual noise; rendering them as directional
# invents a signal from rounding.
FLAT_THRESHOLD = 1e-9


def direction_of(value: float, *, threshold: float = FLAT_THRESHOLD) -> BarDirection:
    if not isinstance(value, (int, float)) or value != value:  # NaN
        return BarDirection.FLAT
    if value > threshold:
        return BarDirection.UP
    if value < -threshold:
        return BarDirection.DOWN
    return BarDirection.FLAT


def _time_of(bar: dict[str, Any]) -> str:
    """Accept either name for the date field.

    market-data calls it `time` (the charting convention); the lake calls it
    `date`. Reading both here is one line and removes a rename step at the
    call site that silently KeyError'd when the upstream name changed.
    """
    value = bar.get("time", bar.get("date"))
    if value is None:
        raise KeyError("bar has neither 'time' nor 'date'")
    return str(value)


def _encode(value: float, *, threshold: float = FLAT_THRESHOLD) -> dict[str, Any]:
    """The four-way encoding, applied once so no caller can forget part of it."""
    d = direction_of(value, threshold=threshold)
    return {
        "direction": d.value,
        "glyph": d.glyph,
        "label": d.label,
        "colorVar": d.css_var,
    }


def candlestick_payload(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OHLC bars with per-bar direction.

    Takes the rows market-data returns, as they are. This used to take a
    DataFrame, which meant the gateway — the one internet-facing service —
    carried pandas solely to turn a list of dicts into a list of dicts, and it
    was never declared as a dependency, so it worked in a shared dev venv and
    ImportError'd in the container.

    Direction is close-vs-OPEN, not close-vs-previous-close: a candle body is
    green when the session closed above where it opened, which is what the shape
    depicts. Using previous close would colour bodies inconsistently with its
    own geometry.
    """
    return [
        {
            "time": _time_of(bar),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            **_encode(float(bar["close"]) - float(bar["open"])),
        }
        for bar in bars
    ]


def volume_payload(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Volume bars coloured by the session's own direction, matching the
    candles above them."""
    return [
        {
            "time": _time_of(bar),
            "value": float(bar["volume"]),
            **_encode(float(bar["close"]) - float(bar["open"])),
        }
        for bar in bars
    ]


def return_bars_payload(
    rows: list[dict[str, Any]],
    *,
    value_key: str = "return_pct",
    label_key: str = "symbol",
) -> list[dict[str, Any]]:
    """Horizontal bars for a screener, extending either side of a centre line.

    Position is the fourth encoding: bar geometry alone distinguishes gainers
    from losers with every colour removed.
    """
    return [
        {
            "label": row[label_key],
            "value": float(row[value_key]),
            "displayValue": f"{float(row[value_key]):+.2f}%",
            **_encode(float(row[value_key])),
        }
        for row in rows
    ]


def verdict_bar_payload(
    *,
    probability_up: float,
    signal: str,
    strength: str,
) -> dict[str, Any]:
    """The single large element at the top of a stock page.

    `magnitude` is distance from even, doubled to span 0..1, so a 50/50 call
    renders as a zero-length bar. That is deliberate: the bar's length should
    encode conviction, and an even call has none.
    """
    edge = probability_up - 0.5
    return {
        "probabilityUp": probability_up,
        "magnitude": min(1.0, abs(edge) * 2),
        "signal": signal,
        "strength": strength,
        # Explicit so the frontend does not re-derive it and drift.
        "extendsRight": edge > 0,
        **_encode(edge, threshold=0.005),
    }


def reliability_payload(bins: list[dict[str, Any]]) -> dict[str, Any]:
    """Reliability diagram data, plus the diagonal it should sit on.

    The reference line is shipped rather than drawn client-side so the chart
    cannot be rendered without the thing that gives it meaning.
    """
    return {
        "bins": [
            {
                "meanPredicted": b["mean_predicted"],
                "observedRate": b["observed_rate"],
                "count": b["count"],
                "gap": b["observed_rate"] - b["mean_predicted"],
                # Over-confident where observed < predicted.
                **_encode(b["observed_rate"] - b["mean_predicted"], threshold=0.02),
            }
            for b in bins
        ],
        "reference": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        "referenceLabel": "perfect calibration",
    }
