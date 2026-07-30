"""
intelligence/signals/generator.py
────────────────────────────────────
Converts raw model probabilities into structured trading signals.

Responsibility:
- Apply confidence thresholds to produce BUY/HOLD/SELL
- Combine signals from multiple models (ensemble voting)
- Add signal metadata (strength, regime context)

This is intentionally kept separate from the model layer —
models produce probabilities, this module interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Signal(str, Enum):
    BUY  = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class SignalResult:
    signal: Signal
    confidence: float        # 0.5 → 1.0 (how confident in the signal)
    probability_up: float    # raw P(price higher in N months)
    strength: str            # "strong" | "moderate" | "weak"
    regime_aligned: bool     # is signal consistent with market regime?
    notes: list[str]


def generate_signal(
    probability_up: float,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    adx: Optional[float] = None,
    trend_consistency: Optional[float] = None,
) -> SignalResult:
    """
    Convert a model probability into a structured signal.

    Signal logic:
        P(up) >= buy_threshold  → BUY
        P(up) <= sell_threshold → SELL
        otherwise               → HOLD

    Confidence = distance from 0.5 (certainty of direction):
        confidence = |P(up) - 0.5| * 2   mapped to [0, 1]
        But we report it as P(up) for BUY, 1-P(up) for SELL.

    Strength thresholds:
        confidence >= 0.70 → strong
        confidence >= 0.60 → moderate
        otherwise          → weak

    Regime alignment:
        A BUY signal is regime-aligned if ADX > 20 AND
        trend_consistency > 0.5 (more up days than down).
        Misaligned signals are still valid but noted.

    Parameters
    ----------
    probability_up : float
        P(price higher in N months) from the ML model.
    buy_threshold : float
        Minimum probability to emit BUY. Default 0.55.
    sell_threshold : float
        Maximum probability to emit SELL. Default 0.45.
    adx : float, optional
        ADX value for regime alignment check.
    trend_consistency : float, optional
        Fraction of recent days with positive returns.
    """
    p = float(probability_up)
    p = max(0.0, min(1.0, p))

    # ── Signal direction ──────────────────────────────────────────────────
    if p >= buy_threshold:
        signal = Signal.BUY
        confidence = p
    elif p <= sell_threshold:
        signal = Signal.SELL
        confidence = 1.0 - p
    else:
        signal = Signal.HOLD
        confidence = 1.0 - abs(p - 0.5) * 2   # highest at exact 0.5

    # ── Signal strength ───────────────────────────────────────────────────
    if confidence >= 0.70:
        strength = "strong"
    elif confidence >= 0.60:
        strength = "moderate"
    else:
        strength = "weak"

    # ── Regime alignment ──────────────────────────────────────────────────
    notes: list[str] = []
    regime_aligned = True

    if signal == Signal.BUY:
        if adx is not None and adx < 20:
            regime_aligned = False
            notes.append("ADX < 20 — weak/ranging market, trend may not sustain.")
        if trend_consistency is not None and trend_consistency < 0.4:
            regime_aligned = False
            notes.append("Low trend consistency — fewer than 40% of recent days were up.")

    elif signal == Signal.SELL:
        if adx is not None and adx < 20:
            regime_aligned = False
            notes.append("ADX < 20 — weak/ranging market.")
        if trend_consistency is not None and trend_consistency > 0.6:
            notes.append("High trend consistency — downside may be limited.")

    if not regime_aligned:
        notes.insert(0, "Signal not aligned with current market regime.")

    return SignalResult(
        signal=signal,
        confidence=round(confidence, 4),
        probability_up=round(p, 4),
        strength=strength,
        regime_aligned=regime_aligned,
        notes=notes,
    )


def ensemble_signal(
    probabilities: list[float],
    weights: Optional[list[float]] = None,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> SignalResult:
    """
    Combine predictions from multiple models into a single signal.

    Uses weighted average of probabilities.
    Equal weights if not specified.

    Parameters
    ----------
    probabilities : list[float]
        P(up) from each model.
    weights : list[float], optional
        Weight for each model. Must sum to 1. Default: equal weights.
    """
    if not probabilities:
        raise ValueError("probabilities list cannot be empty.")

    if weights is None:
        weights = [1.0 / len(probabilities)] * len(probabilities)

    if len(weights) != len(probabilities):
        raise ValueError("weights and probabilities must have same length.")

    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()   # normalise

    p_combined = float(np.dot(weights_arr, probabilities))

    return generate_signal(
        p_combined,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
