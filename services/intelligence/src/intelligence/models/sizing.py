"""
intelligence/risk/sizing.py
──────────────────────────────
Position sizing using the Kelly Criterion and risk-adjusted variants.

The Kelly Criterion answers: given an edge, what fraction of
your capital should you bet to maximise long-run growth?

Full Kelly:
    f* = (bp - q) / b
    where:
        b = net odds (e.g. expected return if right)
        p = probability of winning (P_up from our model)
        q = 1 - p = probability of losing

In practice, Full Kelly is too aggressive (high variance, large drawdowns).
We use Fractional Kelly (half-Kelly is standard in quant finance).

Additional constraints:
    - Max position: cap at 10% of portfolio per stock
    - Min confidence: only size if confidence >= threshold
    - Volatility scaling: reduce size for high-volatility stocks
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSize:
    ticker: str
    kelly_fraction: float        # raw Kelly f*
    recommended_fraction: float  # after half-Kelly + caps + vol scaling
    max_loss_pct: float          # stop loss level
    rationale: str


def kelly_fraction(
    p_win: float,
    win_return: float = 0.15,   # expected gain if prediction correct (15%)
    loss_return: float = 0.08,  # expected loss if wrong (8%)
) -> float:
    """
    Full Kelly Criterion.

    f* = (b*p - q) / b

    where:
        b = win_return / loss_return  (reward-to-risk ratio)
        p = probability of winning
        q = 1 - p

    Example:
        p=0.65, win=15%, loss=8%
        b = 0.15/0.08 = 1.875
        f* = (1.875*0.65 - 0.35) / 1.875 = 0.463 → 46% of capital

    This is why we always use Half-Kelly in practice.

    Parameters
    ----------
    p_win : float
        Model's P(price up) — our edge estimate.
    win_return : float
        Expected % gain if prediction is correct.
    loss_return : float
        Expected % loss if prediction is wrong.

    Returns
    -------
    float
        Optimal fraction of capital (can be negative = short).
        Clamped to [-1, 1].
    """
    if p_win <= 0 or p_win >= 1:
        return 0.0

    p_lose = 1.0 - p_win
    b = win_return / max(loss_return, 1e-6)   # reward-to-risk ratio

    f_star = (b * p_win - p_lose) / b

    return max(-1.0, min(1.0, f_star))


def recommended_position(
    ticker: str,
    p_win: float,
    annualised_volatility: float = 0.25,
    confidence_threshold: float = 0.55,
    kelly_multiplier: float = 0.5,    # half-Kelly
    max_position: float = 0.10,       # max 10% per stock
    win_return: float = 0.15,
    loss_return: float = 0.08,
) -> PositionSize:
    """
    Compute a risk-adjusted position size.

    Steps:
    1. Compute full Kelly fraction
    2. Apply half-Kelly (or custom multiplier)
    3. Scale down for high volatility
    4. Cap at max_position
    5. Zero out if below confidence threshold

    Volatility scaling:
        A stock with 40% annualised vol is twice as risky as one
        with 20% vol. We scale position inversely:
        vol_scale = target_vol / stock_vol
        target_vol = 0.20 (20% is our baseline)

    Parameters
    ----------
    ticker : str
    p_win : float
        Model confidence (P_up for BUY, 1-P_up for SELL).
    annualised_volatility : float
        Realised volatility (rv_63 from features).
    confidence_threshold : float
        Minimum confidence to take a position.
    kelly_multiplier : float
        Fraction of Kelly to use. 0.5 = Half-Kelly.
    max_position : float
        Maximum fraction of portfolio (0.10 = 10%).
    win_return : float
        Expected gain if right.
    loss_return : float
        Expected loss if wrong.

    Returns
    -------
    PositionSize
    """
    # Below threshold → no position
    if p_win < confidence_threshold:
        return PositionSize(
            ticker=ticker,
            kelly_fraction=0.0,
            recommended_fraction=0.0,
            max_loss_pct=0.0,
            rationale=f"Confidence {p_win:.1%} below threshold {confidence_threshold:.1%}.",
        )

    # Step 1: Full Kelly
    f_full = kelly_fraction(p_win, win_return, loss_return)

    # Step 2: Fractional Kelly
    f_fractional = f_full * kelly_multiplier

    # Step 3: Volatility scaling
    # target_vol = 0.20; if stock vol is 0.40, scale by 0.20/0.40 = 0.5
    target_vol = 0.20
    vol_scale = min(1.0, target_vol / max(annualised_volatility, 0.01))
    f_vol_scaled = f_fractional * vol_scale

    # Step 4: Cap at max_position
    f_final = min(abs(f_vol_scaled), max_position)
    f_final = max(0.0, f_final)

    # Step 5: Stop loss = 1.5x ATR or loss_return, whichever is smaller
    stop_loss = loss_return

    notes = []
    if vol_scale < 1.0:
        notes.append(f"Vol-scaled by {vol_scale:.2f} (stock vol {annualised_volatility:.0%} > target 20%).")
    if f_full > max_position / kelly_multiplier:
        notes.append(f"Kelly suggested {f_full:.1%} but capped at {max_position:.0%}.")

    rationale = " ".join(notes) if notes else f"Standard {kelly_multiplier:.0%}-Kelly sizing."

    return PositionSize(
        ticker=ticker,
        kelly_fraction=round(f_full, 4),
        recommended_fraction=round(f_final, 4),
        max_loss_pct=round(stop_loss, 4),
        rationale=rationale,
    )
