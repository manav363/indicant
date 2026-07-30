"""
market_regime/backtest/metrics.py
───────────────────────────────────
Pure functions for computing trading strategy performance metrics.

All functions operate on a Series/array of daily portfolio returns
(with transaction costs already deducted) unless otherwise noted.

Metrics computed:
    Sharpe ratio          — risk-adjusted return (annualised)
    Sortino ratio         — same as Sharpe but penalises only downside vol
    Max drawdown          — largest peak-to-trough decline
    CAGR                  — compound annual growth rate
    Total return          — cumulative return over the full period
    Win rate              — fraction of days with positive return
    Profit factor         — sum of gains / sum of losses
    Turnover              — average absolute position change per day
    Cost-adjusted Sharpe  — Sharpe on returns net of transaction costs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def compute_sharpe(
    returns: np.ndarray | pd.Series,
    annual_trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

        Sharpe = mean(r) / std(r) * sqrt(252)

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns (already net of transaction costs).
    annual_trading_days : int
        Number of trading days per year (default 252 for India).

    Returns
    -------
    float
        0.0 if insufficient data or zero volatility.
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=0)
    if std < 1e-15 or np.isnan(std):
        return 0.0
    return float(r.mean() / std * np.sqrt(annual_trading_days))


def compute_sortino(
    returns: np.ndarray | pd.Series,
    annual_trading_days: int = 252,
) -> float:
    """
    Annualised Sortino ratio — penalises only downside deviation.

        Sortino = mean(r) / downside_std(r) * sqrt(252)

    Downside deviation = std of negative returns only.
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    negative = r[r < 0]
    if len(negative) < 2:
        return 0.0
    downside_std = negative.std(ddof=0)
    if downside_std < 1e-15 or np.isnan(downside_std):
        return 0.0
    return float(r.mean() / downside_std * np.sqrt(annual_trading_days))


def compute_max_drawdown(
    returns: np.ndarray | pd.Series,
) -> float:
    """
    Maximum drawdown — largest peak-to-trough decline.

        cumulative = (1 + r).cumprod()
        peak = expanding_max(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = min(drawdown)

    Returns
    -------
    float
        Negative value (e.g. -0.25 = 25% drawdown).
        0.0 if no drawdown or insufficient data.
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    cumulative = (1.0 + r).cumprod()
    peak = np.maximum.accumulate(cumulative)
    # Avoid division by zero on first element
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (cumulative - peak) / peak, 0.0)
    dd = float(np.min(drawdown))
    return min(dd, 0.0)


def compute_cagr(
    returns: np.ndarray | pd.Series,
    annual_trading_days: int = 252,
) -> float:
    """
    Compound Annual Growth Rate.

        total_return = prod(1 + r) - 1
        years = len(returns) / 252
        CAGR = (1 + total_return)^(1/years) - 1

    Returns 0.0 if the period is too short (< 1 trading day).
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    total_return = float(np.prod(1.0 + r) - 1.0)
    years = len(r) / max(annual_trading_days, 1)
    if years <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0  # total loss
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def compute_total_return(returns: np.ndarray | pd.Series) -> float:
    """Cumulative return over the full period: prod(1 + r) - 1."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 1:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def compute_win_rate(returns: np.ndarray | pd.Series) -> float:
    """Fraction of days with positive return."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def compute_profit_factor(returns: np.ndarray | pd.Series) -> float:
    """
    Profit factor = sum(gains) / sum(losses).

    Returns infinity if there are no losing days.
    Returns 0.0 if there are no winning days.
    """
    r = np.asarray(returns, dtype=np.float64)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def compute_turnover(positions: np.ndarray | pd.Series) -> float:
    """
    Average daily turnover — mean absolute position change.

        turnover = mean(abs(position_t - position_{t-1}))

    Parameters
    ----------
    positions : array-like
        Daily position values (-1, 0, +1, or fractional Kelly amounts).

    Returns
    -------
    float
        Average absolute change per day (0 = no trading).
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 2:
        return 0.0
    changes = np.abs(np.diff(pos, prepend=pos[0]))
    return float(changes.mean())


# ── Convenience aggregator ──────────────────────────────────────────────

def compute_all_metrics(
    returns: np.ndarray | pd.Series,
    positions: np.ndarray | pd.Series | None = None,
    annual_trading_days: int = 252,
) -> dict[str, float]:
    """
    Compute all standard backtest metrics at once.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns (net of transaction costs).
    positions : array-like, optional
        Daily positions for turnover computation.
    annual_trading_days : int
        Trading days per year.

    Returns
    -------
    dict
        Keys: sharpe, sortino, max_dd, cagr, total_return,
              win_rate, profit_factor, turnover, cost_adjusted_sharpe
    """
    r = np.asarray(returns, dtype=np.float64)
    metrics: dict[str, float] = {
        "sharpe": compute_sharpe(r, annual_trading_days),
        "sortino": compute_sortino(r, annual_trading_days),
        "max_dd": compute_max_drawdown(r),
        "cagr": compute_cagr(r, annual_trading_days),
        "total_return": compute_total_return(r),
        "win_rate": compute_win_rate(r),
        "profit_factor": compute_profit_factor(r),
    }

    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64)
        metrics["turnover"] = compute_turnover(pos)
        # Cost-adjusted Sharpe is just Sharpe on returns that already
        # include transaction costs — so it's the same as sharpe here.
        # The *separate* cost_adjusted_sharpe field in the registry
        # exists for cases where raw returns are stored separately.
        # For this engine, returns are always net of costs.
        metrics["cost_adjusted_sharpe"] = metrics["sharpe"]
    else:
        metrics["turnover"] = 0.0
        metrics["cost_adjusted_sharpe"] = metrics["sharpe"]

    return metrics
