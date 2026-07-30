"""Shared error envelope.

Every service returns the same error shape, so the gateway can compose
failures from two upstreams without special-casing either one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    # Client
    SYMBOL_NOT_FOUND = "symbol_not_found"
    SYMBOL_NOT_ELIGIBLE = "symbol_not_eligible"
    INVALID_REQUEST = "invalid_request"
    NOT_A_TRADING_DAY = "not_a_trading_day"

    # Data / upstream
    DATA_UNAVAILABLE = "data_unavailable"
    INSUFFICIENT_HISTORY = "insufficient_history"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"

    # Model
    MODEL_NOT_TRAINED = "model_not_trained"
    MODEL_REFUSED = "model_refused"

    INTERNAL = "internal"


class ErrorEnvelope(BaseModel):
    """`user_message` is what a person reads; `message` is what an engineer
    reads. Keeping them separate stops internal detail leaking into the UI and
    stops the UI's phrasing being the only record of what broke.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    user_message: str
    detail: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def not_eligible(cls, symbol: str, reason: str) -> ErrorEnvelope:
        """A symbol below the quality bar is not an error condition in the
        system — it is a correctly-scoped refusal, and the user gets the real
        reason rather than a generic failure.
        """
        return cls(
            code=ErrorCode.SYMBOL_NOT_ELIGIBLE,
            message=f"{symbol} excluded by eligibility thresholds: {reason}",
            user_message=(
                f"We do not have enough reliable data on {symbol} to give an honest "
                f"read yet — {reason}."
            ),
            detail={"symbol": symbol, "reason": reason},
        )
