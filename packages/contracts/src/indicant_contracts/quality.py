"""Data quality contracts.

Governing principle: unknown quality fails. Bad data is quarantined with its
evidence, never silently dropped and never silently used.

A rule that cannot say *why* it failed is not a rule, so `evidence` is
required on every failing result.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import IntEnum, StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Tier(IntEnum):
    """Rule tiers, ordered by how early they can reject.

    STRUCTURAL failures kill the file. VALIDITY failures kill the row.
    Everything above that annotates rather than rejects, except where a
    threshold escalates it.
    """

    STRUCTURAL = 1
    VALIDITY = 2
    COMPLETENESS = 3
    CONTINUITY = 4
    PLAUSIBILITY = 5
    CROSS_SOURCE = 6


class Severity(StrEnum):
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Verdict(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"

    @property
    def is_usable(self) -> bool:
        """Whether any data from this run may enter the lake.

        QUARANTINED is usable: the clean rows land, the failing rows are held.
        REJECTED is not: the whole file is untrustworthy.
        """
        return self in {Verdict.PASS, Verdict.PASS_WITH_WARNINGS, Verdict.QUARANTINED}


class RuleResult(BaseModel):
    """The outcome of one rule against one ingestion batch.

    `evidence` must be populated on failure. "Rule X failed" without actual
    vs expected is not actionable at 3am.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(pattern=r"^T[1-6]\.[a-z_]+\.[a-z_0-9]+$")
    tier: Tier
    severity: Severity
    passed: bool
    message: str
    affected_symbols: tuple[str, ...] = ()
    affected_rows: int = 0
    evidence: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _failures_carry_evidence(self) -> RuleResult:
        if not self.passed and not self.evidence:
            raise ValueError(f"{self.rule_id} failed without evidence")
        return self

    @property
    def blocks_file(self) -> bool:
        return not self.passed and self.tier is Tier.STRUCTURAL and self.severity is Severity.FATAL

    @property
    def blocks_rows(self) -> bool:
        return not self.passed and self.severity in {Severity.FATAL, Severity.ERROR}


class QualityReport(BaseModel):
    """Everything the gate concluded about one ingestion batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trade_date: date
    started_at: datetime
    finished_at: datetime
    verdict: Verdict
    rows_in: int = Field(ge=0)
    rows_accepted: int = Field(ge=0)
    rows_quarantined: int = Field(ge=0)
    results: tuple[RuleResult, ...]

    @model_validator(mode="after")
    def _rows_reconcile(self) -> QualityReport:
        if self.rows_accepted + self.rows_quarantined > self.rows_in:
            raise ValueError(
                f"accepted {self.rows_accepted} + quarantined {self.rows_quarantined} "
                f"exceeds rows_in {self.rows_in}"
            )
        return self

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class QualityScore(BaseModel):
    """Per-symbol fitness. Drives universe eligibility.

    The five components are deliberately separate rather than a single opaque
    number, so an exclusion can be explained to a user in one sentence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    as_of: date
    history_completeness: float = Field(ge=0, le=1)
    validity_clean_rate: float = Field(ge=0, le=1)
    continuity_clean_rate: float = Field(ge=0, le=1)
    liquidity_adequacy: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    history_days: int = Field(ge=0)
    median_turnover: float = Field(ge=0)

    # Weights sum to 1.0. Continuity is weighted highest because an
    # unexplained prev_close break means the adjustment pipeline is wrong,
    # which corrupts every derived feature downstream.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "history_completeness": 0.25,
        "validity_clean_rate": 0.20,
        "continuity_clean_rate": 0.30,
        "liquidity_adequacy": 0.15,
        "recency": 0.10,
    }

    @property
    def score(self) -> float:
        return round(
            sum(getattr(self, name) * weight for name, weight in self.WEIGHTS.items()),
            6,
        )


class EligibilityThresholds(BaseModel):
    """The bar a symbol must clear to be offered to a user.

    Anything below the bar is not a failure and not a fallback — it is
    correctly out of scope, and the UI says so in plain words.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_score: float = Field(default=0.85, ge=0, le=1)
    min_history_days: int = Field(default=756, ge=1)  # ~3 years
    min_median_turnover: float = Field(default=1e7, ge=0)  # Rs 1 crore
    # Recency is a HARD GATE, not a weighted component. A delisted company can
    # have twenty years of flawless, liquid history — its weighted score stays
    # high, and it would sail into a tradeable universe it left years ago.
    # Weighting alone cannot express "this no longer trades".
    min_recency: float = Field(default=0.5, ge=0, le=1)

    def evaluate(self, score: QualityScore) -> str | None:
        """Return None if eligible, else a human-readable exclusion reason.

        Order is deliberate: the most specific, least alarming explanation
        first. A newly-listed stock should be told it is new, not that its data
        is bad.
        """
        if score.history_days < self.min_history_days:
            return (
                f"only {score.history_days} trading days of history "
                f"(need {self.min_history_days})"
            )
        if score.recency < self.min_recency:
            return "has not traded recently — the listing looks delisted or suspended"
        if score.median_turnover < self.min_median_turnover:
            return (
                f"median daily turnover Rs {score.median_turnover:,.0f} is below the "
                f"Rs {self.min_median_turnover:,.0f} liquidity floor"
            )
        if score.score < self.min_score:
            return f"data quality score {score.score:.2f} is below {self.min_score:.2f}"
        return None


class QuarantineRecord(BaseModel):
    """A held row and the rule that held it. Never deleted — fix the rule,
    then replay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trade_date: date
    symbol: str
    rule_id: str
    reason: str
    payload: dict[str, object]
    quarantined_at: datetime
