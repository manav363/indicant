"""Gate orchestration.

Order matters and is not arbitrary:

1. **Structural** rules run first and short-circuit. If the bytes are not a
   bhavcopy, every later rule would produce noise about a frame that should
   never have been parsed.
2. **Row-level validity** rules run next and partition the frame into accepted
   and quarantined. Later tiers then see only rows that are individually sane,
   so a `high < low` row cannot also trigger a spurious sigma outlier.
3. **Everything else** annotates.

The verdict is deliberately coarse. `PASS_WITH_WARNINGS` and `QUARANTINED` both
let data into the lake; only `REJECTED` does not. Fine-grained severity lives in
the individual results, where it can be read with its evidence.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd
from indicant_contracts import (
    QualityReport,
    RuleResult,
    Severity,
    Verdict,
)

from market_data.quality.rules import (
    ALL_RULES,
    ROW_INDEX_KEY,
    ROW_LEVEL_RULES,
    STRUCTURAL_RULES,
    Rule,
    RuleContext,
)


class GateOutcome:
    """The report plus the two frames it produced.

    Accepted and quarantined are returned separately rather than as a flagged
    single frame so that a caller cannot accidentally write quarantined rows to
    the price lake by forgetting to filter.
    """

    def __init__(
        self,
        *,
        report: QualityReport,
        accepted: pd.DataFrame,
        quarantined: pd.DataFrame,
    ) -> None:
        self.report = report
        self.accepted = accepted
        self.quarantined = quarantined

    @property
    def verdict(self) -> Verdict:
        return self.report.verdict

    @property
    def is_usable(self) -> bool:
        return self.report.verdict.is_usable

    def __repr__(self) -> str:
        return (
            f"GateOutcome({self.report.trade_date} {self.report.verdict.value} "
            f"accepted={len(self.accepted)} quarantined={len(self.quarantined)})"
        )


class QualityGate:
    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        self._rules = tuple(rules) if rules is not None else ALL_RULES

    def run(self, ctx: RuleContext) -> GateOutcome:
        started = datetime.now(UTC)
        run_id = f"{ctx.trade_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        rows_in = len(ctx.df)
        results: list[RuleResult] = []

        # ---- 1. structural: short-circuit on failure -----------------------
        structural = [r(ctx) for r in self._active(STRUCTURAL_RULES)]
        results.extend(structural)
        if any(r.blocks_file for r in structural):
            return self._finish(
                run_id=run_id,
                ctx=ctx,
                started=started,
                results=results,
                accepted=_empty_like(ctx.df),
                quarantined=ctx.df.copy(),
                verdict=Verdict.REJECTED,
                rows_in=rows_in,
            )

        # ---- 2. row-level validity: partition the frame --------------------
        bad_index: set[int] = set()
        reasons: dict[int, str] = {}
        for rule in self._active(ROW_LEVEL_RULES):
            result = rule(ctx)
            results.append(result)
            if result.passed:
                continue
            offending = _offending_index(ctx.df, result)
            for idx in offending:
                bad_index.add(idx)
                # First rule to catch a row owns the explanation. Later rules
                # on the same row are consequences, not causes.
                reasons.setdefault(idx, result.rule_id)

        accepted = ctx.df.drop(index=list(bad_index), errors="ignore").reset_index(drop=True)
        quarantined = ctx.df.loc[sorted(bad_index)].copy()
        if not quarantined.empty:
            quarantined["quarantine_rule_id"] = [reasons[i] for i in sorted(bad_index)]

        # ---- 3. remaining tiers annotate, on clean rows only ---------------
        clean_ctx = _with_frame(ctx, accepted)
        already = {r.rule_id for r in results}
        for rule in self._active(self._rules):
            candidate = rule(clean_ctx)
            if candidate.rule_id in already:
                continue
            results.append(candidate)

        verdict = _verdict(results, quarantined_rows=len(quarantined))
        return self._finish(
            run_id=run_id,
            ctx=ctx,
            started=started,
            results=results,
            accepted=accepted,
            quarantined=quarantined,
            verdict=verdict,
            rows_in=rows_in,
        )

    def _active(self, rules: Sequence[Rule]) -> tuple[Rule, ...]:
        """Respect a caller-supplied rule subset (used by tests)."""
        selected = set(self._rules)
        return tuple(r for r in rules if r in selected)

    def _finish(
        self,
        *,
        run_id: str,
        ctx: RuleContext,
        started: datetime,
        results: Sequence[RuleResult],
        accepted: pd.DataFrame,
        quarantined: pd.DataFrame,
        verdict: Verdict,
        rows_in: int,
    ) -> GateOutcome:
        report = QualityReport(
            run_id=run_id,
            trade_date=ctx.trade_date,
            started_at=started,
            finished_at=datetime.now(UTC),
            verdict=verdict,
            rows_in=rows_in,
            rows_accepted=len(accepted),
            rows_quarantined=len(quarantined),
            results=tuple(results),
        )
        return GateOutcome(report=report, accepted=accepted, quarantined=quarantined)


def _verdict(results: Sequence[RuleResult], *, quarantined_rows: int) -> Verdict:
    if any(r.blocks_file for r in results):
        return Verdict.REJECTED
    if quarantined_rows:
        return Verdict.QUARANTINED
    failures = [r for r in results if not r.passed]
    if any(r.severity in {Severity.FATAL, Severity.ERROR} for r in failures):
        # An ERROR that identified no specific rows is a whole-batch concern
        # (row count, continuity, missing constituents). The data is usable but
        # something needs a human, so it must not read as a clean pass.
        return Verdict.PASS_WITH_WARNINGS
    if failures:
        return Verdict.PASS_WITH_WARNINGS
    return Verdict.PASS


def _offending_index(df: pd.DataFrame, result: RuleResult) -> list[int]:
    """The exact rows a row-level rule objected to.

    Row-level rules record indices under `evidence[ROW_INDEX_KEY]`. Falling
    back to symbol matching would over-quarantine, because one symbol can hold
    both an EQ and a BE row and only one may be malformed — so the fallback is
    a last resort and is reported as such by the caller's tests.
    """
    indices = result.evidence.get(ROW_INDEX_KEY)
    if isinstance(indices, list):
        valid = df.index.intersection(pd.Index([int(i) for i in indices]))
        return valid.tolist()
    if not result.affected_symbols:
        return []
    mask = df["symbol"].astype(str).str.upper().isin(set(result.affected_symbols))
    return df.index[mask].tolist()


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _with_frame(ctx: RuleContext, df: pd.DataFrame) -> RuleContext:
    return RuleContext(
        df=df,
        trade_date=ctx.trade_date,
        expected_trading_day=ctx.expected_trading_day,
        previous_close=ctx.previous_close,
        previous_symbols=ctx.previous_symbols,
        trailing_row_counts=ctx.trailing_row_counts,
        history=ctx.history,
        corporate_actions=ctx.corporate_actions,
        cross_source=ctx.cross_source,
        expected_symbols=ctx.expected_symbols,
    )
