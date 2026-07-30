"""Quarantine store.

Nothing is ever deleted. A quarantined row is held with the rule that held it,
so when a rule turns out to be wrong — or an adjustment bug is fixed — the held
rows can be replayed rather than re-downloaded.

That replay path is the whole point. A gate that drops bad rows forces a full
re-ingest to recover from its own false positives, which in practice means
nobody ever fixes a false positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from indicant_contracts import Dataset, QualityReport

from market_data.quality.gate import GateOutcome
from market_data.quality.rules import RuleContext
from market_data.store.lake import Lake

QUARANTINE_META = ("quarantine_rule_id", "quarantined_at", "run_id")


class QuarantineStore:
    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def write(self, outcome: GateOutcome) -> Path | None:
        """Persist the held rows. Returns None when there is nothing to hold."""
        if outcome.quarantined.empty:
            return None
        payload = outcome.quarantined.copy()
        payload["quarantined_at"] = datetime.now(UTC).isoformat()
        payload["run_id"] = outcome.report.run_id
        return self._lake.write_partition(
            payload,
            dataset=Dataset.QUARANTINE,
            when=outcome.report.trade_date,
        )

    def write_report(self, report: QualityReport) -> Path:
        """Persist the full rule-by-rule report, evidence included."""
        rows = [
            {
                "run_id": report.run_id,
                "trade_date": report.trade_date,
                "verdict": report.verdict.value,
                "rows_in": report.rows_in,
                "rows_accepted": report.rows_accepted,
                "rows_quarantined": report.rows_quarantined,
                "duration_seconds": report.duration_seconds,
                "rule_id": r.rule_id,
                "tier": int(r.tier),
                "severity": r.severity.value,
                "passed": r.passed,
                "message": r.message,
                "affected_rows": r.affected_rows,
                "affected_symbols": ",".join(r.affected_symbols),
                # Evidence is JSON-serialised rather than exploded into columns
                # because its shape is rule-specific by design.
                "evidence": _to_json(r.evidence),
            }
            for r in report.results
        ]
        return self._lake.write_partition(
            pd.DataFrame(rows),
            dataset=Dataset.QUALITY_RUNS,
            when=report.trade_date,
        )

    def read(self, trade_date: date | None = None) -> pd.DataFrame:
        if trade_date is None:
            return self._lake.read_dataset(Dataset.QUARANTINE)
        path = self._lake.paths.file(Dataset.QUARANTINE, when=trade_date)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

    def read_reports(self, trade_date: date | None = None) -> pd.DataFrame:
        if trade_date is None:
            return self._lake.read_dataset(Dataset.QUALITY_RUNS)
        path = self._lake.paths.file(Dataset.QUALITY_RUNS, when=trade_date)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

    def rule_ids_held(self, trade_date: date | None = None) -> dict[str, int]:
        """Which rules are currently holding rows, and how many.

        This is the operational view: a rule at the top of this list with a
        large count is either catching a real upstream problem or is itself
        wrong, and either way it needs looking at.
        """
        held = self.read(trade_date)
        if held.empty or "quarantine_rule_id" not in held.columns:
            return {}
        counts = held["quarantine_rule_id"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def replay(
        self,
        *,
        trade_date: date,
        gate,  # QualityGate — untyped to avoid a circular import
        exclude_rules: Sequence[str] = (),
        context_overrides: dict[str, object] | None = None,
    ) -> GateOutcome | None:
        """Re-run the gate over previously held rows.

        `exclude_rules` is how a false positive gets retired: drop the rule that
        was wrong, replay, and the rows return to the lake without touching the
        network.
        """
        held = self.read(trade_date)
        if held.empty:
            return None

        if exclude_rules and "quarantine_rule_id" in held.columns:
            held = held[~held["quarantine_rule_id"].isin(set(exclude_rules))]
            if held.empty:
                return None

        payload = held.drop(columns=[c for c in QUARANTINE_META if c in held.columns])
        payload = payload.reset_index(drop=True)

        overrides = context_overrides or {}
        ctx = RuleContext(df=payload, trade_date=trade_date, **overrides)  # type: ignore[arg-type]
        return gate.run(ctx)


def _to_json(payload: dict[str, object]) -> str:
    import json

    def _default(value: object) -> object:
        if isinstance(value, date | datetime):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()  # numpy scalar
        return str(value)

    return json.dumps(payload, default=_default, sort_keys=True)
