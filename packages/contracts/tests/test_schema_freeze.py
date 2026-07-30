"""Schema-freeze test.

Snapshots the JSON schema of every contract plus the lake layout. A change to
either fails here, which is how a contract change becomes a deliberate act
instead of an accident that surfaces during integration.

To accept an intended change:

    INDICANT_UPDATE_SCHEMA_SNAPSHOT=1 pytest tests/test_schema_freeze.py

and commit the regenerated snapshot alongside the change.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from indicant_contracts import (
    ADJUSTED_EXTRA_COLUMNS,
    CANONICAL_PRICE_COLUMNS,
    CalibrationBin,
    CorporateAction,
    Dataset,
    EligibilityThresholds,
    ErrorEnvelope,
    ExplanationFact,
    LakePaths,
    MarketRegimeResult,
    ModelCard,
    OHLCVBar,
    Prediction,
    QualityReport,
    QualityScore,
    QuarantineRecord,
    RegimeResult,
    RuleResult,
    SymbolChange,
    SymbolMeta,
    TradingCalendar,
    UniverseSnapshot,
)

SNAPSHOT = Path(__file__).parent / "schema_snapshot.json"

FROZEN_MODELS: tuple[type[BaseModel], ...] = (
    CalibrationBin,
    CorporateAction,
    EligibilityThresholds,
    ErrorEnvelope,
    ExplanationFact,
    LakePaths,
    MarketRegimeResult,
    ModelCard,
    OHLCVBar,
    Prediction,
    QualityReport,
    QualityScore,
    QuarantineRecord,
    RegimeResult,
    RuleResult,
    SymbolChange,
    SymbolMeta,
    TradingCalendar,
    UniverseSnapshot,
)

# A fixed reference date so the layout fingerprint is deterministic.
_REF = date(2020, 6, 15)


def _build_fingerprint() -> dict[str, Any]:
    return {
        "models": {m.__name__: m.model_json_schema() for m in FROZEN_MODELS},
        "canonical_price_columns": list(CANONICAL_PRICE_COLUMNS),
        "adjusted_extra_columns": list(ADJUSTED_EXTRA_COLUMNS),
        "lake_layout": {
            ds.name: {
                "dir": ds.value,
                "file": str(LakePaths(root=Path("/L")).file(ds, when=_REF)),
                "glob": LakePaths(root=Path("/L")).glob(ds),
            }
            for ds in Dataset
        },
    }


def test_schemas_are_frozen() -> None:
    current = _build_fingerprint()

    if os.environ.get("INDICANT_UPDATE_SCHEMA_SNAPSHOT"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip("snapshot regenerated — re-run without the env var to verify")

    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip("snapshot created on first run — commit it")

    expected = json.loads(SNAPSHOT.read_text())

    added = sorted(set(current["models"]) - set(expected["models"]))
    removed = sorted(set(expected["models"]) - set(current["models"]))
    assert not removed, f"contract models removed: {removed}"
    assert not added, f"contract models added without updating the snapshot: {added}"

    for name in sorted(current["models"]):
        assert current["models"][name] == expected["models"][name], (
            f"schema for {name} changed. If intended, regenerate with "
            f"INDICANT_UPDATE_SCHEMA_SNAPSHOT=1 and commit the snapshot."
        )

    assert current["canonical_price_columns"] == expected["canonical_price_columns"]
    assert current["adjusted_extra_columns"] == expected["adjusted_extra_columns"]
    assert current["lake_layout"] == expected["lake_layout"], (
        "lake layout changed — this breaks every existing partition on disk. "
        "A migration is required, not just a snapshot update."
    )


def test_every_public_model_is_frozen() -> None:
    """Any new contract model must be added to FROZEN_MODELS deliberately.

    Without this, a model added to __init__ would escape the freeze silently.
    """
    import indicant_contracts

    exported_models = {
        obj.__name__
        for name in indicant_contracts.__all__
        if isinstance(obj := getattr(indicant_contracts, name), type)
        and issubclass(obj, BaseModel)
    }
    frozen = {m.__name__ for m in FROZEN_MODELS}
    missing = sorted(exported_models - frozen)
    assert not missing, f"exported contract models missing from FROZEN_MODELS: {missing}"
