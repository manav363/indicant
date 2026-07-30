"""Shared contracts for every Indicant service.

One definition per schema, imported by all services. A contract change breaks
both sides' tests at build time rather than at integration time — which is the
whole reason this package exists as a package.
"""

from indicant_contracts.errors import ErrorCode, ErrorEnvelope
from indicant_contracts.intelligence import (
    CalibrationBin,
    CompositeSignal,
    Direction,
    ExplanationFact,
    MarketRegimeResult,
    ModelCard,
    Prediction,
    PrimaryRegime,
    RegimeResult,
    Signal,
    Strength,
)
from indicant_contracts.lake import Dataset, LakePaths
from indicant_contracts.market_data import (
    ADJUSTED_EXTRA_COLUMNS,
    CANONICAL_PRICE_COLUMNS,
    CorporateAction,
    CorporateActionType,
    ListingStatus,
    OHLCVBar,
    Series,
    SymbolChange,
    SymbolMeta,
    TradingCalendar,
    UniverseSnapshot,
)
from indicant_contracts.quality import (
    EligibilityThresholds,
    QualityReport,
    QualityScore,
    QuarantineRecord,
    RuleResult,
    Severity,
    Tier,
    Verdict,
)

__version__ = "2.0.0"

__all__ = [
    "ADJUSTED_EXTRA_COLUMNS",
    "CANONICAL_PRICE_COLUMNS",
    "CalibrationBin",
    "CompositeSignal",
    "CorporateAction",
    "CorporateActionType",
    "Dataset",
    "Direction",
    "EligibilityThresholds",
    "ErrorCode",
    "ErrorEnvelope",
    "ExplanationFact",
    "LakePaths",
    "ListingStatus",
    "MarketRegimeResult",
    "ModelCard",
    "OHLCVBar",
    "Prediction",
    "PrimaryRegime",
    "QualityReport",
    "QualityScore",
    "QuarantineRecord",
    "RegimeResult",
    "RuleResult",
    "Series",
    "Severity",
    "Signal",
    "Strength",
    "SymbolChange",
    "SymbolMeta",
    "Tier",
    "TradingCalendar",
    "UniverseSnapshot",
    "Verdict",
    "__version__",
]
