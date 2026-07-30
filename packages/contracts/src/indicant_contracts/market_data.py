"""Market data contracts.

The canonical OHLCV schema is the single definition of what a price row is,
across both bhavcopy format eras. Anything that reads or writes the lake
agrees on these column names and types or it does not compile.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Canonical column order for the price lake. Ingestion writes exactly these;
# readers select from exactly these. Changing this list is a contract change
# and will fail the schema-freeze test.
CANONICAL_PRICE_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "series",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover",
    "trades",
    "delivery_qty",
    "delivery_pct",
    "isin",
)

# Columns added by the adjustment pipeline, on top of the canonical set.
ADJUSTED_EXTRA_COLUMNS: tuple[str, ...] = ("adj_factor", "adj_close")


class Series(StrEnum):
    """NSE equity series. EQ is the rolling-settlement equity series we model.

    BE/BZ are trade-for-trade (surveillance) series; SM/ST are SME platform.
    We keep them ingested but they are excluded from the eligible universe.
    """

    EQ = "EQ"
    BE = "BE"
    BZ = "BZ"
    SM = "SM"
    ST = "ST"
    OTHER = "OTHER"


class ListingStatus(StrEnum):
    LISTED = "listed"
    DELISTED = "delisted"
    SUSPENDED = "suspended"


class OHLCVBar(BaseModel):
    """One security, one trading day. Mirrors CANONICAL_PRICE_COLUMNS.

    Validity invariants (high >= low, high >= max(open, close), etc.) are
    enforced here so a malformed bar cannot be constructed in the first place.
    The quality gate re-checks them at the dataframe level because bulk
    ingestion never round-trips through this model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: date
    symbol: str = Field(min_length=1, max_length=32)
    series: Series = Series.EQ
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    prev_close: float | None = Field(default=None, gt=0)
    volume: int = Field(ge=0)
    turnover: float = Field(ge=0)
    trades: int | None = Field(default=None, ge=0)
    delivery_qty: int | None = Field(default=None, ge=0)
    delivery_pct: float | None = Field(default=None, ge=0, le=100)
    isin: str | None = None

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _check_ohlc_ordering(self) -> OHLCVBar:
        if self.high < self.low:
            raise ValueError(f"high {self.high} < low {self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high {self.high} < max(open, close) {max(self.open, self.close)}")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low {self.low} > min(open, close) {min(self.open, self.close)}")
        if self.delivery_qty is not None and self.delivery_qty > self.volume:
            raise ValueError(f"delivery_qty {self.delivery_qty} > volume {self.volume}")
        return self


class SymbolMeta(BaseModel):
    """What we know about a security independent of any single trading day.

    `delisted_on` being set is what makes the historical universe
    survivorship-bias-free: the symbol stays in history and stops being
    eligible, rather than vanishing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    isin: str | None = None
    name: str | None = None
    sector: str | None = None
    series: Series = Series.EQ
    status: ListingStatus = ListingStatus.LISTED
    first_seen: date
    last_seen: date
    delisted_on: date | None = None

    @property
    def is_active(self) -> bool:
        return self.status is ListingStatus.LISTED


class SymbolChange(BaseModel):
    """A symbol rename. Naive joins across a rename silently split one
    company's history into two, so this map is required, not optional.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    old_symbol: str
    new_symbol: str
    effective_date: date
    isin: str | None = None


class UniverseSnapshot(BaseModel):
    """Point-in-time universe. `as_of` is load-bearing: asking for the
    universe without a date is how survivorship bias gets in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: date
    index_name: str | None = None
    symbols: tuple[str, ...]
    eligible_symbols: tuple[str, ...]
    excluded: dict[str, str] = Field(
        default_factory=dict,
        description="symbol -> human-readable reason it is not eligible",
    )

    @property
    def total(self) -> int:
        return len(self.symbols)

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_symbols)

    @model_validator(mode="after")
    def _eligible_is_subset(self) -> UniverseSnapshot:
        extra = set(self.eligible_symbols) - set(self.symbols)
        if extra:
            raise ValueError(f"eligible symbols not in universe: {sorted(extra)}")
        return self


class CorporateActionType(StrEnum):
    SPLIT = "split"
    BONUS = "bonus"
    DIVIDEND = "dividend"
    RIGHTS = "rights"
    MERGER = "merger"
    DEMERGER = "demerger"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    OTHER = "other"


class CorporateAction(BaseModel):
    """A corporate action with its price-adjustment ratio.

    `ratio` is the multiplicative factor applied to pre-event prices to make
    them comparable with post-event prices. A 1:2 split has ratio 0.5 — every
    price before the ex-date is halved.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    action_type: CorporateActionType
    ex_date: date
    ratio: float = Field(default=1.0, gt=0)
    raw_text: str | None = None

    @property
    def affects_price(self) -> bool:
        """Dividends are not back-adjusted in this system (total-return
        adjustment is a separate, opt-in concern). Only ratio changes are.
        """
        return self.action_type in {
            CorporateActionType.SPLIT,
            CorporateActionType.BONUS,
            CorporateActionType.RIGHTS,
        } and self.ratio != 1.0


class TradingCalendar(BaseModel):
    """Which days the exchange actually traded. Required to tell 'no file
    because it was a holiday' apart from 'no file because ingestion broke'.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date
    end: date
    trading_days: tuple[date, ...]

    def is_trading_day(self, d: date) -> bool:
        return d in set(self.trading_days)

    @property
    def count(self) -> int:
        return len(self.trading_days)
