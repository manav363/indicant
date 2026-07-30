"""Corporate-action ingestion and ratio parsing.

NSE publishes corporate actions as free text in a `purpose` field, e.g.

    FACE VALUE SPLIT FROM RS 10/- TO RS 2/-
    BONUS 1:1
    RIGHTS 1:4 @ PREMIUM RS 90/-
    ANNUAL GENERAL MEETING / DIVIDEND RS 5.50 PER SHARE

Deriving an adjustment ratio from that is genuinely error-prone, and a wrong
ratio is worse than a missing one: a missing action leaves a Tier-4 break that
gets investigated, while a wrong ratio *launders* a bad price into looking
explained. So parsing is conservative — anything not confidently recognised
becomes `OTHER` with ratio 1.0 and is left for the Tier-4 sweep to surface.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date

import pandas as pd
from indicant_contracts import CorporateAction, CorporateActionType

from market_data._dates import as_date

# "FACE VALUE SPLIT FROM RS 10/- TO RS 2/-"  ->  ratio 2/10 = 0.2
_SPLIT_FV = re.compile(
    r"SPLIT.*?FROM\s*(?:RS\.?\s*)?([\d.]+).*?TO\s*(?:RS\.?\s*)?([\d.]+)",
    re.IGNORECASE | re.DOTALL,
)
# "SUB-DIVISION FROM RS 10 TO RE 1"
_SPLIT_ALT = re.compile(
    r"SUB[\s-]*DIVISION.*?FROM\s*(?:RS\.?|RE\.?\s*)?\s*([\d.]+).*?TO\s*(?:RS\.?|RE\.?\s*)?\s*([\d.]+)",
    re.IGNORECASE | re.DOTALL,
)
# "BONUS 1:1", "BONUS ISSUE 3:5"
_BONUS = re.compile(r"BONUS.*?(\d+)\s*:\s*(\d+)", re.IGNORECASE | re.DOTALL)
# "RIGHTS 1:4"
_RIGHTS = re.compile(r"RIGHTS.*?(\d+)\s*:\s*(\d+)", re.IGNORECASE | re.DOTALL)

_DIVIDEND = re.compile(r"\bDIVIDEND\b", re.IGNORECASE)
_MERGER = re.compile(r"\b(?:AMALGAMATION|MERGER|SCHEME OF ARRANGEMENT)\b", re.IGNORECASE)
_DEMERGER = re.compile(r"\b(?:DEMERGER|SPIN[\s-]?OFF)\b", re.IGNORECASE)
_DELISTING = re.compile(r"\b(?:DELISTING|DELISTED)\b", re.IGNORECASE)
_SYMBOL_CHANGE = re.compile(r"\b(?:CHANGE IN NAME|NAME CHANGE|SYMBOL CHANGE)\b", re.IGNORECASE)


class ParsedAction:
    __slots__ = ("action_type", "confident", "ratio")

    def __init__(
        self, action_type: CorporateActionType, ratio: float, *, confident: bool
    ) -> None:
        self.action_type = action_type
        self.ratio = ratio
        self.confident = confident


def parse_purpose(text: str) -> ParsedAction:
    """Derive an action type and adjustment ratio from NSE purpose text.

    Ratio semantics: the multiplier applied to *pre-ex-date* prices.

    * Split from FV 10 to FV 2 -> price divides by 5 -> ratio 2/10 = 0.2
    * Bonus a:b (a new shares for every b held) -> ratio b/(a+b)
    * Rights a:b at issue price -> approximated as b/(a+b)

    The rights approximation ignores the subscription price, which makes it
    wrong in general. It is deliberately still emitted as *not confident*, so a
    rights issue never silently explains a Tier-4 break.
    """
    if not text or not isinstance(text, str):
        return ParsedAction(CorporateActionType.OTHER, 1.0, confident=False)

    for pattern in (_SPLIT_FV, _SPLIT_ALT):
        if match := pattern.search(text):
            old_fv, new_fv = float(match.group(1)), float(match.group(2))
            if old_fv > 0 and new_fv > 0 and new_fv < old_fv:
                return ParsedAction(
                    CorporateActionType.SPLIT, new_fv / old_fv, confident=True
                )

    if match := _BONUS.search(text):
        new_shares, held = float(match.group(1)), float(match.group(2))
        if new_shares > 0 and held > 0:
            return ParsedAction(
                CorporateActionType.BONUS, held / (new_shares + held), confident=True
            )

    if match := _RIGHTS.search(text):
        new_shares, held = float(match.group(1)), float(match.group(2))
        if new_shares > 0 and held > 0:
            # Not confident: the true ratio depends on the subscription price,
            # which this text may not contain.
            return ParsedAction(
                CorporateActionType.RIGHTS, held / (new_shares + held), confident=False
            )

    for pattern, kind in (
        (_DEMERGER, CorporateActionType.DEMERGER),
        (_MERGER, CorporateActionType.MERGER),
        (_DELISTING, CorporateActionType.DELISTING),
        (_SYMBOL_CHANGE, CorporateActionType.SYMBOL_CHANGE),
        (_DIVIDEND, CorporateActionType.DIVIDEND),
    ):
        if pattern.search(text):
            # None of these carry a price ratio we can derive from text alone.
            return ParsedAction(kind, 1.0, confident=True)

    return ParsedAction(CorporateActionType.OTHER, 1.0, confident=False)


def from_frame(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    ex_date_col: str = "ex_date",
    purpose_col: str = "purpose",
    confident_only: bool = False,
) -> list[CorporateAction]:
    """Build contract objects from an NSE corporate-actions frame.

    `confident_only=True` drops low-confidence parses. Use it when the actions
    will *explain away* a price break — an unexplained break that gets
    investigated is strictly better than a wrongly-explained one.
    """
    if df.empty:
        return []

    required = {symbol_col, ex_date_col, purpose_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"corporate actions frame missing columns: {sorted(missing)}")

    actions: list[CorporateAction] = []
    for _, row in df.iterrows():
        raw = row[purpose_col]
        parsed = parse_purpose("" if pd.isna(raw) else str(raw))
        if confident_only and not parsed.confident:
            continue
        actions.append(
            CorporateAction(
                symbol=str(row[symbol_col]).strip().upper(),
                action_type=parsed.action_type,
                ex_date=as_date(row[ex_date_col]),
                ratio=parsed.ratio,
                raw_text=None if pd.isna(raw) else str(raw),
            )
        )
    return actions


def to_frame(actions: Iterable[CorporateAction]) -> pd.DataFrame:
    rows = [
        {
            "symbol": a.symbol,
            "action_type": a.action_type.value,
            "ex_date": a.ex_date,
            "ratio": a.ratio,
            "affects_price": a.affects_price,
            "raw_text": a.raw_text,
        }
        for a in actions
    ]
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "action_type", "ex_date", "ratio", "affects_price", "raw_text"]
        )
    return pd.DataFrame(rows).sort_values(["symbol", "ex_date"]).reset_index(drop=True)


def symbol_changes_from_actions(
    actions: Sequence[CorporateAction],
) -> pd.DataFrame:
    """Extract rename events.

    NSE's purpose text names the *new* symbol inconsistently, so this returns
    the events for review rather than pretending to resolve them. Silently
    guessing a rename target would corrupt price history in a way that is very
    hard to detect later.
    """
    renames = [a for a in actions if a.action_type is CorporateActionType.SYMBOL_CHANGE]
    return pd.DataFrame(
        [
            {
                "old_symbol": a.symbol,
                "new_symbol": None,
                "effective_date": a.ex_date,
                "raw_text": a.raw_text,
                "needs_review": True,
            }
            for a in renames
        ],
        columns=["old_symbol", "new_symbol", "effective_date", "raw_text", "needs_review"],
    )


def _as_date_legacy(value: object) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]
