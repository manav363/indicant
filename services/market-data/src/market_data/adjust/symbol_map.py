"""Symbol-change resolution.

When a company renames, a naive join splits one company's history into two
shorter series. Both then look like they have insufficient history, both get
excluded from the universe, and the company disappears from the backtest — a
survivorship bias reintroduced by a string mismatch.

This is what v1's `.NS` / `.BO` alias hack was groping toward. The difference is
that a rename has an *effective date*, so resolution has to be point-in-time:
asking "what was this company called in 2015" is a different question from
"what is it called now".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

import pandas as pd
from indicant_contracts import SymbolChange

from market_data._dates import as_date


class SymbolMap:
    """Resolves symbols across renames, in both directions."""

    def __init__(self, changes: Sequence[SymbolChange] = ()) -> None:
        self._changes = sorted(changes, key=lambda c: c.effective_date)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> SymbolMap:
        if df.empty:
            return cls(())
        return cls(
            [
                SymbolChange(
                    old_symbol=str(r["old_symbol"]).upper(),
                    new_symbol=str(r["new_symbol"]).upper(),
                    effective_date=as_date(r["effective_date"]),
                    isin=(None if pd.isna(r.get("isin")) else str(r.get("isin"))),
                )
                for _, r in df.iterrows()
            ]
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "old_symbol": c.old_symbol,
                    "new_symbol": c.new_symbol,
                    "effective_date": c.effective_date,
                    "isin": c.isin,
                }
                for c in self._changes
            ]
        )

    def current_symbol(self, symbol: str, *, as_of: date | None = None) -> str:
        """Follow renames forward to the name in use at `as_of` (default: latest).

        Loop-guarded: a data error creating A->B->A would otherwise hang here,
        and a hang in an ingest loop is indistinguishable from a slow network.
        """
        current = symbol.upper()
        seen = {current}
        for _ in range(len(self._changes) + 1):
            nxt = next(
                (
                    c.new_symbol
                    for c in self._changes
                    if c.old_symbol == current and (as_of is None or c.effective_date <= as_of)
                ),
                None,
            )
            if nxt is None or nxt in seen:
                return current
            current = nxt
            seen.add(current)
        return current

    def history_chain(self, symbol: str) -> tuple[str, ...]:
        """Every name this company has traded under, oldest first.

        Used to stitch a full history: reading prices for the chain rather than
        for one symbol is what stops a rename truncating a series.
        """
        target = self.current_symbol(symbol)
        chain: list[str] = [target]
        changed = True
        while changed:
            changed = False
            for c in self._changes:
                if c.new_symbol in chain and c.old_symbol not in chain:
                    chain.insert(0, c.old_symbol)
                    changed = True
        return tuple(chain)

    def canonicalise(self, symbols: Iterable[str], *, as_of: date | None = None) -> list[str]:
        seen: dict[str, None] = {}
        for s in symbols:
            seen.setdefault(self.current_symbol(s, as_of=as_of), None)
        return list(seen)

    def apply(self, prices: pd.DataFrame, *, as_of: date | None = None) -> pd.DataFrame:
        """Rewrite the symbol column to canonical names.

        Deliberately keeps `original_symbol` — losing the name a row actually
        traded under makes any later disagreement with the exchange
        unreconcilable.
        """
        if prices.empty or not self._changes:
            return prices.copy()
        out = prices.copy()
        out["original_symbol"] = out["symbol"]
        out["symbol"] = [self.current_symbol(str(s), as_of=as_of) for s in out["symbol"]]
        return out

    def __len__(self) -> int:
        return len(self._changes)

