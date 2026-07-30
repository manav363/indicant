"""TTL cache keyed by trading day.

Predictions change once per day, when new prices land. Recomputing one per
request wastes the panel scan behind it; caching for a fixed wall-clock TTL
serves a stale answer across the boundary where new data arrived.

So the key includes the trading day. A new day invalidates every entry for free,
and within a day the TTL only bounds how long a mid-day model redeploy stays
invisible.
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class _Entry:
    value: Any
    stored_at: float
    trading_day: date


class TradingDayCache:
    """In-process TTL cache. Deliberately not Redis.

    One gateway process, a few thousand symbols, values measured in kilobytes.
    A network hop to fetch a cached value that fits in a dict is slower than the
    dict and adds a dependency that can fail.
    """

    def __init__(self, *, ttl_seconds: float = 900.0, max_entries: int = 5_000) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: dict[Hashable, _Entry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, *, trading_day: date) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        # A new trading day invalidates regardless of TTL: the data the value
        # was computed from has been superseded.
        if entry.trading_day != trading_day:
            del self._store[key]
            self.misses += 1
            return None

        if time.time() - entry.stored_at > self._ttl:
            del self._store[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def set(self, key: Hashable, value: Any, *, trading_day: date) -> None:
        if len(self._store) >= self._max:
            self._evict()
        self._store[key] = _Entry(value=value, stored_at=time.time(),
                                  trading_day=trading_day)

    def _evict(self) -> None:
        """Drop expired entries first; if none are expired, drop the oldest.

        Unbounded growth here would be a slow leak in a long-running process,
        and the failure mode (memory pressure hours later) is hard to trace back.
        """
        now = time.time()
        expired = [k for k, e in self._store.items() if now - e.stored_at > self._ttl]
        if expired:
            for k in expired:
                del self._store[k]
            return
        oldest = min(self._store, key=lambda k: self._store[k].stored_at)
        del self._store[oldest]

    def invalidate(self, key: Hashable) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self._ttl,
            "max_entries": self._max,
        }
