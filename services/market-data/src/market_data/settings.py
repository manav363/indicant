"""Service configuration.

Everything that varies between dev and prod is here. Nothing else reads
os.environ, so there is one place to look when behaviour differs by
environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from indicant_contracts import EligibilityThresholds, LakePaths

# NSE archive hosts. Kept as constants rather than settings because a change
# here is a code change with fixture consequences, not a deployment knob.
NSE_ARCHIVE_BASE = "https://nsearchives.nseindia.com"
NSE_BASE = "https://www.nseindia.com"

# The UDiFF cutover. Before this date the legacy bhavcopy format applies;
# on or after, UDiFF. This single date decides which normalizer runs, and it
# is why the legacy path is the primary one — it covers 2006 through mid-2024.
UDIFF_CUTOVER = "2024-07-08"

# NSE rejects requests without a browser-shaped UA. This is not evasion of a
# bot defence — the archive is public and free; the check is a legacy filter
# that also blocks ordinary scripted access to published files.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    lake_root: Path
    request_timeout: float
    max_retries: int
    retry_backoff: float
    rate_limit_seconds: float
    eligibility: EligibilityThresholds

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            lake_root=_env_path("INDICANT_LAKE_ROOT", "./data"),
            request_timeout=float(os.environ.get("INDICANT_HTTP_TIMEOUT", "30")),
            max_retries=int(os.environ.get("INDICANT_MAX_RETRIES", "3")),
            retry_backoff=float(os.environ.get("INDICANT_RETRY_BACKOFF", "2.0")),
            # A 20-year backfill is ~4,700 sequential requests. Deliberately
            # polite: this is a free public archive and hammering it is both
            # rude and the fastest way to get blocked.
            rate_limit_seconds=float(os.environ.get("INDICANT_RATE_LIMIT", "0.35")),
            eligibility=EligibilityThresholds(
                min_score=float(os.environ.get("INDICANT_MIN_QUALITY_SCORE", "0.85")),
                min_history_days=int(os.environ.get("INDICANT_MIN_HISTORY_DAYS", "756")),
                min_median_turnover=float(os.environ.get("INDICANT_MIN_TURNOVER", "1e7")),
            ),
        )

    @property
    def paths(self) -> LakePaths:
        return LakePaths(root=self.lake_root)


def get_settings() -> Settings:
    return Settings.from_env()
