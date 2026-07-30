"""Upstream clients and parallel composition.

A stock page needs the universe (market-data), a prediction (intelligence) and
a regime read (intelligence). Fetching them in sequence makes page latency the
SUM of three round trips; fetching them together makes it the MAX. On a page
where each call is ~200ms that is the difference between 600ms and 200ms, and
the calls have no dependency on each other.

Failures are partial by design. A missing regime read should degrade the page,
not blank it — so each upstream result carries its own success flag and the
composer decides what is essential.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from indicant_contracts import ErrorCode, ErrorEnvelope

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class UpstreamResult:
    """One upstream call. `ok=False` is data, not an exception.

    Raising here would make any single slow or broken upstream take down a page
    that could have rendered most of itself.
    """

    name: str
    ok: bool
    data: Any = None
    error: ErrorEnvelope | None = None
    status_code: int | None = None

    @classmethod
    def failure(
        cls, name: str, code: ErrorCode, message: str, user_message: str,
        status_code: int | None = None,
    ) -> UpstreamResult:
        return cls(
            name=name,
            ok=False,
            error=ErrorEnvelope(code=code, message=message, user_message=user_message),
            status_code=status_code,
        )


class UpstreamClient:
    """Thin async wrapper. Translates transport and HTTP errors into
    `UpstreamResult` so callers never handle two failure shapes.
    """

    def __init__(self, base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def get(
        self, client: httpx.AsyncClient, name: str, path: str,
        params: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        url = f"{self._base_url}{path}"
        try:
            resp = await client.get(url, params=params or {}, timeout=self._timeout)
        except httpx.TimeoutException:
            return UpstreamResult.failure(
                name, ErrorCode.UPSTREAM_UNAVAILABLE,
                f"{name} timed out after {self._timeout}s at {url}",
                "That took too long to load. Please try again.",
            )
        except httpx.RequestError as exc:
            return UpstreamResult.failure(
                name, ErrorCode.UPSTREAM_UNAVAILABLE,
                f"{name} unreachable: {type(exc).__name__}: {exc}",
                "We could not reach part of the system just now.",
            )

        if resp.status_code >= 400:
            # Preserve an upstream's own envelope when it sent one — it knows
            # more about the failure than the gateway does.
            detail = _extract_envelope(resp)
            if detail is not None:
                return UpstreamResult(
                    name=name, ok=False, error=detail, status_code=resp.status_code
                )
            return UpstreamResult.failure(
                name, ErrorCode.UPSTREAM_UNAVAILABLE,
                f"{name} returned HTTP {resp.status_code} for {url}",
                "Something went wrong loading this.",
                status_code=resp.status_code,
            )

        try:
            return UpstreamResult(name=name, ok=True, data=resp.json(),
                                  status_code=resp.status_code)
        except ValueError:
            return UpstreamResult.failure(
                name, ErrorCode.INTERNAL,
                f"{name} returned non-JSON from {url}",
                "Something went wrong loading this.",
            )


def _extract_envelope(resp: httpx.Response) -> ErrorEnvelope | None:
    try:
        payload = resp.json()
    except ValueError:
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        return None
    try:
        return ErrorEnvelope.model_validate(detail)
    except Exception:
        return None


async def gather_upstreams(
    calls: list[tuple[UpstreamClient, str, str, dict[str, Any] | None]],
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, UpstreamResult]:
    """Run every call concurrently and key results by name.

    `return_exceptions=True` so one raising task cannot cancel its siblings —
    the whole point of composing in parallel is that a partial page beats no
    page.
    """
    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        results = await asyncio.gather(
            *[c.get(client, name, path, params) for c, name, path, params in calls],
            return_exceptions=True,
        )
    finally:
        if owns_client:
            await client.aclose()

    out: dict[str, UpstreamResult] = {}
    for (_, name, _, _), result in zip(calls, results, strict=True):
        if isinstance(result, BaseException):
            logger.exception("upstream %s raised", name)
            out[name] = UpstreamResult.failure(
                name, ErrorCode.INTERNAL,
                f"{type(result).__name__}: {result}",
                "Something went wrong loading this.",
            )
        else:
            out[name] = result
    return out


def first_failure(
    results: dict[str, UpstreamResult], *, essential: list[str]
) -> UpstreamResult | None:
    """The first essential upstream that failed, if any.

    Explicit `essential` list: a failed regime read degrades a page, a failed
    prediction blanks it, and the composer must be the one to decide which is
    which rather than treating every upstream as equally load-bearing.
    """
    for name in essential:
        result = results.get(name)
        if result is None or not result.ok:
            return result or UpstreamResult.failure(
                name, ErrorCode.INTERNAL, f"{name} was never called",
                "Something went wrong loading this.",
            )
    return None
