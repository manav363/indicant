"""Composition and cache tests.

Failure handling is the substance here. Fetching in parallel is easy; deciding
which upstream failure degrades a page and which one blanks it is the part that
needs asserting.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import httpx
import pytest
from indicant_contracts import ErrorCode

from gateway.cache import TradingDayCache
from gateway.composition.client import (
    UpstreamClient,
    UpstreamResult,
    first_failure,
    gather_upstreams,
)


def stub_transport(routes: dict[str, tuple[int, object]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        for path, (status, payload) in routes.items():
            if request.url.path == path:
                return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


# ==========================================================================
# Upstream client
# ==========================================================================


class TestUpstreamClient:
    @pytest.mark.asyncio
    async def test_successful_call_returns_data(self) -> None:
        transport = stub_transport({"/universe": (200, {"symbols": ["A"]})})
        async with httpx.AsyncClient(transport=transport) as http:
            result = await UpstreamClient("http://md:8000").get(
                http, "market-data", "/universe"
            )
        assert result.ok
        assert result.data == {"symbols": ["A"]}

    @pytest.mark.asyncio
    async def test_connection_failure_is_data_not_an_exception(self) -> None:
        """Raising would let one broken upstream take down a page that could
        have rendered most of itself."""
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as http:
            result = await UpstreamClient("http://md:8000").get(http, "market-data", "/x")
        assert not result.ok
        assert result.error is not None
        assert result.error.code is ErrorCode.UPSTREAM_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_timeout_is_reported_as_such(self) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as http:
            result = await UpstreamClient("http://md:8000", timeout=5.0).get(
                http, "market-data", "/x"
            )
        assert not result.ok
        assert "timed out after 5.0s" in result.error.message  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_upstream_error_envelope_is_preserved(self) -> None:
        """The upstream knows more about its own failure than the gateway
        does — replacing its envelope loses the reason."""
        payload = {
            "detail": {
                "code": "symbol_not_eligible",
                "message": "NEWCO excluded: only 40 trading days",
                "user_message": "We do not have enough data on NEWCO yet.",
                "detail": {},
            }
        }
        transport = stub_transport({"/predict": (422, payload)})
        async with httpx.AsyncClient(transport=transport) as http:
            result = await UpstreamClient("http://intel:8000").get(
                http, "intelligence", "/predict"
            )
        assert not result.ok
        assert result.error is not None
        assert result.error.code is ErrorCode.SYMBOL_NOT_ELIGIBLE
        assert "NEWCO" in result.error.user_message

    @pytest.mark.asyncio
    async def test_non_json_response_is_handled(self) -> None:
        def html(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway timeout</html>")

        async with httpx.AsyncClient(transport=httpx.MockTransport(html)) as http:
            result = await UpstreamClient("http://md:8000").get(http, "market-data", "/x")
        assert not result.ok
        assert result.error.code is ErrorCode.INTERNAL  # type: ignore[union-attr]


# ==========================================================================
# Parallel composition
# ==========================================================================


class TestGatherUpstreams:
    @pytest.mark.asyncio
    async def test_all_results_are_keyed_by_name(self) -> None:
        transport = stub_transport({
            "/universe": (200, {"symbols": []}),
            "/predict": (200, {"signal": "BUY"}),
        })
        md = UpstreamClient("http://md:8000")
        intel = UpstreamClient("http://intel:8000")
        async with httpx.AsyncClient(transport=transport) as http:
            results = await gather_upstreams(
                [
                    (md, "market-data", "/universe", None),
                    (intel, "intelligence", "/predict", None),
                ],
                client=http,
            )
        assert set(results) == {"market-data", "intelligence"}
        assert all(r.ok for r in results.values())

    @pytest.mark.asyncio
    async def test_one_failure_does_not_cancel_the_others(self) -> None:
        """The whole point of composing in parallel: a partial page beats no
        page."""
        transport = stub_transport({"/universe": (200, {"symbols": []})})
        md = UpstreamClient("http://md:8000")
        intel = UpstreamClient("http://intel:8000")
        async with httpx.AsyncClient(transport=transport) as http:
            results = await gather_upstreams(
                [
                    (md, "market-data", "/universe", None),
                    (intel, "intelligence", "/missing", None),
                ],
                client=http,
            )
        assert results["market-data"].ok
        assert not results["intelligence"].ok

    @pytest.mark.asyncio
    async def test_calls_actually_run_concurrently(self) -> None:
        """Sequential would make page latency the SUM of the round trips
        instead of the MAX."""
        async def slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.15)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(slow_handler)
        clients = [UpstreamClient(f"http://s{i}:8000") for i in range(4)]
        async with httpx.AsyncClient(transport=transport) as http:
            start = asyncio.get_running_loop().time()
            await gather_upstreams(
                [(c, f"svc{i}", "/x", None) for i, c in enumerate(clients)],
                client=http,
            )
            elapsed = asyncio.get_running_loop().time() - start

        # Four 0.15s calls: ~0.15s parallel, ~0.60s sequential.
        assert elapsed < 0.4


class TestFirstFailure:
    def test_returns_none_when_everything_essential_succeeded(self) -> None:
        results = {
            "a": UpstreamResult("a", ok=True, data={}),
            "b": UpstreamResult("b", ok=False),
        }
        assert first_failure(results, essential=["a"]) is None

    def test_a_failed_essential_upstream_is_returned(self) -> None:
        results = {"a": UpstreamResult("a", ok=False)}
        assert first_failure(results, essential=["a"]) is not None

    def test_a_non_essential_failure_only_degrades(self) -> None:
        """A missing regime read degrades a page; a missing prediction blanks
        it. The composer decides which is which."""
        results = {
            "prediction": UpstreamResult("prediction", ok=True, data={}),
            "regime": UpstreamResult("regime", ok=False),
        }
        assert first_failure(results, essential=["prediction"]) is None

    def test_an_uncalled_essential_upstream_is_a_failure(self) -> None:
        assert first_failure({}, essential=["prediction"]) is not None


# ==========================================================================
# Trading-day cache
# ==========================================================================


class TestTradingDayCache:
    @pytest.fixture
    def today(self) -> date:
        return date(2026, 7, 30)

    def test_stores_and_returns(self, today) -> None:
        cache = TradingDayCache()
        cache.set("RELIANCE", {"signal": "BUY"}, trading_day=today)
        assert cache.get("RELIANCE", trading_day=today) == {"signal": "BUY"}

    def test_a_new_trading_day_invalidates_regardless_of_ttl(self, today) -> None:
        """The reason the key includes the day: a wall-clock TTL alone would
        serve yesterday's answer across the boundary where new data landed."""
        cache = TradingDayCache(ttl_seconds=10_000)
        cache.set("RELIANCE", {"signal": "BUY"}, trading_day=today)
        assert cache.get("RELIANCE", trading_day=today + timedelta(days=1)) is None

    def test_expired_entries_are_dropped(self, today) -> None:
        cache = TradingDayCache(ttl_seconds=0.0)
        cache.set("RELIANCE", 1, trading_day=today)
        assert cache.get("RELIANCE", trading_day=today) is None

    def test_miss_on_an_unknown_key(self, today) -> None:
        assert TradingDayCache().get("NOPE", trading_day=today) is None

    def test_capacity_is_bounded(self, today) -> None:
        """Unbounded growth is a slow leak whose symptom appears hours later,
        far from the cause."""
        cache = TradingDayCache(max_entries=10)
        for i in range(50):
            cache.set(f"S{i}", i, trading_day=today)
        assert cache.size <= 10

    def test_hit_rate_is_tracked(self, today) -> None:
        cache = TradingDayCache()
        cache.set("A", 1, trading_day=today)
        cache.get("A", trading_day=today)
        cache.get("B", trading_day=today)
        assert cache.hit_rate == pytest.approx(0.5)

    def test_invalidate_removes_one_key(self, today) -> None:
        cache = TradingDayCache()
        cache.set("A", 1, trading_day=today)
        assert cache.invalidate("A") is True
        assert cache.invalidate("A") is False

    def test_stats_expose_everything_operationally_useful(self, today) -> None:
        assert set(TradingDayCache().stats()) == {
            "size", "hits", "misses", "hit_rate", "ttl_seconds", "max_entries"
        }
