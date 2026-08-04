"""Tests for the provenance-chain routes.

These endpoints exist because the browser can only reach the gateway. Thirteen
built-and-tested upstream routes had no public path, so the evidence tier — the
quality gate, the point-in-time universe, the model card — could not appear on a
screen no matter how good the backend was.

What is worth asserting here is the DEGRADATION shape. Each of these composes
several upstreams, and the wrong failure behaviour turns "one upstream is slow"
into "the whole rail is blank", or worse, into a rail that silently shows zeros
as though they were measurements.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.api.main import app
from gateway.api import terminal


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def route_stub(routes: dict[str, tuple[int, object]]):
    """Patch BOTH upstream clients onto one mock transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        for path, (status, payload) in routes.items():
            if request.url.path == path:
                return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"detail": "not found"})
    return httpx.MockTransport(handler)


@pytest.fixture
def upstreams(monkeypatch):
    """Point every gateway upstream call at a stub transport."""
    def install(routes: dict[str, tuple[int, object]]) -> None:
        transport = route_stub(routes)
        real_init = httpx.AsyncClient.__init__

        def patched(self, *a, **kw):
            kw["transport"] = transport
            real_init(self, *a, **kw)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return install


LAKE = {"has_data": True, "trading_days": 1166, "last_date": "2026-07-30",
        "first_date": "2022-01-03"}
COVERAGE = {"coverage": 0.9914965986394558, "observed": 1166,
            "missing": ["2022-08-08", "2022-08-09"], "unexpected": [],
            "uncurated_years": [2022]}
UNIVERSE = {"as_of": "2026-08-03", "symbols": ["A", "B", "C", "D"],
            "eligible_symbols": ["A"],
            "excluded": {
                "B": "has not traded recently — the listing looks delisted",
                "C": "only 73 trading days of history",
                "D": "data quality score 0.82 is below 0.85",
            }}
MODEL = {"run_id": "20260801_231b", "trained_at": "2026-08-01",
         "model_type": "stacked", "n_features": 94, "universe_size": 30,
         "permutation_p_value": 0.0597, "is_significant": False}


class TestChain:
    def test_returns_all_five_stages_in_one_call(self, client, upstreams) -> None:
        """One call, not five — the rail is on every screen."""
        upstreams({"/health": (200, LAKE),
                   "/internal/quality/coverage": (200, COVERAGE),
                   "/universe": (200, UNIVERSE),
                   "/model/current": (200, MODEL)})
        r = client.get("/api/chain")
        assert r.status_code == 200
        body = r.json()
        assert body["source"]["value"] == 1166
        assert body["universe"]["eligible"] == 1
        assert body["universe"]["seen"] == 4
        assert body["model"]["pValue"] == pytest.approx(0.0597)

    def test_significance_is_precomputed_not_left_to_the_client(self, client, upstreams) -> None:
        """If each client re-derives `p < 0.05` they will eventually disagree
        with the backend about what the model claims."""
        upstreams({"/health": (200, LAKE),
                   "/internal/quality/coverage": (200, COVERAGE),
                   "/universe": (200, UNIVERSE),
                   "/model/current": (200, MODEL)})
        assert client.get("/api/chain").json()["model"]["isSignificant"] is False

    def test_a_dead_upstream_degrades_the_rail_rather_than_blanking_it(
        self, client, upstreams
    ) -> None:
        """The lake being unreachable must not hide the universe count."""
        upstreams({"/internal/quality/coverage": (200, COVERAGE),
                   "/universe": (200, UNIVERSE),
                   "/model/current": (200, MODEL)})
        body = client.get("/api/chain").json()
        assert "lake" in body["degraded"]
        assert body["universe"]["eligible"] == 1     # still reported

    def test_untrained_model_is_reported_as_untrained_not_as_zero(
        self, client, upstreams
    ) -> None:
        """`pValue: 0` would render as a wildly significant model. None is the
        only honest value for "there is no model"."""
        upstreams({"/health": (200, LAKE),
                   "/internal/quality/coverage": (200, COVERAGE),
                   "/universe": (200, UNIVERSE),
                   "/model/current": (503, {"detail": "no model"})})
        m = client.get("/api/chain").json()["model"]
        assert m["trained"] is False
        assert m["pValue"] is None
        assert m["isSignificant"] is None


class TestProvenance:
    def test_composes_lineage_and_quality(self, client, upstreams) -> None:
        upstreams({
            "/symbols/RELIANCE/meta": (200, {"symbol": "RELIANCE", "isin": "INE002A01018",
                                             "series": "EQ", "status": "listed",
                                             "first_seen": "2022-01-03",
                                             "last_seen": "2026-07-30"}),
            "/symbols/RELIANCE/quality": (200, {"symbol": "RELIANCE",
                                                "history_completeness": 1.0,
                                                "validity_clean_rate": 1.0,
                                                "continuity_clean_rate": 1.0,
                                                "liquidity_adequacy": 1.0,
                                                "recency": 1.0,
                                                "history_days": 1166,
                                                "median_turnover": 15045389416.3}),
        })
        body = client.get("/api/provenance/reliance").json()
        assert body["symbol"] == "RELIANCE"          # normalised
        assert body["meta"]["isin"] == "INE002A01018"
        assert len(body["components"]) == 5
        assert body["historyDays"] == 1166

    def test_missing_quality_still_returns_lineage(self, client, upstreams) -> None:
        """Half the rail is better than none — a symbol with no quality row
        still has a listing history worth showing."""
        upstreams({"/symbols/X/meta": (200, {"symbol": "X", "series": "EQ",
                                             "status": "listed",
                                             "first_seen": "2022-01-03",
                                             "last_seen": "2026-07-30"})})
        body = client.get("/api/provenance/X").json()
        assert body["meta"]["series"] == "EQ"
        assert body["components"] == []
        assert "quality" in body["degraded"]


class TestGate:
    def test_missing_days_are_listed_not_just_counted(self, client, upstreams) -> None:
        """A missing trading day that is merely counted is a missing day nobody
        can go and look at."""
        upstreams({"/internal/quality/coverage": (200, COVERAGE), "/health": (200, LAKE)})
        body = client.get("/api/gate").json()
        assert body["missing"] == ["2022-08-08", "2022-08-09"]
        assert body["expected"] == 1168          # observed + missing
        assert len(body["tiers"]) == 6

    def test_coverage_upstream_failure_is_a_503_with_a_reason(
        self, client, upstreams
    ) -> None:
        upstreams({"/health": (200, LAKE)})
        r = client.get("/api/gate")
        assert r.status_code == 503
        assert "user_message" in r.json()["detail"]


class TestUniverseDetail:
    def test_refusals_are_grouped_by_reason(self, client, upstreams) -> None:
        """2,169 individual refusal strings is not a screen. The shape of the
        refusal is the thing worth seeing."""
        upstreams({"/universe": (200, UNIVERSE)})
        body = client.get("/api/universe/detail").json()
        assert body["seen"] == 4
        assert body["eligible"] == 1
        assert body["excluded"] == 3
        labels = {g["reason"] for g in body["groups"]}
        assert "Delisted or suspended" in labels
        assert "Insufficient history" in labels
        assert "Quality score below floor" in labels

    def test_groups_are_ranked_and_keep_concrete_examples(self, client, upstreams) -> None:
        upstreams({"/universe": (200, UNIVERSE)})
        groups = client.get("/api/universe/detail").json()["groups"]
        assert groups == sorted(groups, key=lambda g: -g["count"])
        # Every group carries a real symbol, so the number stays checkable.
        assert all(g["examples"] for g in groups)


class TestModelCard:
    def test_returns_the_card(self, client, upstreams) -> None:
        upstreams({"/model/current": (200, MODEL)})
        body = client.get("/api/model").json()
        assert body["runId"] == "20260801_231b"
        assert body["nFeatures"] == 94
        assert body["isSignificant"] is False

    def test_untrained_is_503_not_an_empty_card(self, client, upstreams) -> None:
        """"No model" and "a model that reports nothing" must not look alike."""
        upstreams({"/model/current": (503, {"detail": {"code": "model_not_trained",
                                                       "message": "none",
                                                       "user_message": "Not trained yet."}})})
        assert client.get("/api/model").status_code == 503
