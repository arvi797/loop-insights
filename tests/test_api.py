"""API-level tests with the network and LLM mocked.

Exercises the HTTP contract end-to-end: validation errors, the collaboration
endpoint over mocked GitHub responses, cache HIT/MISS semantics, and the narrative
endpoint with a fake LLM provider.
"""


import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, get_app_settings, get_service
from app.config import Settings
from app.github.client import GitHubClient
from app.llm.provider import OpenAIProvider
from app.models import EvidenceItem, NarrativeDraft
from app.service import InsightService
from app.store.cache import InsightCache


class _FakeTransport(httpx.MockTransport):
    """Serves canned GitHub responses so the service can run without network."""

    def __init__(self):
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/repos/octo/demo"):
            return httpx.Response(200, json={"full_name": "octo/demo"})
        if path.endswith("/commits"):
            return httpx.Response(200, json=[
                {"author": {"login": "alice"}, "commit": {"author": {"name": "alice"}}},
                {"author": {"login": "bob"}, "commit": {"author": {"name": "bob"}}},
            ])
        if path.endswith("/pulls") and "/pulls/" not in path:
            return httpx.Response(200, json=[{
                "number": 1, "user": {"login": "alice"},
                "created_at": "2099-01-01T00:00:00Z",
                "merged_at": "2099-01-01T05:00:00Z",
                "updated_at": "2099-01-01T05:00:00Z",
            }])
        if "/pulls/1/reviews" in path:
            return httpx.Response(200, json=[
                {"user": {"login": "bob"}, "submitted_at": "2099-01-01T02:00:00Z"}
            ])
        return httpx.Response(404, json={"message": "not found"})


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Wire app state manually (we bypass lifespan in TestClient-with-overrides).
    fake_http = httpx.AsyncClient(
        base_url="https://api.github.com", transport=_FakeTransport()
    )
    gh = GitHubClient(token=None, client=fake_http)
    cache = InsightCache(str(tmp_path / "t.db"))
    service = InsightService(gh, cache)

    test_settings = Settings(default_repo="octo/demo", openai_api_key="x")
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_app_settings] = lambda: test_settings

    # raise_server_exceptions=False so our exception handlers produce real HTTP codes.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_bad_repo_is_400(client):
    r = client.get("/v1/insights/collaboration", params={"repo": "../etc/passwd"})
    assert r.status_code == 400
    assert "owner/name" in r.json()["error"]


def test_collaboration_and_cache(client):
    params = {
        "repo": "octo/demo",
        "period_start": "2099-01-01T00:00:00Z",
        "period_end": "2099-01-31T00:00:00Z",
    }
    r1 = client.get("/v1/insights/collaboration", params=params)
    assert r1.status_code == 200
    assert r1.headers["X-Cache"] == "MISS"
    body = r1.json()
    assert body["total_commits"] == 2
    assert body["total_prs_merged"] == 1
    assert body["total_reviews"] == 1
    assert body["busiest_reviewer"] == "bob"

    # Second identical request is served from cache.
    r2 = client.get("/v1/insights/collaboration", params=params)
    assert r2.headers["X-Cache"] == "HIT"


def test_narrative_with_fake_llm(client, monkeypatch):
    async def fake_draft(self, system, user):
        return NarrativeDraft(
            summary="Bob did the only review.",
            root_cause_hypothesis=None,
            confidence=0.9,
            evidence=[EvidenceItem(metric="total_reviews", value=1, detail="one review")],
        )

    monkeypatch.setattr(OpenAIProvider, "draft_narrative", fake_draft)
    monkeypatch.setattr(OpenAIProvider, "__init__", lambda self, *a, **k: None)

    params = {
        "repo": "octo/demo",
        "period_start": "2099-01-01T00:00:00Z",
        "period_end": "2099-01-31T00:00:00Z",
    }
    r = client.get("/v1/insights/narrative", params=params)
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["evidence"][0]["metric"] == "total_reviews"
    # confidence capped by signal strength (thin data) below the model's 0.9
    assert body["confidence"] <= 0.9
