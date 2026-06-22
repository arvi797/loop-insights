"""Async GitHub REST client.

Scope is deliberately narrow: fetch the commits, merged PRs, and reviews needed to
compute collaboration-health metrics over a time window. Handles pagination,
rate-limit signalling, and transient-error retries.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubError(Exception):
    """Upstream GitHub failure surfaced to the caller."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimitError(GitHubError):
    """Raised when GitHub reports the rate limit is exhausted."""


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        max_pages: int = 10,
        client: httpx.AsyncClient | None = None,
    ):
        self._token = token
        self._max_pages = max_pages
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "loop-insights/0.1",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Allow injection of a client for testing; otherwise own one.
        self._client = client or httpx.AsyncClient(
            base_url=GITHUB_API, headers=headers, timeout=30.0
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        resp = await self._client.get(path, params=params)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset", "unknown")
            raise RateLimitError(
                f"GitHub rate limit exhausted; resets at epoch {reset}. "
                "Set GITHUB_TOKEN to raise the limit.",
                status=403,
            )
        if resp.status_code == 404:
            raise GitHubError("Repository not found or not accessible.", status=404)
        if resp.status_code >= 400:
            raise GitHubError(
                f"GitHub returned {resp.status_code}: {resp.text[:200]}",
                status=resp.status_code,
            )
        return resp

    async def _paginate(
        self, path: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Follow Link-header pagination up to max_pages."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        items: list[dict[str, Any]] = []
        page = 1
        while page <= self._max_pages:
            params["page"] = page
            resp = await self._get(path, params)
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            # Stop when GitHub signals no "next" relation.
            if 'rel="next"' not in resp.headers.get("Link", ""):
                break
            page += 1
        return items

    async def get_repo(self, owner: str, name: str) -> dict[str, Any]:
        resp = await self._get(f"/repos/{owner}/{name}")
        return resp.json()

    async def list_commits(
        self, owner: str, name: str, since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        return await self._paginate(
            f"/repos/{owner}/{name}/commits",
            {"since": _iso(since), "until": _iso(until)},
        )

    async def list_pulls(
        self, owner: str, name: str, state: str = "closed"
    ) -> list[dict[str, Any]]:
        """List PRs, newest first. Caller filters by merge time within the window."""
        return await self._paginate(
            f"/repos/{owner}/{name}/pulls",
            {"state": state, "sort": "updated", "direction": "desc"},
        )

    async def list_reviews(
        self, owner: str, name: str, pr_number: int
    ) -> list[dict[str, Any]]:
        return await self._paginate(
            f"/repos/{owner}/{name}/pulls/{pr_number}/reviews"
        )


def _iso(dt: datetime) -> str:
    """GitHub expects ISO-8601 in UTC with a trailing Z."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
