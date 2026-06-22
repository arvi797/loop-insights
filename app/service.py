"""Application service: orchestrates GitHub fetch -> metrics, with caching.

This is the boundary between the HTTP layer (thin) and the domain (pure metrics +
upstream client). It also owns input validation for the repo identifier and window,
which keeps untrusted input from reaching the upstream client unchecked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from app.github.client import GitHubClient
from app.metrics.collaboration import _parse_dt, compute_collaboration_health
from app.models import CollaborationHealth, Narrative, RepoRef
from app.store.cache import InsightCache

logger = logging.getLogger(__name__)

# owner/name segments: GitHub allows alphanumerics, hyphen, underscore, dot.
# Anchoring the regex prevents path traversal / SSRF via a crafted "repo" value.
_SEGMENT = r"[A-Za-z0-9._-]+"
_REPO_RE = re.compile(rf"^(?P<owner>{_SEGMENT})/(?P<name>{_SEGMENT})$")

MAX_WINDOW_DAYS = 365
# Cap PRs inspected per request: each costs a reviews call, so this bounds fan-out.
MAX_PULLS_INSPECTED = 150


class ValidationError(ValueError):
    """Bad client input (400)."""


def parse_repo(value: str) -> RepoRef:
    match = _REPO_RE.match(value.strip())
    if not match:
        raise ValidationError(
            "repo must be in 'owner/name' form using letters, digits, '.', '_', '-'."
        )
    return RepoRef(owner=match.group("owner"), name=match.group("name"))


def resolve_window(
    days: int | None, start: datetime | None, end: datetime | None
) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    if start and end:
        if start >= end:
            raise ValidationError("period_start must be before period_end.")
    elif start and not end:
        end = now
    else:
        window = days or 30
        if window < 1 or window > MAX_WINDOW_DAYS:
            raise ValidationError(f"days must be between 1 and {MAX_WINDOW_DAYS}.")
        end = now
        start = now - timedelta(days=window)
    if (end - start).days > MAX_WINDOW_DAYS:
        raise ValidationError(f"window must not exceed {MAX_WINDOW_DAYS} days.")
    return start, end


class InsightService:
    def __init__(self, client: GitHubClient, cache: InsightCache):
        self._client = client
        self._cache = cache

    async def get_health(
        self, repo: RepoRef, period_start: datetime, period_end: datetime
    ) -> tuple[CollaborationHealth, bool]:
        """Return (report, cache_hit)."""
        cache_key = InsightCache.key(
            repo.full_name, period_start.isoformat(), period_end.isoformat()
        )
        cached = self._cache.get_health(cache_key)
        if cached is not None:
            return cached, True

        health = await self._compute(repo, period_start, period_end)
        self._cache.put_health(cache_key, health)
        return health, False

    async def get_narrative(
        self,
        repo: RepoRef,
        period_start: datetime,
        period_end: datetime,
        synthesize: Callable[[CollaborationHealth], Awaitable[Narrative]],
    ) -> tuple[Narrative, bool]:
        """Return (narrative, cache_hit).

        The caller supplies how to synthesize (so the LLM provider stays out of the
        service); the service owns the single cache path for both report and
        narrative.
        """
        cache_key = InsightCache.key(
            repo.full_name, period_start.isoformat(), period_end.isoformat()
        )
        cached = self._cache.get_narrative(cache_key)
        if cached is not None:
            return cached, True

        health, _ = await self.get_health(repo, period_start, period_end)
        narrative = await synthesize(health)
        self._cache.put_narrative(cache_key, narrative)
        return narrative, False

    async def _compute(
        self, repo: RepoRef, period_start: datetime, period_end: datetime
    ) -> CollaborationHealth:
        # Validate the repo exists before fanning out into many calls.
        await self._client.get_repo(repo.owner, repo.name)

        commits = await self._client.list_commits(
            repo.owner, repo.name, period_start, period_end
        )
        merged_pulls, truncated = await self._pulls_merged_in_window(
            repo, period_start, period_end
        )
        pulls_with_reviews = await self._attach_reviews(repo, merged_pulls)

        return compute_collaboration_health(
            repo=repo.full_name,
            period_start=period_start,
            period_end=period_end,
            commits=commits,
            pulls_with_reviews=pulls_with_reviews,
            truncated=truncated,
        )

    async def _pulls_merged_in_window(
        self, repo: RepoRef, start: datetime, end: datetime
    ) -> tuple[list[dict], bool]:
        """Filter closed PRs down to those merged within [start, end].

        PRs come back sorted by 'updated desc'; once we pass PRs updated before the
        window we can stop, avoiding a full-history scan. Returns (pulls, truncated)
        where `truncated` is True if the window held more PRs than the per-request cap.
        """
        pulls = await self._client.list_pulls(repo.owner, repo.name, state="closed")
        in_window: list[dict] = []
        truncated = False
        for pull in pulls:
            updated = _parse_dt(pull.get("updated_at"))
            if updated and updated < start:
                break  # everything older is also outside the window
            merged_at = _parse_dt(pull.get("merged_at"))
            if merged_at and start <= merged_at <= end:
                in_window.append(pull)
            if len(in_window) >= MAX_PULLS_INSPECTED:
                truncated = True
                logger.warning(
                    "Hit MAX_PULLS_INSPECTED=%d for %s; report covers the most "
                    "recent merged PRs in the window.",
                    MAX_PULLS_INSPECTED,
                    repo.full_name,
                )
                break
        return in_window, truncated

    async def _attach_reviews(
        self, repo: RepoRef, pulls: list[dict]
    ) -> list[tuple[dict, list[dict]]]:
        """Fetch reviews for each PR concurrently, with bounded parallelism."""
        semaphore = asyncio.Semaphore(8)

        async def fetch(pull: dict) -> tuple[dict, list[dict]]:
            async with semaphore:
                reviews = await self._client.list_reviews(
                    repo.owner, repo.name, pull["number"]
                )
            return pull, reviews

        return await asyncio.gather(*(fetch(p) for p in pulls))
