"""HTTP layer.

Thin by design: parse and validate input, call the service, serialise. All domain
logic lives below this layer. Two insight endpoints (numbers, then grounded
narrative) plus a health check.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Response
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.github.client import GitHubClient, GitHubError, RateLimitError
from app.llm.narrative import synthesize_narrative
from app.llm.provider import (
    LLMNotConfiguredError,
    build_judge_provider,
    build_provider,
)
from app.models import CollaborationHealth, Narrative
from app.service import (
    InsightService,
    ValidationError,
    parse_repo,
    resolve_window,
)
from app.store.cache import InsightCache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app.state.settings = settings
    app.state.client = GitHubClient(
        token=settings.github_token, max_pages=settings.max_pages
    )
    app.state.cache = InsightCache(settings.sqlite_path)
    logger.info("loop-insights started (default repo: %s)", settings.default_repo)
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="Loop Insights",
    version="0.1.0",
    summary="Collaboration-health insights from GitHub, with a grounded LLM narrative.",
    lifespan=lifespan,
)


def get_service() -> InsightService:
    return InsightService(app.state.client, app.state.cache)


def get_app_settings() -> Settings:
    return app.state.settings


# --- Error handling: map domain errors to sensible HTTP semantics ---


@app.exception_handler(ValidationError)
async def _validation_handler(_req, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(RateLimitError)
async def _rate_limit_handler(_req, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error": str(exc)})


@app.exception_handler(GitHubError)
async def _github_handler(_req, exc: GitHubError):
    status = 404 if exc.status == 404 else 502
    return JSONResponse(status_code=status, content={"error": str(exc)})


@app.exception_handler(LLMNotConfiguredError)
async def _llm_config_handler(_req, exc: LLMNotConfiguredError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


# --- Routes ---


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/insights/collaboration", response_model=CollaborationHealth, tags=["insights"])
async def collaboration_insight(
    response: Response,
    service: Annotated[InsightService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    repo: Annotated[str | None, Query(description="owner/name; defaults to DEFAULT_REPO")] = None,
    days: Annotated[int | None, Query(ge=1, le=365, description="trailing window in days")] = None,
    period_start: Annotated[datetime | None, Query(description="ISO-8601 window start")] = None,
    period_end: Annotated[datetime | None, Query(description="ISO-8601 window end")] = None,
) -> CollaborationHealth:
    """Collaboration-health report over a period.

    Surfaces top contributors, review-load distribution, review concentration
    (bus-factor), PR time-to-first-review / time-to-merge, and unreviewed merges.
    """
    repo_ref = parse_repo(repo or settings.default_repo)
    start, end = resolve_window(days, period_start, period_end)
    health, cache_hit = await service.get_health(repo_ref, start, end)
    response.headers["X-Cache"] = "HIT" if cache_hit else "MISS"
    return health


@app.get("/v1/insights/narrative", response_model=Narrative, tags=["insights"])
async def narrative_insight(
    response: Response,
    service: Annotated[InsightService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    repo: Annotated[str | None, Query(description="owner/name; defaults to DEFAULT_REPO")] = None,
    days: Annotated[int | None, Query(ge=1, le=365)] = None,
    period_start: Annotated[datetime | None, Query()] = None,
    period_end: Annotated[datetime | None, Query()] = None,
) -> Narrative:
    """LLM-synthesized narrative over the numbers.

    Returns a summary, an evidence-backed root-cause hypothesis where the signal
    supports one, and a confidence score reconciled against the data's signal
    strength. `grounded` reports whether every cited figure traces back to the
    source metrics.
    """
    repo_ref = parse_repo(repo or settings.default_repo)
    start, end = resolve_window(days, period_start, period_end)

    # Build providers eagerly so a missing key fails fast (503) before any work.
    # The judge runs on a separate model from the writer (see build_judge_provider).
    provider = build_provider(settings)
    judge_provider = build_judge_provider(settings)
    narrative, cache_hit = await service.get_narrative(
        repo_ref,
        start,
        end,
        lambda health: synthesize_narrative(
            provider, health, judge_provider=judge_provider
        ),
    )
    response.headers["X-Cache"] = "HIT" if cache_hit else "MISS"
    return narrative
