"""Compute the collaboration-health report from raw GitHub payloads.

Pure functions over already-fetched data: no network here, so the maths is
deterministic and unit-testable against fixtures. This separation is what lets the
eval harness and tests assert "the numbers add up" without hitting GitHub.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime
from typing import Any

from app.models import (
    CollaborationHealth,
    ContributorStat,
    ReviewLoad,
)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # GitHub timestamps look like 2024-01-02T03:04:05Z
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _commit_author_login(commit: dict[str, Any]) -> str | None:
    """Prefer the linked GitHub account; fall back to the git commit author name."""
    author = commit.get("author")
    if isinstance(author, dict) and author.get("login"):
        return author["login"]
    commit_obj = commit.get("commit", {})
    git_author = commit_obj.get("author", {}) if isinstance(commit_obj, dict) else {}
    return git_author.get("name")


def compute_collaboration_health(
    *,
    repo: str,
    period_start: datetime,
    period_end: datetime,
    commits: list[dict[str, Any]],
    pulls_with_reviews: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    top_n: int = 10,
    truncated: bool = False,
) -> CollaborationHealth:
    """Build the report.

    Args:
        commits: raw commit objects already filtered to the window by the API call.
        pulls_with_reviews: (pull, reviews) pairs for PRs merged within the window.
        top_n: cap on how many contributors/reviewers to surface.
        truncated: True if the PR set was capped before covering the full window.
    """
    commit_counts: Counter[str] = Counter()
    for c in commits:
        login = _commit_author_login(c)
        if login:
            commit_counts[login] += 1

    merged_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    hours_to_first_review: list[float] = []
    hours_to_merge: list[float] = []
    unreviewed_merges = 0

    for pull, reviews in pulls_with_reviews:
        author = (pull.get("user") or {}).get("login")
        if author:
            merged_counts[author] += 1

        created = _parse_dt(pull.get("created_at"))
        merged = _parse_dt(pull.get("merged_at"))
        if created and merged:
            hours_to_merge.append((merged - created).total_seconds() / 3600.0)

        # A reviewer is anyone who submitted a review; don't count self-reviews
        # or the author's own approval toward "review load".
        review_times: list[datetime] = []
        saw_external_review = False
        for r in reviews:
            reviewer = (r.get("user") or {}).get("login")
            submitted = _parse_dt(r.get("submitted_at"))
            if reviewer and reviewer != author:
                review_counts[reviewer] += 1
                saw_external_review = True
            if submitted:
                review_times.append(submitted)

        if not saw_external_review:
            unreviewed_merges += 1

        if created and review_times:
            first = min(review_times)
            if first >= created:
                hours_to_first_review.append((first - created).total_seconds() / 3600.0)

    total_reviews = sum(review_counts.values())
    busiest_reviewer, busiest_count = (
        review_counts.most_common(1)[0] if review_counts else (None, 0)
    )
    review_concentration = (
        round(busiest_count / total_reviews, 3) if total_reviews else 0.0
    )

    return CollaborationHealth(
        repo=repo,
        period_start=period_start,
        period_end=period_end,
        total_commits=sum(commit_counts.values()),
        total_prs_merged=sum(merged_counts.values()),
        total_reviews=total_reviews,
        contributors=_top_contributors(
            commit_counts, merged_counts, review_counts, top_n
        ),
        review_load=[
            ReviewLoad(reviewer=login, reviews=n)
            for login, n in review_counts.most_common(top_n)
        ],
        contributor_count=len(set(commit_counts) | set(merged_counts)),
        reviewer_count=len(review_counts),
        review_concentration=review_concentration,
        busiest_reviewer=busiest_reviewer,
        median_hours_to_first_review=_median(hours_to_first_review),
        median_hours_to_merge=_median(hours_to_merge),
        unreviewed_merges=unreviewed_merges,
        pulls_analyzed=len(pulls_with_reviews),
        truncated=truncated,
    )


def _top_contributors(
    commit_counts: Counter[str],
    merged_counts: Counter[str],
    review_counts: Counter[str],
    top_n: int,
) -> list[ContributorStat]:
    logins = set(commit_counts) | set(merged_counts) | set(review_counts)
    stats = [
        ContributorStat(
            login=login,
            commits=commit_counts.get(login, 0),
            prs_merged=merged_counts.get(login, 0),
            reviews_submitted=review_counts.get(login, 0),
        )
        for login in logins
    ]
    # Rank by overall involvement, commits as the primary signal.
    stats.sort(key=lambda s: (s.commits, s.prs_merged, s.reviews_submitted), reverse=True)
    return stats[:top_n]


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None
