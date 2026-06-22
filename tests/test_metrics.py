"""Metric-correctness tests — the 'numbers add up' guarantee.

These run against hand-built payloads with known answers, so a regression in the
counting logic fails loudly without touching the network.
"""

from datetime import UTC, datetime

from app.metrics.collaboration import compute_collaboration_health


def _dt(day: int, hour: int = 0) -> str:
    return datetime(2024, 1, day, hour, tzinfo=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _commit(login: str) -> dict:
    return {"author": {"login": login}, "commit": {"author": {"name": login}}}


def _pull(number: int, author: str, created_day: int, merged_day: int) -> dict:
    return {
        "number": number,
        "user": {"login": author},
        "created_at": _dt(created_day),
        "merged_at": _dt(merged_day),
        "updated_at": _dt(merged_day),
    }


def _review(reviewer: str, day: int, hour: int = 0) -> dict:
    return {"user": {"login": reviewer}, "submitted_at": _dt(day, hour)}


START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, tzinfo=UTC)


def test_basic_counts_add_up():
    commits = [_commit("alice"), _commit("alice"), _commit("bob")]
    pulls = [
        (_pull(1, "alice", 2, 3), [_review("bob", 2, 12)]),
        (_pull(2, "bob", 4, 6), [_review("alice", 5)]),
    ]
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=commits, pulls_with_reviews=pulls,
    )
    assert health.total_commits == 3
    assert health.total_prs_merged == 2
    assert health.total_reviews == 2
    assert health.contributor_count == 2  # alice, bob
    assert health.reviewer_count == 2

    by_login = {c.login: c for c in health.contributors}
    assert by_login["alice"].commits == 2
    assert by_login["alice"].prs_merged == 1
    assert by_login["bob"].reviews_submitted == 1


def test_review_concentration_flags_bottleneck():
    # Carol does 3 of 4 reviews -> concentration 0.75.
    pulls = [
        (_pull(1, "alice", 2, 3), [_review("carol", 2)]),
        (_pull(2, "alice", 2, 3), [_review("carol", 2)]),
        (_pull(3, "bob", 2, 3), [_review("carol", 2)]),
        (_pull(4, "bob", 2, 3), [_review("dave", 2)]),
    ]
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=pulls,
    )
    assert health.busiest_reviewer == "carol"
    assert health.review_concentration == 0.75


def test_self_review_not_counted_and_unreviewed_merges():
    # PR author approves own PR -> not a review; PR with no reviews -> unreviewed.
    pulls = [
        (_pull(1, "alice", 2, 3), [_review("alice", 2)]),  # self-review only
        (_pull(2, "bob", 2, 3), []),                        # no reviews
    ]
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=pulls,
    )
    assert health.total_reviews == 0
    assert health.unreviewed_merges == 2


def test_timing_medians():
    pulls = [
        (_pull(1, "alice", 1, 2), [_review("bob", 1, 12)]),  # 12h to first review, 24h to merge
        (_pull(2, "alice", 1, 3), [_review("bob", 2, 0)]),   # 24h to first review, 48h to merge
    ]
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=pulls,
    )
    assert health.median_hours_to_first_review == 18.0  # median(12, 24)
    assert health.median_hours_to_merge == 36.0          # median(24, 48)


def test_pulls_analyzed_and_truncation_flag():
    pulls = [
        (_pull(1, "alice", 2, 3), [_review("bob", 2)]),
        (_pull(2, "bob", 2, 3), [_review("alice", 2)]),
    ]
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=pulls, truncated=True,
    )
    assert health.pulls_analyzed == 2
    assert health.truncated is True

    # Default: not truncated.
    health2 = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=pulls,
    )
    assert health2.truncated is False


def test_empty_window_is_safe():
    health = compute_collaboration_health(
        repo="o/r", period_start=START, period_end=END,
        commits=[], pulls_with_reviews=[],
    )
    assert health.total_commits == 0
    assert health.review_concentration == 0.0
    assert health.busiest_reviewer is None
    assert health.median_hours_to_merge is None
