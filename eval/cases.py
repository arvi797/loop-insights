"""Evaluation cases: fixtures with known properties the narrative must respect.

Each case pairs a CollaborationHealth input with assertions about the narrative we'd
accept. This is the suite you'd run before changing the prompt or swapping the model
(per the assignment's 'eval harness' bonus).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.models import (
    CollaborationHealth,
    ContributorStat,
    ReviewLoad,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, tzinfo=UTC)


@dataclass
class EvalCase:
    name: str
    health: CollaborationHealth
    # The narrative must be grounded (no fabricated figures).
    expect_grounded: bool = True
    # Confidence must fall within these bounds (signal-strength expectation).
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    # Substrings we expect to appear in summary+hypothesis (case-insensitive),
    # used as a soft behavioural check that the model surfaced the key signal.
    expect_keywords: list[str] = field(default_factory=list)
    # A root-cause hypothesis is warranted for this case.
    expect_hypothesis: bool = False


def _health(**kw) -> CollaborationHealth:
    base = dict(
        repo="o/r", period_start=START, period_end=END,
        total_commits=0, total_prs_merged=0, total_reviews=0,
        contributors=[], review_load=[], contributor_count=0, reviewer_count=0,
        review_concentration=0.0, busiest_reviewer=None, unreviewed_merges=0,
    )
    base.update(kw)
    return CollaborationHealth(**base)


CASES: list[EvalCase] = [
    EvalCase(
        name="review_bottleneck",
        health=_health(
            total_commits=80, total_prs_merged=30, total_reviews=40,
            contributors=[
                ContributorStat(login="carol", commits=50, prs_merged=15),
                ContributorStat(login="dave", commits=30, prs_merged=15),
            ],
            review_load=[ReviewLoad(reviewer="carol", reviews=34),
                         ReviewLoad(reviewer="dave", reviews=6)],
            contributor_count=2, reviewer_count=2,
            review_concentration=0.85, busiest_reviewer="carol",
            unreviewed_merges=3,
        ),
        expect_grounded=True,
        min_confidence=0.5,  # strong signal + volume -> should be reasonably confident
        expect_keywords=["carol"],
        expect_hypothesis=True,
    ),
    EvalCase(
        name="healthy_balanced_team",
        health=_health(
            total_commits=60, total_prs_merged=25, total_reviews=30,
            contributors=[
                ContributorStat(login="a", commits=20, prs_merged=8),
                ContributorStat(login="b", commits=20, prs_merged=9),
                ContributorStat(login="c", commits=20, prs_merged=8),
            ],
            review_load=[ReviewLoad(reviewer="a", reviews=11),
                         ReviewLoad(reviewer="b", reviews=10),
                         ReviewLoad(reviewer="c", reviews=9)],
            contributor_count=3, reviewer_count=3,
            review_concentration=0.37, busiest_reviewer="a",
            unreviewed_merges=0,
        ),
        expect_grounded=True,
    ),
    EvalCase(
        name="thin_data_low_confidence",
        health=_health(
            total_commits=2, total_prs_merged=1, total_reviews=0,
            contributors=[ContributorStat(login="solo", commits=2, prs_merged=1)],
            contributor_count=1, reviewer_count=0,
            unreviewed_merges=1,
        ),
        expect_grounded=True,
        max_confidence=0.5,  # not enough data to be confident about anything
    ),
]
