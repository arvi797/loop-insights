"""Domain models shared across the service.

These are the contract: the metrics layer produces them, the API serialises them,
and the LLM narrative is grounded strictly against them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RepoRef(BaseModel):
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class ContributorStat(BaseModel):
    """Per-author activity within the analysis window."""

    login: str
    commits: int = 0
    prs_merged: int = 0
    reviews_submitted: int = 0


class ReviewLoad(BaseModel):
    """How review work is distributed across the team."""

    reviewer: str
    reviews: int


class CollaborationHealth(BaseModel):
    """The primary insight: a collaboration-health report over a period.

    Every field here is a number a reviewer can verify against GitHub, and every
    figure the LLM is allowed to cite must trace back to this object.
    """

    repo: str
    period_start: datetime
    period_end: datetime

    total_commits: int
    total_prs_merged: int
    total_reviews: int

    contributors: list[ContributorStat] = Field(default_factory=list)
    review_load: list[ReviewLoad] = Field(default_factory=list)

    # Derived collaboration signals
    contributor_count: int = 0
    reviewer_count: int = 0
    # Share of all reviews done by the single busiest reviewer (0..1).
    # A high value is the classic review-bottleneck / bus-factor signal.
    review_concentration: float = 0.0
    busiest_reviewer: str | None = None

    # PR flow timings (hours), median over merged PRs in the window.
    median_hours_to_first_review: float | None = None
    median_hours_to_merge: float | None = None

    # Number of merged PRs that received no review at all.
    unreviewed_merges: int = 0

    # Coverage: how many merged PRs the report is based on, and whether the window
    # held more than the per-request cap. When `truncated` is true the totals are a
    # lower bound over the most recent `pulls_analyzed` PRs, not the full window —
    # surfaced so neither the caller nor the LLM mistakes a partial count for total.
    pulls_analyzed: int = 0
    truncated: bool = False


class EvidenceItem(BaseModel):
    """One link in the evidence chain: a claim tied to a specific metric value."""

    metric: str = Field(description="Name of the metric this claim rests on.")
    value: float | int | str = Field(description="The value as found in the data.")
    detail: str = Field(description="What this number shows, in one phrase.")


class NarrativeDraft(BaseModel):
    """The narrative as produced by the LLM — exactly the fields the model fills.

    This is what we hand the provider as its output schema. Keeping it separate from
    Narrative means the model is never asked to self-report its own grounding verdict;
    that is set downstream by the validator.
    """

    summary: str
    root_cause_hypothesis: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class Narrative(NarrativeDraft):
    """A validated narrative: the LLM draft plus grounding metadata.

    `grounded` and `grounding_warnings` are set by the grounding validator, not the
    model — they record whether every cited figure traces back to the source metrics.
    """

    grounded: bool = True
    grounding_warnings: list[str] = Field(default_factory=list)
