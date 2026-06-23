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

    Deliberately has NO confidence field: an LLM's self-reported confidence is
    uncalibrated, so confidence is computed by the system (see compute_confidence),
    not asked of the model. The model only narrates; the system scores.
    """

    summary: str
    root_cause_hypothesis: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    """One judged claim from the prose summary/hypothesis."""

    claim: str = Field(description="The specific claim, quoted from the narrative.")
    supported: bool = Field(description="Whether the metrics support this claim.")
    reason: str = Field(description="Why it is or isn't supported, in one phrase.")


class FaithfulnessVerdict(BaseModel):
    """An LLM judge's assessment of how faithful the prose is to the metrics.

    The score is a 1–5 Likert rating (discrete scale — LLMs rate far more
    consistently on an anchored rubric than on a raw 0–1 float).
    """

    score: int = Field(ge=1, le=5, description="1=unsupported claims, 5=fully grounded.")
    claims: list[ClaimCheck] = Field(default_factory=list)


class Narrative(NarrativeDraft):
    """A validated narrative: the LLM draft plus system-computed trust metadata.

    `confidence`, `grounded`, and `grounding_warnings` are all set downstream — never
    by the narrating model. Confidence is derived from the data's signal strength,
    numeric grounding, and (when available) the faithfulness judge.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool = True
    grounding_warnings: list[str] = Field(default_factory=list)
    # 1–5 faithfulness score from the LLM judge, if it ran (null if no judge).
    faithfulness: int | None = None
