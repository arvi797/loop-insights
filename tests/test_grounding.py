"""Tests for the grounding validator — the trust layer.

If the LLM cites a fabricated number, `grounded` must be False and confidence must
be penalised. If the data is thin, confidence must be capped regardless of what the
model claimed.
"""

from datetime import UTC, datetime

from app.llm.grounding import signal_strength, validate_grounding
from app.models import (
    CollaborationHealth,
    ContributorStat,
    EvidenceItem,
    Narrative,
    ReviewLoad,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, tzinfo=UTC)


def _health(**overrides) -> CollaborationHealth:
    base = dict(
        repo="o/r", period_start=START, period_end=END,
        total_commits=40, total_prs_merged=20, total_reviews=20,
        contributors=[ContributorStat(login="carol", commits=30, prs_merged=10)],
        review_load=[ReviewLoad(reviewer="carol", reviews=15)],
        contributor_count=4, reviewer_count=3,
        review_concentration=0.75, busiest_reviewer="carol",
        unreviewed_merges=2,
    )
    base.update(overrides)
    return CollaborationHealth(**base)


def test_grounded_narrative_passes():
    health = _health()
    narrative = Narrative(
        summary="Reviews are concentrated on carol.",
        root_cause_hypothesis="carol is a review bottleneck.",
        confidence=0.8,
        evidence=[
            EvidenceItem(metric="review_concentration", value=0.75, detail="carol's share"),
            EvidenceItem(metric="total_reviews", value=20, detail="reviews in window"),
        ],
    )
    result = validate_grounding(narrative, health)
    assert result.grounded is True
    assert result.grounding_warnings == []


def test_fabricated_number_is_flagged_and_confidence_penalised():
    health = _health()
    narrative = Narrative(
        summary="Carol did 99 reviews.",  # 99 is not in the data
        confidence=0.9,
        evidence=[EvidenceItem(metric="reviews:carol", value=99, detail="made up")],
    )
    result = validate_grounding(narrative, health)
    assert result.grounded is False
    assert result.grounding_warnings
    # ungrounded -> confidence halved on top of the signal-strength cap
    assert result.confidence < 0.9


def test_percent_form_of_ratio_is_accepted():
    health = _health()
    # Model expresses review_concentration 0.75 as "75" (percent) — should still ground.
    narrative = Narrative(
        summary="75% of reviews are carol's.",
        confidence=0.7,
        evidence=[EvidenceItem(metric="review_concentration", value=75, detail="percent")],
    )
    result = validate_grounding(narrative, health)
    assert result.grounded is True


def test_thin_data_caps_confidence():
    # Almost no activity -> signal strength low -> confidence capped even if model said 0.95.
    health = _health(
        total_commits=1, total_prs_merged=1, total_reviews=0,
        review_concentration=0.0, unreviewed_merges=0, review_load=[],
        contributors=[ContributorStat(login="solo", commits=1)],
    )
    narrative = Narrative(
        summary="Tiny window.", confidence=0.95,
        evidence=[EvidenceItem(metric="total_commits", value=1, detail="one commit")],
    )
    result = validate_grounding(narrative, health)
    assert result.confidence <= signal_strength(health) + 1e-9
    assert result.confidence < 0.95


def test_missing_evidence_is_warned():
    health = _health()
    narrative = Narrative(summary="No evidence given.", confidence=0.6, evidence=[])
    result = validate_grounding(narrative, health)
    assert result.grounded is False
    assert any("no evidence" in w.lower() for w in result.grounding_warnings)
