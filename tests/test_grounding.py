"""Tests for the grounding validator — the trust layer.

If the LLM cites a fabricated number, `grounded` must be False and confidence must
be penalised. If the data is thin, confidence must be low regardless of the prose.
Confidence is computed by the system (never the model), so these tests construct a
NarrativeDraft (no confidence field) and assert on the validated Narrative.
"""

from datetime import UTC, datetime

from app.llm.grounding import compute_confidence, signal_strength, validate_grounding
from app.models import (
    ClaimCheck,
    CollaborationHealth,
    ContributorStat,
    EvidenceItem,
    FaithfulnessVerdict,
    NarrativeDraft,
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
    draft = NarrativeDraft(
        summary="Reviews are concentrated on carol.",
        root_cause_hypothesis="carol is a review bottleneck.",
        evidence=[
            EvidenceItem(metric="review_concentration", value=0.75, detail="carol's share"),
            EvidenceItem(metric="total_reviews", value=20, detail="reviews in window"),
        ],
    )
    result = validate_grounding(draft, health)
    assert result.grounded is True
    assert result.grounding_warnings == []
    # strong signal + volume -> confident
    assert result.confidence == signal_strength(health)


def test_fabricated_number_is_flagged_and_confidence_penalised():
    health = _health()
    draft = NarrativeDraft(
        summary="Carol did 99 reviews.",  # 99 is not in the data
        evidence=[EvidenceItem(metric="reviews:carol", value=99, detail="made up")],
    )
    result = validate_grounding(draft, health)
    assert result.grounded is False
    assert result.grounding_warnings
    # ungrounded -> confidence is half of what the data alone would support
    assert result.confidence == round(signal_strength(health) * 0.5, 3)


def test_percent_form_of_ratio_is_accepted():
    health = _health()
    # Model expresses review_concentration 0.75 as "75" (percent) — should still ground.
    draft = NarrativeDraft(
        summary="75% of reviews are carol's.",
        evidence=[EvidenceItem(metric="review_concentration", value=75, detail="percent")],
    )
    result = validate_grounding(draft, health)
    assert result.grounded is True


def test_thin_data_gives_low_confidence():
    # Almost no activity -> signal strength low -> confidence low regardless of prose.
    health = _health(
        total_commits=1, total_prs_merged=1, total_reviews=0,
        review_concentration=0.0, unreviewed_merges=0, review_load=[],
        contributors=[ContributorStat(login="solo", commits=1)],
    )
    draft = NarrativeDraft(
        summary="Tiny window.",
        evidence=[EvidenceItem(metric="total_commits", value=1, detail="one commit")],
    )
    result = validate_grounding(draft, health)
    assert result.confidence <= 0.1
    assert result.confidence == signal_strength(health)


def test_missing_evidence_is_warned():
    health = _health()
    draft = NarrativeDraft(summary="No evidence given.", evidence=[])
    result = validate_grounding(draft, health)
    assert result.grounded is False
    assert any("no evidence" in w.lower() for w in result.grounding_warnings)


def test_unfaithful_prose_lowers_confidence_via_judge():
    # Numbers are clean, but the judge flags an unsupported prose claim. Confidence
    # must drop even though numeric grounding passes — this is the gap numeric
    # grounding alone can't catch (irrelevant/omitted-number claims).
    health = _health()
    draft = NarrativeDraft(
        summary="The team is understaffed and collapsing.",  # cause not in data
        evidence=[EvidenceItem(metric="review_concentration", value=0.75, detail="share")],
    )
    bad_verdict = FaithfulnessVerdict(
        score=2,
        claims=[
            ClaimCheck(
                claim="team is understaffed",
                supported=False,
                reason="staffing not in metrics",
            )
        ],
    )
    result = validate_grounding(draft, health, bad_verdict)
    # numeric check passes, so still "grounded" on numbers...
    assert result.faithfulness == 2
    # ...but confidence is scaled down by the 2/5 faithfulness multiplier
    assert result.confidence == round(signal_strength(health) * (2 / 5), 3)
    assert any("Unsupported prose claim" in w for w in result.grounding_warnings)


def test_compute_confidence_is_independent_of_model():
    # Same data + same checks -> same confidence, no model input anywhere.
    health = _health()
    good = FaithfulnessVerdict(score=5, claims=[])
    assert compute_confidence(health, True, good) == signal_strength(health)
    assert compute_confidence(health, False, good) == round(signal_strength(health) * 0.5, 3)
