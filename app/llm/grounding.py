"""Grounding validation for the LLM narrative.

The narrative endpoint's whole value rests on trust: a root-cause story is only
useful if its numbers are real and its prose doesn't overreach. We never ask the
model how confident it is (that signal is uncalibrated). Instead:

  * numeric grounding — every number cited in the evidence chain must match a source
    metric (deterministic, exact);
  * the confidence score is COMPUTED from the data's signal strength, then penalised
    for a fabricated number or for unfaithful prose (the latter via the LLM judge's
    1–5 faithfulness score — see judge.py).

Numeric grounding alone is blind to omission (citing no number) and irrelevance (a
real number that doesn't support the claim); the judge and the data-derived
confidence are what cover those gaps. This is the piece that turns "an LLM call" into
a grounded, auditable insight.
"""

from __future__ import annotations

from app.models import (
    CollaborationHealth,
    FaithfulnessVerdict,
    Narrative,
    NarrativeDraft,
)


def collect_source_values(health: CollaborationHealth) -> dict[str, float]:
    """Flatten the metrics into the set of numbers the narrative is allowed to cite."""
    values: dict[str, float] = {
        "total_commits": health.total_commits,
        "total_prs_merged": health.total_prs_merged,
        "total_reviews": health.total_reviews,
        "contributor_count": health.contributor_count,
        "reviewer_count": health.reviewer_count,
        "review_concentration": health.review_concentration,
        "unreviewed_merges": health.unreviewed_merges,
        # Coverage figures are citable too — the narrative is expected to reference
        # them when the data is truncated. (bool is coerced to 0/1 for matching.)
        "pulls_analyzed": health.pulls_analyzed,
        "truncated": float(health.truncated),
    }
    if health.median_hours_to_first_review is not None:
        values["median_hours_to_first_review"] = health.median_hours_to_first_review
    if health.median_hours_to_merge is not None:
        values["median_hours_to_merge"] = health.median_hours_to_merge
    for c in health.contributors:
        values[f"commits:{c.login}"] = c.commits
        values[f"prs_merged:{c.login}"] = c.prs_merged
    for r in health.review_load:
        values[f"reviews:{r.reviewer}"] = r.reviews
    return values


def _matches_any(value: float, source: dict[str, float], tol: float = 0.01) -> bool:
    """A cited numeric value is grounded if it equals some source figure.

    Tolerance covers float rounding and the model expressing a ratio as a percent
    (e.g. review_concentration 0.62 cited as 62)."""
    candidates = set(source.values())
    for v in list(candidates):
        candidates.add(round(v * 100, 2))  # ratio -> percent
    return any(abs(value - cand) <= tol for cand in candidates)


def validate_grounding(
    draft: NarrativeDraft,
    health: CollaborationHealth,
    faithfulness: FaithfulnessVerdict | None = None,
) -> Narrative:
    """Promote an LLM draft to a validated Narrative.

    Checks each evidence number against the source metrics, folds in the optional
    prose-faithfulness verdict, and computes a confidence the system stands behind —
    the narrating model is never asked for its own confidence.
    """
    source = collect_source_values(health)

    # Track numeric-grounding failures explicitly (not by re-parsing warning text), so
    # the confidence logic can't silently break if a warning message is reworded.
    numeric_warnings: list[str] = []
    for item in draft.evidence:
        numeric = _as_number(item.value)
        if numeric is None:
            continue  # qualitative evidence — nothing numeric to verify
        if not _matches_any(numeric, source):
            numeric_warnings.append(
                f"Evidence cites {item.metric}={item.value}, which does not match "
                "any value in the source metrics."
            )
    if not draft.evidence:
        numeric_warnings.append("Narrative provided no evidence chain.")

    # The judge's unsupported-prose findings are warnings too, but they don't flip the
    # numeric-grounding flag — they feed confidence via the faithfulness score instead.
    prose_warnings: list[str] = []
    if faithfulness is not None:
        prose_warnings = [
            f"Unsupported prose claim: {c.claim} ({c.reason})"
            for c in faithfulness.claims
            if not c.supported
        ]

    warnings = numeric_warnings + prose_warnings
    numerically_grounded = not numeric_warnings
    confidence = compute_confidence(health, numerically_grounded, faithfulness)

    return Narrative(
        summary=draft.summary,
        root_cause_hypothesis=draft.root_cause_hypothesis,
        confidence=confidence,
        evidence=draft.evidence,
        grounded=len(warnings) == 0,
        grounding_warnings=warnings,
        faithfulness=faithfulness.score if faithfulness else None,
    )


def signal_strength(health: CollaborationHealth) -> float:
    """A deterministic 0..1 estimate of how much the data actually supports a claim.

    Confidence shouldn't outrun the evidence. Thin windows (few PRs/commits) or a
    flat review distribution can't support a strong root-cause story, regardless of
    how assertive the model sounds.
    """
    volume = min(1.0, (health.total_commits + health.total_prs_merged) / 50.0)
    # A clear bottleneck signal (high concentration) or unreviewed merges strengthen
    # the case for a root-cause narrative...
    bottleneck = max(
        health.review_concentration,
        min(1.0, health.unreviewed_merges / max(health.total_prs_merged, 1)),
    )
    # ...but only insofar as there's enough data to observe it. A 1-of-1 unreviewed
    # merge looks like 100% concentration yet proves nothing — gate by volume so a
    # ratio over tiny N can't masquerade as a strong signal.
    bottleneck *= volume
    return round(0.5 * volume + 0.5 * bottleneck, 3)


def compute_confidence(
    health: CollaborationHealth,
    numerically_grounded: bool,
    faithfulness: FaithfulnessVerdict | None = None,
) -> float:
    """Confidence the SYSTEM stands behind — never the model's self-report.

    Starts from how strongly the data supports a claim (signal_strength), then applies
    penalties for the two ways a narrative can be untrustworthy:
      * a fabricated number in the evidence chain (hard 0.5x penalty), and
      * unfaithful prose, scaled by the judge's 1–5 score (5 -> 1.0x ... 1 -> 0.2x).
    An LLM's own confidence is uncalibrated, so it is deliberately not an input.
    """
    confidence = signal_strength(health)
    if not numerically_grounded:
        confidence *= 0.5
    if faithfulness is not None:
        confidence *= faithfulness.score / 5.0  # 1–5 rubric -> 0.2 .. 1.0 multiplier
    return round(confidence, 3)


def _as_number(value: float | int | str) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None
