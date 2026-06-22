"""Grounding validation for the LLM narrative.

The narrative endpoint's whole value rests on trust: a root-cause story is only
useful if its numbers are real. Rather than trust the model's self-reported
confidence, we independently verify the output against the source metrics and
reconcile confidence with a deterministic signal-strength score.

This is the piece that turns "an LLM call" into "a grounded, auditable insight".
"""

from __future__ import annotations

from app.models import CollaborationHealth, Narrative, NarrativeDraft


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
    draft: NarrativeDraft, health: CollaborationHealth
) -> Narrative:
    """Promote an LLM draft to a validated Narrative.

    Checks each evidence item against the source metrics, flags fabrications, and
    reconciles the draft's confidence with the data's actual signal strength.
    """
    source = collect_source_values(health)
    warnings: list[str] = []

    for item in draft.evidence:
        numeric = _as_number(item.value)
        if numeric is None:
            continue  # qualitative evidence — nothing numeric to verify
        if not _matches_any(numeric, source):
            warnings.append(
                f"Evidence cites {item.metric}={item.value}, which does not match "
                "any value in the source metrics."
            )

    if not draft.evidence:
        warnings.append("Narrative provided no evidence chain.")

    grounded = len(warnings) == 0
    confidence = _reconcile_confidence(draft.confidence, health, grounded)

    return Narrative(
        summary=draft.summary,
        root_cause_hypothesis=draft.root_cause_hypothesis,
        confidence=confidence,
        evidence=draft.evidence,
        grounded=grounded,
        grounding_warnings=warnings,
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


def _reconcile_confidence(
    model_confidence: float, health: CollaborationHealth, grounded: bool
) -> float:
    """Cap the model's confidence at what the evidence supports; penalise if ungrounded.

    No artificial floor: if the data is thin (low signal strength), low confidence is
    the honest answer, even if the model sounded sure.
    """
    ceiling = signal_strength(health)
    reconciled = min(model_confidence, ceiling)
    if not grounded:
        reconciled *= 0.5
    return round(reconciled, 3)


def _as_number(value: float | int | str) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None
