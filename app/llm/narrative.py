"""Synthesize a grounded narrative over the collaboration-health numbers.

Flow: serialise the metrics -> ask the LLM for a structured narrative using ONLY
those numbers -> validate the result against the source (grounding.py) -> return.
The prompt and the validator are deliberately separate: the prompt asks for good
behaviour, the validator enforces it.
"""

from __future__ import annotations

import json

from app.llm.grounding import collect_source_values, validate_grounding
from app.llm.provider import LLMProvider
from app.models import CollaborationHealth, Narrative

SYSTEM_PROMPT = """\
You are a senior engineering-operations analyst. You are given a JSON object of \
collaboration metrics computed from a GitHub repository over a time period.

Write a short, factual narrative about what is interesting or unusual in the data.

Hard rules:
- Use ONLY numbers present in the provided metrics. Never invent figures, names, or \
trends that are not in the data.
- Every number you mention in `evidence` must be copied exactly from the metrics.
- The main quantitative claim in your `summary` MUST also appear as an item in \
`evidence`. Don't headline a number you don't back up.
- If the data is thin or the signal is weak, say so and lower your confidence.
- If `truncated` is true, the totals are a LOWER BOUND over only the most recent \
`pulls_analyzed` merged PRs, not the full window. Say so explicitly and lower your \
confidence accordingly — do not present capped totals as complete.
- Offer a root-cause hypothesis ONLY when a metric supports it (e.g. high \
review_concentration or many unreviewed_merges suggests a review bottleneck). \
If nothing supports a hypothesis, set root_cause_hypothesis to null.
- `confidence` (0..1) must reflect how strongly the data supports your claims, not \
how confident you sound.

Return strict JSON with this shape:
{
  "summary": "2-4 sentences on what stands out",
  "root_cause_hypothesis": "one sentence, or null",
  "confidence": 0.0,
  "evidence": [{"metric": "metric_name", "value": <number from data>, "detail": "what it shows"}]
}
"""


def _metrics_payload(health: CollaborationHealth) -> str:
    """The exact numbers the model may cite — also what grounding validates against."""
    payload = {
        "repo": health.repo,
        "period_start": health.period_start.isoformat(),
        "period_end": health.period_end.isoformat(),
        "available_metrics": collect_source_values(health),
        "top_contributors": [c.model_dump() for c in health.contributors],
        "review_load": [r.model_dump() for r in health.review_load],
        "busiest_reviewer": health.busiest_reviewer,
        "coverage": {
            "pulls_analyzed": health.pulls_analyzed,
            "truncated": health.truncated,
        },
    }
    return json.dumps(payload, indent=2)


async def synthesize_narrative(
    provider: LLMProvider, health: CollaborationHealth
) -> Narrative:
    user_prompt = (
        "Here are the collaboration metrics. Write the narrative.\n\n"
        + _metrics_payload(health)
    )
    # The provider returns a schema-enforced draft (structured output). The grounding
    # validator then promotes it to a Narrative with verified grounding metadata.
    draft = await provider.draft_narrative(SYSTEM_PROMPT, user_prompt)
    return validate_grounding(draft, health)
