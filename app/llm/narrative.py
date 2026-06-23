"""Synthesize a grounded narrative over the collaboration-health numbers.

Flow: serialise the metrics -> ask the LLM for a structured narrative using ONLY
those numbers -> validate the result against the source (grounding.py) -> return.
The prompt and the validator are deliberately separate: the prompt asks for good
behaviour, the validator enforces it.
"""

from __future__ import annotations

import json

from app.llm.grounding import collect_source_values, validate_grounding
from app.llm.judge import judge_faithfulness
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
- Do not assert causes the data can't show (e.g. WHY reviews are slow). Describe what \
the metrics show; only hypothesise a cause a metric directly supports.
- If `truncated` is true, the totals are a LOWER BOUND over only the most recent \
`pulls_analyzed` merged PRs, not the full window. Say so explicitly — do not present \
capped totals as complete.
- Offer a root-cause hypothesis ONLY when a metric supports it (e.g. high \
review_concentration or many unreviewed_merges suggests a review bottleneck). \
If nothing supports a hypothesis, set root_cause_hypothesis to null.

Do NOT report a confidence score — the system computes that separately.

Return strict JSON with this shape:
{
  "summary": "2-4 sentences on what stands out",
  "root_cause_hypothesis": "one sentence, or null",
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
    provider: LLMProvider, health: CollaborationHealth, *, judge: bool = True
) -> Narrative:
    """Draft the narrative, optionally run the faithfulness judge, then validate.

    Two layers of trust: the deterministic grounding check on the evidence numbers
    (always), and the LLM judge on the prose (when `judge` is on). Confidence is then
    computed by the system from data signal + both checks.
    """
    user_prompt = (
        "Here are the collaboration metrics. Write the narrative.\n\n"
        + _metrics_payload(health)
    )
    draft = await provider.draft_narrative(SYSTEM_PROMPT, user_prompt)

    verdict = await judge_faithfulness(provider, draft, health) if judge else None
    return validate_grounding(draft, health, verdict)
