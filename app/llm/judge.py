"""LLM-as-judge: faithfulness check on the narrative prose.

Deterministic grounding (grounding.py) verifies the *numbers* in the evidence chain —
exact, free, the right tool for figures. But it can't judge the *prose*: a summary can
cite all-correct numbers and still overreach in words ("the team is collapsing",
"reviews are slow because the team is understaffed" — a cause the data doesn't show).

Judging language faithfulness is what an LLM is genuinely good at, so this second
layer hands the judge the prose AND the real metrics and asks it to flag any claim the
data doesn't support, scored on a 1–5 rubric (discrete scales are far more consistent
for LLM raters than a raw 0–1 float).

This layer is optional: it needs an LLM key. Without one, the narrative still returns
with numeric grounding only.
"""

from __future__ import annotations

import json

from app.llm.grounding import collect_source_values
from app.llm.provider import LLMProvider
from app.models import CollaborationHealth, FaithfulnessVerdict, NarrativeDraft

JUDGE_SYSTEM_PROMPT = """\
You are a skeptical fact-checker. You are given (1) a narrative written about a \
GitHub repository's collaboration metrics, and (2) the actual metrics it should be \
based on. Your job is to find claims the metrics do NOT support.

Be adversarial: default to skepticism. A claim is supported ONLY if it follows \
directly from the provided metrics. Treat as UNSUPPORTED: invented causes (e.g. \
"because the team is understaffed" when staffing isn't in the data), exaggerations \
("the team is collapsing"), trends not present in a single-window snapshot, and any \
number that isn't in the metrics.

Score faithfulness on this 1–5 rubric:
  5 = every claim is directly supported by the metrics
  4 = supported, with one minor hedge or soft overreach
  3 = mostly supported, but contains a noticeable unsupported claim
  2 = several unsupported claims, or one clearly wrong number
  1 = central claim is unsupported or fabricated

List each distinct claim you assessed with whether it is supported and why.
"""


def _judge_payload(narrative: NarrativeDraft, health: CollaborationHealth) -> str:
    return json.dumps(
        {
            "narrative": {
                "summary": narrative.summary,
                "root_cause_hypothesis": narrative.root_cause_hypothesis,
            },
            "actual_metrics": collect_source_values(health),
        },
        indent=2,
    )


async def judge_faithfulness(
    provider: LLMProvider, narrative: NarrativeDraft, health: CollaborationHealth
) -> FaithfulnessVerdict:
    """Ask the LLM judge to rate how faithful the prose is to the metrics."""
    user = (
        "Assess the narrative against the actual metrics, then score it.\n\n"
        + _judge_payload(narrative, health)
    )
    return await provider.judge_faithfulness(JUDGE_SYSTEM_PROMPT, user)
