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
You are a skeptical fact-checker. You are given a narrative about a GitHub repo's \
collaboration metrics, the narrative's `evidence` pairings (metric, value, detail), \
and the `actual_metrics`. Your job is to find claims the metrics do NOT support.

Check the evidence pairings too, not just the prose. For each evidence item, the \
`value` must match the named `metric` in actual_metrics, AND the `detail` text must \
describe what that metric actually is. Flag "right number, wrong label/conclusion": \
e.g. a value that is really someone's review count but is labelled or described as \
"median hours to merge". This mislabelling is a failure even though the number is real.

Be adversarial about FACTS, but fair about HYPOTHESES.

A factual statement (a number, a count, "X did Y reviews") is SUPPORTED only if it \
matches the metrics exactly. Mark a factual statement UNSUPPORTED if it cites a \
number not in the data, states a trend a single snapshot can't show, or asserts a \
cause the data has no field for (e.g. "because the team is understaffed" — staffing \
isn't measured; "the team is collapsing" — an exaggeration).

A hypothesis is DIFFERENT and is explicitly wanted. A statement that is (a) hedged as \
a possibility ("may", "suggests", "likely") AND (b) tied to a metric that plausibly \
implies it (e.g. "high unreviewed_merges may indicate a review bottleneck") is \
SUPPORTED — do NOT flag it merely for being an interpretation. Only flag a hypothesis \
if it rests on a metric that isn't present or that doesn't plausibly relate to it.

Score faithfulness on this 1–5 rubric:
  5 = all facts correct; any hypothesis is hedged and metric-grounded
  4 = facts correct, with one soft overreach or weakly-linked hypothesis
  3 = contains one unsupported factual claim or an ungrounded hypothesis
  2 = several unsupported claims, or one clearly wrong number
  1 = a central factual claim is fabricated or contradicts the metrics

List each distinct claim you assessed, whether it is supported, and why.
"""


def _judge_payload(narrative: NarrativeDraft, health: CollaborationHealth) -> str:
    return json.dumps(
        {
            "narrative": {
                "summary": narrative.summary,
                "root_cause_hypothesis": narrative.root_cause_hypothesis,
                # The evidence pairings are included so the judge can catch a claim that
                # cites a real number under the wrong label or draws the wrong
                # conclusion from it ("right number, wrong story") — not just prose that
                # invents facts. Numeric grounding checks the value; the judge checks
                # that the value actually supports what the detail text claims.
                "evidence": [e.model_dump() for e in narrative.evidence],
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
