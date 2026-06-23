"""Run the narrative eval suite.

Two modes:
  * default — uses a deterministic stub LLM, so the suite runs offline in CI and
    exercises the grounding/confidence logic without spending tokens.
  * --live  — uses the configured real provider, to check the actual model+prompt
    before you ship a prompt or model change.

Usage:
    python -m eval.run_eval            # offline, deterministic
    python -m eval.run_eval --live     # hits the configured LLM
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.llm.grounding import signal_strength
from app.llm.narrative import synthesize_narrative
from app.llm.provider import build_provider
from app.models import (
    CollaborationHealth,
    EvidenceItem,
    FaithfulnessVerdict,
    Narrative,
    NarrativeDraft,
)
from eval.cases import CASES, EvalCase


class StubProvider:
    """A deterministic 'model' that cites real numbers from the metrics.

    Mimics a well-behaved LLM: surfaces the busiest reviewer when concentration is
    high and proposes a bottleneck hypothesis, citing only numbers from the data.
    The judge stub returns a perfect faithfulness score, so the offline suite
    exercises the data-derived confidence path without spending tokens.
    """

    async def draft_narrative(self, system: str, user: str) -> NarrativeDraft:
        import json

        payload = json.loads(user[user.index("{") :])
        metrics = payload["available_metrics"]
        concentration = metrics.get("review_concentration", 0.0)
        busiest = payload.get("busiest_reviewer")

        evidence = [
            EvidenceItem(
                metric="total_reviews",
                value=metrics.get("total_reviews", 0),
                detail="reviews in window",
            )
        ]
        hypothesis = None
        summary = "Activity summary for the period."
        if concentration >= 0.5 and busiest:
            evidence.append(
                EvidenceItem(
                    metric="review_concentration",
                    value=concentration,
                    detail=f"{busiest}'s share of reviews",
                )
            )
            hypothesis = f"{busiest} is a review bottleneck."
            summary = f"Reviews are concentrated on {busiest}."
        return NarrativeDraft(
            summary=summary,
            root_cause_hypothesis=hypothesis,
            evidence=evidence,
        )

    async def judge_faithfulness(self, system: str, user: str) -> FaithfulnessVerdict:
        # The stub narrative only cites real metrics, so it is faithful by construction.
        return FaithfulnessVerdict(score=5, claims=[])


async def _synthesize(provider, health: CollaborationHealth) -> Narrative:
    return await synthesize_narrative(provider, health)


def _check(case: EvalCase, narrative: Narrative) -> list[str]:
    failures: list[str] = []
    if narrative.grounded != case.expect_grounded:
        failures.append(
            f"grounded={narrative.grounded}, expected {case.expect_grounded} "
            f"(warnings: {narrative.grounding_warnings})"
        )
    if not (case.min_confidence <= narrative.confidence <= case.max_confidence):
        failures.append(
            f"confidence {narrative.confidence} outside "
            f"[{case.min_confidence}, {case.max_confidence}] "
            f"(signal_strength={signal_strength(case.health)})"
        )
    text = f"{narrative.summary} {narrative.root_cause_hypothesis or ''}".lower()
    for kw in case.expect_keywords:
        if kw.lower() not in text:
            failures.append(f"expected keyword '{kw}' not found in narrative")
    if case.expect_hypothesis and not narrative.root_cause_hypothesis:
        failures.append("expected a root-cause hypothesis, got none")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use the configured LLM")
    args = parser.parse_args()

    provider = build_provider(get_settings()) if args.live else StubProvider()
    mode = "LIVE" if args.live else "STUB"
    print(f"Running {len(CASES)} eval cases [{mode}]\n")

    total_failures = 0
    for case in CASES:
        narrative = await _synthesize(provider, case.health)
        failures = _check(case, narrative)
        status = "PASS" if not failures else "FAIL"
        print(f"  [{status}] {case.name}  (confidence={narrative.confidence})")
        for f in failures:
            print(f"        - {f}")
        total_failures += len(failures)

    print()
    if total_failures:
        print(f"❌ {total_failures} assertion(s) failed.")
        return 1
    print("✅ all eval cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
