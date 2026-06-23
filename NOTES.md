# Submission notes

## 1. How to run it locally

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env            # runs as-is; add OPENAI_API_KEY for the narrative endpoint
uvicorn app.api.main:app --reload
```

- No GitHub token required — it falls back to unauthenticated access (60 req/hr).
  Add a read-only `GITHUB_TOKEN` to raise the limit.
- The narrative endpoint needs `OPENAI_API_KEY` (or `GOOGLE_API_KEY` +
  `LLM_PROVIDER=gemini`). Without one, that endpoint returns `503`; the
  collaboration endpoint still works.
- `pytest` for tests; `python -m eval.run_eval` for the narrative eval suite.
- Docker: `docker compose up --build`.

## 2. Architecture & main decisions

The code is layered so the domain logic is testable without the network or an LLM:

```
api/        thin HTTP layer (FastAPI): validate input, call service, serialise
service     orchestration: GitHub fetch -> metrics, caching, input validation
github/     async REST client (pagination, rate-limit handling, retries)
metrics/    pure functions over raw payloads -> the report (deterministic, unit-tested)
llm/        provider seam (OpenAI/Gemini) + narrative writer + judge + grounding
store/      SQLite cache keyed by (repo, window) with a TTL
eval/       offline + live evaluation suite for the narrative
```

**The decision I care most about is how the narrative is made trustworthy.** An LLM
narrative over numbers is only useful if you can trust it, so the model's output is
never taken at face value — and the model is never asked how confident it is, because
self-reported LLM confidence is uncalibrated. Instead there are **two grounding
layers plus a computed confidence** (`llm/grounding.py`, `llm/judge.py`):

1. **Numeric grounding (deterministic).** Every number the model cites in `evidence`
   must match a source metric exactly. A fabricated figure sets `grounded=false` and
   is reported in `grounding_warnings`. This is exact, free, and unit-tested — the
   right tool for verifying *numbers*.
2. **Faithfulness judge (LLM-as-judge).** Numeric grounding is blind to prose that
   cites real numbers but overreaches — e.g. "reviews are slow *because the team is
   understaffed*" (staffing isn't in the data). A second LLM reads the prose plus the
   real metrics and scores faithfulness 1–5, flagging unsupported claims. It runs on a
   **different model from the writer** (`gpt-4.1` judging `gpt-5.5` by default) so it
   isn't grading its own output. A discrete 1–5 rubric is used deliberately — LLMs
   rate far more consistently on an anchored scale than on a raw 0–1 float.
3. **Computed confidence.** `confidence` is derived by the system from the data's
   signal strength (volume × how clear the bottleneck signal is), then penalised for a
   fabricated number (0.5×) and scaled by the judge's faithfulness (score/5). So a
   model can't sound 95%-sure about a 3-PR window, and a fluent-but-unfounded story
   can't score high. The prompt *asks* for good behaviour; the two layers *enforce* it.

**Other decisions.** Metrics are pure functions over already-fetched payloads, which
is what lets the tests and eval suite assert the numbers add up without hitting
GitHub. Reviews are fetched concurrently with a bounded semaphore. The SQLite cache
persists computed reports (and their narratives) so repeat queries for the same
window don't re-hit the upstream API or re-invoke the LLM — the `X-Cache` header makes
hits visible. Input is validated at the service boundary: the `repo` value is matched
against an anchored regex so a crafted value can't become a path-traversal / SSRF
vector, and the window is bounded.

**LLM provider & structured output.** OpenAI and Gemini sit behind one `LLMProvider`
seam (`draft_narrative` + `judge_faithfulness`), both asked for structured output
using the Pydantic model directly — OpenAI via `chat.completions.parse`, Gemini via
`response_schema` — so there's no hand-written JSON schema to drift. The writer's
output schema (`NarrativeDraft`, no confidence field) is deliberately separate from
the validated `Narrative`, so the model is never asked for its own grounding verdict
or confidence; the system sets those. Writer defaults to `gpt-5.5` (best
instruction-following), judge to `gpt-4.1` (`JUDGE_MODEL`) — distinct models so the
judge isn't self-grading; point `JUDGE_MODEL` at a non-OpenAI model for a fully
cross-family check. Reasoning models pin `temperature`, so the provider omits the
parameter for the `gpt-5.x` / o-series families rather than sending a value they'd
reject. Both narrative endpoints fail fast with `503` if their model's key is absent.

**Coverage / truncation.** Upstream fan-out is bounded by `MAX_PULLS_INSPECTED`. When
the window holds more PRs than the cap, the report sets `truncated=true` and reports
`pulls_analyzed`; the narrative prompt is told to treat the totals as a lower bound
and lower its confidence. This avoids the trap of presenting a capped count as a
complete one — the kind of silent truncation that quietly corrupts a metric.

## 3. What I'd do next with another day

- **Background sync worker.** Today the first query for a cold window fans out into
  many GitHub calls. I'd add a worker that periodically pulls PRs/reviews into the
  SQLite store and serve all queries from there, with the on-demand path as fallback.
  The store schema is already the natural seam for this.
- **GraphQL for the upstream fetch.** GitHub's REST API needs one call per PR's
  reviews; a single GraphQL query could pull PRs + reviews together and cut the
  fan-out dramatically. I stayed on REST for clarity under time pressure.
- **ETag / conditional requests** against GitHub to avoid spending rate limit on
  unchanged data (the client is structured to add this).
- **A second integration (GitLab)** behind the same metrics interface, to prove the
  boundary. The `RepoRef` → report flow is provider-agnostic by design.
- **A small React page** with a date-range picker and the two tables.
- **Calibrate the confidence weights** (the ÷50 volume scale, the 50/50 split) against
  labelled outcomes; today they're transparent, defensible heuristics, not fitted.
- **A judge eval set** — a handful of deliberately-overreaching narratives with known
  verdicts, run against the judge to catch prompt regressions (the judge is itself an
  LLM, so it deserves its own regression net).

## 4. What I used AI for

I built this with an AI coding assistant (Claude Code), the way the brief encourages.
I used it to scaffold the layout, draft the FastAPI boilerplate and the async GitHub
client, and accelerate the tests. I directed the design decisions — the grounding /
confidence-reconciliation approach, the choice of collaboration-health metrics, the
layering — and reviewed everything on a re-read. Two bugs I caught and fixed that way
are worth calling out because they're the kind of thing that slips through:

1. An **in-memory SQLite** cache opened a fresh connection per call, so the table
   vanished between operations — the live smoke test surfaced it; fixed by holding one
   shared connection for `:memory:`.
2. The **signal-strength** score initially let a 1-of-1 unreviewed merge read as a
   100% bottleneck signal — the eval suite caught it; fixed by gating the bottleneck
   term by data volume, so a ratio over tiny N can't masquerade as a strong signal.
3. The **faithfulness judge** initially flagged legitimate, metric-grounded
   hypotheses as "unsupported" — penalising the narrative for producing the very
   root-cause hypothesis the task asks for. A live run across several repos surfaced
   the inconsistency (equivalent hypotheses scored 5/5 on one repo, flagged on
   another); fixed by teaching the judge prompt to distinguish a *fabricated fact or
   cause* (flag) from a *hedged hypothesis tied to a real metric* (allow), then
   re-verifying it still catches genuine overreach.

All three are in the git history. Catching #3 only by actually running the pipeline
end-to-end — not by reading the code — is the reason the live eval matters.

## Things I deliberately did *not* do

- **No background sync / no real DB beyond SQLite** — on-demand fetch + a TTL cache is
  enough to demonstrate the caching story at this scope; the design note above is how
  I'd extend it.
- **No frontend** — chose to spend the time on the grounding/eval layer instead, since
  that's where the interesting judgement is.
- **No auth on the service itself** — it's a local, read-only analytics service over
  public data; adding API-key auth would be the first step before any real deployment.

## A note on verification

The endpoints, tests, and eval suite were run live end-to-end against the real GitHub
API with the real models (writer `gpt-5.5`, judge `gpt-4.1`) across several repos of
different shapes — healthy, bottlenecked, truncated, and empty — to confirm the
confidence scores track the data and the judge flags overreach without flagging valid
hypotheses. The README example is genuine output, not illustrative. The **Docker path
was verified too**: the image builds, `docker compose up` boots the service, the
container healthcheck reports `healthy`, and the real collaboration endpoint serves
data from inside the container. The same build + `/health` smoke test runs in CI on
every push (see below).

## CI

`.github/workflows/ci.yml` runs on every push / PR to `main`:

1. **Lint + test + eval** — `ruff`, the full `pytest` suite, and the eval harness in
   offline stub mode (no API keys needed — the stub provider stands in for the LLM).
2. **Docker build + smoke test** — builds the image and boots the container with *no*
   secrets, then polls `/health`. This doubles as a guarantee that the liveness path
   never grows a hidden dependency on a key.

The narrative endpoint's behaviour against a *real* model is checked locally with
`python -m eval.run_eval --live` rather than in CI, to keep API keys out of the
pipeline.
