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
llm/        provider abstraction (OpenAI/Gemini) + narrative + grounding validator
store/      SQLite cache keyed by (repo, window) with a TTL
eval/       offline + live evaluation suite for the narrative
```

**The decision I care most about is how the narrative is grounded.** An LLM
narrative over numbers is only useful if you can trust it, so the model's output is
never taken at face value. The flow is: compute the metrics → hand the LLM *only*
those numbers and ask for structured JSON → **validate every cited figure against
the source metrics** (`llm/grounding.py`). If the model invents a number, `grounded`
is set to `false`, the discrepancy is reported in `grounding_warnings`, and
confidence is penalised. Separately, the model's self-reported confidence is **capped
by a deterministic signal-strength score** derived from the data's volume and how
clear the bottleneck signal is — so a model can't sound 95%-sure about a 3-PR window.
The prompt *asks* for good behaviour; the validator *enforces* it. That separation is
the point.

**Other decisions.** Metrics are pure functions over already-fetched payloads, which
is what lets the tests and eval suite assert the numbers add up without hitting
GitHub. Reviews are fetched concurrently with a bounded semaphore. The SQLite cache
persists computed reports (and their narratives) so repeat queries for the same
window don't re-hit the upstream API or re-invoke the LLM — the `X-Cache` header makes
hits visible. Input is validated at the service boundary: the `repo` value is matched
against an anchored regex so a crafted value can't become a path-traversal / SSRF
vector, and the window is bounded.

**LLM provider & structured output.** Both OpenAI and Gemini are driven through a
one-method `LLMProvider` seam and asked for structured output using the Pydantic
model directly (`NarrativeDraft`) — OpenAI via `chat.completions.parse`, Gemini via
`response_schema` — so there's no hand-written JSON schema to drift. The model output
schema (`NarrativeDraft`) is deliberately separate from the validated `Narrative`, so
the LLM is never asked to self-report its own grounding verdict; the validator sets
that. The default model is `gpt-5.5` (best instruction-following for the grounding +
calibration task); it's one env var to switch to `gpt-4.1` if latency matters.
Reasoning models pin `temperature`, so the provider omits the parameter for the
`gpt-5.x` / o-series families rather than sending a value they'd reject.

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
- **Richer eval**: a faithfulness check using an LLM-as-judge on the prose summary,
  not just the numeric evidence chain.

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

Both are in the git history. The eval suite catching the second one is exactly why
it's in the repo.

## Things I deliberately did *not* do

- **No background sync / no real DB beyond SQLite** — on-demand fetch + a TTL cache is
  enough to demonstrate the caching story at this scope; the design note above is how
  I'd extend it.
- **No frontend** — chose to spend the time on the grounding/eval layer instead, since
  that's where the interesting judgement is.
- **No auth on the service itself** — it's a local, read-only analytics service over
  public data; adding API-key auth would be the first step before any real deployment.

## A note on verification

The endpoints, tests, and eval suite were run live against the real GitHub API and the
real OpenAI model (`gpt-5.5`) — the example output in the README is genuine, not
illustrative. The **Docker path was verified too**: the image builds, `docker compose
up` boots the service, the container healthcheck reports `healthy`, and the real
collaboration endpoint serves data from inside the container. The same build +
`/health` smoke test runs in CI on every push (see below).

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
