# Loop Insights

[![CI](https://github.com/arvi797/loop-insights/actions/workflows/ci.yml/badge.svg)](https://github.com/arvi797/loop-insights/actions/workflows/ci.yml)

A small service that reads collaboration data from the **GitHub API** and surfaces
insights over a time period, including an **LLM-synthesized narrative** with a
root-cause hypothesis, a confidence score, and an evidence chain that traces every
claim back to the underlying numbers.

What makes the narrative trustworthy rather than just plausible is a **two-layer
grounding** check: a deterministic pass that verifies every cited number against the
source metrics, and an **LLM-as-judge** (running on a *different* model from the
writer) that checks the prose doesn't overreach. The confidence score is then
**computed by the service** from the data — never self-reported by the model.

- `GET /v1/insights/collaboration` — the numbers (top contributors, review-load
  distribution, review concentration, PR timings).
- `GET /v1/insights/narrative` — a grounded narrative over those numbers.

---

## 60-second quickstart

Requires Python 3.11+ (3.12 recommended). No GitHub token is needed to try it —
the service falls back to unauthenticated access (60 req/hr).

```bash
# 1. Install (uv — recommended)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
#    ...or with plain pip:
#    python -m venv .venv && source .venv/bin/activate
#    pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # works as-is for the collaboration endpoint
                              # add OPENAI_API_KEY to enable the narrative endpoint

# 3. Run
uvicorn app.api.main:app --reload

# 4. Query (defaults to fastapi/fastapi, trailing 30 days)
curl "http://localhost:8000/v1/insights/collaboration?repo=psf/requests&days=90"
curl "http://localhost:8000/v1/insights/narrative?repo=psf/requests&days=90"
```

Interactive API docs (Swagger) are at `http://localhost:8000/docs`.

### With Docker

```bash
cp .env.example .env          # add your keys
docker compose up --build
# service on http://localhost:8000
```

---

## Endpoints

### `GET /v1/insights/collaboration`

Collaboration-health report over a window.

| Query param    | Default       | Notes                                    |
|----------------|---------------|------------------------------------------|
| `repo`         | `DEFAULT_REPO`| `owner/name`                             |
| `days`         | `30`          | trailing window, 1–365                   |
| `period_start` | —             | ISO-8601; overrides `days` (with `period_end`) |
| `period_end`   | now           | ISO-8601                                 |

Response header `X-Cache: HIT|MISS` indicates whether the result came from the
local store. Example body (truncated):

```json
{
  "repo": "psf/requests",
  "total_commits": 62,
  "total_prs_merged": 60,
  "total_reviews": 63,
  "contributors": [{"login": "nateprewitt", "commits": 37, "prs_merged": 35, "reviews_submitted": 27}],
  "review_load": [{"reviewer": "nateprewitt", "reviews": 27}, {"reviewer": "sigmavirus24", "reviews": 21}],
  "contributor_count": 16,
  "reviewer_count": 9,
  "review_concentration": 0.429,
  "busiest_reviewer": "nateprewitt",
  "median_hours_to_first_review": 3.17,
  "median_hours_to_merge": 3.22,
  "unreviewed_merges": 14,
  "pulls_analyzed": 60,
  "truncated": false
}
```

(Real output for `psf/requests` over a 90-day window.)

### `GET /v1/insights/narrative`

Same query params. The writer model explains the numbers; the service then verifies
the result (numeric grounding + an LLM judge on the prose) and computes a confidence:

```json
{
  "summary": "The period shows substantial PR throughput, with 60 merged PRs and 62 commits. Review activity is concentrated, with review_concentration at 0.429 and the two largest review loads at 27 and 21. The median workflow was fast, with 3.17 hours to first review and 3.22 hours to merge, but 14 merged PRs had no recorded review.",
  "root_cause_hypothesis": "Review activity may depend heavily on a small set of reviewers, supported by review_concentration of 0.429 and the top review counts of 27 and 21.",
  "confidence": 0.715,
  "evidence": [
    {"metric": "review_concentration", "value": 0.429, "detail": "review activity is concentrated among fewer reviewers"},
    {"metric": "reviews:nateprewitt", "value": 27, "detail": "largest individual review load"},
    {"metric": "unreviewed_merges", "value": 14, "detail": "merged PRs with no recorded review"}
  ],
  "grounded": true,
  "grounding_warnings": [],
  "faithfulness": 5
}
```

(Real output for `psf/requests` over 90 days — writer `gpt-5.5`, judge `gpt-4.1`.)
`confidence`, `grounded`, `grounding_warnings`, and `faithfulness` are all set by the
service, never by the writer model:

- **`grounded`** — did every number cited in `evidence` match a source metric exactly?
- **`faithfulness`** — the judge's 1–5 rating of how well the *prose* is supported by
  the metrics (`null` if no judge ran).
- **`confidence`** — computed from the data's signal strength, then penalised for a
  fabricated number or for unfaithful prose. See [NOTES.md](./NOTES.md).

Requires `OPENAI_API_KEY` (or `GOOGLE_API_KEY` with `LLM_PROVIDER=gemini`); returns
`503` if no key is configured.

---

## The metric: collaboration health

Beyond "top contributors", the report surfaces **how collaboration is distributed**:

- **`review_concentration`** — the share of all reviews done by the single busiest
  reviewer (0–1). A high value is the classic **bus-factor / review-bottleneck**
  signal: one person is the gate for most changes.
- **`median_hours_to_first_review` / `median_hours_to_merge`** — where PR time goes.
- **`unreviewed_merges`** — merged PRs that received no external review.
- **`pulls_analyzed` / `truncated`** — coverage. To bound upstream fan-out, at most
  `MAX_PULLS_INSPECTED` PRs are analyzed per request; if the window held more,
  `truncated` is `true` and the totals are a lower bound. This is surfaced in the
  response (and to the narrative) so a capped count is never mistaken for a complete one.

These were chosen because they give the LLM a *real* root-cause story to find
(e.g. "reviews concentrated on one person → bottleneck → slow time-to-first-review"),
rather than a flat leaderboard. Self-reviews are excluded from review counts.

---

## Testing & evaluation

```bash
pytest                       # unit + API tests (metrics correctness, grounding, HTTP)
python -m eval.run_eval      # narrative eval suite (offline, deterministic stub LLM)
python -m eval.run_eval --live   # same suite against the configured real LLM
```

---

## Configuration

All config is environment-driven (see `.env.example`). Secrets never live in code or
the repo.

| Variable        | Purpose                                       |
|-----------------|-----------------------------------------------|
| `GITHUB_TOKEN`  | Read-only PAT; blank = unauthenticated (60/hr)|
| `LLM_PROVIDER`  | `openai` or `gemini`                          |
| `OPENAI_API_KEY`/`GOOGLE_API_KEY` | provider key                |
| `OPENAI_MODEL`  | writer model; default `gpt-5.5`               |
| `JUDGE_MODEL`   | faithfulness-judge model; default `gpt-4.1` (different from the writer, so it isn't grading itself) |
| `LLM_TEMPERATURE` | sampling temp; ignored by reasoning models that pin it |
| `DEFAULT_REPO`  | repo used when `?repo=` is omitted            |
| `DATABASE_URL`  | SQLite path for the cache                     |
| `MAX_PAGES`     | per-request upstream pagination cap           |

See [NOTES.md](./NOTES.md) for architecture, trade-offs, and what I'd do next.
