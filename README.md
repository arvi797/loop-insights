# Loop Insights

A small service that reads collaboration data from the **GitHub API** and surfaces
insights over a time period, including an **LLM-synthesized narrative** with a
root-cause hypothesis, a confidence score, and an evidence chain that traces every
claim back to the underlying numbers.

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
  "total_commits": 64,
  "total_prs_merged": 62,
  "total_reviews": 66,
  "contributors": [{"login": "nateprewitt", "commits": 37, "prs_merged": 35, "reviews_submitted": 29}],
  "review_load": [{"reviewer": "nateprewitt", "reviews": 29}, {"reviewer": "sigmavirus24", "reviews": 21}],
  "contributor_count": 19,
  "reviewer_count": 10,
  "review_concentration": 0.439,
  "busiest_reviewer": "nateprewitt",
  "median_hours_to_first_review": 3.36,
  "median_hours_to_merge": 3.7,
  "unreviewed_merges": 14,
  "pulls_analyzed": 62,
  "truncated": false
}
```

(Real output for `psf/requests` over a 90-day window.)

### `GET /v1/insights/narrative`

Same query params. Calls the configured LLM to explain the numbers, then validates
the output against the source metrics:

```json
{
  "summary": "The collaboration data shows that a small number of contributors are responsible for the majority of both code contributions and reviews. Notably, 29 out of 66 total reviews were performed by a single contributor, and the top two reviewers together accounted for 50 of the 66 reviews. Additionally, 14 out of 62 merged pull requests were merged without review...",
  "root_cause_hypothesis": "The high review concentration (0.439) and the large number of unreviewed merges (14) suggest a bottleneck where review responsibilities are concentrated among a few individuals.",
  "confidence": 0.72,
  "evidence": [
    {"metric": "review_concentration", "value": 0.439, "detail": "Reviews are concentrated among a small subset of reviewers."},
    {"metric": "reviews:nateprewitt", "value": 29, "detail": "One contributor performed 29 of 66 reviews."},
    {"metric": "unreviewed_merges", "value": 14, "detail": "14 of 62 merged PRs were merged without review."}
  ],
  "grounded": true,
  "grounding_warnings": []
}
```

(Real `gpt-5.5` output over the numbers above. `confidence` and `grounded` are set by the
service's validator, not the model — see NOTES.)

`grounded` and `grounding_warnings` are set by the service, not the model — see
NOTES. Requires `OPENAI_API_KEY` (or `GOOGLE_API_KEY` with `LLM_PROVIDER=gemini`);
returns `503` if no key is configured.

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
| `OPENAI_MODEL`  | default `gpt-5.5`; any structured-output model (e.g. `gpt-4.1`) |
| `LLM_TEMPERATURE` | sampling temp; ignored by reasoning models that pin it |
| `DEFAULT_REPO`  | repo used when `?repo=` is omitted            |
| `DATABASE_URL`  | SQLite path for the cache                     |
| `MAX_PAGES`     | per-request upstream pagination cap           |

See [NOTES.md](./NOTES.md) for architecture, trade-offs, and what I'd do next.
