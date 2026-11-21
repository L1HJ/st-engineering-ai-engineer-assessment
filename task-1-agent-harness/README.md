# Operations Assistant Agent Harness

A local, single-process Agent Harness for an operations assistant. It exposes a FastAPI API and a terminal REPL, runs an explicit LLM-tool loop against an OpenAI-compatible Kimi model, persists state in SQLite, records an append-only execution trace, pauses for explicit approval before creating an incident, and supports multi-turn conversations. All tools, datasets, and incidents are local mocks.

- Version: 0.1.0
- Runtime: Python 3.11+
- Default test suite: `201 passed, 1 deselected`

## What Is Delivered

- HTTP API and a terminal REPL sharing the same database and configuration.
- Three mock tools: knowledge-base search, service status lookup, and approval-gated incident creation.
- A bounded ReAct loop with state management, retries, timeouts, malformed-response handling, and context compaction.
- SQLite persistence for conversations, runs, events, approvals, and incidents.
- Explicit approval before any `create_incident` call, with original-argument binding and idempotent writes.
- Structured JSON logs and a queryable ordered trace per run.
- Multi-turn conversations with isolated per-turn runs.
- A Postman collection with automated assertions.
- Automated tests plus a multi-turn conversation example in `examples/`.

## Approach

### Execution model

There is no hidden agent framework. `AgentRunner` owns an explicit state machine:

```text
RUNNING -> COMPLETED
        -> WAITING_APPROVAL -> (approval) -> RUNNING
        -> FAILED
        -> LIMIT_EXCEEDED
```

Each decision step sends the system prompt, the user objective, any context summary, and the rounds accumulated so far to the model:

1. A final answer completes the run with `COMPLETED`.
2. A tool call is validated, persisted with its unfinished round, executed with bounded retries, validated again, and completed with a standard observation. The observation is included in the next model request.
3. A valid `create_incident` call pauses the run at `WAITING_APPROVAL` without invoking the handler. Approval resumes the same round with the stored arguments; rejection finishes safely without an incident or another model call.

### Budgets and failure handling

- Maximum 20 decision steps and 120 seconds of monotonic active time per run. Approval waiting is excluded.
- Every model and tool attempt is clipped to the remaining active-time budget.
- Bounded retries use exponential Full Jitter backoff and honor `Retry-After`. Retries are persisted as `retry_scheduled` events.
- The first malformed model response receives one safe protocol correction; a second consecutive malformed response fails the run.
- Exhausted tool retries produce a sanitized failure observation instead of crashing the run.
- When 15 complete ReAct rounds exist and another decision is required, the oldest 10 rounds are folded into a summary and 5 recent rounds are kept. Compaction uses the same timeout, retry, and budget rules and never deletes existing events, approvals, or incidents.
- Limit and failure outcomes use stable codes such as `MAX_STEPS_EXCEEDED`, `ACTIVE_TIME_LIMIT_EXCEEDED`, and `CONTEXT_COMPACTION_FAILED`.

### Tool integration and validation

- Tool arguments and outputs are validated with Pydantic schemas; only registered tools are exposed.
- Mock datasets are validated at startup. Invalid JSON, schema violations, duplicates, or non-UTC timestamps fail startup instead of loading empty data.
- Each observation uses an envelope containing `tool_call_id`, `ok`, and either `data` or a sanitized `error` object.
- The complete UTF-8 JSON observation envelope must fit within 16 KiB.
- API errors do not expose provider or database exception details.

### Safety controls

- `create_incident` cannot execute without approval, and no API bypasses the approval gate.
- Approval stores normalized Canonical JSON arguments and their SHA-256 hash. The approval endpoint accepts only a decision and rejects replacement arguments with HTTP 422.
- Approval uses conditional database transitions. Incident writes are idempotent under the `run_id:tool_call_id` key.
- A `WAITING_APPROVAL` run and its arguments survive restart. A leftover `RUNNING` record found at startup is marked `FAILED / RUN_INTERRUPTED` rather than replayed.
- Logs redact sensitive keys and remove configured credentials. The POC has no authentication, so it should be bound to a trusted local interface only.

### Multi-turn conversations

A conversation owns one run per turn. Only the user message, the assistant final answer, the conversation summary, and a status note for failed or limited turns are inherited; tool observations, events, and trace data are never replayed.

- `POST /api/v1/conversations` creates a conversation.
- `POST /api/v1/conversations/{id}/messages` executes one turn synchronously and returns the same `RunView` as `/api/v1/runs`. A repeated `client_message_id` replays the original run with HTTP 200.
- `GET /api/v1/conversations/{id}` returns metadata and recent turns.
- `GET /api/v1/conversations/{id}/messages?limit=20&before_turn=13` pages through history.

A partial unique index allows only one active turn per conversation; concurrent messages receive `409 CONVERSATION_BUSY`. Conversation-level compaction folds older turns into a summary after 12 completed turns, and the conversation is capped at 50 turns.

## Data Model

| Table | Purpose |
| --- | --- |
| `conversations` | Title, status, conversation-level summary, and the last summary-covered turn index. |
| `runs` | Objective, state, counters, budgets, final result, error, and compacted context snapshot; turns add `conversation_id`, `turn_index`, and `client_message_id`. |
| `run_events` | Ordered per-run sequence, event type, JSON payload, latency, and timestamp. Append-only. |
| `approvals` | Pending/decided state, original Canonical JSON arguments, and SHA-256 binding hash. |
| `incidents` | Created mock incident and unique `run_id:tool_call_id` idempotency key. |

SQLite is initialized at startup with idempotent table creation and lightweight column/index migration for the conversation extension.

## Technology Stack

- Python 3.11+
- FastAPI and Uvicorn
- Pydantic v2 and pydantic-settings
- OpenAI Python SDK with the Kimi OpenAI-compatible API
- SQLAlchemy and SQLite
- pytest, FastAPI TestClient, and deterministic test fakes

## Project Layout

```text
src/agent_harness/   API, runner, adapter, tools, persistence, observability
tests/               Fake-based unit and integration tests, opt-in Kimi smoke test
data/                Versioned knowledge-base and service mock datasets
docs/                Technical contract, roadmap, plan, and extension design
postman/             Collection and local environment with automated tests
examples/            Multi-turn conversation example with per-turn runs and traces
pyproject.toml       Package metadata, dependencies, and pytest configuration
.env.example         Complete non-secret configuration template
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and replace both required Kimi placeholders. Never commit `.env` or real credentials.

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `KIMI_API_KEY` | Yes | None | Non-empty Kimi API key. |
| `KIMI_MODEL` | Yes | None | Kimi model name; no model is hard-coded. |
| `KIMI_BASE_URL` | No | `https://api.moonshot.cn/v1` | HTTP(S) OpenAI-compatible base URL. |
| `DATABASE_URL` | No | `sqlite:///./agent_harness.db` | SQLite database URL. |
| `LLM_TIMEOUT_SECONDS` | No | `30` | Timeout for one model HTTP attempt. |
| `LLM_MAX_RETRIES` | No | `3` | Maximum extra model retries after the first attempt. |
| `TOOL_TIMEOUT_SECONDS` | No | `5` | Timeout for one tool attempt, clipped to the remaining run budget. |
| `TOOL_MAX_RETRIES` | No | `2` | Maximum extra tool retries after the first attempt. |
| `RETRY_BACKOFF_BASE_SECONDS` | No | `0.5` | Base delay for exponential Full Jitter backoff. |
| `RETRY_BACKOFF_MAX_SECONDS` | No | `8` | Backoff cap; must be at least the base. |
| `AGENT_MAX_STEPS` | No | `20` | Maximum number of decision steps. |
| `AGENT_ACTIVE_TIME_LIMIT_SECONDS` | No | `120` | Active execution budget, excluding approval waits. |
| `CONTEXT_WINDOW_ROUNDS` | No | `15` | Complete-round threshold that triggers compaction. |
| `CONTEXT_COMPACT_ROUNDS` | No | `10` | Oldest rounds folded into the new summary; must be smaller than the window. |
| `CONVERSATION_WINDOW_TURNS` | No | `12` | Completed turns that trigger conversation-level compaction. |
| `CONVERSATION_COMPACT_TURNS` | No | `8` | Oldest turns folded into the conversation summary. |
| `CONVERSATION_MAX_TURNS` | No | `50` | Maximum turns per conversation. |
| `MALFORMED_RESPONSE_LIMIT` | No | `2` | Consecutive malformed-response limit; one safe correction is attempted. |
| `TOOL_RESULT_MAX_BYTES` | No | `16384` | Maximum UTF-8 JSON bytes for the complete observation envelope. |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

All settings are validated together at startup. Diagnostics name invalid fields without echoing API keys.

## Terminal REPL (Recommended)

```bash
python -m agent_harness.chat
```

Recommended for interactive use. Each launch starts a new conversation. Tool calls and retries are printed as indented steps, replies render a safe Markdown subset, and incident creation opens an arrow-key selector (or `y`/`n`). Commands: `/help`, `/new`, `/id`, `/history [limit]`, `/trace`, `/exit`. The last conversation is remembered in `.agent_harness_chat.json`; use `--continue` to resume it, `--conversation-id` to resume a specific conversation, or `--title` to name a new one. Up/down arrows browse persisted input history.

## Running the HTTP API

```bash
uvicorn agent_harness.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The API is the base interface for automation and direct HTTP integration. Use one worker. SQLite tables are created during startup, and interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## API Reference

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/runs` | Create and synchronously execute a single run. |
| GET | `/api/v1/runs/{run_id}` | Read a persisted run and its pending approval. |
| GET | `/api/v1/runs/{run_id}/events` | Read the ordered trace. |
| POST | `/api/v1/runs/{run_id}/approvals/{approval_id}` | Approve or reject an incident action. |
| POST | `/api/v1/conversations` | Create a conversation. |
| POST | `/api/v1/conversations/{id}/messages` | Execute one conversation turn. |
| GET | `/api/v1/conversations/{id}` | Read conversation metadata and recent turns. |
| GET | `/api/v1/conversations/{id}/messages` | Page through conversation history. |
| GET | `/api/v1/health` | Check SQLite health only; never calls Kimi. |

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"objective":"Investigate elevated errors in checkout"}'
```

The objective is trimmed and must contain 1–4000 characters. A successful request returns HTTP 201 with `status`, `final_answer`, and `error`; an incident action returns `WAITING_APPROVAL` with a pending approval ID and stored arguments.

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/runs/RUN_ID/approvals/APPROVAL_ID \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve"}'
```

All API errors use the envelope `{error: {code, message, details}}`.

## Postman

Import `postman/agent-harness.postman_collection.json` and `postman/agent-harness.postman_environment.json`, select the `Agent Harness - Local` environment, and run folders in collection order with the API running. The collection covers health, direct answer creation, run and trace queries, incident approval, and incident rejection. Create requests save `run_id` and `approval_id` for later requests.

```bash
npx --yes newman run postman/agent-harness.postman_collection.json \
  --environment postman/agent-harness.postman_environment.json
```

Postman and Newman runs call the configured real Kimi model and are delivery checks, not part of the offline test suite.

## Tests

```bash
pytest
```

The default suite uses fake Kimi, fake clock, fake sleeper, and temporary SQLite databases. An autouse fixture blocks socket connections and real retry sleeps, so accidental external requests fail the tests. Coverage includes the ReAct loop, tool validation and allowlisting, retries and backoff, malformed responses, step and time limits, approval pause/resume and rejection, incident idempotency, context compaction, conversation ordering and isolation, restart persistence, and REPL rendering.

Run the explicit real-provider smoke test only with valid local credentials:

```bash
RUN_KIMI_SMOKE=1 pytest -m kimi_smoke
```

This test makes a real Kimi request and may incur provider cost. It asserts controlled run and trace behavior rather than exact free text.

## Multi-Turn Conversation Example

`examples/multi-turn-conversation.json` is a deterministic three-turn session generated offline with fake model and clock fakes; no real provider call is made. It contains the conversation record, every turn with its run, and the ordered trace for each run.

- Turn 1: an ambiguous request; the assistant asks for clarification and the turn completes.
- Turn 2: `get_service_status` is called for `checkout`, and the answer uses the observation.
- Turn 3: `create_incident` pauses for approval; after approval the run resumes and completes.

The example is a review artifact. The SQLite `run_events` table remains the runtime source of truth.

## Limitations

- Single-process synchronous HTTP execution with SQLite; no queue, horizontal scaling, authentication, or high availability.
- Only waiting approvals and historical records survive restart. Execution does not resume from an arbitrary process crash, and incident write versus observation persistence has a small manual-reconciliation window.
- The knowledge base uses deterministic keyword matching rather than semantic retrieval. All tools and incidents are local mocks.
- Context budgets count rounds and turns rather than tokens. Compaction failure terminates the run instead of using a local summary fallback.
- No SSE streaming. The REPL polls persisted run events, so updates are slightly delayed.
- Model decisions and answer quality are nondeterministic; the test suite validates harness behavior, not model quality.

## Future Improvements

- SSE streaming and asynchronous runs with cancellation.
- Token-aware context budgets.
- PostgreSQL and distributed recovery.
- Third-party incident reconciliation.
- OpenTelemetry metrics and cost tracking.
- Authenticated approvals and role-based access.
- Hybrid retrieval over operational documentation.
- Real operations integrations.

## Documentation

- `docs/technical-design.en.md`: authoritative implementation contract.
- `docs/roadmap.en.md`: phase scope, dependencies, and acceptance status.
- `docs/agent-harness-plan.en.md`: overall project plan.
- `docs/multi-turn-chat.en.md`: multi-turn conversation extension design.
