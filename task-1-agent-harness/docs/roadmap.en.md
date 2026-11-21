# Operations Assistant Agent Harness Implementation Roadmap

- Document status: Implementation baseline
- Version: 1.0
- Last updated: 2026-11-21
- Technical implementation contract: `docs/technical-design.en.md`

> Extension note: multi-turn conversations (P7a) and the minimal terminal REPL (P7c) are implemented; see `docs/multi-turn-chat.en.md`. The P1–P6 scope and acceptance conclusions in this document remain frozen.

## 1. Document Purpose

This document independently describes the implementation path for the Operations Assistant Agent Harness from the current design baseline to a submittable version, including the project background, delivery scope, technical baseline, phase dependencies, work items for each phase, testing, exit criteria, design traceability, risks, definition of done, and release checklist.

This document is not merely a collection of tasks excerpted from other documents. Readers can rely on this document alone to understand "what must be delivered, in what order, and how to prove that each phase is complete"; specific fields, algorithms, and interface details uniformly follow the technical design document.

## 2. Project Background and Delivery Goals

This task requires building an Operations Assistant Agent Harness. The system receives user goals through an API, uses the Kimi model to drive an LLM–tool execution loop, and uses three Mock tools:

1. `search_knowledge_base(query)`: Searches internal documents.
2. `get_service_status(service_name)`: Retrieves service status.
3. `create_incident(title, description, severity)`: Creates an incident; user approval is required before invocation.

The final version must demonstrate the following capabilities rather than merely provide one successful model call:

- The Agent loop and state management are explainable, persistable, and recoverable.
- Tool inputs and outputs are strictly validated.
- Timeouts, retries, backoff, tool errors, and malformed model responses are handled deterministically.
- Steps, active time, retries, and format corrections are all bounded.
- Incident creation is protected by server-side approval, original-argument binding, and idempotency constraints.
- Every execution has structured logs and a queryable Trace.
- Important success and failure paths have automated tests.
- The code, Mock Dataset, documentation, environment variable template, and Postman Collection can be submitted and reproduced together.

## 3. Scope and Deliverables

### 3.1 Scope of This Iteration

- A single-user, single-process, locally running POC.
- Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, and SQLite.
- Kimi OpenAI-compatible API, with the model name configured through an environment variable.
- Synchronous Run API: Returns after execution reaches a terminal state, hits a limit, or begins waiting for approval.
- Three deterministic Mock tools and local JSON datasets.
- Persistence of Runs, Run Events, Approvals, and Incidents.
- Recovery of Runs awaiting approval after a service restart.
- Unit tests, API integration tests, concurrent approval tests, restart recovery tests, and an optional Kimi Smoke Test.
- A complete README, `.env.example`, Postman Collection, and descriptions of limitations and future improvements.

### 3.2 Non-Goals

- Multi-tenancy, user login, role-based permissions, or public production deployment.
- Distributed queues, multiple Workers, horizontal scaling, and high availability.
- Integration with real knowledge bases, monitoring systems, or incident platforms.
- Vector databases and complex semantic retrieval.
- Parallel tools, multiple Agents, or a general-purpose workflow platform.
- Automatic execution recovery from an arbitrary process crash point.

### 3.3 Final Deliverables Checklist

| Deliverable | Target Location | Completion Criteria |
| --- | --- | --- |
| Agent Harness source code | `src/agent_harness/` | API, ReAct, tools, state, retries, approvals, and Compact are operational |
| Automated tests | `tests/` | Default tests have no real external dependencies and cover both success and failure paths |
| Knowledge base Mock Dataset | `data/knowledge_base.json` | Stable format, sufficient content to demonstrate retrieval, and published with the repository |
| Service status Mock Dataset | `data/services.json` | Covers healthy, degraded, unavailable, maintenance, and unknown service paths |
| Project dependencies and metadata | `pyproject.toml` | Editable installation, with explicit test dependencies and pytest configuration |
| Environment variable template | `.env.example` | Lists all variables, default values, and non-real examples; contains no secrets |
| Usage documentation | `README.md` | Installation, configuration, startup, testing, API, approvals, and troubleshooting are reproducible |
| Technical design | `docs/technical-design.en.md` | Fully defines the implementation contract, database, interfaces, security, and limitations |
| Implementation Roadmap | `docs/roadmap.en.md` | Complete phases, dependencies, statuses, traceability, and acceptance criteria |
| Overall plan | `docs/agent-harness-plan.en.md` | Retains the project proposal background and remains aligned with the implementation baseline |
| Postman Collection | `postman/agent-harness.postman_collection.json` | Covers creation, queries, Trace, approval, rejection, and health checks |
| Multi-turn conversation example | `examples/multi-turn-conversation.json` | One deterministic session with three turns, per-turn runs, and ordered traces, including tool use and approval |
| Submission notes | Relevant sections of the README and documentation | Includes the technology stack, database, limitations, future improvements, and operating instructions |

## 4. Frozen Technical Baseline

The following decisions are shared dependencies across all phases and must not be silently changed by localized implementation.

### 4.1 Technology Stack and Framework Boundaries

| Item | Decision |
| --- | --- |
| Model | Kimi API, accessed through the OpenAI Python SDK and a custom Base URL |
| Agent pattern | Independently implemented structured ReAct; no parsing of free-form `Action:` text |
| Agent framework | No ready-made execution frameworks such as LangChain AgentExecutor |
| API | FastAPI + Uvicorn, single Worker, local execution |
| Validation | Pydantic v2, covering the API, tool inputs and outputs, and model actions |
| Persistence | SQLAlchemy + SQLite, with automatic table creation at startup |
| Logging and Trace | JSON application logs + SQLite `run_events` |
| Testing | pytest, FastAPI TestClient, Fake Kimi, Fake Clock, and Fake Tools |

Ready-made Agent frameworks are not used so that the loop, state, failures, safety approvals, and execution limits can all be directly demonstrated, fault-injected, and verified, and to prevent framework default policies from overriding the precise requirements of the Harness. General-purpose SDKs and foundational libraries may be used, but they cannot replace the Harness core.

### 4.2 Core Execution Conventions

- One Run corresponds to one user goal, and the same Run advances serially.
- Each well-formed model response can produce only one final answer or one tool call.
- Multiple tool calls, unknown tools, empty responses, and invalid argument JSON are all countable malformed responses.
- When tool argument JSON is valid but does not conform to the Schema, the Handler is not executed and a controlled input validation error is returned to the model.
- Tool output that does not conform to the Schema or exceeds the size boundary does not enter the model context.
- A valid tool action and its incomplete Round must be persisted before the Handler can execute, ensuring that the execution intent remains auditable after a crash.
- All state, event, round, and approval updates are completed through explicit transactions.
- The complete execution history is retained permanently; Compact only changes the context sent to the model.
- `create_incident` can only execute using the original arguments stored in the database after approval.
- At most one incident can be created for the same `run_id + tool_call_id`.

### 4.3 Default Execution Budget

| Configuration | Default Value | Completion Semantics |
| --- | --- | --- |
| LLM timeout | 30 seconds | Per HTTP Attempt, and no more than the Run's remaining active budget |
| LLM max_retry | 3 | At most 3 retries after the initial call, for at most 4 calls in total |
| Tool timeout | 5 seconds | Per Handler Attempt, and no more than the Run's remaining active budget |
| Tool max_retry | 2 | At most 2 retries after the initial call, for at most 3 calls in total |
| Backoff | 0.5-second base, 8-second cap | Exponential backoff + Full Jitter, with valid Retry-After values taking precedence |
| Agent max steps | 20 | Number of normal decisions; malformed responses also count as steps, while Compact does not |
| Active time | 120 seconds | Includes model calls, tools, retries, backoff, and Compact; excludes waiting for approval |
| Consecutive malformed responses | 2 | The first triggers one correction; the second terminates execution |
| Tool result size | 16 KiB | Exceeding the limit causes a controlled failure; truncated JSON is not sent to the model |

### 4.4 Compact Baseline

- The uncompressed window is 15 rounds. Each compaction processes the earliest 10 rounds, and the 5 most recent rounds are retained when Compact is committed.
- When 15 completed Rounds already exist and another model decision is still required, Compact runs before the 16th normal ReAct Round is produced; malformed responses count only toward Steps, not Rounds.
- Compact generates a new summary from the previous summary together with the earliest 10 rounds and, upon success, retains the 5 most recent rounds within a transaction.
- The next decision is then initiated; after the 16th normal Round is completed, the snapshot contains 6 rounds.
- The trigger is determined by the number of completed Rounds, not unconditionally by the 16th HTTP request or `step_count=15`.
- This trigger ensures that the uncompressed history sent to the model never exceeds 15 rounds and avoids the counting error of assuming "16 minus 10 equals 5."
- Compact uses the same model timeout and retry budget and does not increase the Agent Step count.
- Compact failure must not silently delete history; exhaustion of active time takes precedence and leads to `LIMIT_EXCEEDED`, while other permanent failures fail with `CONTEXT_COMPACTION_FAILED`.

### 4.5 State Baseline

Run states:

```text
RUNNING -> COMPLETED
RUNNING -> FAILED
RUNNING -> LIMIT_EXCEEDED
RUNNING -> WAITING_APPROVAL -> RUNNING
RUNNING -> WAITING_APPROVAL -> COMPLETED
Orphaned RUNNING -> FAILED (service startup recovery, error code RUN_INTERRUPTED)
```

Approval states:

```text
PENDING -> APPROVED
PENDING -> REJECTED
```

Terminal states cannot advance further. Repeated approvals, races between approval and rejection, and repeated recovery must all be prevented by conditional updates and database constraints.

## 5. Implementation Strategy

### 5.1 Vertical Slices Rather Than Horizontal Layering

Each phase forms a runnable, testable thin slice, avoiding a situation in which "the code is run for the first time only after all of it has been written":

1. Basic end-to-end loop: First connect the API to the model, state, events, and queries.
2. Read-only tool ReAct: Add two side-effect-free tools and a real multi-step loop.
3. Reliability: Add failure handling, retries, and limits to the already operational loop.
4. Approval and side effects: Add the incident tool and complete safety boundaries to the stable loop.
5. Compact: Add precise context compaction for multi-round execution.
6. Delivery hardening: Complete end-to-end regression, Postman, README, and submission checks.

### 5.2 Phase Rules

- Define observable outcomes for each phase before implementing the corresponding code.
- Complete tests and code within the same phase; do not leave failure paths until the end.
- All exit criteria for the previous phase must be satisfied before a new phase begins.
- Default tests do not access Kimi, perform real Sleeps, or depend on local database files.
- No safety invariant may be bypassed to temporarily make a Demo work.
- When documentation and implementation disagree, first determine whether this is a design change or a code defect, then update the corresponding authoritative document accordingly.
- Do not introduce queues, permission systems, or complex infrastructure beyond the POC goals.

## 6. Phase Overview and Dependencies

| Phase | Name | Dependencies | Current Status | Demonstrable Outcome |
| --- | --- | --- | --- | --- |
| P0 | Design baseline | None | Completed | Complete design goals, technical contract, and implementation path |
| P1 | Runnable basic end-to-end loop | P0 | Completed | The API creates a Run, the model answers directly, and state and Trace are queryable |
| P2 | Read-only tool ReAct | P1 | Completed | Knowledge base/status queries can execute over multiple steps and persist observations |
| P3 | Reliability and execution limits | P2 | Completed | Failures, retries, malformed responses, steps, and time are all controlled |
| P4 | Approval and incident side effects | P3 | Completed | Approval, rejection, recovery, idempotency, and concurrency safety are complete |
| P5 | Context Compact | P4 | Completed | Compaction follows the 15/10/5 rule before the 16th normal Round is produced, without losing history |
| P6 | End-to-end delivery and release | P1–P5 | Completed | Full testing, real Kimi Smoke, documentation, Postman, Dataset, and submission package are complete |

Dependency relationships:

```mermaid
flowchart LR
    P0[P0 Design Baseline] --> P1[P1 Basic End-to-End Loop]
    P1 --> P2[P2 Read-Only ReAct]
    P2 --> P3[P3 Reliability]
    P3 --> P4[P4 Approval and Incidents]
    P4 --> P5[P5 Compact]
    P5 --> P6[P6 Delivery and Release]
```

The only permitted statuses are: `Not Started`, `In Progress`, `Completed`, and `Blocked`. P0–P6 have all been implemented and have passed local delivery acceptance; GitHub repository creation, Git commits, and remote upload were not performed in the current environment.

## 7. P0: Design Baseline

### 7.1 Goal

Turn the design goals into a shared contract that is implementable, testable, and free of critical ambiguities, avoiding ad hoc decisions about safety or state semantics during coding.

### 7.2 Work Items

- [x] P0-1: Confirm the tool surface, delivery criteria, and acceptance checks.
- [x] P0-2: Determine the Kimi OpenAI-compatible integration and the Python/FastAPI/SQLite technology stack.
- [x] P0-3: Clarify the reasons for independently implementing ReAct and not using an existing Agent execution framework.
- [x] P0-4: Determine the 15/10/5 Compact, timeout, retry, step, and active-time budgets.
- [x] P0-5: Determine the incident approval, original-argument binding, conditional update, and idempotency strategies.
- [x] P0-6: Write an independent, complete technical design document.
- [x] P0-7: Write an independent, complete Roadmap and establish design traceability and release acceptance.

### 7.3 Outputs

- `docs/agent-harness-plan.en.md`
- `docs/technical-design.en.md`
- `docs/roadmap.en.md`
- Project directory skeleton and references to the design contract.

### 7.4 Verification

- The documentation covers all functional and output scope.
- Retry counts, format correction, the Compact trigger, approval concurrency, and restart boundaries are unambiguous.
- Both the technical document and the Roadmap can be read independently and do not require "additional details in the preceding document" to be understood.

### 7.5 Exit Criteria

- [x] Technology choices, scope, non-goals, and architecture are clear.
- [x] The database, API, error codes, and testing strategy are clear.
- [x] Implementation phases, dependencies, and acceptance criteria are clear.

Status: **Completed**.

## 8. P1: Runnable Basic End-to-End Loop

### 8.1 Goal

Establish a minimal but real end-to-end path: the client creates a Run, the Kimi Adapter returns a direct final answer, the Runner persists state and events, and the client can query the Run and Trace.

Tool calls are not enabled in this phase, and a Mock final answer must not replace the production Kimi Adapter; Fake Kimi is used only for testing.

### 8.2 Dependencies

- P0 is fully complete.
- A Kimi API Key and model name for local development are available for manual Smoke testing; automated tests do not require real credentials.

### 8.3 Work Items

#### Project and Configuration

- [x] P1-1: Create `pyproject.toml`, the Src Layout, development dependencies, and pytest configuration.
- [x] P1-2: Implement Settings loading and startup validation, covering all environment variables and their boundaries.
- [x] P1-3: Create `.gitignore` to exclude `.env`, virtual environments, caches, and local databases.

#### Data and Observability

- [x] P1-4: Implement the `runs` and `run_events` SQLAlchemy Models and SQLite initialization.
- [x] P1-5: Implement the Run Repository, event sequence numbers, transactions, and queries by Sequence.
- [x] P1-6: Implement the JSON Logger, shared correlation fields, and basic sanitization.
- [x] P1-7: Establish Run/Error/API Pydantic Schemas and a standardized HTTP Error Envelope.

#### Model and Runner

- [x] P1-8: Use `AsyncOpenAI` to implement the Kimi Adapter's basic requests and final-answer normalization, without hardcoding the model name.
- [x] P1-9: Implement Clock/Sleeper injection boundaries to reserve interfaces for subsequent reliability testing.
- [x] P1-10: Implement Runner creation of Runs, direct final answers, basic step/time checks, and terminal-state persistence.
- [x] P1-11: Reserve the `context_summary`, `recent_rounds_json`, and consecutive malformed count fields, but do not execute Compact in this phase.

#### API

- [x] P1-12: Implement `POST /api/v1/runs`.
- [x] P1-13: Implement `GET /api/v1/runs/{run_id}`.
- [x] P1-14: Implement `GET /api/v1/runs/{run_id}/events`.
- [x] P1-15: Implement `GET /api/v1/health`, checking only the API and database without calling Kimi.

### 8.4 Phase Outputs

- An installable, startable FastAPI project.
- Persistable direct-answer Runs.
- Queryable Run Events and JSON logs.
- Fake Kimi, temporary SQLite, and the first set of test utilities.

### 8.5 Phase Tests

- A direct final answer returns 201, and the Run is `COMPLETED`.
- The `run_created`, model response, and `run_finished` events occur in the correct order.
- GET Run and Trace do not advance execution.
- Empty and overlength goals return a standardized 422.
- A nonexistent Run returns a standardized 404.
- Unknown routes and incorrect methods return standardized 404/405 responses respectively, and 405 retains the `Allow` Header.
- Healthy and unhealthy database states return 200/503 respectively.
- Logs and API responses do not contain the Fake API Key.
- Fake Clock can drive basic budget checks without real waiting.

### 8.6 Exit Criteria

- [x] A fresh environment can be installed and started using the agreed commands.
- [x] Automated tests pass across the full creation, query, and Trace path for direct answers.
- [x] History remains queryable after an SQLite restart.
- [x] Configuration errors have safe, clear diagnostics.
- [x] All P1 tests pass, and the default tests make no external network access.

Status: **Completed**.

## 9. P2: Read-Only Tool ReAct

### 9.1 Goal

Add `search_knowledge_base` and `get_service_status` to the basic end-to-end loop, forming a real multi-step "model action—tool observation—another decision" path, and complete tool input/output validation and the Mock Dataset.

`create_incident` is not registered in this phase, preventing any bypass path before the approval capability is complete.

### 9.2 Dependencies

- All P1 exit criteria are satisfied.
- The Run, events, model Adapter, and test doubles are reusable.

### 9.3 Work Items

#### Mock Dataset and Registry

- [x] P2-1: Write `knowledge_base.json`, covering short documents on troubleshooting, alert handling, incident severity classification, and related topics.
- [x] P2-2: Write `services.json`, covering healthy, degraded, unavailable, and maintenance states.
- [x] P2-3: Implement Dataset Schema validation at startup, as well as the Tool Registry, allowlist, JSON Schema exposure, and unified execution context.
- [x] P2-4: Implement knowledge base keyword retrieval, deterministic ordering, a maximum of 5 results, and empty results.
- [x] P2-5: Implement service status queries and a valid `unknown` observation for unknown services.
- [x] P2-6: Implement Pydantic input and output validation for both tools.

#### ReAct and Context

- [x] P2-7: Implement Final Answer/Tool Action normalization and the single-tool-call constraint.
- [x] P2-8: Implement the Round representation of Assistant Tool Calls and Tool Messages.
- [x] P2-9: Implement standard success/failure observation Envelopes.
- [x] P2-10: Implement message construction from the System Prompt, user goal, and recent Rounds.
- [x] P2-11: Implement normal read-only tool dispatch; persist the action and incomplete Round first, execute the Handler, then validate the output and complete the original Round.
- [x] P2-12: Pass both read-only tool Schemas to Kimi, leaving the model to decide the next step itself.

### 9.4 Phase Outputs

- Two usable Mock tools and their Datasets.
- Runnable multi-step read-only ReAct.
- Auditable tool arguments, observations, final answers, and round snapshots.

### 9.5 Phase Tests

- Knowledge base retrieval hits, case insensitivity, ordering, empty results, and the maximum of 5 results.
- Every service status enumeration and unknown-service behavior.
- Duplicate document IDs, duplicate service names, invalid JSON, invalid statuses, and additional fields cause startup validation to fail rather than silently loading empty data.
- Blank inputs, overlength inputs, additional fields, and incorrect types do not execute the Handler.
- Outputs with missing fields, incorrect types, or additional invalid structures do not enter the model context.
- The model can query the service first and then the knowledge base, and the second-round messages contain the first-round observation.
- No tool executes when one response contains multiple tool calls.
- Unknown tools and invalid argument JSON have explicit classifications.
- Run Events are consistent with Round snapshots, and the complete Trace can be queried in order.

### 9.6 Exit Criteria

- [x] End-to-end success paths for both read-only tools pass.
- [x] Multi-step observations accurately enter the next model request.
- [x] The tool allowlist and bidirectional Schema validation are effective.
- [x] There is no executable entry point for `create_incident`.
- [x] All P1–P2 tests pass.

Status: **Completed**.

## 10. P3: Reliability and Execution Limits

### 10.1 Goal

Harden the operational ReAct into a bounded, diagnosable, fault-injectable execution system, fully covering model and tool timeouts, retries, backoff, format errors, steps, and active time.

### 10.2 Dependencies

- The P2 read-only ReAct success path is stable.
- Fake Kimi, Fake Handler, Fake Clock, and Fake Sleeper can control every Attempt.

### 10.3 Work Items

#### Model Reliability

- [x] P3-1: Implement classification of LLM network, Timeout, HTTP, and SDK exceptions.
- [x] P3-2: Implement bounded retries for 408/409/429/5xx and network errors.
- [x] P3-3: Implement no-retry behavior and safe error mapping for 400/401/403/404/422.
- [x] P3-4: Implement exponential backoff, Full Jitter, Retry-After, and `retry_scheduled` events.
- [x] P3-5: Implement unified malformed-response classification for empty responses, multiple calls, unknown tools, invalid JSON, and duplicate Tool Call IDs.
- [x] P3-6: Implement "correct the first occurrence, fail on the second consecutive occurrence" and reset the count after a valid response.

#### Tool Reliability

- [x] P3-7: Implement read-only tool timeouts and at most two additional retries.
- [x] P3-8: Return a safe failure observation after retries are exhausted, without passing through exceptions or Stack Traces.
- [x] P3-9: Implement the 16 KiB tool output limit and controlled Run failure for invalid output.
- [x] P3-10: Distinguish input validation errors, transient execution errors, permanent errors, and output trust errors.

#### Global Budget

- [x] P3-11: Complete step counting and all boundaries, including counting malformed responses as steps and excluding retries from the step count.
- [x] P3-12: Complete Monotonic active-time accumulation and persistence.
- [x] P3-13: Cap every model/tool Attempt timeout at the remaining active budget.
- [x] P3-14: Check the budget before backoff to prevent the retry mechanism from bypassing the overall 120-second limit.
- [x] P3-15: Complete `MAX_STEPS_EXCEEDED`, `ACTIVE_TIME_LIMIT_EXCEEDED`, and terminal-state events.

### 10.4 Phase Outputs

- Complete error classification, retry, backoff, and format correction mechanisms.
- Bounded model, tool, and Run execution paths.
- A repeatable fault test suite with no real waiting.

### 10.5 Phase Tests

- The model succeeds after an initial Timeout/429/5xx, with accurate Attempt and backoff events.
- The model fails with an explicit error after retries are exhausted.
- 401/400 cause no additional Attempts.
- Both Retry-After and the Run's remaining time are respected.
- The first malformed response can be corrected, the second consecutive malformed response fails, and a valid response resets the count.
- A tool succeeds after a transient failure; after persistent failures, the model receives a safe failure observation.
- Invalid and oversized tool outputs do not enter the model Messages.
- A final answer at step 20 can succeed; requiring step 21 results in `MAX_STEPS_EXCEEDED`.
- Time exhaustion during the model, tools, or backoff all results in `ACTIVE_TIME_LIMIT_EXCEEDED`.
- All failed Runs retain existing events and a safe error summary.

### 10.6 Exit Criteria

- [x] All external waits have timeouts, and all retries and corrections have upper bounds.
- [x] Step and active-time limits cannot be bypassed by retries or format errors.
- [x] Default tests perform no real Sleeps, and critical failure paths are repeatable.
- [x] Errors have safe representations for the API, logs, and model.
- [x] All P1–P3 tests pass.

Status: **Completed**.

## 11. P4: Approval and Incident Side Effects

### 11.1 Goal

Add `create_incident`, completing the full safety loop from incident request, pause, approval/rejection, and synchronous recovery to idempotent writes, and verify concurrency and restart scenarios.

### 11.2 Dependencies

- The P3 loop, state, error, and budget behavior is stable.
- Read-only tool path regression passes.
- The Runner already has transactional state and event Repositories.

### 11.3 Work Items

#### Data and Tools

- [x] P4-1: Implement the `approvals` and `incidents` Models, foreign keys, unique constraints, and partial unique indexes.
- [x] P4-2: Implement incident input/output Schemas and the Incident Mock Handler.
- [x] P4-3: Implement Canonical JSON, SHA-256 argument hashes, and idempotency keys.
- [x] P4-4: Implement Incident lookup before creation, rereading after unique conflicts, and conflict detection for the same key with different arguments.

#### Pause and Recovery

- [x] P4-5: A valid incident request creates only a Pending Approval and does not execute the Handler.
- [x] P4-6: Within a transaction, verify the saved Tool Call and incomplete Round, update the Run to the waiting state, and write the approval event.
- [x] P4-7: Implement conditional approval updates, Run/Approval matching, and argument hash verification.
- [x] P4-8: After approval, execute only with the stored arguments, save the observation, and restore the original Round.
- [x] P4-9: Implement rejection without execution or incident creation, followed by normal Run completion.
- [x] P4-10: Incident tool retries always reuse the same idempotency key, and a new incident action requires new approval.

#### API and Recovery

- [x] P4-11: Implement the approval API, accepting only `approve/reject` and no replacement arguments.
- [x] P4-12: Implement `pending_approval` in the Run View.
- [x] P4-13: Implement 409 responses for repeated approvals and invalid states.
- [x] P4-14: Ensure that only one operation succeeds when approval and rejection occur concurrently.
- [x] P4-15: Preserve Runs awaiting approval at startup and mark orphaned RUNNING as `RUN_INTERRUPTED`.
- [x] P4-16: Verify that the model, tools, and backoff after approval remain subject to the remaining step and active budgets, and that waiting time is not counted toward the budget.

### 11.4 Phase Outputs

- The third Mock tool and incident database.
- A complete approval API and Run pause/recovery.
- Idempotency, concurrency, and restart safety tests.

### 11.5 Phase Tests

- Without approval, the Handler is called zero times and the Incident table is empty.
- Approval creates exactly one incident, and the execution arguments are exactly the same as when approval was requested.
- Additional tool arguments submitted by the client return 422 and do not affect the original arguments.
- After rejection, the Handler is not called, no incident is created, and the Run is `COMPLETED`.
- Repeated approval, repeated rejection, and rejection after approval all return 409.
- Only one of concurrent approval and rejection succeeds, and the final state is unique.
- Retrying with the same idempotency key returns the original incident; using the same key with different arguments fails without overwriting the original record.
- After a service restart, the pending approval arguments and incomplete Round can be recovered and execution can continue.
- Orphaned RUNNING does not automatically replay the incident tool and is marked `RUN_INTERRUPTED` at startup.
- Time spent waiting for approval does not increase `active_elapsed_ms`.
- Boundaries such as approval at step 20 and requiring step 21 after approval conform to the technical contract.

### 11.6 Exit Criteria

- [x] All incident side effects occur only after server-side approval.
- [x] Original-argument binding, conditional updates, and idempotency uniqueness constraints are all effective simultaneously.
- [x] Concurrency, repeated-request, and restart recovery tests pass.
- [x] No background execution occurs during approval, and all budget limits still apply after approval.
- [x] All P1–P4 tests pass.

Status: **Completed**.

## 12. P5: Context Compact

### 12.1 Goal

Implement precise 15/10/5 context compaction on top of complete approval and multi-step ReAct, ensuring bounded model context while retaining the complete audit history.

### 12.2 Dependencies

- P4 Round, approval recovery, and transaction behavior is stable.
- Fake Kimi can distinguish normal decision requests from summarization requests.
- A test script can construct at least 16 normal ReAct Rounds and independently inject malformed responses to distinguish Steps from Rounds.

### 12.3 Work Items

- [x] P5-1: Implement deterministic conversion from Rounds to summarization input.
- [x] P5-2: Implement the Compact Prompt combining the previous summary + earliest 10 rounds, and validate that summaries are nonempty.
- [x] P5-3: Trigger Compact when 15 completed Rounds already exist and another decision is still required, rather than performing pointless compaction after Round 15 terminates.
- [x] P5-4: A successful Compact transaction updates the summary, Round snapshot, and events together.
- [x] P5-5: Exactly the 5 most recent rounds are retained when Compact is committed, and the next decision is initiated afterward.
- [x] P5-6: Construct the summary message as context data without elevating it to an instruction that can override the System Prompt.
- [x] P5-7: Compact reuses the LLM timeout, retries, backoff, and remaining-time budget but does not increase the Step count.
- [x] P5-8: Compact failure leaves the previous summary and rounds unchanged; active-time exhaustion leads to the limit terminal state, while other permanent failures lead to `CONTEXT_COMPACTION_FAILED`.
- [x] P5-9: Ensure that Compact does not delete or rewrite any Run Event, Approval, or Incident.
- [x] P5-10: Record the compacted round range, summary length, elapsed time, and Attempt count.

### 12.4 Phase Outputs

- A persistable summary and bounded model context.
- A precise compaction path before the 16th normal Round is produced.
- Compact success, failure, and information-integrity tests.

### 12.5 Phase Tests

- Compact does not run early when there are fewer than 15 completed Rounds; it triggers exactly once when 15 completed Rounds exist and the 16th normal Round is about to be produced.
- The previous summary and earliest 10 rounds enter the summarization request, and only the 5 most recent rounds are retained after Compact is committed.
- The next decision request contains the new summary and 5 uncompressed Rounds; after the 16th normal Round is completed, the snapshot contains 6 rounds.
- Malformed responses increase Steps but not Rounds and do not incorrectly trigger Compact early.
- Summarization requests carry no tools and do not increase `step_count`.
- Tool arguments, key observations, errors, approval facts, and incident IDs are retained according to the summary contract.
- The content and order of `run_events` existing before Compact remain unchanged; only Compact, retry, and terminal-state events are appended.
- Compact Timeout/5xx responses are retried, and the previous context remains completely unchanged upon final failure.
- Empty summaries and persistence failures do not cause silent history loss.
- Compact consumes active time, while waiting for approval still does not.
- A long Run containing an approval pause/recovery still compacts correctly after recovery.

### 12.6 Exit Criteria

- [x] All tests for the 15/10/5 rule, trigger point, and counting pass.
- [x] The model context is bounded, and the complete history is always queryable.
- [x] Compact failure does not damage the previous context or bypass the budget.
- [x] All P1–P5 tests pass.

Status: **Completed**.

## 13. P6: End-to-End Delivery and Release

### 13.1 Goal

Organize the implementation that has passed phased acceptance into a complete submission package that others can install, run, test, and review, and complete final end-to-end regression.

### 13.2 Dependencies

- All P1–P5 exit criteria are satisfied.
- All core functional and safety tests have passed.

### 13.3 Work Items

#### End-to-End and Quality

- [x] P6-1: Execute the complete "knowledge base retrieval → service diagnosis → incident request → approval → final answer" scenario.
- [x] P6-2: Execute complete scenarios for incident rejection, persistent tool failure, malformed model responses, step limits, time limits, and Compact failure.
- [x] P6-3: Check state transitions, event Sequences, log correlation fields, and sensitive-information sanitization.
- [x] P6-4: Run all default tests and confirm no external network access, real waiting, or test-order dependencies.
- [x] P6-5: Run the optional Kimi Smoke Test with explicit credentials and marker, record model configuration compatibility, and do not save secrets.
- [x] P6-6: Check project dependencies, the Python version, imports, and startup with a fresh temporary database.

#### Documentation and Examples

- [x] P6-7: Complete the README's architecture summary, directory overview, installation, configuration, startup, testing, and API examples.
- [x] P6-8: Complete the README's approval workflow, Trace queries, troubleshooting, and security notes.
- [x] P6-9: Complete `.env.example`, consistent with all variables in the technical design and containing no real secrets.
- [x] P6-10: Complete the Postman Collection and environment variables, covering health, creation, queries, Trace, approval, and rejection.
- [x] P6-11: Configure example variables for Run ID and Approval ID and the necessary Tests in Postman to avoid manual copying alone.
- [x] P6-12: Reconcile the technical design, Roadmap, and actual implementation; if a legitimate design change has occurred, update the contract before completing acceptance.
- [x] P6-13: Verify limitations, future improvements, the Mock Dataset publication method, and submission notes.

#### Release Checks

- [x] P6-14: Confirm that `.env`, databases, caches, logs, and local secrets have not entered the submission content.
- [x] P6-15: Reproduce installation, startup, and testing from a clean environment according to the README.
- [x] P6-16: Check every item in the Section 17 release checklist with sign-off rigor, without skipping failed items.

### 13.4 Phase Outputs

- Complete submittable code and Mock Dataset.
- The full automated test suite and optional Smoke Test.
- README, environment variable template, Postman Collection, and consistent design documents.
- Reproducible release acceptance results.

### 13.5 Phase Tests and Acceptance

- The default `pytest` suite passes completely.
- Every command in the README is executable in a clean environment.
- Postman can independently complete both the approval and rejection workflows.
- A real Kimi Smoke is not required for default test success, but configuration errors are clearly explained.
- All delivery outputs map to specific files and verification results.
- The local submission candidate contains no real credentials, local databases, or meaningless temporary files.

### 13.6 Exit Criteria

- [x] Functionality, reliability, security, observability, testing, and documentation all satisfy the design contract.
- [x] All phases remain Completed, with no known unaddressed high-priority defects.
- [x] A third party can run the deliverables using only the repository instructions.
- [x] Every item in the release checklist is checked.

Status: **Completed** (local delivery and release acceptance completed; GitHub remote publication not performed).

## 14. Capability Traceability Matrix

| Capability | Implementation Phase | Primary Test or Acceptance Evidence | Current Status |
| --- | --- | --- | --- |
| API receives user goals | P1 | Run creation, goal validation, and 404 tests | Completed |
| LLM–tool execution loop | P1, P2 | Direct answers and multi-step knowledge base/service diagnosis | Completed |
| Three Mock tools | P2, P4 | Tests for two read-only tools and incident approval tests | Completed |
| Tool input validation | P2, P4 | Blank values, types, lengths, enumerations, and additional fields | Completed |
| Tool output validation | P2, P3, P4 | Invalid output, 16 KiB limit, and incident output | Completed |
| Agent state persistence | P1, P4 | Run queries, SQLite restart, and recovery while awaiting approval | Completed |
| Complete execution history | P1, P2, P5 | Trace ordering and unchanged pre-Compact events | Completed |
| Tool errors and timeouts | P3 | Fake Handler transient/persistent failures | Completed |
| Model errors and timeouts | P3 | Timeout, network, 429, 5xx, and 401/400 | Completed |
| Bounded retries and backoff | P3 | Attempt counts, Jitter, Retry-After, and budget capping | Completed |
| Malformed LLM responses | P2, P3 | Empty responses, multiple calls, unknown tools, invalid JSON, and duplicate IDs | Completed |
| Prevention of infinite loops | P1, P3 | Steps 20/21, 120 seconds, and correction and retry limits | Completed |
| User approval before incident invocation | P4 | Zero side effects without approval, approval, and rejection | Completed |
| Approval concurrency and idempotency | P4 | Repeated approvals, races, same-key retries, and conflicts | Completed |
| Recovery while awaiting approval | P4 | Recovery by a new App instance and execution with the original arguments | Completed |
| Context Compact | P5 | Before the 16th normal Round, the earliest 10 rounds, and retention of 5 rounds | Completed |
| Logs or Trace | P1, P3, P4, P5, P6 | JSON fields, Run Events, sanitization, and the multi-turn conversation example | Completed |
| Successful execution tests | P1, P2, P4, P6 | Direct answers, read-only and multi-step execution, and end-to-end approval | Completed |
| Tool failure tests | P3, P6 | Success after retry, persistent failure, and invalid output | Completed |
| Approval tests | P4, P6 | Approval, rejection, repetition, concurrency, and restart | Completed |
| Execution limit tests | P3, P6 | Steps, active time, and format correction limits | Completed |
| Clear operating and testing instructions | P6 | Reproduction from a clean environment according to the README | Completed |
| Database and technology stack documentation | P0, P6 | Reconciliation of the technical design with actual implementation | Completed |
| Environment variable inventory | P0, P1, P6 | Consistency among the technical design, `.env.example`, and README | Completed |
| Postman Collection | P6 | Creation, queries, Trace, approval, rejection, and health | Completed |
| Mock Dataset publication | P2, P6 | JSON included in the local submission candidate and loading tests | Completed |
| Current limitations | P0, P6 | Documentation consistent with actual behavior | Completed |
| Future improvements | P0, P6 | Improvement items are not mixed into the current scope | Completed |

This matrix's status must be updated whenever a phase is completed; do not modify only the phase heading while leaving traceability information outdated.

## 15. Testing and Quality Strategy

### 15.1 Test Levels

| Level | Focus | Prohibited Dependencies |
| --- | --- | --- |
| Unit tests | Schemas, tools, retries, state, Compact, idempotency, and safety boundaries | Real Kimi, real Sleep, shared databases |
| API integration tests | HTTP contracts, Runs, Trace, approvals, error codes, and database persistence | External services |
| Concurrency tests | Approval/rejection races, repeated approvals, conditional updates, and unique constraints | Reliance solely on application-layer locks |
| Restart tests | New App/Repository instances loading Runs awaiting approval and handling orphaned RUNNING | In-memory state |
| End-to-end tests | Multiple tools, multiple steps, approvals, Compact, and final answers | Unbounded free-running execution |
| Optional Smoke | Real Kimi authentication, model name, and protocol compatibility | Fixed free-text assertions |

### 15.2 Test Infrastructure Requirements

- Fake Kimi supports returning responses or exceptions in sequence and records input Messages.
- Fake Clock controls Monotonic and UTC separately and can verify that waiting for approval is not timed.
- Fake Sleeper records every backoff without real waiting.
- Fake Random provides deterministic Jitter.
- Fake Tool supports failure on the Nth attempt, persistent failure, Timeout, and invalid output.
- Each test uses an independent temporary SQLite database with the same constraints enabled.
- Tests do not depend on execution order and do not share mutable Fake queues.

### 15.3 Safety Regressions That Must Always Be Maintained

- The number of unapproved incident Handler calls must be 0.
- Repeated approvals must not increase the Incident count.
- Clients must not be able to replace approved arguments.
- Tool results must not bypass the allowlist or Schema.
- Logs, Traces, summaries, and API errors must not contain secrets.
- Step, time, retry, and format correction limits must not be disabled by model output.
- Compact must not delete the complete history.

## 16. Definition of Done

### 16.1 Every Work Item

A work item can be checked only when all of the following conditions are satisfied:

- The implementation conforms to the technical contract and does not substitute Prompts for server-side controls.
- Corresponding success and failure tests have been added and pass.
- Relevant state, events, and errors are observable.
- It contains no real secrets, real waiting, or implicit external dependencies.
- It does not affect tests from previous phases.
- Necessary documentation and examples have been updated accordingly.

### 16.2 Every Phase

- All phase work items are complete.
- All phase tests and exit criteria are satisfied.
- Newly identified limitations are recorded rather than hidden under "handle later."
- The phase status and traceability matrix have been updated.
- No safety or data integrity checks were skipped to advance to the next phase.

### 16.3 Final Version

- All functionality, testing, and delivery criteria are satisfied.
- The default tests pass on the first run from a clean environment.
- The README, environment variables, Postman, Dataset, and code are consistent.
- The technical design, Roadmap, and actual behavior are consistent.
- Current limitations are truthful and explicit, and no unimplemented capability is described as supported.
- The release checklist is fully complete.

## 17. Release Acceptance Checklist

### 17.1 Functionality

- [x] User goals of 1–4000 characters can be submitted through the API.
- [x] Kimi can answer directly and perform multi-step ReAct based on tool observations.
- [x] All three Mock tools—knowledge base, service status, and incident—are available.
- [x] Tool inputs, outputs, and model actions are all validated.
- [x] Run state and the complete Trace are persistable and queryable.
- [x] Timeouts, retries, backoff, and malformed responses are all handled according to the contract.
- [x] Maximum step, active-time, and format correction limits are effective.
- [x] No incident is created without approval; approval, rejection, recovery, and idempotency are effective.
- [x] The 15/10/5 Compact before the 16th normal Round is effective, and no complete history is lost.

### 17.2 Testing and Quality

- [x] Successful execution, tool failures, approvals, and execution limits all have automated tests.
- [x] Model failures, invalid outputs, repeated approvals, concurrency, and restarts all have tests.
- [x] Default tests do not access real Kimi or perform real Sleeps.
- [x] All tests pass, with no cases dependent on execution order.
- [x] JSON logs and Traces can be correlated and contain no sensitive information.

### 17.3 Documentation and Submission Artifacts

- [x] The README covers architecture, installation, configuration, startup, testing, API, and approvals.
- [x] `.env.example` lists all environment variables and contains no real values.
- [x] The Postman Collection covers Runs, Trace, approval, rejection, and health checks.
- [x] The Mock Dataset is included in the local submission candidate and can be published with the GitHub repository; no Hugging Face link is provided.
- [x] The documentation includes the database design, technology stack, current limitations, and future improvements.
- [x] The technical design and Roadmap are independently complete and consistent with the implementation.
- [x] The code has been organized into a local candidate that can be submitted through a GitHub repository link; remote publication has not yet been performed.
- [x] The submission candidate contains no `.env`, local databases, caches, temporary debugging artifacts, or secrets.

Checking items in this section indicates that local delivery and release-readiness checks have passed; it does not mean that a GitHub repository has been created, a commit has been generated, or content has been uploaded remotely. P6 can be marked Completed only when all of the above items pass.

## 18. Risks and Mitigations

| Risk | Early Warning Sign | Prevention/Mitigation | Corresponding Phase |
| --- | --- | --- | --- |
| Kimi tool-call compatibility differences | SDK response structures or argument shapes differ from assumptions | Isolate the protocol in the Adapter; use Smoke only to verify the protocol; the server still validates the single-call constraint | P1–P3, P6 |
| Retries bypass the total time limit | Multiple Attempts/Sleeps take far longer than 120 seconds | Cap the remaining budget before every timeout and Sleep; regress with Fake Clock | P3 |
| Infinite correction of malformed responses | Consecutive empty responses continue to trigger requests | Correct the first occurrence and fail on the second; malformed responses also count as steps | P3 |
| Tool results pollute the context | Invalid fields or oversized results enter Messages | Output Schema and a hard 16 KiB boundary; do not append observations before failure | P2–P3 |
| Duplicate incident side effects | Repeated approvals or timeout retries create multiple incidents | Conditional updates, argument hashes, idempotency keys, and unique constraints | P4 |
| Approved arguments are replaced | The client or model submits a new description/severity | Approval requests accept only a Decision, and Canonical JSON is reread at execution time | P4 |
| Side-effect state is unknown after a crash | The process exits between Insert and Tool Result | No automatic replay; idempotency keys assist reconciliation; documentation explicitly describes the window | P4, P6 |
| Multiple instances misuse recovery logic | Two processes mark each other's RUNNING state or advance Runs | Restrict to a single Worker; startup recovery is only for the single-instance POC | P1, P4 |
| Compact information loss | The summary omits approvals, incident IDs, or errors | Summary content contract and targeted tests; the database remains the source of truth | P5 |
| Compact counting ambiguity | Counts differ from expectations at the 16th Round or when malformed responses are included | Trigger by completed Round count; test 5 rounds at commit, 6 after appending, and Step/Round differences | P5 |
| Unstable or slow tests | Real Sleep, real APIs, or random Jitter causes fluctuations | Fake Clock/Sleeper/Random/Kimi, with Smoke explicitly isolated | All phases |
| Secret leakage | SDK objects, Headers, or environment variables enter logs | Standardized sanitization, controlled errors, repository scanning, and manual inspection | All phases |
| Documentation and code drift | API fields, default values, or state semantics are inconsistent | Design change rules, pre-release contract reconciliation, and Postman verification | P6 |
| POC scope creep | Queues, permissions, or vector stores are introduced while core test coverage is insufficient | Adhere to non-goals and place production capabilities in future improvements | All phases |

## 19. Change Control and Status Maintenance

### 19.1 Which Changes Constitute Design Changes

The following changes cannot be treated directly as localized code modifications:

- Default budgets, retry counts, steps, or format correction semantics.
- The Compact window, trigger point, retained round count, or failure policy.
- Run/Approval states and transitions.
- Approval argument binding, idempotency keys, or incident side-effect boundaries.
- API paths, request/response fields, and error codes.
- Database fields, constraints, and transaction boundaries.
- Service restart recovery commitments.

### 19.2 Change Procedure

1. Explain the reason for the change, its impact, and alternatives.
2. Update the authoritative contract in the technical design first.
3. Update the Roadmap's work items, traceability matrix, and acceptance criteria.
4. Update or add tests that demonstrate the change.
5. Implement the code and run full regression.
6. Update affected deliverables such as the README, `.env.example`, and Postman.

### 19.3 Status Update Rules

- A phase can change from `In Progress` to `Completed` only when all exit criteria are satisfied.
- When a defect is found in a previous phase, mark that phase as `In Progress` again or record `Blocked`; do not retain a false completed status.
- Checking a work item must be supported by code, tests, or inspectable documentation as evidence.
- Whenever a phase is completed, update the Section 6 overview and Section 14 traceability matrix at the same time.
- Do not substitute commit counts or lines of code for acceptance results.

## 20. Current Limitations and Future Roadmap

### 20.1 Limitations Accepted in This Iteration and Requiring Explicit Disclosure

- The round window is not Token-aware and cannot precisely control model Token costs.
- Single-process synchronous HTTP execution is unsuitable for horizontal scaling of long-running tasks.
- SQLite is suitable for a local POC but not for high-concurrency production writes.
- Only Runs awaiting approval are recovered; resuming from an arbitrary crash point is not supported.
- A small crash window requiring manual reconciliation still exists between the incident write and observation persistence.
- Knowledge base retrieval uses keyword matching, and its effectiveness depends on data quality.
- The incident platform is a local Mock and does not represent real third-party transaction and idempotency protocols.
- The API has no authentication or authorization and can only be bound to a trusted local environment.
- Compact failure terminates the Run, with no local summarization fallback.
- Model selection and final text are nondeterministic, and Harness tests are not equivalent to model quality evaluation.

### 20.2 Ordered Improvements After This Iteration

1. Token-aware context budgets, summary evaluation, and local summarization fallback.
2. Asynchronous Runs, background tasks, polling/Webhooks, and cancellable execution.
3. PostgreSQL, Worker Leases, distributed locks, and complete crash recovery.
4. Incident platform reconciliation tasks and real third-party idempotency protocols.
5. OpenTelemetry, Metrics, alerts, and cost observability.
6. User authentication, approval authorization, approver identity, and auditing.
7. Vector/hybrid retrieval and an offline operations evaluation set.
8. Integration with real monitoring, knowledge base, and incident systems.

These improvements are not part of the completion criteria for the current P1–P6, preventing production-oriented expansion from replacing the core Harness capabilities required by this task.

## 21. Current Progress and Next Steps

Local implementation and delivery acceptance for P0–P6 are now complete. The basic API, SQLite Run/Trace, Kimi Adapter, three Mock tools, bidirectional Schema validation, multi-step ReAct, model/tool reliability, execution budgets, approval pause/recovery, original-argument binding, concurrency safety, incident idempotency, restart recovery, and 15/10/5 Compact have all been implemented. The default automated test result is **179 passed, 1 deselected**, using Fake Kimi/Clock/Sleeper, with an autouse fixture blocking real network connections and real backoff waits; the explicit real-Kimi marker Smoke runs separately. Real `kimi-k2.6` has verified direct answers, read-only multi-step ReAct, and the complete workflow of knowledge base retrieval, service diagnosis, incident approval, and post-approval execution. Newman executed 10 requests and 10 assertions against the real API, with no failures in either the approval or rejection workflow. A clean Python 3.11.5 environment completed editable installation, dependency checks, default tests, real Uvicorn startup, and fresh SQLite table creation verification. The submission candidate excludes real credentials, local databases, logs, and caches.

There are no new local implementation phases after P6. The only remaining work is optional external publication: create a GitHub repository, generate a Git commit, upload the content, and provide the repository link; these operations were not performed in the current environment and cannot be considered complete. See Section 20 for runtime limitations and subsequent feature improvements.
