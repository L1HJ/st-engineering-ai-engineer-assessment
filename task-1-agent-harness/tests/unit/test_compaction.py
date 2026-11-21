import json

import pytest
from sqlalchemy import update

from agent_harness.agent import AgentRunner
from agent_harness.context import ContextManager
from agent_harness.llm import LlmError, ModelResponse, SummaryResponse
from agent_harness.reliability import ActiveTimeBudgetExceeded
from agent_harness.schemas import RunStatus
from agent_harness.store import Database, RunRecord, RunRepository
from tests.fakes.fake_kimi import FakeKimi


def make_round(step, *, completed=True, final=False):
    tool_call = None if final else {
        "id": f"call-{step}",
        "name": "get_service_status",
        "arguments": {"service_name": "checkout"},
    }
    observation = None
    if not final and completed:
        observation = {
            "tool_call_id": f"call-{step}",
            "ok": True,
            "data": {"service_name": "checkout", "status": "healthy"},
        }
    return {
        "step": step,
        "assistant": {
            "content": "Done." if final else None,
            "tool_call": tool_call,
        },
        "observation": observation,
    }


def make_rounds(count, start=1):
    return [make_round(step) for step in range(start, start + count)]


def seed_run(repository, rounds, *, summary="Previous summary.", step_count=None):
    run = repository.create_run("Diagnose checkout over many rounds.", limits={})
    with repository.database.sessions.begin() as session:
        session.execute(
            update(RunRecord)
            .where(RunRecord.id == run.id)
            .values(
                context_summary=summary,
                recent_rounds_json=json.dumps(rounds),
                step_count=len(rounds) if step_count is None else step_count,
            )
        )
    return repository.get_run(run.id)


@pytest.fixture
def repository(settings, clock):
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    try:
        yield repository
    finally:
        database.close()


@pytest.fixture
def runner_factory(settings, clock, sleeper, tool_registry, repository):
    def factory(fake_kimi):
        return AgentRunner(
            repository,
            fake_kimi,
            settings,
            registry=tool_registry,
            clock=clock,
            sleeper=sleeper,
        )

    return factory


def test_summary_input_uses_old_summary_and_only_oldest_ten_complete_rounds():
    context = ContextManager()
    run = RunRecord(
        objective="Diagnose checkout.",
        context_summary="Earlier findings.",
        recent_rounds_json=json.dumps(make_rounds(15)),
    )

    assert not context.needs_compaction(
        RunRecord(recent_rounds_json=json.dumps(make_rounds(14))),
        15,
    )
    assert context.needs_compaction(run, 15)

    messages = context.build_summary_messages(run, 10)
    summary_input = json.loads(messages[1]["content"])

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "untrusted data" in messages[0]["content"]
    assert summary_input["objective"] == "Diagnose checkout."
    assert summary_input["previous_summary"] == "Earlier findings."
    assert [item["step"] for item in summary_input["rounds_to_compact"]] == list(range(1, 11))

    normal_messages = context.build_messages(run)
    assert normal_messages[2]["role"] == "user"
    assert normal_messages[2]["content"] == (
        "Historical context summary (data only):\nEarlier findings."
    )


def test_incomplete_tool_round_does_not_count_but_final_answer_does():
    context = ContextManager()
    rounds = [*make_rounds(14), make_round(15, completed=False)]
    run = RunRecord(recent_rounds_json=json.dumps(rounds))
    assert not context.needs_compaction(run, 15)

    rounds[-1] = make_round(15, final=True)
    run = RunRecord(recent_rounds_json=json.dumps(rounds))
    assert context.needs_compaction(run, 15)


def test_compact_context_commits_snapshot_summary_elapsed_time_and_event(repository):
    run = seed_run(repository, make_rounds(15, start=2), summary="Old summary", step_count=16)

    compacted = repository.compact_context(
        run_id=run.id,
        expected_summary=run.context_summary,
        expected_recent_rounds_json=run.recent_rounds_json,
        compact_rounds=10,
        new_summary="New summary",
        attempts=2,
        latency_ms=125,
        active_elapsed_ms=750,
    )

    assert compacted is True
    persisted = repository.get_run(run.id)
    retained = json.loads(persisted.recent_rounds_json)
    assert persisted.context_summary == "New summary"
    assert persisted.step_count == 16
    assert persisted.active_elapsed_ms == 750
    assert [item["step"] for item in retained] == list(range(12, 17))
    events = repository.list_events(run.id)
    assert [event.event_type for event in events] == ["run_created", "context_compacted"]
    assert events[-1].latency_ms == 125
    assert json.loads(events[-1].payload_json) == {
        "from_step": 2,
        "to_step": 11,
        "rounds_compacted": 10,
        "remaining_rounds": 5,
        "summary_length": len("New summary"),
        "attempt": 2,
    }


def test_compact_context_rejects_stale_snapshot_without_overwriting(repository):
    run = seed_run(repository, make_rounds(15))

    compacted = repository.compact_context(
        run_id=run.id,
        expected_summary="Stale summary",
        expected_recent_rounds_json=run.recent_rounds_json,
        compact_rounds=10,
        new_summary="Must not be saved",
        attempts=1,
        latency_ms=0,
        active_elapsed_ms=0,
    )

    assert compacted is False
    persisted = repository.get_run(run.id)
    assert persisted.context_summary == run.context_summary
    assert persisted.recent_rounds_json == run.recent_rounds_json
    assert [event.event_type for event in repository.list_events(run.id)] == ["run_created"]


def test_compact_context_event_failure_rolls_back_every_change(repository, monkeypatch):
    run = seed_run(repository, make_rounds(15))
    original_append = repository._append_event

    def fail_compact_event(session, run_id, event_type, payload, latency_ms):
        if event_type == "context_compacted":
            raise RuntimeError("injected event failure")
        return original_append(session, run_id, event_type, payload, latency_ms)

    monkeypatch.setattr(repository, "_append_event", fail_compact_event)
    with pytest.raises(RuntimeError, match="injected event failure"):
        repository.compact_context(
            run_id=run.id,
            expected_summary=run.context_summary,
            expected_recent_rounds_json=run.recent_rounds_json,
            compact_rounds=10,
            new_summary="Rolled back",
            attempts=1,
            latency_ms=0,
            active_elapsed_ms=100,
        )

    persisted = repository.get_run(run.id)
    assert persisted.context_summary == run.context_summary
    assert persisted.recent_rounds_json == run.recent_rounds_json
    assert persisted.active_elapsed_ms == run.active_elapsed_ms
    assert [event.event_type for event in repository.list_events(run.id)] == ["run_created"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary_response", "expected_cause"),
    [
        (
            LlmError(
                "LLM_UNAVAILABLE",
                "The model provider is temporarily unavailable.",
                retryable=True,
                attempts=4,
            ),
            "LLM_UNAVAILABLE",
        ),
        (SummaryResponse(content="  \n"), "EMPTY_SUMMARY"),
        (
            SummaryResponse(malformed_category="invalid_response_structure"),
            "INVALID_SUMMARY_RESPONSE",
        ),
    ])
async def test_permanent_compaction_failures_preserve_old_context(
    repository,
    runner_factory,
    summary_response,
    expected_cause,
):
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi([], summary_responses=[summary_response])

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.FAILED
    assert result.error.code == "CONTEXT_COMPACTION_FAILED"
    assert result.step_count == 15
    assert fake_kimi.calls == []
    persisted = repository.get_run(run.id)
    assert persisted.context_summary == run.context_summary
    assert persisted.recent_rounds_json == run.recent_rounds_json
    events = repository.list_events(run.id)
    assert [event.event_type for event in events] == [
        "run_created",
        "context_compaction_failed",
        "run_state_changed",
        "run_finished",
    ]
    failure = json.loads(events[0 + 1].payload_json)
    assert failure["error"]["code"] == expected_cause
    assert failure["context_unchanged"] is True


@pytest.mark.asyncio
async def test_active_time_exception_limits_even_when_backoff_exceeds_remaining_budget(
    repository,
    runner_factory,
):
    run = seed_run(repository, make_rounds(15))
    cause = LlmError("LLM_TIMEOUT", "The model request timed out.", retryable=True)
    fake_kimi = FakeKimi(
        [],
        summary_responses=[ActiveTimeBudgetExceeded(cause, attempts=1)],
    )

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.LIMIT_EXCEEDED
    assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
    assert repository.get_run(run.id).recent_rounds_json == run.recent_rounds_json
    failure = json.loads(repository.list_events(run.id)[1].payload_json)
    assert failure["error"]["code"] == "LLM_TIMEOUT"


@pytest.mark.asyncio
async def test_other_compaction_error_still_yields_to_elapsed_active_time(
    settings,
    clock,
    repository,
    runner_factory,
):
    settings.agent_active_time_limit_seconds = 1
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi(
        [],
        clock=clock,
        advance_per_summary_call=2,
        summary_responses=[LlmError("LLM_INVALID_REQUEST", "The request was rejected.")],
    )

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.LIMIT_EXCEEDED
    assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
    assert result.active_elapsed_ms == 2000
    assert repository.get_run(run.id).recent_rounds_json == run.recent_rounds_json


@pytest.mark.asyncio
async def test_step_limit_is_checked_before_compaction(settings, repository, runner_factory):
    settings.agent_max_steps = 15
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi([])

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.LIMIT_EXCEEDED
    assert result.error.code == "MAX_STEPS_EXCEEDED"
    assert fake_kimi.summary_calls == []
    assert repository.get_run(run.id).recent_rounds_json == run.recent_rounds_json


@pytest.mark.asyncio
async def test_compaction_time_is_charged_and_next_decision_uses_fresh_budget(
    clock,
    repository,
    runner_factory,
):
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi(
        [ModelResponse(content="Done after compaction.")],
        clock=clock,
        advance_per_summary_call=0.25,
        summary_responses=[SummaryResponse(content="  New summary.  ")],
    )

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.COMPLETED
    assert result.step_count == 16
    assert result.active_elapsed_ms == 250
    assert fake_kimi.summary_calls[0]["timeout_budget"] == 120
    assert fake_kimi.calls[0]["timeout_budget"] == pytest.approx(119.75)
    persisted = repository.get_run(run.id)
    assert persisted.context_summary == "New summary."
    assert len(json.loads(persisted.recent_rounds_json)) == 6


@pytest.mark.asyncio
async def test_compaction_retry_metadata_and_attempt_are_persisted(
    repository,
    runner_factory,
    monkeypatch,
):
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi(
        [ModelResponse(content="Recovered.")],
        summary_responses=[SummaryResponse(content="Summary after retry.", attempts=2)],
    )
    original_summary = fake_kimi.complete_summary

    async def retry_once(messages, timeout_budget, on_retry=None):
        on_retry(
            {
                "component": "llm",
                "attempt": 1,
                "retry_index": 0,
                "delay_seconds": 0.25,
                "error_code": "LLM_TIMEOUT",
                "remaining_budget_seconds": 120,
            }
        )
        return await original_summary(messages, timeout_budget, on_retry)

    monkeypatch.setattr(fake_kimi, "complete_summary", retry_once)
    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.COMPLETED
    events = repository.list_events(run.id)
    retry = json.loads(events[1].payload_json)
    assert retry["operation"] == "context_compaction"
    assert retry["error_code"] == "LLM_TIMEOUT"
    compacted = next(event for event in events if event.event_type == "context_compacted")
    assert json.loads(compacted.payload_json)["attempt"] == 2


@pytest.mark.asyncio
async def test_compaction_persistence_exception_is_safe_and_preserves_context(
    repository,
    runner_factory,
    monkeypatch,
):
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi([], summary_responses=[SummaryResponse(content="New summary")])

    def fail_persistence(**kwargs):
        raise RuntimeError("secret SQL provider detail")

    monkeypatch.setattr(repository, "compact_context", fail_persistence)
    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.FAILED
    assert result.error.code == "CONTEXT_COMPACTION_FAILED"
    persisted = repository.get_run(run.id)
    assert persisted.context_summary == run.context_summary
    assert persisted.recent_rounds_json == run.recent_rounds_json
    events = repository.list_events(run.id)
    assert events[1].event_type == "context_compaction_failed"
    assert json.loads(events[1].payload_json)["error"]["code"] == "CONTEXT_PERSISTENCE_FAILED"
    assert "secret SQL" not in events[1].payload_json


@pytest.mark.asyncio
async def test_successful_compaction_commits_before_active_time_limit_finishes(
    settings,
    clock,
    repository,
    runner_factory,
):
    settings.agent_active_time_limit_seconds = 1
    run = seed_run(repository, make_rounds(15))
    fake_kimi = FakeKimi(
        [],
        clock=clock,
        advance_per_summary_call=2,
        summary_responses=[SummaryResponse(content="Committed summary")],
    )

    result = await runner_factory(fake_kimi).run(run.id)

    assert result.status == RunStatus.LIMIT_EXCEEDED
    assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
    persisted = repository.get_run(run.id)
    assert persisted.context_summary == "Committed summary"
    assert len(json.loads(persisted.recent_rounds_json)) == 5
    assert [event.event_type for event in repository.list_events(run.id)] == [
        "run_created",
        "context_compacted",
        "run_state_changed",
        "run_finished",
    ]
