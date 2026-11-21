import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlalchemy import update

from agent_harness.agent import AgentRunner
from agent_harness.llm import KimiAdapter, ModelResponse, RawToolCall, SummaryResponse
from agent_harness.schemas import RunStatus
from agent_harness.store import Database, RunRecord, RunRepository
from agent_harness.tools import TemporaryToolError
from tests.fakes.fake_kimi import FakeKimi


@pytest.mark.asyncio
async def test_fake_clock_drives_active_time_limit_without_real_waiting(
    settings, clock, sleeper, tool_registry
):
    settings.agent_active_time_limit_seconds = 1
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    fake_kimi = FakeKimi(
        [ModelResponse(content="This answer arrived too late.")],
        clock=clock,
        advance_per_call=2,
    )
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Check a slow model response.")

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
        assert result.step_count == 1
        assert result.active_elapsed_ms >= 2000
        assert fake_kimi.calls[0]["timeout_budget"] == 1
        events = repository.list_events(result.id)
        assert [event.event_type for event in events] == [
            "run_created",
            "llm_response",
            "run_state_changed",
            "run_finished",
        ]
        assert sleeper.calls == []
    finally:
        database.close()


@pytest.mark.asyncio
async def test_step_limit_is_enforced_before_another_model_call(
    settings, clock, sleeper, tool_registry
):
    settings.agent_max_steps = 1
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    run = repository.create_run("Check step enforcement.", limits={})
    with database.sessions.begin() as session:
        session.execute(update(RunRecord).where(RunRecord.id == run.id).values(step_count=1))
    fake_kimi = FakeKimi([ModelResponse(content="This call must not happen.")])
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.run(run.id)

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "MAX_STEPS_EXCEEDED"
        assert result.step_count == 1
        assert fake_kimi.calls == []
        assert [event.event_type for event in repository.list_events(run.id)] == [
            "run_created",
            "run_state_changed",
            "run_finished",
        ]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_final_answer_on_last_allowed_step_completes(
    settings, clock, sleeper, tool_registry
):
    settings.agent_max_steps = 1
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    fake_kimi = FakeKimi([ModelResponse(content="Final answer within the step budget.")])
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Answer within one step.")

        assert result.status == RunStatus.COMPLETED
        assert result.step_count == 1
        assert result.final_answer == "Final answer within the step budget."
        assert len(fake_kimi.calls) == 1
    finally:
        database.close()


def tool_decision(index):
    return ModelResponse(
        tool_calls=(
            RawToolCall(
                id=f"call-{index}",
                name="get_service_status",
                arguments='{"service_name":"checkout"}',
            ),
        )
    )


@pytest.mark.asyncio
async def test_twentieth_step_can_finish_successfully(
    settings, clock, sleeper, tool_registry
):
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    fake_kimi = FakeKimi(
        [*[tool_decision(index) for index in range(1, 20)], ModelResponse(content="Done at 20.")],
        summary_responses=[SummaryResponse(content="Compacted context.")],
    )
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Use the complete step budget.")

        assert result.status == RunStatus.COMPLETED
        assert result.step_count == 20
        assert len(fake_kimi.calls) == 20
        assert len([
            event for event in repository.list_events(result.id)
            if event.event_type == "tool_result"
        ]) == 19
    finally:
        database.close()


@pytest.mark.asyncio
async def test_twentieth_tool_finishes_before_twenty_first_decision_is_rejected(
    settings, clock, sleeper, tool_registry
):
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    fake_kimi = FakeKimi(
        [tool_decision(index) for index in range(1, 21)],
        summary_responses=[SummaryResponse(content="Compacted context.")],
    )
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Stop before decision twenty-one.")

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "MAX_STEPS_EXCEEDED"
        assert result.step_count == 20
        assert len(fake_kimi.calls) == 20
        events = repository.list_events(result.id)
        assert len([event for event in events if event.event_type == "tool_result"]) == 20
        assert events[-1].event_type == "run_finished"
        persisted_rounds = json.loads(repository.get_run(result.id).recent_rounds_json)
        assert len(persisted_rounds) == 10
        assert [round_record["step"] for round_record in persisted_rounds] == list(range(11, 21))
        assert persisted_rounds[-1]["observation"]["ok"] is True
    finally:
        database.close()


@pytest.mark.asyncio
async def test_runner_persists_model_retry_event_and_attempt_count(
    settings, clock, sleeper, tool_registry
):
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)

    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            message = SimpleNamespace(content="Recovered answer.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FlakyCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = KimiAdapter(
        api_key="unused",
        model="fake-model",
        base_url="https://api.moonshot.cn/v1",
        client=client,
        clock=clock,
        sleeper=sleeper,
        random_source=lambda: 0.5,
    )
    runner = AgentRunner(
        repository,
        llm,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Recover a model timeout.")

        assert result.status == RunStatus.COMPLETED
        assert result.step_count == 1
        assert result.active_elapsed_ms == 250
        assert completions.calls == 2
        events = repository.list_events(result.id)
        assert [event.event_type for event in events] == [
            "run_created",
            "retry_scheduled",
            "llm_response",
            "run_state_changed",
            "run_finished",
        ]
        retry = json.loads(events[1].payload_json)
        assert retry["component"] == "llm"
        assert retry["error_code"] == "LLM_TIMEOUT"
        assert retry["delay_seconds"] == 0.25
        assert json.loads(events[2].payload_json)["attempt"] == 2
    finally:
        database.close()


@pytest.mark.asyncio
async def test_model_backoff_beyond_active_budget_finishes_limit_without_waiting(
    settings, clock, sleeper, tool_registry
):
    settings.agent_active_time_limit_seconds = 1
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)

    class TimeoutCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise TimeoutError("temporary timeout")

    completions = TimeoutCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = KimiAdapter(
        api_key="unused",
        model="fake-model",
        base_url="https://api.moonshot.cn/v1",
        client=client,
        retry_backoff_base_seconds=2,
        retry_backoff_max_seconds=2,
        clock=clock,
        sleeper=sleeper,
        random_source=lambda: 1.0,
    )
    runner = AgentRunner(
        repository,
        llm,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Bound model backoff.")

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
        assert result.step_count == 0
        assert completions.calls == 1
        assert sleeper.calls == []
        events = repository.list_events(result.id)
        assert [event.event_type for event in events] == [
            "run_created",
            "llm_error",
            "run_state_changed",
            "run_finished",
        ]
        assert json.loads(events[1].payload_json)["error_code"] == "LLM_TIMEOUT"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_tool_timeout_that_exhausts_active_budget_preserves_original_error(
    settings, clock, sleeper, tool_registry
):
    settings.agent_active_time_limit_seconds = 1
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    definition = tool_registry._definitions["get_service_status"]
    calls = 0

    async def slow_timeout(typed_input, context):
        nonlocal calls
        calls += 1
        clock.advance(2)
        raise TimeoutError("sensitive timeout detail")

    tool_registry._definitions["get_service_status"] = replace(
        definition, handler=slow_timeout
    )
    fake_kimi = FakeKimi([tool_decision(1)])
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        result = await runner.create_and_run("Bound a slow tool.")

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
        assert result.step_count == 1
        assert calls == 1
        assert len(fake_kimi.calls) == 1
        events = repository.list_events(result.id)
        assert [event.event_type for event in events] == [
            "run_created",
            "llm_response",
            "tool_call",
            "tool_error",
            "run_state_changed",
            "run_finished",
        ]
        tool_error = json.loads(events[3].payload_json)
        assert tool_error["error"]["code"] == "TOOL_TIMEOUT"
        assert "sensitive" not in tool_error["error"]["message"]
        persisted_rounds = json.loads(repository.get_run(result.id).recent_rounds_json)
        assert persisted_rounds[-1]["observation"] is None
    finally:
        database.close()


@pytest.mark.asyncio
async def test_tool_backoff_beyond_active_budget_does_not_sleep(
    settings, clock, sleeper, tool_registry
):
    settings.agent_active_time_limit_seconds = 1
    settings.retry_backoff_base_seconds = 2
    settings.retry_backoff_max_seconds = 2
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    definition = tool_registry._definitions["get_service_status"]

    async def temporary_failure(typed_input, context):
        raise TemporaryToolError("temporary failure")

    tool_registry._definitions["get_service_status"] = replace(
        definition, handler=temporary_failure
    )
    fake_kimi = FakeKimi([tool_decision(1)])
    runner = AgentRunner(
        repository,
        fake_kimi,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
        random_source=lambda: 1.0,
    )
    try:
        result = await runner.create_and_run("Bound tool backoff.")

        assert result.status == RunStatus.LIMIT_EXCEEDED
        assert result.error.code == "ACTIVE_TIME_LIMIT_EXCEEDED"
        assert sleeper.calls == []
        events = repository.list_events(result.id)
        assert "retry_scheduled" not in [event.event_type for event in events]
        tool_error = next(event for event in events if event.event_type == "tool_error")
        assert json.loads(tool_error.payload_json)["error"]["code"] == "TOOL_EXECUTION_FAILED"
    finally:
        database.close()
