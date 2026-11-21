import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent_harness.llm import ModelResponse, RawToolCall
from agent_harness.main import create_app
from agent_harness.tools import TemporaryToolError
from tests.fakes.fake_kimi import FakeKimi


def tool_response(call_id, name, arguments, content=None, raw_arguments=None):
    return ModelResponse(
        content=content,
        tool_calls=(
            RawToolCall(
                id=call_id,
                name=name,
                arguments=raw_arguments if raw_arguments is not None else json.dumps(arguments),
            ),
        ),
    )


@pytest.fixture
def react_app_factory(settings, tool_registry, clock, sleeper):
    def factory(responses):
        fake_kimi = FakeKimi(responses)
        app = create_app(
            settings=settings,
            llm=fake_kimi,
            tool_registry=tool_registry,
            clock=clock,
            sleeper=sleeper,
        )
        return app, fake_kimi

    return factory


def test_multi_step_react_uses_observations_and_persists_consistent_rounds(react_app_factory):
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-status", "get_service_status", {"service_name": "checkout"}),
            tool_response("call-kb", "search_knowledge_base", {"query": "checkout errors"}),
            ModelResponse(content="Checkout is degraded because upstream payment errors are rising."),
        ]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Diagnose checkout errors."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    run = response.json()
    assert response.status_code == 201
    assert run["status"] == "COMPLETED"
    assert run["step_count"] == 3
    assert len(fake_kimi.calls) == 3
    assert all(len(call["tool_schemas"]) == 3 for call in fake_kimi.calls)

    second_request = fake_kimi.calls[1]["messages"]
    assert [message["role"] for message in second_request] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_request[2]["tool_calls"][0]["id"] == "call-status"
    first_observation = json.loads(second_request[3]["content"])
    assert first_observation["ok"] is True
    assert first_observation["data"]["status"] == "degraded"

    third_request = fake_kimi.calls[2]["messages"]
    assert [message["role"] for message in third_request] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    second_observation = json.loads(third_request[5]["content"])
    assert second_observation["data"]["results"][0]["id"] == "kb-001"

    assert [event["event_type"] for event in trace["events"]] == [
        "run_created",
        "llm_response",
        "tool_call",
        "tool_result",
        "llm_response",
        "tool_call",
        "tool_result",
        "llm_response",
        "run_state_changed",
        "run_finished",
    ]
    assert [event["sequence"] for event in trace["events"]] == list(range(1, 11))

    persisted = app.state.repository.get_run(run["id"])
    rounds = json.loads(persisted.recent_rounds_json)
    assert len(rounds) == 3
    assert rounds[0]["observation"]["data"]["status"] == "degraded"
    assert rounds[1]["observation"]["data"]["results"][0]["id"] == "kb-001"
    assert rounds[2]["assistant"]["content"] == run["final_answer"]
    tool_events = [event for event in trace["events"] if event["event_type"] == "tool_result"]
    assert tool_events[0]["payload"]["observation"] == rounds[0]["observation"]
    assert tool_events[1]["payload"]["observation"] == rounds[1]["observation"]


@pytest.mark.asyncio
async def test_action_and_unfinished_round_are_saved_before_handler_execution(
    react_app_factory, tool_registry, monkeypatch
):
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-status", "get_service_status", {"service_name": "checkout"}),
            ModelResponse(content="Done."),
        ]
    )
    original_execute = tool_registry.execute
    observed_states = []

    async def inspect_before_execution(name, typed_input, context, timeout_budget):
        run = app.state.repository.get_run(context.run_id)
        rounds = json.loads(run.recent_rounds_json)
        events = app.state.repository.list_events(context.run_id)
        observed_states.append(
            {
                "observation": rounds[-1]["observation"],
                "last_event_type": events[-1].event_type,
                "tool_call_id": rounds[-1]["assistant"]["tool_call"]["id"],
            }
        )
        return await original_execute(name, typed_input, context, timeout_budget)

    monkeypatch.setattr(tool_registry, "execute", inspect_before_execution)
    with TestClient(app):
        result = await app.state.runner.create_and_run("Verify action persistence.")

    assert result.status == "COMPLETED"
    assert observed_states == [
        {"observation": None, "last_event_type": "tool_call", "tool_call_id": "call-status"}
    ]
    assert len(fake_kimi.calls) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "checkout", "extra": True},
        {"query": 42},
    ])
def test_invalid_tool_input_never_executes_handler_and_becomes_an_observation(
    react_app_factory, tool_registry, monkeypatch, arguments
):
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-invalid", "search_knowledge_base", arguments),
            ModelResponse(content="The search arguments were invalid."),
        ]
    )
    execute = AsyncMock(wraps=tool_registry.execute)
    monkeypatch.setattr(tool_registry, "execute", execute)

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Search operational docs."})

    assert response.json()["status"] == "COMPLETED"
    assert execute.await_count == 0
    next_messages = fake_kimi.calls[1]["messages"]
    observation = json.loads(next_messages[-1]["content"])
    assert observation["ok"] is False
    assert observation["error"]["code"] == "TOOL_INPUT_VALIDATION_ERROR"


@pytest.mark.parametrize(
    "raw_output",
    [
        {},
        {"results": [{"id": "kb-x", "title": 42, "content": "Wrong type"}]},
        {
            "results": [
                {"id": "kb-x", "title": "Title", "content": "Content", "extra": True}
            ]
        },
    ])
def test_invalid_tool_output_fails_run_without_entering_model_context(
    react_app_factory, tool_registry, monkeypatch, raw_output
):
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-kb", "search_knowledge_base", {"query": "checkout"}),
            ModelResponse(content="This response must not be requested."),
        ]
    )
    monkeypatch.setattr(tool_registry, "execute", AsyncMock(return_value=raw_output))

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Validate tool output."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "TOOL_OUTPUT_VALIDATION_ERROR"
    assert len(fake_kimi.calls) == 1
    assert "tool_result" not in [event["event_type"] for event in trace["events"]]
    persisted = app.state.repository.get_run(response.json()["id"])
    assert json.loads(persisted.recent_rounds_json)[-1]["observation"] is None


def test_multiple_tool_calls_are_rejected_without_executing_any_tool(
    react_app_factory, tool_registry, monkeypatch
):
    response = ModelResponse(
        tool_calls=(
            RawToolCall("call-1", "get_service_status", '{"service_name":"checkout"}'),
            RawToolCall("call-2", "search_knowledge_base", '{"query":"checkout"}'),
        )
    )
    app, fake_kimi = react_app_factory([response, response])
    execute = AsyncMock(wraps=tool_registry.execute)
    monkeypatch.setattr(tool_registry, "execute", execute)

    with TestClient(app) as client:
        result = client.post("/api/v1/runs", json={"objective": "Use only one tool."})
        trace = client.get(f"/api/v1/runs/{result.json()['id']}/events").json()

    assert result.json()["status"] == "FAILED"
    assert result.json()["error"]["code"] == "MALFORMED_LLM_RESPONSE"
    assert execute.await_count == 0
    assert trace["events"][-3]["event_type"] == "malformed_llm_response"
    assert trace["events"][-3]["payload"]["category"] == "multiple_tool_calls"
    assert "tool_call" not in [event["event_type"] for event in trace["events"]]


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (ModelResponse(), "empty_response"),
        (
            tool_response(None, "search_knowledge_base", {"query": "checkout"}),
            "missing_tool_call_id",
        ),
        (
            tool_response("call-x", None, {"query": "checkout"}),
            "missing_tool_name",
        ),
        (tool_response("call-x", "restart_service", {"service_name": "checkout"}), "unknown_tool"),
        (
            tool_response(
                "call-x",
                "search_knowledge_base",
                None,
                raw_arguments='{"query":',
            ),
            "invalid_arguments_json",
        ),
        (
            tool_response("call-x", "search_knowledge_base", None, raw_arguments="[]"),
            "arguments_not_object",
        ),
    ])
def test_unknown_tools_and_malformed_arguments_have_explicit_categories(
    react_app_factory, tool_registry, monkeypatch, response, category
):
    app, fake_kimi = react_app_factory([response, response])
    execute = AsyncMock(wraps=tool_registry.execute)
    monkeypatch.setattr(tool_registry, "execute", execute)

    with TestClient(app) as client:
        result = client.post("/api/v1/runs", json={"objective": "Validate model actions."})
        trace = client.get(f"/api/v1/runs/{result.json()['id']}/events").json()

    assert result.json()["status"] == "FAILED"
    assert result.json()["step_count"] == 2
    assert execute.await_count == 0
    assert trace["events"][-3]["payload"]["category"] == category


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_malformed_model_responses(
    react_app_factory, tool_registry, monkeypatch, constant
):
    response = tool_response(
        "call-nonfinite",
        "search_knowledge_base",
        None,
        raw_arguments=f'{{"query":{constant}}}',
    )
    app, _ = react_app_factory([response, response])
    execute = AsyncMock(wraps=tool_registry.execute)
    monkeypatch.setattr(tool_registry, "execute", execute)

    with TestClient(app) as client:
        result = client.post("/api/v1/runs", json={"objective": "Reject nonstandard JSON."})
        trace = client.get(f"/api/v1/runs/{result.json()['id']}/events").json()

    assert result.json()["status"] == "FAILED"
    assert result.json()["error"]["code"] == "MALFORMED_LLM_RESPONSE"
    assert result.json()["step_count"] == 2
    assert execute.await_count == 0
    malformed_events = [
        event for event in trace["events"] if event["event_type"] == "malformed_llm_response"
    ]
    assert [event["payload"]["category"] for event in malformed_events] == [
        "invalid_arguments_json",
        "invalid_arguments_json",
    ]
    assert not [event for event in trace["events"] if event["event_type"] == "tool_call"]


def test_duplicate_tool_call_id_is_rejected_before_second_execution(
    react_app_factory, tool_registry, monkeypatch
):
    app, fake_kimi = react_app_factory(
        [
            tool_response("duplicate-id", "get_service_status", {"service_name": "checkout"}),
            tool_response("duplicate-id", "search_knowledge_base", {"query": "checkout"}),
            tool_response("duplicate-id", "search_knowledge_base", {"query": "checkout"}),
        ]
    )
    execute = AsyncMock(wraps=tool_registry.execute)
    monkeypatch.setattr(tool_registry, "execute", execute)

    with TestClient(app) as client:
        result = client.post("/api/v1/runs", json={"objective": "Enforce unique call IDs."})
        trace = client.get(f"/api/v1/runs/{result.json()['id']}/events").json()

    assert result.json()["status"] == "FAILED"
    assert execute.await_count == 1
    assert trace["events"][-3]["payload"]["category"] == "duplicate_tool_call_id"


def test_first_malformed_response_is_corrected_without_creating_a_round(react_app_factory):
    app, fake_kimi = react_app_factory(
        [ModelResponse(), ModelResponse(content="Recovered final answer.")]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Recover the protocol."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["step_count"] == 2
    assert len(fake_kimi.calls) == 2
    correction = fake_kimi.calls[1]["messages"][-1]
    assert correction["role"] == "user"
    assert "category: empty_response" in correction["content"]
    malformed_event = trace["events"][1]
    assert malformed_event["event_type"] == "malformed_llm_response"
    assert malformed_event["payload"]["consecutive_count"] == 1
    assert malformed_event["payload"]["will_correct"] is True
    persisted = app.state.repository.get_run(response.json()["id"])
    rounds = json.loads(persisted.recent_rounds_json)
    assert len(rounds) == 1
    assert rounds[0]["assistant"]["content"] == "Recovered final answer."


def test_second_consecutive_malformed_response_fails_without_third_decision(react_app_factory):
    app, fake_kimi = react_app_factory([ModelResponse(), ModelResponse()])

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Bound corrections."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "MALFORMED_LLM_RESPONSE"
    assert response.json()["step_count"] == 2
    assert len(fake_kimi.calls) == 2
    malformed_events = [
        event for event in trace["events"] if event["event_type"] == "malformed_llm_response"
    ]
    assert [event["payload"]["will_correct"] for event in malformed_events] == [True, False]
    assert [event["payload"]["consecutive_count"] for event in malformed_events] == [1, 2]


def test_valid_tool_action_resets_consecutive_malformed_count(react_app_factory):
    app, fake_kimi = react_app_factory(
        [
            ModelResponse(),
            tool_response("call-valid", "get_service_status", {"service_name": "checkout"}),
            ModelResponse(),
            ModelResponse(content="Recovered after separate malformed responses."),
        ]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Reset malformed count."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["step_count"] == 4
    malformed_events = [
        event for event in trace["events"] if event["event_type"] == "malformed_llm_response"
    ]
    assert [event["payload"]["consecutive_count"] for event in malformed_events] == [1, 1]
    assert all(event["payload"]["will_correct"] for event in malformed_events)


def test_temporary_tool_failure_retries_and_records_successful_attempt(
    react_app_factory, tool_registry, settings, sleeper
):
    settings.retry_backoff_base_seconds = 0
    definition = tool_registry._definitions["get_service_status"]
    original_handler = definition.handler
    calls = 0

    async def flaky_handler(typed_input, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TemporaryToolError("temporary failure")
        return await original_handler(typed_input, context)

    tool_registry._definitions["get_service_status"] = replace(
        definition, handler=flaky_handler
    )
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-retry", "get_service_status", {"service_name": "checkout"}),
            ModelResponse(content="Recovered after a tool retry."),
        ]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Retry the status query."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "COMPLETED"
    assert calls == 2
    assert sleeper.calls == [0]
    retry_events = [
        event for event in trace["events"] if event["event_type"] == "retry_scheduled"
    ]
    assert len(retry_events) == 1
    assert retry_events[0]["payload"]["component"] == "tool"
    assert retry_events[0]["payload"]["tool_call_id"] == "call-retry"
    result_event = next(event for event in trace["events"] if event["event_type"] == "tool_result")
    assert result_event["payload"]["attempt"] == 2


def test_exhausted_tool_retries_become_safe_observation_for_next_decision(
    react_app_factory, tool_registry, settings, sleeper
):
    settings.retry_backoff_base_seconds = 0
    definition = tool_registry._definitions["get_service_status"]
    calls = 0

    async def failing_handler(typed_input, context):
        nonlocal calls
        calls += 1
        raise TemporaryToolError("sensitive-tool-detail")

    tool_registry._definitions["get_service_status"] = replace(
        definition, handler=failing_handler
    )
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-failed", "get_service_status", {"service_name": "checkout"}),
            ModelResponse(content="The status query is temporarily unavailable."),
        ]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Handle tool failure."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "COMPLETED"
    assert calls == 3
    assert sleeper.calls == [0, 0]
    observation = json.loads(fake_kimi.calls[1]["messages"][-1]["content"])
    assert observation == {
        "tool_call_id": "call-failed",
        "ok": False,
        "error": {
            "code": "TOOL_EXECUTION_FAILED",
            "message": "The tool execution failed safely.",
            "retryable": False,
        },
    }
    assert len([event for event in trace["events"] if event["event_type"] == "retry_scheduled"]) == 2
    tool_error = next(event for event in trace["events"] if event["event_type"] == "tool_error")
    assert tool_error["payload"]["attempt"] == 3
    assert "sensitive-tool-detail" not in response.text
    assert "sensitive-tool-detail" not in json.dumps(trace)


def test_utf8_oversized_tool_observation_fails_without_entering_model_context(
    react_app_factory, tool_registry, settings, monkeypatch
):
    raw_output = {
        "service_name": "checkout",
        "found": True,
        "status": "degraded",
        "description": "é" * 100,
        "updated_at": "2026-11-21T08:00:00Z",
    }
    observation = {"tool_call_id": "call-large", "ok": True, "data": raw_output}
    serialized = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
    character_count = len(serialized)
    byte_count = len(serialized.encode("utf-8"))
    settings.tool_result_max_bytes = (character_count + byte_count) // 2
    assert character_count < settings.tool_result_max_bytes < byte_count

    monkeypatch.setattr(tool_registry, "execute", AsyncMock(return_value=raw_output))
    app, fake_kimi = react_app_factory(
        [
            tool_response("call-large", "get_service_status", {"service_name": "checkout"}),
            ModelResponse(content="This response must not be requested."),
        ]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Bound tool output size."})
        trace = client.get(f"/api/v1/runs/{response.json()['id']}/events").json()

    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "TOOL_OUTPUT_TOO_LARGE"
    assert len(fake_kimi.calls) == 1
    assert "tool_result" not in [event["event_type"] for event in trace["events"]]
    tool_error = next(event for event in trace["events"] if event["event_type"] == "tool_error")
    assert tool_error["payload"]["error"]["code"] == "TOOL_OUTPUT_TOO_LARGE"
    persisted = app.state.repository.get_run(response.json()["id"])
    assert json.loads(persisted.recent_rounds_json)[-1]["observation"] is None
