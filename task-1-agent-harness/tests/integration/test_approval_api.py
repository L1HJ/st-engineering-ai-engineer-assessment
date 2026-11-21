import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from agent_harness.llm import ModelResponse, RawToolCall
from agent_harness.main import create_app
from agent_harness.store import ApprovalRecord, IncidentRecord
from agent_harness.tools import TemporaryToolError
from tests.fakes.fake_kimi import FakeKimi


INCIDENT_ARGUMENTS = {
    "title": "Checkout outage",
    "description": "Checkout requests are failing at an elevated rate.",
    "severity": "P2",
}


def incident_response(call_id="call-incident", arguments=None):
    return ModelResponse(
        tool_calls=(
            RawToolCall(
                id=call_id,
                name="create_incident",
                arguments=json.dumps(arguments or INCIDENT_ARGUMENTS),
            ),
        )
    )


@pytest.fixture
def approval_app_factory(settings, tool_registry, clock, sleeper):
    def factory(responses, *, random_source=None):
        fake_kimi = FakeKimi(responses)
        app = create_app(
            settings=settings,
            llm=fake_kimi,
            tool_registry=tool_registry,
            clock=clock,
            sleeper=sleeper,
            random_source=random_source,
        )
        return app, fake_kimi

    return factory


def incident_count(app, run_id=None):
    statement = select(func.count()).select_from(IncidentRecord)
    if run_id is not None:
        statement = statement.where(IncidentRecord.run_id == run_id)
    with app.state.database.sessions() as session:
        return session.scalar(statement)


def approval_count(app):
    with app.state.database.sessions() as session:
        return session.scalar(select(func.count()).select_from(ApprovalRecord))


def create_waiting_run(client):
    response = client.post("/api/v1/runs", json={"objective": "Create a checkout incident."})
    assert response.status_code == 201
    assert response.json()["status"] == "WAITING_APPROVAL"
    return response.json()


def approval_url(run):
    return f"/api/v1/runs/{run['id']}/approvals/{run['pending_approval']['id']}"


def test_approval_pauses_without_side_effects_then_resumes_original_round(
    approval_app_factory, tool_registry, clock
):
    definition = tool_registry._definitions["create_incident"]
    handler = AsyncMock(wraps=definition.handler)
    tool_registry._definitions["create_incident"] = replace(definition, handler=handler)
    app, fake_kimi = approval_app_factory(
        [incident_response(arguments={**INCIDENT_ARGUMENTS, "title": "  Checkout outage  "}),
         ModelResponse(content="The incident was created.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        assert run["step_count"] == 1
        assert run["pending_approval"]["arguments"]["title"] == "Checkout outage"
        assert run["final_answer"] is None
        assert handler.await_count == 0
        assert incident_count(app) == 0
        assert len(fake_kimi.calls) == 1

        active_elapsed_ms = run["active_elapsed_ms"]
        clock.advance(600)
        restored = client.get(f"/api/v1/runs/{run['id']}").json()
        assert restored["active_elapsed_ms"] == active_elapsed_ms
        assert restored["pending_approval"] == run["pending_approval"]

        invalid = client.post(
            approval_url(run),
            json={"decision": "approve", "arguments": {"title": "Replacement"}},
        )
        assert invalid.status_code == 422
        assert client.get(f"/api/v1/runs/{run['id']}").json()["status"] == "WAITING_APPROVAL"

        approved = client.post(approval_url(run), json={"decision": "approve"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert approved.status_code == 200
    assert approved.json()["status"] == "COMPLETED"
    assert approved.json()["step_count"] == 2
    assert approved.json()["pending_approval"] is None
    assert approved.json()["active_elapsed_ms"] == active_elapsed_ms
    assert handler.await_count == 1
    typed_input = handler.await_args.args[0]
    assert typed_input.model_dump() == INCIDENT_ARGUMENTS
    assert incident_count(app) == 1
    assert len(fake_kimi.calls) == 2
    observation = json.loads(fake_kimi.calls[1]["messages"][-1]["content"])
    assert observation["ok"] is True
    assert observation["data"]["incident_id"].startswith("inc_")
    assert [event["event_type"] for event in trace["events"]] == [
        "run_created",
        "llm_response",
        "tool_call",
        "approval_requested",
        "run_state_changed",
        "approval_decided",
        "run_state_changed",
        "tool_result",
        "llm_response",
        "run_state_changed",
        "run_finished",
    ]
    state_events = [
        event["payload"]
        for event in trace["events"]
        if event["event_type"] == "run_state_changed"
    ]
    assert state_events == [
        {
            "from_status": "RUNNING",
            "to_status": "WAITING_APPROVAL",
            "reason": "approval_requested",
        },
        {
            "from_status": "WAITING_APPROVAL",
            "to_status": "RUNNING",
            "reason": "approval_approved",
        },
        {
            "from_status": "RUNNING",
            "to_status": "COMPLETED",
            "reason": "COMPLETED",
        },
    ]
    assert len([
        event for event in trace["events"] if event["event_type"] == "llm_response"
    ]) == 2
    assert len([
        event for event in trace["events"] if event["event_type"] == "tool_call"
    ]) == 1


def test_rejection_completes_without_handler_incident_or_another_model_call(
    approval_app_factory, tool_registry
):
    definition = tool_registry._definitions["create_incident"]
    handler = AsyncMock(wraps=definition.handler)
    tool_registry._definitions["create_incident"] = replace(definition, handler=handler)
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This response must not be requested.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        rejected = client.post(approval_url(run), json={"decision": "reject"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()
        repeated_reject = client.post(approval_url(run), json={"decision": "reject"})
        approve_after_reject = client.post(approval_url(run), json={"decision": "approve"})

    assert rejected.status_code == 200
    assert rejected.json()["status"] == "COMPLETED"
    assert rejected.json()["step_count"] == 1
    assert rejected.json()["final_answer"] == (
        "Incident creation was not approved, so no incident was created."
    )
    assert rejected.json()["error"] is None
    assert rejected.json()["pending_approval"] is None
    assert rejected.json()["completed_at"].endswith("Z")
    assert handler.await_count == 0
    assert incident_count(app) == 0
    assert len(fake_kimi.calls) == 1
    assert [event["event_type"] for event in trace["events"]][-5:] == [
        "approval_requested",
        "run_state_changed",
        "approval_decided",
        "run_state_changed",
        "run_finished",
    ]
    assert trace["events"][-2]["payload"] == {
        "from_status": "WAITING_APPROVAL",
        "to_status": "COMPLETED",
        "reason": "approval_rejected",
    }
    assert "tool_result" not in [event["event_type"] for event in trace["events"]]
    for repeated in (repeated_reject, approve_after_reject):
        assert repeated.status_code == 409
        assert repeated.json()["error"]["code"] == "APPROVAL_ALREADY_DECIDED"


def test_invalid_incident_input_creates_no_approval_and_returns_safe_observation(
    approval_app_factory, tool_registry
):
    definition = tool_registry._definitions["create_incident"]
    handler = AsyncMock(wraps=definition.handler)
    tool_registry._definitions["create_incident"] = replace(definition, handler=handler)
    app, fake_kimi = approval_app_factory(
        [incident_response(arguments={"title": "Missing fields"}),
         ModelResponse(content="The incident arguments were invalid.")]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Create an incident."})

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["pending_approval"] is None
    assert approval_count(app) == 0
    assert incident_count(app) == 0
    assert handler.await_count == 0
    observation = json.loads(fake_kimi.calls[1]["messages"][-1]["content"])
    assert observation["error"]["code"] == "TOOL_INPUT_VALIDATION_ERROR"


def test_approval_request_validation_and_missing_resources_do_not_change_state(
    approval_app_factory
):
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="Unused final answer.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        for payload in ({"decision": "execute"}, {}, {"decision": "approve", "title": "Other"}):
            response = client.post(approval_url(run), json=payload)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "VALIDATION_ERROR"

        missing_approval = client.post(
            f"/api/v1/runs/{run['id']}/approvals/{uuid4()}",
            json={"decision": "approve"},
        )
        missing_run = client.post(
            f"/api/v1/runs/{uuid4()}/approvals/{run['pending_approval']['id']}",
            json={"decision": "approve"},
        )
        other_run = app.state.repository.create_run("Unrelated run.", limits={})
        mismatched = client.post(
            f"/api/v1/runs/{other_run.id}/approvals/{run['pending_approval']['id']}",
            json={"decision": "reject"},
        )
        restored = client.get(f"/api/v1/runs/{run['id']}").json()

    assert missing_approval.status_code == 404
    assert missing_approval.json()["error"]["code"] == "APPROVAL_NOT_FOUND"
    assert missing_run.status_code == 404
    assert missing_run.json()["error"]["code"] == "RUN_NOT_FOUND"
    assert mismatched.status_code == 404
    assert mismatched.json()["error"]["code"] == "APPROVAL_NOT_FOUND"
    assert restored["status"] == "WAITING_APPROVAL"
    assert restored["pending_approval"]["arguments"] == INCIDENT_ARGUMENTS
    assert app.state.repository.get_run(other_run.id).status == "RUNNING"
    assert len(fake_kimi.calls) == 1
    assert incident_count(app) == 0


@pytest.mark.parametrize(
    "decisions",
    [("approve", "reject"), ("approve", "approve"), ("reject", "reject")],
)
def test_concurrent_decisions_have_exactly_one_winner(approval_app_factory, decisions):
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="The approved flow finished.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        barrier = Barrier(2)

        def decide(decision):
            barrier.wait()
            return client.post(approval_url(run), json={"decision": decision})

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(decide, decisions))
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert sorted(response.status_code for response in responses) == [200, 409]
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json()["error"]["code"] == "APPROVAL_ALREADY_DECIDED"
    assert len([
        event for event in trace["events"] if event["event_type"] == "approval_decided"
    ]) == 1
    assert len([
        event for event in trace["events"] if event["event_type"] == "run_finished"
    ]) == 1
    winner_index = next(
        index for index, response in enumerate(responses) if response.status_code == 200
    )
    winner = decisions[winner_index]
    assert incident_count(app) == (1 if winner == "approve" else 0)
    assert len(fake_kimi.calls) == (2 if winner == "approve" else 1)


def test_waiting_run_survives_restart_and_running_run_is_interrupted(
    settings, tool_registry, clock, sleeper
):
    first_kimi = FakeKimi([incident_response(), incident_response()])
    first_app = create_app(
        settings=settings,
        llm=first_kimi,
        tool_registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    with TestClient(first_app) as first_client:
        waiting = create_waiting_run(first_client)
        interrupted = create_waiting_run(first_client)
        first_app.state.repository.decide_approval(
            interrupted["id"],
            interrupted["pending_approval"]["id"],
            "approve",
        )
        waiting_before_restart = first_client.get(f"/api/v1/runs/{waiting['id']}").json()

    second_kimi = FakeKimi([ModelResponse(content="Resumed after restart.")])
    second_app = create_app(
        settings=settings,
        llm=second_kimi,
        tool_registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )
    with TestClient(second_app) as second_client:
        restored = second_client.get(f"/api/v1/runs/{waiting['id']}").json()
        legacy = second_client.get(f"/api/v1/runs/{interrupted['id']}").json()
        interrupted_trace = second_client.get(
            f"/api/v1/runs/{interrupted['id']}/events"
        ).json()
        approved = second_client.post(approval_url(restored), json={"decision": "approve"})

    assert restored["status"] == "WAITING_APPROVAL"
    assert restored["pending_approval"] == waiting_before_restart["pending_approval"]
    assert legacy["status"] == "FAILED"
    assert legacy["error"]["code"] == "RUN_INTERRUPTED"
    assert interrupted_trace["events"][-2]["event_type"] == "run_state_changed"
    assert interrupted_trace["events"][-2]["payload"] == {
        "from_status": "RUNNING",
        "to_status": "FAILED",
        "reason": "RUN_INTERRUPTED",
    }
    assert approved.status_code == 200
    assert approved.json()["status"] == "COMPLETED"
    assert approved.json()["final_answer"] == "Resumed after restart."
    assert len(second_kimi.calls) == 1
    assert incident_count(second_app, interrupted["id"]) == 0
    assert incident_count(second_app) == 1


def test_same_idempotency_key_with_different_arguments_fails_without_overwrite(
    approval_app_factory
):
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This response must not be requested.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        existing = IncidentRecord(
            id="inc_existing",
            run_id=run["id"],
            tool_call_id="call-incident",
            title="Original incident",
            description=INCIDENT_ARGUMENTS["description"],
            severity=INCIDENT_ARGUMENTS["severity"],
            status="OPEN",
            idempotency_key=f"{run['id']}:call-incident",
            created_at=app.state.repository._now(),
        )
        with app.state.database.sessions.begin() as session:
            session.add(existing)
        approved = client.post(approval_url(run), json={"decision": "approve"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert approved.status_code == 200
    assert approved.json()["status"] == "FAILED"
    assert approved.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(fake_kimi.calls) == 1
    assert incident_count(app) == 1
    with app.state.database.sessions() as session:
        persisted = session.get(IncidentRecord, "inc_existing")
        assert persisted.title == "Original incident"
    assert trace["events"][-3]["event_type"] == "tool_error"
    assert trace["events"][-3]["payload"]["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert trace["events"][-1]["event_type"] == "run_finished"


def test_each_new_incident_action_requires_a_new_approval(approval_app_factory):
    app, fake_kimi = approval_app_factory(
        [
            incident_response("call-incident-1"),
            incident_response("call-incident-2", {**INCIDENT_ARGUMENTS, "title": "Second"}),
            ModelResponse(content="Both incidents were created."),
        ]
    )

    with TestClient(app) as client:
        first = create_waiting_run(client)
        second = client.post(approval_url(first), json={"decision": "approve"}).json()
        assert second["status"] == "WAITING_APPROVAL"
        assert second["pending_approval"]["id"] != first["pending_approval"]["id"]
        assert second["pending_approval"]["tool_call_id"] == "call-incident-2"
        assert incident_count(app) == 1
        completed = client.post(approval_url(second), json={"decision": "approve"})

    assert completed.json()["status"] == "COMPLETED"
    assert incident_count(app) == 2
    assert approval_count(app) == 2
    assert len(fake_kimi.calls) == 3


def test_incident_retry_reuses_idempotency_key_and_records_one_incident(
    approval_app_factory, tool_registry, settings
):
    settings.retry_backoff_base_seconds = 0
    definition = tool_registry._definitions["create_incident"]
    original_handler = definition.handler
    calls = []

    async def flaky_handler(typed_input, context):
        calls.append(context.idempotency_key)
        if len(calls) == 1:
            raise TemporaryToolError("Temporary storage failure.")
        return await original_handler(typed_input, context)

    tool_registry._definitions["create_incident"] = replace(definition, handler=flaky_handler)
    app, _ = approval_app_factory([incident_response(), ModelResponse(content="Recovered.")])

    with TestClient(app) as client:
        run = create_waiting_run(client)
        approved = client.post(approval_url(run), json={"decision": "approve"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert approved.json()["status"] == "COMPLETED"
    assert calls == [f"{run['id']}:call-incident"] * 2
    assert incident_count(app) == 1
    result = next(event for event in trace["events"] if event["event_type"] == "tool_result")
    assert result["payload"]["attempt"] == 2


def test_approved_incident_timeout_exhausts_remaining_active_budget(
    approval_app_factory, tool_registry, settings, clock, sleeper
):
    settings.agent_active_time_limit_seconds = 1
    definition = tool_registry._definitions["create_incident"]
    calls = 0

    async def slow_timeout(_typed_input, _context):
        nonlocal calls
        calls += 1
        clock.advance(2)
        raise TimeoutError("Sensitive timeout detail.")

    tool_registry._definitions["create_incident"] = replace(definition, handler=slow_timeout)
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This decision must not happen.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        approved = client.post(approval_url(run), json={"decision": "approve"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert approved.status_code == 200
    assert approved.json()["status"] == "LIMIT_EXCEEDED"
    assert approved.json()["error"]["code"] == "ACTIVE_TIME_LIMIT_EXCEEDED"
    assert calls == 1
    assert len(fake_kimi.calls) == 1
    assert sleeper.calls == []
    assert incident_count(app) == 0
    assert trace["events"][-3]["event_type"] == "tool_error"
    assert trace["events"][-3]["payload"]["error"]["code"] == "TOOL_TIMEOUT"
    assert "Sensitive" not in json.dumps(trace)
    assert trace["events"][-1]["event_type"] == "run_finished"


def test_approved_incident_backoff_cannot_exceed_remaining_active_budget(
    approval_app_factory, tool_registry, settings, sleeper
):
    settings.agent_active_time_limit_seconds = 1
    settings.retry_backoff_base_seconds = 2
    settings.retry_backoff_max_seconds = 2
    definition = tool_registry._definitions["create_incident"]
    handler = AsyncMock(side_effect=TemporaryToolError("Temporary storage failure."))
    tool_registry._definitions["create_incident"] = replace(definition, handler=handler)
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This decision must not happen.")],
        random_source=lambda: 1.0,
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        approved = client.post(approval_url(run), json={"decision": "approve"})
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert approved.status_code == 200
    assert approved.json()["status"] == "LIMIT_EXCEEDED"
    assert approved.json()["error"]["code"] == "ACTIVE_TIME_LIMIT_EXCEEDED"
    assert handler.await_count == 1
    assert len(fake_kimi.calls) == 1
    assert sleeper.calls == []
    assert incident_count(app) == 0
    assert "retry_scheduled" not in [event["event_type"] for event in trace["events"]]
    assert trace["events"][-3]["event_type"] == "tool_error"
    assert trace["events"][-3]["payload"]["error"]["code"] == "TOOL_EXECUTION_FAILED"
    assert trace["events"][-1]["event_type"] == "run_finished"


def test_approval_on_last_step_executes_tool_but_rejects_next_model_decision(
    approval_app_factory, settings
):
    settings.agent_max_steps = 1
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This decision must not happen.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        approved = client.post(approval_url(run), json={"decision": "approve"})

    assert approved.status_code == 200
    assert approved.json()["status"] == "LIMIT_EXCEEDED"
    assert approved.json()["error"]["code"] == "MAX_STEPS_EXCEEDED"
    assert approved.json()["step_count"] == 1
    assert incident_count(app) == 1
    assert len(fake_kimi.calls) == 1


def test_waiting_time_does_not_consume_remaining_active_budget(
    approval_app_factory, settings, clock
):
    settings.agent_active_time_limit_seconds = 1
    app, fake_kimi = approval_app_factory(
        [incident_response(), ModelResponse(content="This response is over budget.")]
    )

    with TestClient(app) as client:
        run = create_waiting_run(client)
        clock.advance(120)
        fake_kimi._clock = clock
        fake_kimi._advance_per_call = 2
        approved = client.post(approval_url(run), json={"decision": "approve"})

    assert approved.json()["status"] == "LIMIT_EXCEEDED"
    assert approved.json()["error"]["code"] == "ACTIVE_TIME_LIMIT_EXCEEDED"
    assert 2000 <= approved.json()["active_elapsed_ms"] < 10000
    assert incident_count(app) == 1
    assert len(fake_kimi.calls) == 2
