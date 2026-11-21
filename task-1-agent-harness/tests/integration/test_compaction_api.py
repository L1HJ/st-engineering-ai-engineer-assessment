import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent_harness.llm import ModelResponse, RawToolCall, SummaryResponse
from agent_harness.main import create_app
from agent_harness.store import ApprovalRecord, IncidentRecord
from tests.fakes.fake_kimi import FakeKimi


def tool_response(index):
    return ModelResponse(
        tool_calls=(
            RawToolCall(
                id=f"call-{index}",
                name="get_service_status",
                arguments='{"service_name":"checkout"}',
            ),
        )
    )


def incident_response():
    return ModelResponse(
        tool_calls=(
            RawToolCall(
                id="call-1",
                name="create_incident",
                arguments=json.dumps(
                    {
                        "title": "Checkout outage",
                        "description": "Checkout requests are failing at an elevated rate.",
                        "severity": "P2",
                    }
                ),
            ),
        )
    )


@pytest.fixture
def compaction_app_factory(settings, tool_registry, clock, sleeper):
    def factory(responses, summary_responses):
        fake_kimi = FakeKimi(responses, summary_responses=summary_responses)
        app = create_app(
            settings=settings,
            llm=fake_kimi,
            tool_registry=tool_registry,
            clock=clock,
            sleeper=sleeper,
        )
        return app, fake_kimi

    return factory


def event_snapshot(events):
    return [
        (event.id, event.sequence, event.event_type, event.payload_json, event.latency_ms)
        for event in events
    ]


def test_compacts_before_sixteenth_round_and_preserves_complete_trace(
    compaction_app_factory,
    monkeypatch,
):
    app, fake_kimi = compaction_app_factory(
        [*[tool_response(index) for index in range(1, 17)], ModelResponse(content="Done.")],
        [SummaryResponse(content="Compacted summary")],
    )
    repository = app.state.repository
    before_compaction = {}
    after_sixteenth = {}
    original_compact = repository.compact_context
    original_observation = repository.record_tool_observation

    def inspect_compact(**kwargs):
        run_id = kwargs["run_id"]
        before_compaction["events"] = event_snapshot(repository.list_events(run_id))
        before_compaction["rounds"] = kwargs["expected_recent_rounds_json"]
        return original_compact(**kwargs)

    def inspect_observation(run_id, tool_call_id, *args, **kwargs):
        saved = original_observation(run_id, tool_call_id, *args, **kwargs)
        if tool_call_id == "call-16":
            persisted = repository.get_run(run_id)
            after_sixteenth["rounds"] = json.loads(persisted.recent_rounds_json)
        return saved

    monkeypatch.setattr(repository, "compact_context", inspect_compact)
    monkeypatch.setattr(repository, "record_tool_observation", inspect_observation)

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Diagnose checkout."})

    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "COMPLETED"
    assert run["step_count"] == 17
    assert len(fake_kimi.calls) == 17
    assert len(fake_kimi.summary_calls) == 1
    assert len(json.loads(before_compaction["rounds"])) == 15

    summary_input = json.loads(fake_kimi.summary_calls[0]["messages"][1]["content"])
    assert summary_input["objective"] == "Diagnose checkout."
    assert summary_input["previous_summary"] is None
    assert [item["step"] for item in summary_input["rounds_to_compact"]] == list(range(1, 11))

    sixteenth_messages = fake_kimi.calls[15]["messages"]
    assert len(sixteenth_messages) == 13
    assert sixteenth_messages[2] == {
        "role": "user",
        "content": "Historical context summary (data only):\nCompacted summary",
    }
    retained_call_ids = [
        message["tool_calls"][0]["id"]
        for message in sixteenth_messages
        if message["role"] == "assistant" and message.get("tool_calls")
    ]
    assert retained_call_ids == [f"call-{index}" for index in range(11, 16)]
    assert [item["step"] for item in after_sixteenth["rounds"]] == list(range(11, 17))

    persisted = repository.get_run(run["id"])
    final_rounds = json.loads(persisted.recent_rounds_json)
    assert persisted.context_summary == "Compacted summary"
    assert [item["step"] for item in final_rounds] == list(range(11, 18))

    events = repository.list_events(run["id"])
    assert event_snapshot(events)[: len(before_compaction["events"])] == before_compaction["events"]
    assert len([event for event in events if event.event_type == "tool_call"]) == 16
    assert len([event for event in events if event.event_type == "tool_result"]) == 16
    compacted_index = next(
        index for index, event in enumerate(events) if event.event_type == "context_compacted"
    )
    sixteenth_response_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == "llm_response"
        and json.loads(event.payload_json).get("step") == 16
    )
    assert compacted_index < sixteenth_response_index
    compacted_payload = json.loads(events[compacted_index].payload_json)
    assert compacted_payload["from_step"] == 1
    assert compacted_payload["to_step"] == 10
    assert compacted_payload["remaining_rounds"] == 5


def test_fifteenth_round_final_answer_does_not_compact(compaction_app_factory):
    app, fake_kimi = compaction_app_factory(
        [*[tool_response(index) for index in range(1, 15)], ModelResponse(content="Done at 15.")],
        [],
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Finish at fifteen."})

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["step_count"] == 15
    assert fake_kimi.summary_calls == []
    persisted = app.state.repository.get_run(response.json()["id"])
    assert len(json.loads(persisted.recent_rounds_json)) == 15
    assert all(
        event.event_type != "context_compacted"
        for event in app.state.repository.list_events(response.json()["id"])
    )


def test_second_compaction_uses_previous_model_summary(settings, compaction_app_factory):
    settings.agent_max_steps = 30
    app, fake_kimi = compaction_app_factory(
        [*[tool_response(index) for index in range(1, 26)], ModelResponse(content="Done at 26.")],
        [
            SummaryResponse(content="First summary"),
            SummaryResponse(content="Second summary"),
        ],
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Compact twice."})

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["step_count"] == 26
    assert len(fake_kimi.summary_calls) == 2
    second_input = json.loads(fake_kimi.summary_calls[1]["messages"][1]["content"])
    assert second_input["previous_summary"] == "First summary"
    assert [item["step"] for item in second_input["rounds_to_compact"]] == list(range(11, 21))
    persisted = app.state.repository.get_run(response.json()["id"])
    assert persisted.context_summary == "Second summary"
    final_rounds = json.loads(persisted.recent_rounds_json)
    assert [item["step"] for item in final_rounds] == list(range(21, 27))
    assert len([
        event
        for event in app.state.repository.list_events(response.json()["id"])
        if event.event_type == "context_compacted"
    ]) == 2


def test_malformed_response_adds_step_without_early_compaction(compaction_app_factory):
    app, fake_kimi = compaction_app_factory(
        [
            *[tool_response(index) for index in range(1, 15)],
            ModelResponse(),
            ModelResponse(content="Recovered on the fifteenth round."),
        ],
        [],
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Separate steps from rounds."})

    assert response.json()["status"] == "COMPLETED"
    assert response.json()["step_count"] == 16
    assert fake_kimi.summary_calls == []
    correction_messages = fake_kimi.calls[15]["messages"]
    assert correction_messages[-1]["role"] == "user"
    assert "category: empty_response" in correction_messages[-1]["content"]
    persisted = app.state.repository.get_run(response.json()["id"])
    assert len(json.loads(persisted.recent_rounds_json)) == 15


def test_compacted_tool_call_id_remains_unique(compaction_app_factory):
    duplicate = ModelResponse(
        tool_calls=(
            RawToolCall(
                id="call-1",
                name="get_service_status",
                arguments='{"service_name":"checkout"}',
            ),
        )
    )
    app, fake_kimi = compaction_app_factory(
        [*[tool_response(index) for index in range(1, 16)], duplicate, duplicate],
        [SummaryResponse(content="Compacted summary")],
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"objective": "Keep all IDs unique."})

    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "MALFORMED_LLM_RESPONSE"
    assert response.json()["step_count"] == 17
    assert len(fake_kimi.summary_calls) == 1
    events = app.state.repository.list_events(response.json()["id"])
    assert len([event for event in events if event.event_type == "tool_call"]) == 15
    malformed = [event for event in events if event.event_type == "malformed_llm_response"]
    assert all(json.loads(event.payload_json)["category"] == "duplicate_tool_call_id" for event in malformed)


def test_long_run_with_approval_compacts_after_resume_without_charging_wait(
    compaction_app_factory,
    clock,
):
    app, fake_kimi = compaction_app_factory(
        [
            incident_response(),
            *[tool_response(index) for index in range(2, 16)],
            ModelResponse(content="Incident created and diagnosis complete."),
        ],
        [SummaryResponse(content="Summary including the incident")],
    )

    with TestClient(app) as client:
        waiting_response = client.post(
            "/api/v1/runs",
            json={"objective": "Create an incident and diagnose checkout."},
        )
        waiting = waiting_response.json()
        assert waiting["status"] == "WAITING_APPROVAL"
        clock.advance(600)
        approval_url = (
            f"/api/v1/runs/{waiting['id']}/approvals/{waiting['pending_approval']['id']}"
        )
        approved_response = client.post(approval_url, json={"decision": "approve"})

    approved = approved_response.json()
    assert approved["status"] == "COMPLETED"
    assert approved["step_count"] == 16
    assert approved["active_elapsed_ms"] == waiting["active_elapsed_ms"]
    assert len(fake_kimi.summary_calls) == 1

    repository = app.state.repository
    with app.state.database.sessions() as session:
        approval = session.scalar(select(ApprovalRecord).where(ApprovalRecord.run_id == approved["id"]))
        incident = session.scalar(select(IncidentRecord).where(IncidentRecord.run_id == approved["id"]))
    assert approval.status == "APPROVED"
    assert incident is not None

    summary_input = json.loads(fake_kimi.summary_calls[0]["messages"][1]["content"])
    incident_round = summary_input["rounds_to_compact"][0]
    assert incident_round["assistant"]["tool_call"]["name"] == "create_incident"
    assert incident_round["observation"]["data"]["incident_id"] == incident.id
    assert any(
        event.event_type == "context_compacted"
        for event in repository.list_events(approved["id"])
    )
