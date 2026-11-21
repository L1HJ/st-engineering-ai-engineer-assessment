import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from agent_harness.llm import LlmError, ModelResponse, RawToolCall, SummaryResponse
from agent_harness.main import create_app
from agent_harness.store import Database, IncidentRecord
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
def conversation_factory(settings, tool_registry, clock, sleeper):
    def factory(responses, *, summary_responses=None):
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


def create_conversation(client, title="checkout triage"):
    response = client.post("/api/v1/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()


def send_message(client, conversation_id, content, client_message_id=None):
    payload = {"content": content}
    if client_message_id is not None:
        payload["client_message_id"] = client_message_id
    return client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json=payload,
    )


def seed_completed_turns(app, conversation_id, count, *, final_answer_prefix="answer"):
    run_ids = []
    for index in range(1, count + 1):
        record, _ = app.state.repository.create_conversation_turn(
            conversation_id,
            f"question {index}",
            None,
            {},
            app.state.settings.conversation_max_turns,
        )
        app.state.repository.finish_run(
            record.id,
            status="COMPLETED",
            active_elapsed_ms=0,
            final_answer=f"{final_answer_prefix} {index}",
        )
        run_ids.append(record.id)
    return run_ids


def incident_count(app):
    with app.state.database.sessions() as session:
        return session.scalar(select(func.count()).select_from(IncidentRecord))


def test_create_conversation_returns_active_with_zero_turns(client):
    conversation = create_conversation(client)

    assert conversation["status"] == "ACTIVE"
    assert conversation["turn_count"] == 0
    assert conversation["turns"] == []
    assert conversation["title"] == "checkout triage"
    assert conversation["created_at"].endswith("Z")
    assert conversation["updated_at"].endswith("Z")


def test_first_message_completes_and_is_listed_as_turn(conversation_factory):
    app, fake_kimi = conversation_factory(
        [ModelResponse(content="Checkout is degraded.")]
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        response = send_message(client, conversation["id"], "Check checkout.")
        run = response.json()
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()
        detail = client.get(f"/api/v1/conversations/{conversation['id']}").json()

    assert response.status_code == 201
    assert run["status"] == "COMPLETED"
    assert run["final_answer"] == "Checkout is degraded."
    assert history["count"] == 1
    turn = history["turns"][0]
    assert turn["turn_index"] == 1
    assert turn["run_id"] == run["id"]
    assert turn["status"] == "COMPLETED"
    assert turn["user_content"] == "Check checkout."
    assert turn["assistant_content"] == "Checkout is degraded."
    assert turn["error"] is None
    assert turn["completed_at"].endswith("Z")
    assert detail["turn_count"] == 1
    assert len(detail["turns"]) == 1
    assert len(fake_kimi.calls) == 1


def test_ambiguous_question_gets_clarification_then_continues(conversation_factory):
    app, fake_kimi = conversation_factory(
        [
            ModelResponse(content="Which service do you mean?"),
            ModelResponse(content="Checkout is degraded."),
        ]
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        first = send_message(client, conversation["id"], "That service is down again.")
        second = send_message(client, conversation["id"], "Checkout.")

    assert first.json()["status"] == "COMPLETED"
    assert first.json()["final_answer"] == "Which service do you mean?"
    assert second.json()["status"] == "COMPLETED"
    second_request = fake_kimi.calls[1]["messages"]
    contents = [message["content"] for message in second_request]
    assert "That service is down again." in contents
    assert "Which service do you mean?" in contents
    assert contents[-1] == "Checkout."


def test_second_turn_does_not_inherit_tool_messages(conversation_factory):
    app, fake_kimi = conversation_factory(
        [
            ModelResponse(
                tool_calls=(
                    RawToolCall(
                        id="call-status",
                        name="get_service_status",
                        arguments=json.dumps({"service_name": "checkout"}),
                    ),
                )
            ),
            ModelResponse(content="Checkout is degraded."),
            ModelResponse(content="Payments is down."),
        ]
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        send_message(client, conversation["id"], "Check checkout.")
        send_message(client, conversation["id"], "And payments?")

    second_turn_messages = fake_kimi.calls[2]["messages"]
    assert all(message["role"] != "tool" for message in second_turn_messages)
    assert all("tool_calls" not in message for message in second_turn_messages)
    contents = [message["content"] for message in second_turn_messages]
    assert "Checkout is degraded." in contents
    assert "And payments?" == contents[-1]


def test_failed_turn_is_reported_as_data_and_next_turn_continues(conversation_factory):
    app, fake_kimi = conversation_factory(
        [
            LlmError(
                "LLM_AUTHENTICATION_FAILED",
                "The model provider rejected the configured credentials.",
            ),
            ModelResponse(content="Recovered."),
        ]
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        failed = send_message(client, conversation["id"], "Check checkout.")
        recovered = send_message(client, conversation["id"], "Try again.")
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()

    assert failed.json()["status"] == "FAILED"
    assert failed.json()["error"]["code"] == "LLM_AUTHENTICATION_FAILED"
    assert recovered.json()["status"] == "COMPLETED"
    turns = history["turns"]
    assert [turn["status"] for turn in turns] == ["FAILED", "COMPLETED"]
    assert turns[0]["assistant_content"] is None
    assert turns[0]["error"]["code"] == "LLM_AUTHENTICATION_FAILED"
    history_messages = fake_kimi.calls[1]["messages"]
    failed_notes = [
        message["content"]
        for message in history_messages
        if "Historical turn data" in (message["content"] or "")
    ]
    assert len(failed_notes) == 1
    assert "FAILED" in failed_notes[0]
    assert "LLM_AUTHENTICATION_FAILED" in failed_notes[0]


def test_approval_blocks_new_messages_until_decided(conversation_factory):
    app, _ = conversation_factory([incident_response()])

    with TestClient(app) as client:
        conversation = create_conversation(client)
        waiting = send_message(client, conversation["id"], "Create an incident.")
        blocked = send_message(client, conversation["id"], "Anything new?")

    waiting_run = waiting.json()
    assert waiting_run["status"] == "WAITING_APPROVAL"
    assert blocked.status_code == 409
    assert blocked.json()["error"]["code"] == "CONVERSATION_BUSY"
    details = blocked.json()["error"]["details"]
    assert details["active_run_id"] == waiting_run["id"]
    assert details["active_status"] == "WAITING_APPROVAL"
    assert details["pending_approval_id"] == waiting_run["pending_approval"]["id"]


def test_approved_turn_completes_and_creates_exactly_one_incident(conversation_factory):
    app, fake_kimi = conversation_factory(
        [incident_response(), ModelResponse(content="The incident was created.")]
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        waiting = send_message(client, conversation["id"], "Create an incident.").json()
        approved = client.post(
            f"/api/v1/runs/{waiting['id']}/approvals/"
            f"{waiting['pending_approval']['id']}",
            json={"decision": "approve"},
        )
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()

    assert approved.status_code == 200
    assert approved.json()["status"] == "COMPLETED"
    assert approved.json()["final_answer"] == "The incident was created."
    assert incident_count(app) == 1
    assert history["turns"][0]["status"] == "COMPLETED"
    assert history["turns"][0]["assistant_content"] == "The incident was created."
    assert len(fake_kimi.calls) == 2


class _BlockingKimi(FakeKimi):
    def __init__(self, responses, started: threading.Event, release: threading.Event):
        super().__init__(responses)
        self._started = started
        self._release = release

    async def complete_decision(self, messages, tool_schemas, timeout_budget, on_retry=None):
        self._started.set()
        await asyncio.to_thread(self._release.wait, 5)
        return await super().complete_decision(
            messages, tool_schemas, timeout_budget, on_retry
        )


def test_concurrent_messages_allow_only_one_active_turn(
    settings, tool_registry, clock, sleeper
):
    started = threading.Event()
    release = threading.Event()
    fake_kimi = _BlockingKimi(
        [ModelResponse(content="First turn done.")],
        started,
        release,
    )
    app = create_app(
        settings=settings,
        llm=fake_kimi,
        tool_registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                send_message, client, conversation["id"], "First message."
            )
            assert started.wait(5)
            second = send_message(client, conversation["id"], "Second message.")
            release.set()
            first_response = first.result(timeout=5)

    assert first_response.status_code == 201
    assert first_response.json()["status"] == "COMPLETED"
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "CONVERSATION_BUSY"
    assert len(fake_kimi.calls) == 1


def test_duplicate_client_message_id_replays_without_another_model_call(
    conversation_factory,
):
    app, fake_kimi = conversation_factory([ModelResponse(content="Done.")])

    with TestClient(app) as client:
        conversation = create_conversation(client)
        first = send_message(
            client,
            conversation["id"],
            "Check checkout.",
            client_message_id="tui-0001",
        )
        replay = send_message(
            client,
            conversation["id"],
            "A changed payload is ignored on replay.",
            client_message_id="tui-0001",
        )
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()

    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    assert history["count"] == 1
    assert len(fake_kimi.calls) == 1


def test_conversation_compaction_summarizes_oldest_turns_and_keeps_recent_ones(
    conversation_factory, settings
):
    settings.conversation_window_turns = 3
    settings.conversation_compact_turns = 2
    settings.conversation_max_turns = 10
    app, fake_kimi = conversation_factory(
        [ModelResponse(content="answer 4")],
        summary_responses=[SummaryResponse(content="Summary of turns 1 and 2.")],
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        seed_completed_turns(app, conversation["id"], 3)
        response = send_message(client, conversation["id"], "question 4")
        detail = client.get(f"/api/v1/conversations/{conversation['id']}").json()
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()

    assert response.json()["status"] == "COMPLETED"
    assert len(fake_kimi.summary_calls) == 1
    summary_input = json.loads(fake_kimi.summary_calls[0]["messages"][1]["content"])
    assert [turn["turn_index"] for turn in summary_input["turns_to_compact"]] == [1, 2]
    stored = app.state.repository.get_conversation(conversation["id"])
    assert stored.context_summary == "Summary of turns 1 and 2."
    assert stored.summary_through_turn_index == 2
    request_contents = [
        message["content"] for message in fake_kimi.calls[0]["messages"]
    ]
    assert any("Summary of turns 1 and 2." in (content or "") for content in request_contents)
    assert not any(content == "question 1" for content in request_contents)
    assert not any(content == "answer 1" for content in request_contents)
    assert "question 3" in request_contents
    assert "answer 3" in request_contents
    assert detail["turn_count"] == 4
    assert history["count"] == 4


def test_compaction_failure_fails_the_turn_and_keeps_history_unchanged(
    conversation_factory, settings
):
    settings.conversation_window_turns = 3
    settings.conversation_compact_turns = 2
    settings.conversation_max_turns = 10
    app, fake_kimi = conversation_factory(
        [ModelResponse(content="This decision must not happen.")],
        summary_responses=[
            LlmError("LLM_UNAVAILABLE", "The model provider is unavailable.")
        ],
    )

    with TestClient(app) as client:
        conversation = create_conversation(client)
        seed_completed_turns(app, conversation["id"], 3)
        response = send_message(client, conversation["id"], "question 4")
        run = response.json()
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()
        trace = client.get(f"/api/v1/runs/{run['id']}/events").json()

    assert response.status_code == 201
    assert run["status"] == "FAILED"
    assert run["error"]["code"] == "CONVERSATION_COMPACTION_FAILED"
    assert fake_kimi.calls == []
    stored = app.state.repository.get_conversation(conversation["id"])
    assert stored.context_summary is None
    assert stored.summary_through_turn_index == 0
    assert history["count"] == 4
    assert history["turns"][-1]["status"] == "FAILED"
    assert "conversation_compaction_failed" in [
        event["event_type"] for event in trace["events"]
    ]


def test_turn_limit_rejects_new_messages(conversation_factory, settings):
    settings.conversation_window_turns = 2
    settings.conversation_compact_turns = 1
    settings.conversation_max_turns = 3
    app, fake_kimi = conversation_factory([ModelResponse(content="Unused.")])

    with TestClient(app) as client:
        conversation = create_conversation(client)
        seed_completed_turns(app, conversation["id"], 3)
        response = send_message(client, conversation["id"], "question 4")
        history = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
        ).json()

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONVERSATION_TURN_LIMIT_EXCEEDED"
    assert history["count"] == 3
    assert fake_kimi.calls == []


def test_history_pagination_respects_limit_and_before_turn(conversation_factory):
    app, _ = conversation_factory([])

    with TestClient(app) as client:
        conversation = create_conversation(client)
        seed_completed_turns(app, conversation["id"], 3)
        latest = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages?limit=2"
        ).json()
        before_three = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages"
            "?limit=2&before_turn=3"
        ).json()
        invalid_limit = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages?limit=0"
        )
        invalid_before = client.get(
            f"/api/v1/conversations/{conversation['id']}/messages?before_turn=0"
        )
        missing = client.get("/api/v1/conversations/does-not-exist/messages")
        missing_detail = client.get("/api/v1/conversations/does-not-exist")

    assert [turn["turn_index"] for turn in latest["turns"]] == [2, 3]
    assert latest["count"] == 2
    assert [turn["turn_index"] for turn in before_three["turns"]] == [1, 2]
    assert invalid_limit.status_code == 422
    assert invalid_before.status_code == 422
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"
    assert missing_detail.status_code == 404
    assert missing_detail.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"


def test_migration_is_idempotent_on_empty_and_legacy_databases(tmp_path):
    legacy_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(legacy_path)
    connection.execute(
        """
        CREATE TABLE runs (
            id VARCHAR(36) PRIMARY KEY,
            objective TEXT NOT NULL,
            status VARCHAR(32) NOT NULL,
            step_count INTEGER NOT NULL,
            active_elapsed_ms INTEGER NOT NULL,
            context_summary TEXT,
            recent_rounds_json TEXT NOT NULL,
            consecutive_malformed_responses INTEGER NOT NULL,
            final_answer TEXT,
            error_code VARCHAR(64),
            error_message TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            completed_at DATETIME
        )
        """
    )
    connection.commit()
    connection.close()

    for database_url in (
        f"sqlite:///{legacy_path}",
        f"sqlite:///{tmp_path / 'empty.db'}",
    ):
        database = Database(database_url)
        database.initialize()
        database.initialize()
        with database.engine.connect() as conn:
            columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(runs)"))
            }
            indexes = {
                row[1] for row in conn.execute(text("PRAGMA index_list(runs)"))
            }
        database.close()

        assert {"conversation_id", "turn_index", "client_message_id"} <= columns
        assert {
            "uq_runs_conversation_turn",
            "uq_runs_conversation_client_message",
            "uq_runs_active_turn",
        } <= indexes
