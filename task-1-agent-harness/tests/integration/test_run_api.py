import logging
from uuid import uuid4

from fastapi.testclient import TestClient

from agent_harness.llm import ModelResponse
from agent_harness.main import create_app
from agent_harness.observability import JsonFormatter
from tests.fakes.fake_kimi import FakeKimi


def test_direct_final_answer_creates_completed_run_and_ordered_trace(client, fake_kimi):
    response = client.post(
        "/api/v1/runs",
        json={"objective": "  Investigate checkout availability.  "},
    )

    assert response.status_code == 201
    run = response.json()
    assert run["objective"] == "Investigate checkout availability."
    assert run["status"] == "COMPLETED"
    assert run["step_count"] == 1
    assert run["final_answer"] == "The service is healthy."
    assert run["error"] is None
    assert run["pending_approval"] is None
    assert run["created_at"].endswith("Z")
    assert run["completed_at"].endswith("Z")
    assert len(fake_kimi.calls) == 1
    assert [
        schema["function"]["name"] for schema in fake_kimi.calls[0]["tool_schemas"]
    ] == ["search_knowledge_base", "get_service_status", "create_incident"]
    assert fake_kimi.calls[0]["messages"][1]["content"] == run["objective"]

    trace_response = client.get(f"/api/v1/runs/{run['id']}/events")
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["run_id"] == run["id"]
    assert trace["count"] == 4
    assert [event["sequence"] for event in trace["events"]] == [1, 2, 3, 4]
    assert [event["event_type"] for event in trace["events"]] == [
        "run_created",
        "llm_response",
        "run_state_changed",
        "run_finished",
    ]
    assert trace["events"][0]["payload"]["objective"] == run["objective"]
    assert trace["events"][1]["payload"]["step"] == 1
    assert trace["events"][2]["payload"] == {
        "from_status": "RUNNING",
        "to_status": "COMPLETED",
        "reason": "COMPLETED",
    }
    assert trace["events"][3]["payload"]["status"] == "COMPLETED"


def test_get_run_and_trace_never_advance_execution(client, fake_kimi):
    created = client.post("/api/v1/runs", json={"objective": "Check storage."}).json()

    first_run = client.get(f"/api/v1/runs/{created['id']}")
    second_run = client.get(f"/api/v1/runs/{created['id']}")
    first_trace = client.get(f"/api/v1/runs/{created['id']}/events").json()
    second_trace = client.get(f"/api/v1/runs/{created['id']}/events").json()

    assert first_run.status_code == 200
    assert first_run.json() == second_run.json()
    assert first_trace == second_trace
    assert len(fake_kimi.calls) == 1


def test_sqlite_history_survives_application_restart(settings, clock, sleeper):
    first_kimi = FakeKimi([ModelResponse(content="Persisted answer.")])
    first_app = create_app(settings=settings, llm=first_kimi, clock=clock, sleeper=sleeper)
    with TestClient(first_app) as first_client:
        created = first_client.post(
            "/api/v1/runs", json={"objective": "Verify durable history."}
        ).json()

    second_kimi = FakeKimi([])
    second_app = create_app(settings=settings, llm=second_kimi, clock=clock, sleeper=sleeper)
    with TestClient(second_app) as second_client:
        restored = second_client.get(f"/api/v1/runs/{created['id']}")
        trace = second_client.get(f"/api/v1/runs/{created['id']}/events")

    assert restored.status_code == 200
    assert restored.json()["status"] == "COMPLETED"
    assert restored.json()["final_answer"] == "Persisted answer."
    assert trace.status_code == 200
    assert trace.json()["count"] == 4
    assert second_kimi.calls == []


def test_request_validation_uses_uniform_error_envelope(client):
    requests = [
        {"objective": "   "},
        {"objective": "x" * 4001},
        {"objective": "Valid", "unexpected": True},
        {"objective": 42},
    ]

    for payload in requests:
        response = client.post("/api/v1/runs", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "The request failed validation."
        assert isinstance(body["error"]["details"], list)
        assert all(set(detail) == {"field", "message"} for detail in body["error"]["details"])


def test_missing_run_and_trace_return_uniform_404(client):
    run_id = str(uuid4())
    for path in (f"/api/v1/runs/{run_id}", f"/api/v1/runs/{run_id}/events"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "RUN_NOT_FOUND",
                "message": "The requested run does not exist.",
                "details": None,
            }
        }


def test_framework_http_errors_use_uniform_envelope(client):
    missing = client.get("/api/v1/not-a-route")
    assert missing.status_code == 404
    assert missing.json() == {
        "error": {
            "code": "NOT_FOUND",
            "message": "The requested API route does not exist.",
            "details": None,
        }
    }

    not_allowed = client.delete("/api/v1/health")
    assert not_allowed.status_code == 405
    assert not_allowed.json() == {
        "error": {
            "code": "METHOD_NOT_ALLOWED",
            "message": "The HTTP method is not supported for this route.",
            "details": None,
        }
    }
    assert "GET" in not_allowed.headers["allow"]


def test_api_trace_and_logs_do_not_leak_credentials(settings, clock, sleeper):
    secret = settings.kimi_api_key.get_secret_value()
    failing_kimi = FakeKimi([RuntimeError(f"Provider rejected {secret}")])
    app = create_app(settings=settings, llm=failing_kimi, clock=clock, sleeper=sleeper)
    stream = logging.StreamHandler()
    from io import StringIO

    output = StringIO()
    stream.stream = output
    stream.setFormatter(JsonFormatter([secret]))
    logger = logging.getLogger("agent_harness")
    logger.addHandler(stream)
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/runs", json={"objective": "Diagnose API errors."})
            trace = client.get(f"/api/v1/runs/{response.json()['id']}/events")
    finally:
        logger.removeHandler(stream)

    assert response.status_code == 201
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert secret not in response.text
    assert secret not in trace.text
    assert secret not in output.getvalue()
