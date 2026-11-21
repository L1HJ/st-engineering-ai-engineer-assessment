import os

import pytest
from fastapi.testclient import TestClient

from agent_harness.config import Settings
from agent_harness.main import create_app

pytestmark = pytest.mark.kimi_smoke


def test_real_kimi_direct_answer(tmp_path):
    if os.environ.get("RUN_KIMI_SMOKE") != "1":
        pytest.skip("Set RUN_KIMI_SMOKE=1 and select -m kimi_smoke to call the real API.")

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'kimi_smoke.db'}")
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/api/v1/runs",
            json={"objective": "Reply with a short acknowledgment. Do not use any tools."},
        )
        assert response.status_code == 201
        run = response.json()
        assert run["status"] == "COMPLETED"
        assert run["error"] is None
        assert run["final_answer"] is not None and run["final_answer"].strip()
        assert run["step_count"] == 1

        trace_response = client.get(f"/api/v1/runs/{run['id']}/events")
        assert trace_response.status_code == 200
        trace = trace_response.json()

    assert [event["sequence"] for event in trace["events"]] == list(
        range(1, trace["count"] + 1)
    )
    assert trace["events"][-1]["event_type"] == "run_finished"
