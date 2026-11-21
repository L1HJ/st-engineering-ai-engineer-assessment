import socket

import pytest
from fastapi.testclient import TestClient

from agent_harness.config import Settings
from agent_harness.llm import ModelResponse
from agent_harness.main import create_app
from agent_harness.reliability import SystemSleeper
from agent_harness.tools import ToolRegistry
from tests.fakes.fake_clock import FakeClock, FakeSleeper
from tests.fakes.fake_kimi import FakeKimi


@pytest.fixture(autouse=True)
def block_real_network_and_backoff(request, monkeypatch):
    if request.node.get_closest_marker("kimi_smoke") is not None:
        return

    def network_blocked(*args, **kwargs):
        raise AssertionError("Default tests must not open network connections.")

    async def backoff_blocked(*args, **kwargs):
        raise AssertionError("Default tests must inject Fake Sleeper instead of real backoff.")

    monkeypatch.setattr(socket, "create_connection", network_blocked)
    monkeypatch.setattr(socket.socket, "connect", network_blocked)
    monkeypatch.setattr(SystemSleeper, "sleep", backoff_blocked)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeper(clock: FakeClock) -> FakeSleeper:
    return FakeSleeper(clock)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        kimi_api_key="fake-secret-api-key",
        kimi_model="fake-kimi-model",
        database_url=f"sqlite:///{tmp_path / 'agent_harness_test.db'}",
    )


@pytest.fixture
def fake_kimi() -> FakeKimi:
    return FakeKimi([ModelResponse(content="The service is healthy.")])


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry.from_default_files()


@pytest.fixture
def app(settings, fake_kimi, tool_registry, clock, sleeper):
    return create_app(
        settings=settings,
        llm=fake_kimi,
        tool_registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
    )


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
