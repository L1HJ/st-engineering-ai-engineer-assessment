from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from agent_harness.llm import KimiAdapter, LlmError
from agent_harness.reliability import ActiveTimeBudgetExceeded
from tests.fakes.fake_clock import FakeClock, FakeSleeper


@pytest.mark.asyncio
async def test_kimi_adapter_uses_configured_model_and_normalizes_final_answer():
    requests = []

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content="A direct answer.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    adapter = KimiAdapter(
        api_key="unused-by-fake-client",
        model="configured-kimi-model",
        base_url="https://api.moonshot.cn/v1",
        client=client,
    )
    messages = [{"role": "user", "content": "Check the service."}]

    response = await adapter.complete_decision(messages, [], timeout_budget=12.5)

    assert response.content == "A direct answer."
    assert response.tool_calls == ()
    assert requests == [
        {
            "model": "configured-kimi-model",
            "messages": messages,
            "timeout": 12.5,
        }
    ]


class ScriptedCompletions:
    def __init__(self, items):
        self._items = list(items)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def completion(content="Recovered answer."):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def status_error(status, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.moonshot.cn/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers)
    return APIStatusError("Provider error", response=response, body=None)


def build_adapter(items, clock, sleeper, **overrides):
    completions = ScriptedCompletions(items)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    options = {
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_backoff_base_seconds": 0.5,
        "retry_backoff_max_seconds": 8,
        "random_source": lambda: 0.5,
    }
    options.update(overrides)
    adapter = KimiAdapter(
        api_key="unused-by-fake-client",
        model="configured-kimi-model",
        base_url="https://api.moonshot.cn/v1",
        client=client,
        clock=clock,
        sleeper=sleeper,
        **options,
    )
    return adapter, completions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (TimeoutError("Read timeout"), "LLM_TIMEOUT"),
        (ConnectionError("Connection failed"), "LLM_UNAVAILABLE"),
        (status_error(408), "LLM_UNAVAILABLE"),
        (status_error(409), "LLM_UNAVAILABLE"),
        (status_error(429), "LLM_RATE_LIMITED"),
        (status_error(503), "LLM_UNAVAILABLE"),
    ],
)
async def test_transient_model_errors_retry_then_succeed(error, expected_code):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([error, completion()], clock, sleeper)
    retries = []

    response = await adapter.complete_decision([], [], 10, on_retry=retries.append)

    assert response.content == "Recovered answer."
    assert response.attempts == 2
    assert len(completions.requests) == 2
    assert completions.requests[0]["timeout"] == 10
    assert completions.requests[1]["timeout"] == pytest.approx(9.75)
    assert sleeper.calls == [0.25]
    assert retries == [
        {
            "component": "llm",
            "attempt": 1,
            "retry_index": 0,
            "delay_seconds": 0.25,
            "error_code": expected_code,
            "remaining_budget_seconds": 10,
        }
    ]


@pytest.mark.asyncio
async def test_model_retry_exhaustion_preserves_safe_error_and_attempt_count():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter(
        [TimeoutError("secret provider detail")] * 4,
        clock,
        sleeper,
    )

    with pytest.raises(LlmError) as captured:
        await adapter.complete_decision([], [], 20)

    assert captured.value.code == "LLM_TIMEOUT"
    assert captured.value.attempts == 4
    assert "secret" not in captured.value.message
    assert len(completions.requests) == 4
    assert len(sleeper.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, "LLM_INVALID_REQUEST"),
        (401, "LLM_AUTHENTICATION_FAILED"),
        (403, "LLM_AUTHENTICATION_FAILED"),
        (404, "LLM_INVALID_REQUEST"),
        (422, "LLM_INVALID_REQUEST"),
        (418, "LLM_REQUEST_FAILED"),
    ],
)
async def test_permanent_model_http_errors_are_not_retried(status, expected_code):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([status_error(status)], clock, sleeper)

    with pytest.raises(LlmError) as captured:
        await adapter.complete_decision([], [], 10)

    assert captured.value.code == expected_code
    assert captured.value.http_status == status
    assert captured.value.attempts == 1
    assert len(completions.requests) == 1
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_retry_after_takes_precedence_and_remaining_budget_clips_next_timeout():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter(
        [status_error(429, "2"), completion()],
        clock,
        sleeper,
    )
    retries = []

    response = await adapter.complete_decision([], [], 10, on_retry=retries.append)

    assert response.attempts == 2
    assert sleeper.calls == [2]
    assert retries[0]["delay_seconds"] == 2
    assert completions.requests[1]["timeout"] == 8


@pytest.mark.asyncio
async def test_retry_after_beyond_run_budget_does_not_sleep_or_start_new_attempt():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([status_error(429, "2")], clock, sleeper)
    retries = []

    with pytest.raises(ActiveTimeBudgetExceeded) as captured:
        await adapter.complete_decision([], [], 1, on_retry=retries.append)

    assert isinstance(captured.value.cause, LlmError)
    assert captured.value.cause.code == "LLM_RATE_LIMITED"
    assert len(completions.requests) == 1
    assert sleeper.calls == []
    assert retries == []


@pytest.mark.asyncio
async def test_full_jitter_uses_exponential_cap():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, _ = build_adapter(
        [TimeoutError(), TimeoutError(), completion()],
        clock,
        sleeper,
        max_retries=2,
        retry_backoff_max_seconds=0.8,
        random_source=lambda: 1.0,
    )

    response = await adapter.complete_decision([], [], 10)

    assert response.attempts == 3
    assert sleeper.calls == [0.5, 0.8]


@pytest.mark.asyncio
async def test_invalid_sdk_response_structure_is_malformed_without_http_retry():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([SimpleNamespace(choices=[])], clock, sleeper)
    empty = await adapter.complete_decision([], [], 10)
    assert empty.malformed_category is None

    adapter, completions = build_adapter([SimpleNamespace()], clock, sleeper)
    malformed = await adapter.complete_decision([], [], 10)

    assert malformed.malformed_category == "invalid_response_structure"
    assert len(completions.requests) == 1
    assert sleeper.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("summary timeout"), status_error(503)])
async def test_summary_reuses_retries_and_never_sends_tools(error):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter(
        [error, completion("  Updated context summary.  ")],
        clock,
        sleeper,
    )
    retries = []
    messages = [{"role": "user", "content": '{"rounds_to_compact":[]}'}]

    response = await adapter.complete_summary(messages, 10, on_retry=retries.append)

    assert response.content == "  Updated context summary.  "
    assert response.attempts == 2
    assert len(completions.requests) == 2
    assert all("tools" not in request for request in completions.requests)
    assert completions.requests[0]["messages"] == messages
    assert completions.requests[0]["timeout"] == 10
    assert completions.requests[1]["timeout"] == pytest.approx(9.75)
    assert sleeper.calls == [0.25]
    assert retries[0]["component"] == "llm"


@pytest.mark.asyncio
async def test_summary_permanent_http_error_is_not_retried():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([status_error(400)], clock, sleeper)

    with pytest.raises(LlmError) as captured:
        await adapter.complete_summary([], 10)

    assert captured.value.code == "LLM_INVALID_REQUEST"
    assert captured.value.attempts == 1
    assert len(completions.requests) == 1
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_summary_empty_and_invalid_structures_do_not_retry():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter, completions = build_adapter([SimpleNamespace(choices=[])], clock, sleeper)

    empty = await adapter.complete_summary([], 10)

    assert empty.content is None
    assert empty.malformed_category is None
    assert len(completions.requests) == 1

    adapter, completions = build_adapter([SimpleNamespace()], clock, sleeper)
    malformed = await adapter.complete_summary([], 10)

    assert malformed.malformed_category == "invalid_response_structure"
    assert len(completions.requests) == 1
    assert sleeper.calls == []
