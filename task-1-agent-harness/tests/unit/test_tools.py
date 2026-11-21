from dataclasses import replace

import pytest

from agent_harness.reliability import ActiveTimeBudgetExceeded
from agent_harness.tools import (
    KnowledgeDocument,
    TemporaryToolError,
    ToolContext,
    ToolExecutionError,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
)
from tests.fakes.fake_clock import FakeClock, FakeSleeper


def execution_context(**overrides) -> ToolContext:
    clock = overrides.pop("clock", None) or FakeClock()
    options = {
        "run_id": "run-test",
        "tool_call_id": "call-test",
        "idempotency_key": "run-test:call-test",
        "clock": clock,
        "random_source": lambda: 0.5,
    }
    options.update(overrides)
    return ToolContext(**options)


async def execute_validated(registry: ToolRegistry, name: str, arguments: dict):
    typed_input = registry.validate_input(name, arguments)
    raw_output = await registry.execute(name, typed_input, execution_context(), 5)
    return registry.validate_output(name, raw_output)


def test_registry_exposes_read_only_and_approval_tool_schemas(tool_registry):
    assert tool_registry.names == (
        "search_knowledge_base",
        "get_service_status",
        "create_incident",
    )
    assert not tool_registry.requires_approval("search_knowledge_base")
    assert not tool_registry.requires_approval("get_service_status")
    assert tool_registry.requires_approval("create_incident")
    for schema in tool_registry.schemas():
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["additionalProperties"] is False
        assert schema["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_knowledge_search_is_case_insensitive_and_returns_empty_results(tool_registry):
    response = await execute_validated(
        tool_registry,
        "search_knowledge_base",
        {"query": "  CHECKOUT ERRORS  "},
    )

    assert response["results"]
    assert response["results"][0]["id"] == "kb-001"
    empty = await execute_validated(
        tool_registry,
        "search_knowledge_base",
        {"query": "no-such-runbook-token"},
    )
    assert empty == {"results": []}


@pytest.mark.asyncio
async def test_knowledge_search_uses_unicode_casefold():
    registry = ToolRegistry(
        [KnowledgeDocument(id="kb-unicode", title="Straße runbook", content="Escalation guide.")],
        [],
    )

    response = await execute_validated(
        registry,
        "search_knowledge_base",
        {"query": "STRASSE"},
    )

    assert [result["id"] for result in response["results"]] == ["kb-unicode"]


@pytest.mark.asyncio
async def test_knowledge_search_sorts_by_score_then_id_and_limits_results():
    documents = [
        KnowledgeDocument(id="kb-001", title="Guide", content="error error error"),
        KnowledgeDocument(id="kb-002", title="Guide", content="error error"),
        *[
            KnowledgeDocument(id=f"kb-{index:03d}", title="Guide", content="error")
            for index in range(3, 9)
        ],
    ]
    registry = ToolRegistry(documents, [])

    response = await execute_validated(
        registry,
        "search_knowledge_base",
        {"query": "error"},
    )

    assert [result["id"] for result in response["results"]] == [
        "kb-001",
        "kb-002",
        "kb-003",
        "kb-004",
        "kb-005",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_name", "status"),
    [
        ("api-gateway", "healthy"),
        ("checkout", "degraded"),
        ("payments", "outage"),
        ("inventory", "maintenance"),
    ],
)
async def test_service_status_returns_each_known_state(tool_registry, service_name, status):
    response = await execute_validated(
        tool_registry,
        "get_service_status",
        {"service_name": service_name},
    )

    assert response["service_name"] == service_name
    assert response["found"] is True
    assert response["status"] == status
    assert response["updated_at"].endswith("Z")


@pytest.mark.asyncio
async def test_unknown_service_is_a_valid_unknown_observation(tool_registry):
    response = await execute_validated(
        tool_registry,
        "get_service_status",
        {"service_name": "  missing-service  "},
    )

    assert response == {
        "service_name": "missing-service",
        "found": False,
        "status": "unknown",
        "description": "No status snapshot is available for the requested service.",
        "updated_at": None,
    }


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_knowledge_base", {"query": "   "}),
        ("search_knowledge_base", {"query": "x" * 501}),
        ("search_knowledge_base", {"query": "valid", "extra": True}),
        ("search_knowledge_base", {"query": 42}),
        ("get_service_status", {"service_name": "   "}),
        ("get_service_status", {"service_name": "x" * 101}),
        ("get_service_status", {"service_name": "checkout", "extra": True}),
        ("get_service_status", {"service_name": 42}),
    ])
def test_tool_inputs_reject_blank_long_extra_and_wrong_typed_values(
    tool_registry, name, arguments
):
    with pytest.raises(ToolInputValidationError):
        tool_registry.validate_input(name, arguments)


@pytest.mark.parametrize(
    ("name", "raw_output"),
    [
        ("search_knowledge_base", {}),
        (
            "search_knowledge_base",
            {"results": [{"id": "kb-x", "title": 42, "content": "Invalid"}]},
        ),
        (
            "search_knowledge_base",
            {
                "results": [
                    {"id": "kb-x", "title": "Title", "content": "Content", "extra": True}
                ]
            },
        ),
        (
            "get_service_status",
            {"service_name": "checkout", "found": True, "status": "degraded"},
        ),
        (
            "get_service_status",
            {
                "service_name": "checkout",
                "found": 1,
                "status": "degraded",
                "description": "Invalid boolean",
                "updated_at": None,
            },
        ),
        (
            "get_service_status",
            {
                "service_name": "checkout",
                "found": True,
                "status": "degraded",
                "description": "Numeric timestamp",
                "updated_at": 0,
            },
        ),
        (
            "get_service_status",
            {
                "service_name": "checkout",
                "found": True,
                "status": "degraded",
                "description": "Non-RFC timestamp",
                "updated_at": "2026-11-21 08:00:00Z",
            },
        ),
        (
            "get_service_status",
            {
                "service_name": "checkout",
                "found": True,
                "status": "degraded",
                "description": "Extra field",
                "updated_at": None,
                "extra": True,
            },
        ),
    ])
def test_tool_outputs_reject_missing_wrong_typed_and_extra_fields(
    tool_registry, name, raw_output
):
    with pytest.raises(ToolOutputValidationError):
        tool_registry.validate_output(name, raw_output)


def install_handler(registry, handler, name="get_service_status"):
    definition = registry._definitions[name]
    registry._definitions[name] = replace(definition, handler=handler)


def service_output():
    return {
        "service_name": "checkout",
        "found": True,
        "status": "degraded",
        "description": "Elevated error rate",
        "updated_at": "2026-11-21T08:00:00Z",
    }


@pytest.mark.asyncio
async def test_temporary_tool_error_retries_then_succeeds(tool_registry):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = 0

    async def handler(typed_input, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TemporaryToolError("sensitive upstream detail")
        return service_output()

    install_handler(tool_registry, handler)
    retries = []
    attempts = []
    typed_input = tool_registry.validate_input(
        "get_service_status", {"service_name": "checkout"}
    )

    output = await tool_registry.execute(
        "get_service_status",
        typed_input,
        execution_context(
            clock=clock,
            sleeper=sleeper,
            on_retry=retries.append,
            on_attempt=attempts.append,
        ),
        10,
    )

    assert output == service_output()
    assert calls == 2
    assert attempts == [1, 2]
    assert sleeper.calls == [0.25]
    assert retries[0]["component"] == "tool"
    assert retries[0]["error_code"] == "TOOL_EXECUTION_FAILED"


@pytest.mark.asyncio
async def test_tool_retry_exhaustion_returns_safe_classified_error(tool_registry):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = 0

    async def handler(typed_input, context):
        nonlocal calls
        calls += 1
        raise TemporaryToolError("sensitive upstream detail")

    install_handler(tool_registry, handler)
    typed_input = tool_registry.validate_input(
        "get_service_status", {"service_name": "checkout"}
    )

    with pytest.raises(ToolExecutionError) as captured:
        await tool_registry.execute(
            "get_service_status",
            typed_input,
            execution_context(clock=clock, sleeper=sleeper),
            10,
        )

    assert captured.value.code == "TOOL_EXECUTION_FAILED"
    assert captured.value.attempts == 3
    assert "sensitive" not in captured.value.message
    assert calls == 3
    assert sleeper.calls == [0.25, 0.5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected_code", "expected_calls"),
    [
        (RuntimeError("Programming error"), "TOOL_EXECUTION_FAILED", 1),
        (TimeoutError("Handler timeout"), "TOOL_TIMEOUT", 3),
    ],
)
async def test_tool_permanent_and_timeout_errors_have_distinct_retry_behavior(
    tool_registry, exception, expected_code, expected_calls
):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = 0

    async def handler(typed_input, context):
        nonlocal calls
        calls += 1
        raise exception

    install_handler(tool_registry, handler)
    typed_input = tool_registry.validate_input(
        "get_service_status", {"service_name": "checkout"}
    )

    with pytest.raises(ToolExecutionError) as captured:
        await tool_registry.execute(
            "get_service_status",
            typed_input,
            execution_context(clock=clock, sleeper=sleeper),
            10,
        )

    assert captured.value.code == expected_code
    assert calls == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured_timeout", "run_budget", "expected"), [(5, 2, 2), (0.5, 2, 0.5)])
async def test_tool_wait_for_uses_attempt_and_run_timeouts(
    tool_registry, monkeypatch, configured_timeout, run_budget, expected
):
    timeouts = []

    async def fake_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        awaitable.close()
        raise TimeoutError()

    monkeypatch.setattr("agent_harness.tools.asyncio.wait_for", fake_wait_for)
    typed_input = tool_registry.validate_input(
        "get_service_status", {"service_name": "checkout"}
    )

    with pytest.raises(ToolExecutionError) as captured:
        await tool_registry.execute(
            "get_service_status",
            typed_input,
            execution_context(max_retries=0, timeout_seconds=configured_timeout),
            run_budget,
        )

    assert captured.value.code == "TOOL_TIMEOUT"
    assert timeouts == [expected]


@pytest.mark.asyncio
async def test_tool_backoff_never_exceeds_remaining_run_budget(tool_registry):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = 0

    async def handler(typed_input, context):
        nonlocal calls
        calls += 1
        raise TemporaryToolError()

    install_handler(tool_registry, handler)
    typed_input = tool_registry.validate_input(
        "get_service_status", {"service_name": "checkout"}
    )

    with pytest.raises(ActiveTimeBudgetExceeded) as captured:
        await tool_registry.execute(
            "get_service_status",
            typed_input,
            execution_context(
                clock=clock,
                sleeper=sleeper,
                random_source=lambda: 1.0,
                retry_backoff_base_seconds=2,
                retry_backoff_max_seconds=2,
            ),
            1,
        )

    assert isinstance(captured.value.cause, ToolExecutionError)
    assert calls == 1
    assert sleeper.calls == []
