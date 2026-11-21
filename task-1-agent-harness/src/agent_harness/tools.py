import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, field_serializer, field_validator

from agent_harness.reliability import (
    ActiveTimeBudgetExceeded,
    RandomSource,
    RetryBackoff,
    RetryCallback,
    Sleeper,
    SystemSleeper,
)
from agent_harness.schemas import format_utc
from agent_harness.store import IdempotencyConflictError


class DatasetValidationError(RuntimeError):
    pass


class UnknownToolError(RuntimeError):
    pass


class ToolInputValidationError(RuntimeError):
    def __init__(self, fields: tuple[str, ...]) -> None:
        super().__init__("Tool input validation failed.")
        self.fields = fields


class ToolOutputValidationError(RuntimeError):
    pass


class TemporaryToolError(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool, attempts: int) -> None:
        messages = {
            "TOOL_TIMEOUT": "The tool did not finish within the configured budget.",
            "TOOL_EXECUTION_FAILED": "The tool execution failed safely.",
            "IDEMPOTENCY_CONFLICT": "The idempotency key is bound to different arguments.",
        }
        super().__init__(messages[code])
        self.code = code
        self.message = messages[code]
        self.retryable = retryable
        self.attempts = attempts


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


_RFC3339_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})"
)


def _parse_utc_rfc3339(value: Any) -> datetime:
    if not isinstance(value, str) or _RFC3339_PATTERN.fullmatch(value) is None:
        raise ValueError("Service timestamps must be RFC 3339 strings.")
    normalized = value[:10] + "T" + value[11:]
    if normalized[-1] in {"Z", "z"}:
        normalized = normalized[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError("Service timestamps must be valid RFC 3339 timestamps.") from None
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timedelta(0)
        or normalized.endswith("-00:00")
    ):
        raise ValueError("Service timestamps must use UTC.")
    return timestamp


class KnowledgeDocument(StrictModel):
    id: StrictStr = Field(min_length=1, max_length=64)
    title: StrictStr = Field(min_length=1, max_length=200)
    content: StrictStr = Field(min_length=1, max_length=2000)


class KnowledgeSearchInput(StrictModel):
    query: StrictStr = Field(min_length=1, max_length=500)


class KnowledgeSearchOutput(StrictModel):
    results: list[KnowledgeDocument]


class ServiceSnapshot(StrictModel):
    service_name: StrictStr = Field(min_length=1, max_length=100)
    status: Literal["healthy", "degraded", "outage", "maintenance"]
    description: StrictStr = Field(min_length=1, max_length=1000)
    updated_at: datetime

    @field_validator("updated_at", mode="before")
    @classmethod
    def parse_utc_rfc3339(cls, value: Any) -> datetime:
        return _parse_utc_rfc3339(value)


class ServiceStatusInput(StrictModel):
    service_name: StrictStr = Field(min_length=1, max_length=100)


class ServiceStatusOutput(StrictModel):
    service_name: StrictStr = Field(min_length=1, max_length=100)
    found: StrictBool
    status: Literal["healthy", "degraded", "outage", "maintenance", "unknown"]
    description: StrictStr = Field(min_length=1, max_length=1000)
    updated_at: datetime | None

    @field_validator("updated_at", mode="before")
    @classmethod
    def parse_optional_utc_rfc3339(cls, value: Any) -> datetime | None:
        if value is None:
            return None
        return _parse_utc_rfc3339(value)

    @field_serializer("updated_at")
    def serialize_updated_at(self, value: datetime | None) -> str | None:
        return format_utc(value) if value is not None else None


class IncidentCreateInput(StrictModel):
    title: StrictStr = Field(min_length=1, max_length=200)
    description: StrictStr = Field(min_length=1, max_length=4000)
    severity: Literal["P1", "P2", "P3", "P4"]


class IncidentCreateOutput(StrictModel):
    incident_id: StrictStr = Field(min_length=4, max_length=40, pattern="^inc_")
    status: Literal["OPEN"]
    severity: Literal["P1", "P2", "P3", "P4"]
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_utc_rfc3339(cls, value: Any) -> datetime:
        return _parse_utc_rfc3339(value)

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return format_utc(value)


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    tool_call_id: str
    idempotency_key: str
    clock: Any
    repository: Any | None = None
    timeout_seconds: float = 5
    max_retries: int = 2
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 8
    sleeper: Sleeper | None = None
    random_source: RandomSource | None = None
    on_retry: RetryCallback | None = None
    on_attempt: Callable[[int], None] | None = None


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    read_only: bool = True
    requires_approval: bool = False


class ToolRegistry:
    def __init__(
        self,
        knowledge_documents: list[KnowledgeDocument],
        services: list[ServiceSnapshot],
    ) -> None:
        self._knowledge_documents = tuple(knowledge_documents)
        self._services = {service.service_name: service for service in services}
        self._definitions = {
            "search_knowledge_base": ToolDefinition(
                name="search_knowledge_base",
                description="Search internal operations documentation by keywords.",
                input_model=KnowledgeSearchInput,
                output_model=KnowledgeSearchOutput,
                handler=self._search_knowledge_base,
            ),
            "get_service_status": ToolDefinition(
                name="get_service_status",
                description="Retrieve the current deterministic status snapshot for a service.",
                input_model=ServiceStatusInput,
                output_model=ServiceStatusOutput,
                handler=self._get_service_status,
            ),
            "create_incident": ToolDefinition(
                name="create_incident",
                description="Create an incident after explicit user approval.",
                input_model=IncidentCreateInput,
                output_model=IncidentCreateOutput,
                handler=self._create_incident,
                read_only=False,
                requires_approval=True,
            ),
        }

    @classmethod
    def from_files(cls, knowledge_base_path: Path, services_path: Path) -> "ToolRegistry":
        documents = _load_dataset(knowledge_base_path, KnowledgeDocument, "Knowledge base")
        services = _load_dataset(services_path, ServiceSnapshot, "Service status")
        _reject_duplicates(
            (document.id for document in documents),
            "Knowledge base document IDs must be unique.",
        )
        _reject_duplicates(
            (service.service_name for service in services),
            "Service names must be unique.",
        )
        return cls(documents, services)

    @classmethod
    def from_default_files(cls) -> "ToolRegistry":
        data_directory = Path(__file__).resolve().parents[2] / "data"
        return cls.from_files(
            data_directory / "knowledge_base.json",
            data_directory / "services.json",
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def _get_definition(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            raise UnknownToolError(f"Tool {name!r} is not registered.") from None

    def requires_approval(self, name: str) -> bool:
        return self._get_definition(name).requires_approval

    def schemas(self) -> list[dict[str, Any]]:
        return [self.get_schema(name) for name in self.names]

    def get_schema(self, name: str) -> dict[str, Any]:
        definition = self._get_definition(name)
        return {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_model.model_json_schema(),
            },
        }

    def validate_input(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        definition = self._get_definition(name)
        try:
            return definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            fields = tuple(
                ".".join(str(part) for part in error["loc"]) or "arguments"
                for error in exc.errors()
            )
            raise ToolInputValidationError(fields) from None

    async def execute(
        self,
        name: str,
        typed_input: BaseModel,
        context: ToolContext,
        timeout_budget: float,
    ) -> dict[str, Any]:
        definition = self._get_definition(name)
        deadline = context.clock.monotonic() + max(0, timeout_budget)
        backoff = RetryBackoff(
            context.retry_backoff_base_seconds,
            context.retry_backoff_max_seconds,
            context.clock,
            context.sleeper or SystemSleeper(),
            context.random_source,
        )
        for attempt in range(1, context.max_retries + 2):
            remaining = (
                max(0, timeout_budget)
                if attempt == 1
                else deadline - context.clock.monotonic()
            )
            if remaining <= 0:
                raise ActiveTimeBudgetExceeded(attempts=attempt - 1)
            if context.on_attempt is not None:
                context.on_attempt(attempt)
            try:
                return await asyncio.wait_for(
                    definition.handler(typed_input, context),
                    timeout=min(context.timeout_seconds, remaining),
                )
            except Exception as exc:
                error = _classify_tool_error(exc, attempt)
                if not error.retryable or attempt > context.max_retries:
                    raise error from exc
                await backoff.wait(
                    component="tool",
                    attempt=attempt,
                    retry_index=attempt - 1,
                    error_code=error.code,
                    deadline=deadline,
                    on_retry=context.on_retry,
                    cause=error,
                )
        raise AssertionError("The tool retry loop exited unexpectedly.")

    def validate_output(self, name: str, raw_output: Any) -> dict[str, Any]:
        definition = self._get_definition(name)
        try:
            output = definition.output_model.model_validate(raw_output)
        except ValidationError:
            raise ToolOutputValidationError("Tool output failed schema validation.") from None
        return output.model_dump(mode="json")

    async def _search_knowledge_base(
        self,
        typed_input: BaseModel,
        _: ToolContext,
    ) -> dict[str, Any]:
        assert isinstance(typed_input, KnowledgeSearchInput)
        keywords = typed_input.query.casefold().split()
        matches: list[tuple[int, KnowledgeDocument]] = []
        for document in self._knowledge_documents:
            searchable_text = f"{document.title} {document.content}".casefold()
            score = sum(searchable_text.count(keyword) for keyword in keywords)
            if score > 0:
                matches.append((score, document))
        matches.sort(key=lambda match: (-match[0], match[1].id))
        return {
            "results": [document.model_dump() for _, document in matches[:5]],
        }

    async def _get_service_status(
        self,
        typed_input: BaseModel,
        _: ToolContext,
    ) -> dict[str, Any]:
        assert isinstance(typed_input, ServiceStatusInput)
        service_name = typed_input.service_name
        service = self._services.get(service_name)
        if service is None:
            return {
                "service_name": service_name,
                "found": False,
                "status": "unknown",
                "description": "No status snapshot is available for the requested service.",
                "updated_at": None,
            }
        return {
            "service_name": service.service_name,
            "found": True,
            "status": service.status,
            "description": service.description,
            "updated_at": format_utc(service.updated_at),
        }


    async def _create_incident(
        self,
        typed_input: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        assert isinstance(typed_input, IncidentCreateInput)
        if context.repository is None:
            raise RuntimeError("Incident persistence is not configured.")
        incident = context.repository.find_or_create_incident(
            run_id=context.run_id,
            tool_call_id=context.tool_call_id,
            idempotency_key=context.idempotency_key,
            arguments=typed_input.model_dump(mode="json"),
        )
        return {
            "incident_id": incident.id,
            "status": incident.status,
            "severity": incident.severity,
            "created_at": format_utc(incident.created_at),
        }


def _classify_tool_error(exc: Exception, attempt: int) -> ToolExecutionError:
    if isinstance(exc, IdempotencyConflictError):
        return ToolExecutionError("IDEMPOTENCY_CONFLICT", retryable=False, attempts=attempt)
    if isinstance(exc, TimeoutError):
        return ToolExecutionError("TOOL_TIMEOUT", retryable=True, attempts=attempt)
    if isinstance(exc, (TemporaryToolError, ConnectionError)):
        return ToolExecutionError("TOOL_EXECUTION_FAILED", retryable=True, attempts=attempt)
    return ToolExecutionError("TOOL_EXECUTION_FAILED", retryable=False, attempts=attempt)


def _load_dataset(path: Path, model: type[BaseModel], label: str) -> list[Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DatasetValidationError(f"{label} dataset could not be loaded as valid JSON.") from None
    if not isinstance(raw, list):
        raise DatasetValidationError(f"{label} dataset must be a top-level array.")
    try:
        return [model.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise DatasetValidationError(
            f"{label} dataset failed schema validation with {len(exc.errors())} error(s)."
        ) from None


def _reject_duplicates(values: Any, message: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise DatasetValidationError(message)
        seen.add(value)
