import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_serializer, field_validator


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        objective = value.strip()
        if not 1 <= len(objective) <= 4000:
            raise ValueError("Objective must contain between 1 and 4000 non-whitespace characters.")
        return objective


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        title = value.strip()
        if not title:
            return None
        if len(title) > 200:
            raise ValueError("Title must contain at most 200 characters.")
        return title


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    client_message_id: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()
        if not 1 <= len(content) <= 4000:
            raise ValueError("Content must contain between 1 and 4000 non-whitespace characters.")
        return content

    @field_validator("client_message_id")
    @classmethod
    def validate_client_message_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        client_message_id = value.strip()
        if not client_message_id:
            return None
        if len(client_message_id) > 128:
            raise ValueError("client_message_id must contain at most 128 characters.")
        return client_message_id


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]


class FinalAnswer(BaseModel):
    content: StrictStr

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("Final answer content must not be empty.")
        return content


class ToolAction(BaseModel):
    tool_call_id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)
    arguments: dict[str, Any]

    @field_validator("tool_call_id", "name")
    @classmethod
    def trim_identifiers(cls, value: str) -> str:
        return value.strip()


class PendingApprovalView(BaseModel):
    id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_at: datetime

    @classmethod
    def from_record(cls, record: Any) -> "PendingApprovalView":
        return cls(
            id=record.id,
            tool_call_id=record.tool_call_id,
            tool_name=record.tool_name,
            arguments=json.loads(record.arguments_json),
            requested_at=record.requested_at,
        )

    @field_serializer("requested_at")
    def serialize_requested_at(self, value: datetime) -> str:
        return format_utc(value)


class RunErrorView(BaseModel):
    code: str
    message: str


class RunView(BaseModel):
    id: str
    objective: str
    status: RunStatus
    step_count: int
    active_elapsed_ms: int
    final_answer: str | None
    error: RunErrorView | None
    pending_approval: PendingApprovalView | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_record(cls, record: Any, pending_approval: Any | None = None) -> "RunView":
        error = None
        if record.error_code:
            error = RunErrorView(code=record.error_code, message=record.error_message)
        pending = (
            PendingApprovalView.from_record(pending_approval)
            if record.status == RunStatus.WAITING_APPROVAL and pending_approval is not None
            else None
        )
        return cls(
            id=record.id,
            objective=record.objective,
            status=record.status,
            step_count=record.step_count,
            active_elapsed_ms=record.active_elapsed_ms,
            final_answer=record.final_answer,
            error=error,
            pending_approval=pending,
            created_at=record.created_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
        )

    @field_serializer("created_at", "updated_at", "completed_at")
    def serialize_timestamps(self, value: datetime | None) -> str | None:
        return format_utc(value) if value is not None else None


class TurnView(BaseModel):
    turn_index: int
    run_id: str
    status: RunStatus
    user_content: str
    assistant_content: str | None
    error: RunErrorView | None
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_record(cls, record: Any) -> "TurnView":
        error = None
        if record.error_code:
            error = RunErrorView(code=record.error_code, message=record.error_message)
        assistant_content = (
            record.final_answer if record.status == RunStatus.COMPLETED else None
        )
        return cls(
            turn_index=record.turn_index,
            run_id=record.id,
            status=record.status,
            user_content=record.objective,
            assistant_content=assistant_content,
            error=error,
            created_at=record.created_at,
            completed_at=record.completed_at,
        )

    @field_serializer("created_at", "completed_at")
    def serialize_timestamps(self, value: datetime | None) -> str | None:
        return format_utc(value) if value is not None else None


class ConversationView(BaseModel):
    id: str
    title: str | None
    status: Literal["ACTIVE", "ARCHIVED"]
    turn_count: int
    turns: list[TurnView] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(
        cls,
        record: Any,
        turn_count: int,
        turns: list[TurnView] | None = None,
    ) -> "ConversationView":
        return cls(
            id=record.id,
            title=record.title,
            status=record.status,
            turn_count=turn_count,
            turns=turns or [],
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @field_serializer("created_at", "updated_at")
    def serialize_timestamps(self, value: datetime) -> str:
        return format_utc(value)


class TurnPageView(BaseModel):
    conversation_id: str
    count: int
    turns: list[TurnView]


class RunEventView(BaseModel):
    sequence: int
    event_type: str
    payload: dict[str, Any]
    latency_ms: int | None
    created_at: datetime

    @classmethod
    def from_record(cls, record: Any) -> "RunEventView":
        return cls(
            sequence=record.sequence,
            event_type=record.event_type,
            payload=json.loads(record.payload_json),
            latency_ms=record.latency_ms,
            created_at=record.created_at,
        )

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return format_utc(value)


class TraceView(BaseModel):
    run_id: str
    count: int
    events: list[RunEventView]


class HealthView(BaseModel):
    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"
    llm_configured: bool


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
