import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_harness.agent import AgentRunner, Clock, Sleeper
from agent_harness.config import Settings, load_settings
from agent_harness.conversation import ConversationService
from agent_harness.llm import DecisionClient, KimiAdapter
from agent_harness.observability import configure_logging
from agent_harness.reliability import RandomSource
from agent_harness.schemas import (
    ApprovalDecisionRequest,
    ConversationView,
    CreateConversationRequest,
    CreateRunRequest,
    ErrorBody,
    ErrorResponse,
    HealthView,
    RunEventView,
    RunView,
    SendMessageRequest,
    TraceView,
    TurnPageView,
    TurnView,
)
from agent_harness.store import (
    ApprovalDecisionError,
    ConversationArchivedError,
    ConversationBusyError,
    ConversationNotFoundError,
    ConversationTurnLimitError,
    Database,
    RunRepository,
)
from agent_harness.tools import ToolRegistry

logger = logging.getLogger("agent_harness.api")
_application: FastAPI | None = None


class ApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def create_app(
    settings: Settings | None = None,
    llm: DecisionClient | None = None,
    tool_registry: ToolRegistry | None = None,
    database: Database | None = None,
    repository: RunRepository | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    random_source: RandomSource | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level, [settings.kimi_api_key.get_secret_value()])
    if repository is not None:
        database = repository.database
    else:
        database = database or Database(settings.database_url)
        repository = RunRepository(database, clock.utc_now if clock else None)
    llm = llm or KimiAdapter(
        api_key=settings.kimi_api_key.get_secret_value(),
        model=settings.kimi_model,
        base_url=str(settings.kimi_base_url),
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_base_seconds=settings.retry_backoff_base_seconds,
        retry_backoff_max_seconds=settings.retry_backoff_max_seconds,
        clock=clock,
        sleeper=sleeper,
        random_source=random_source,
    )
    tool_registry = tool_registry or ToolRegistry.from_default_files()
    runner = AgentRunner(
        repository,
        llm,
        settings,
        registry=tool_registry,
        clock=clock,
        sleeper=sleeper,
        random_source=random_source,
    )
    conversation_service = ConversationService(
        repository,
        llm,
        settings,
        runner,
        clock=clock,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        interrupted_count = repository.mark_interrupted_runs()
        if interrupted_count:
            logger.warning(
                "Interrupted runs reconciled during startup.",
                extra={"event_type": "startup_reconciliation", "details": {"count": interrupted_count}},
            )
        yield
        close = getattr(llm, "aclose", None)
        if close is not None:
            await close()
        database.close()

    app = FastAPI(
        title="Operations Assistant Agent Harness",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.repository = repository
    app.state.llm = llm
    app.state.tool_registry = tool_registry
    app.state.runner = runner
    app.state.conversation_service = conversation_service

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            code = "NOT_FOUND"
            message = "The requested API route does not exist."
        elif exc.status_code == 405:
            code = "METHOD_NOT_ALLOWED"
            message = "The HTTP method is not supported for this route."
        else:
            code = "HTTP_ERROR"
            message = "The request could not be completed."
        return _error_response(
            exc.status_code,
            code,
            message,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        return _error_response(
            422,
            "VALIDATION_ERROR",
            "The request failed validation.",
            details,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        logger.error(
            "Unhandled API error.",
            extra={"event_type": "internal_error", "error_code": "INTERNAL_ERROR"},
        )
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "The request could not be completed because of an internal error.",
        )

    @app.post("/api/v1/runs", response_model=RunView, status_code=201)
    async def create_run(request: CreateRunRequest) -> RunView:
        return await runner.create_and_run(request.objective)

    @app.post("/api/v1/conversations", response_model=ConversationView, status_code=201)
    def create_conversation(request: CreateConversationRequest) -> ConversationView:
        record = conversation_service.create_conversation(request.title)
        return ConversationView.from_record(record, turn_count=0)

    @app.post(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=RunView,
        status_code=201,
    )
    async def send_conversation_message(
        conversation_id: str,
        request: SendMessageRequest,
        response: Response,
    ) -> RunView:
        try:
            view, replayed = await conversation_service.send_message(
                conversation_id,
                request.content,
                request.client_message_id,
            )
        except ConversationNotFoundError:
            raise ApiError(
                404,
                "CONVERSATION_NOT_FOUND",
                "The requested conversation does not exist.",
            ) from None
        except ConversationArchivedError:
            raise ApiError(
                409,
                "CONVERSATION_ARCHIVED",
                "The conversation is archived and cannot accept new messages.",
            ) from None
        except ConversationBusyError as exc:
            details = {
                "active_run_id": exc.active_run_id,
                "active_status": exc.active_status,
            }
            if exc.pending_approval_id is not None:
                details["pending_approval_id"] = exc.pending_approval_id
            raise ApiError(
                409,
                "CONVERSATION_BUSY",
                "The conversation already has an active turn.",
                details,
            ) from None
        except ConversationTurnLimitError:
            raise ApiError(
                409,
                "CONVERSATION_TURN_LIMIT_EXCEEDED",
                "The conversation reached its turn limit.",
            ) from None
        if replayed:
            response.status_code = 200
        return view

    @app.get(
        "/api/v1/conversations/{conversation_id}",
        response_model=ConversationView,
    )
    def get_conversation(conversation_id: str) -> ConversationView:
        try:
            record = conversation_service.require_conversation(conversation_id)
        except ConversationNotFoundError:
            raise ApiError(
                404,
                "CONVERSATION_NOT_FOUND",
                "The requested conversation does not exist.",
            ) from None
        turns = conversation_service.recent_turns(conversation_id, 20)
        return ConversationView.from_record(
            record,
            turn_count=repository.count_conversation_turns(conversation_id),
            turns=[TurnView.from_record(turn) for turn in turns],
        )

    @app.get(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=TurnPageView,
    )
    def get_conversation_messages(
        conversation_id: str,
        limit: int = Query(20, ge=1, le=100),
        before_turn: int | None = Query(None, ge=1),
    ) -> TurnPageView:
        try:
            turns = conversation_service.list_turns_page(
                conversation_id,
                limit,
                before_turn,
            )
        except ConversationNotFoundError:
            raise ApiError(
                404,
                "CONVERSATION_NOT_FOUND",
                "The requested conversation does not exist.",
            ) from None
        return TurnPageView(
            conversation_id=conversation_id,
            count=len(turns),
            turns=[TurnView.from_record(turn) for turn in turns],
        )

    @app.get("/api/v1/runs/{run_id}", response_model=RunView)
    def get_run(run_id: str) -> RunView:
        record = repository.get_run(run_id)
        if record is None:
            raise ApiError(404, "RUN_NOT_FOUND", "The requested run does not exist.")
        pending = repository.get_pending_approval(run_id) if record.status == "WAITING_APPROVAL" else None
        return RunView.from_record(record, pending)

    @app.post("/api/v1/runs/{run_id}/approvals/{approval_id}", response_model=RunView)
    async def decide_approval(
        run_id: str,
        approval_id: str,
        request: ApprovalDecisionRequest,
    ) -> RunView:
        try:
            return await runner.decide_approval(run_id, approval_id, request.decision)
        except ApprovalDecisionError as exc:
            if exc.code == "RUN_NOT_FOUND":
                raise ApiError(404, exc.code, "The requested run does not exist.") from None
            if exc.code == "APPROVAL_NOT_FOUND":
                raise ApiError(404, exc.code, "The requested approval does not exist.") from None
            if exc.code == "APPROVAL_ALREADY_DECIDED":
                raise ApiError(409, exc.code, "The approval has already been decided.") from None
            raise ApiError(
                409,
                "RUN_INVALID_STATE",
                "The run is not waiting for this approval decision.",
            ) from None

    @app.get("/api/v1/runs/{run_id}/events", response_model=TraceView)
    def get_run_events(run_id: str) -> TraceView:
        if repository.get_run(run_id) is None:
            raise ApiError(404, "RUN_NOT_FOUND", "The requested run does not exist.")
        events = repository.list_events(run_id)
        return TraceView(
            run_id=run_id,
            count=len(events),
            events=[RunEventView.from_record(event) for event in events],
        )

    @app.get("/api/v1/health", response_model=HealthView)
    def health() -> HealthView:
        try:
            database.health()
        except Exception:
            logger.error(
                "Database health check failed.",
                extra={"event_type": "health_check", "error_code": "DATABASE_UNHEALTHY"},
            )
            raise ApiError(
                503,
                "DATABASE_UNHEALTHY",
                "The database health check failed.",
            ) from None
        return HealthView(
            llm_configured=bool(
                settings.kimi_api_key.get_secret_value().strip() and settings.kimi_model.strip()
            )
        )

    return app


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json"),
        headers=headers,
    )


def __getattr__(name: str) -> Any:
    if name == "app":
        global _application
        if _application is None:
            _application = create_app()
        return _application
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")
