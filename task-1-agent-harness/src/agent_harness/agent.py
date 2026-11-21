import json
import logging
from collections.abc import Callable

from agent_harness.config import Settings
from agent_harness.context import ContextManager, ConversationHistory
from agent_harness.llm import DecisionClient, LlmError, ModelResponse
from agent_harness.reliability import (
    ActiveTimeBudgetExceeded,
    Clock,
    RandomSource,
    Sleeper,
    SystemClock,
    SystemSleeper,
)
from agent_harness.schemas import FinalAnswer, RunStatus, RunView, ToolAction
from agent_harness.store import RunRecord, RunRepository
from agent_harness.tools import (
    ToolContext,
    ToolExecutionError,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
)

logger = logging.getLogger("agent_harness.runner")


class RunNotFoundError(LookupError):
    pass


class MalformedResponseError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _reject_nonfinite_json_constant(_constant: str) -> None:
    raise ValueError("Non-finite numbers are not valid JSON.")


class AgentRunner:
    def __init__(
        self,
        repository: RunRepository,
        llm: DecisionClient,
        settings: Settings,
        registry: ToolRegistry,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
        context: ContextManager | None = None,
        random_source: RandomSource | None = None,
    ) -> None:
        self._repository = repository
        self._llm = llm
        self._settings = settings
        self._registry = registry
        self._clock = clock or SystemClock()
        self._sleeper = sleeper or SystemSleeper()
        self._context = context or ContextManager()
        self._random_source = random_source

    async def create_and_run(self, objective: str) -> RunView:
        record = self._repository.create_run(
            objective.strip(),
            limits={
                "agent_max_steps": self._settings.agent_max_steps,
                "agent_active_time_limit_seconds": (
                    self._settings.agent_active_time_limit_seconds
                ),
                "llm_timeout_seconds": self._settings.llm_timeout_seconds,
            },
        )
        logger.info(
            "Run created.",
            extra={"run_id": record.id, "event_type": "run_created", "status": record.status},
        )
        return await self.run(record.id)

    async def decide_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
    ) -> RunView:
        self._repository.decide_approval(
            run_id,
            approval_id,
            decision,
            final_answer="Incident creation was not approved, so no incident was created.",
        )
        if decision == "reject":
            rejected = self._require_run(run_id)
            self._log_finished(rejected)
            return self._run_view(rejected)

        started_at = self._clock.monotonic()
        run = self._require_run(run_id)
        starting_elapsed_ms = run.active_elapsed_ms

        def elapsed_ms() -> int:
            segment_ms = max(0, int((self._clock.monotonic() - started_at) * 1000))
            return starting_elapsed_ms + segment_ms

        approval = self._repository.get_approval(run_id, approval_id)
        if approval is None or approval.status != "APPROVED":
            return self._finish_internal_error(run_id, elapsed_ms())
        try:
            arguments = json.loads(approval.arguments_json)
            typed_input = self._registry.validate_input(approval.tool_name, arguments)
        except (json.JSONDecodeError, ToolInputValidationError):
            return self._finish_internal_error(run_id, elapsed_ms())
        action = ToolAction(
            tool_call_id=approval.tool_call_id,
            name=approval.tool_name,
            arguments=arguments,
        )
        limit_ms = int(self._settings.agent_active_time_limit_seconds * 1000)
        result = await self._handle_tool_action(
            run,
            action,
            assistant_content=None,
            latency_ms=0,
            model_attempts=1,
            elapsed_ms=elapsed_ms,
            limit_ms=limit_ms,
            approved_resume=True,
        )
        if result is not None:
            return result
        return await self.run(run_id)

    async def run(self, run_id: str) -> RunView:
        initial_run = self._require_run(run_id)
        if initial_run.status != RunStatus.RUNNING:
            return self._run_view(initial_run)

        started_at = self._clock.monotonic()
        starting_elapsed_ms = initial_run.active_elapsed_ms
        history = self._conversation_history(initial_run)

        def elapsed_ms() -> int:
            segment_ms = max(0, int((self._clock.monotonic() - started_at) * 1000))
            return starting_elapsed_ms + segment_ms

        limit_ms = int(self._settings.agent_active_time_limit_seconds * 1000)
        pending_correction: str | None = None
        while True:
            run = self._require_run(run_id)
            if run.status != RunStatus.RUNNING:
                return self._run_view(run)
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            if run.step_count >= self._settings.agent_max_steps:
                return self._finish_limit(run.id, "MAX_STEPS_EXCEEDED", elapsed_ms())

            if self._context.needs_compaction(run, self._settings.context_window_rounds):
                result = await self._compact_context(run, elapsed_ms, limit_ms)
                if result is not None:
                    return result
                run = self._require_run(run_id)

            remaining_seconds = max(0, (limit_ms - elapsed_ms()) / 1000)
            decision_started_at = self._clock.monotonic()

            def on_llm_retry(payload: dict) -> None:
                self._repository.append_event(
                    run.id,
                    "retry_scheduled",
                    payload,
                    active_elapsed_ms=elapsed_ms(),
                )

            try:
                response = await self._llm.complete_decision(
                    messages=self._context.build_messages(run, pending_correction, history),
                    tool_schemas=self._registry.schemas(),
                    timeout_budget=remaining_seconds,
                    on_retry=on_llm_retry,
                )
            except ActiveTimeBudgetExceeded as exc:
                latency_ms = self._latency_since(decision_started_at)
                if isinstance(exc.cause, LlmError):
                    self._record_llm_error(run.id, exc.cause, latency_ms, elapsed_ms())
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            except LlmError as exc:
                latency_ms = self._latency_since(decision_started_at)
                self._record_llm_error(run.id, exc, latency_ms, elapsed_ms())
                if elapsed_ms() >= limit_ms:
                    return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
                return self._finish_failed(run.id, exc.code, exc.message, elapsed_ms())
            except Exception:
                latency_ms = self._latency_since(decision_started_at)
                self._repository.append_event(
                    run.id,
                    "llm_error",
                    {"error_code": "INTERNAL_ERROR", "attempt": 1},
                    latency_ms,
                    active_elapsed_ms=elapsed_ms(),
                )
                logger.error(
                    "Model decision failed with an unexpected error.",
                    extra={
                        "run_id": run.id,
                        "event_type": "llm_error",
                        "error_code": "INTERNAL_ERROR",
                        "latency_ms": latency_ms,
                    },
                )
                if elapsed_ms() >= limit_ms:
                    return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
                return self._finish_failed(
                    run.id,
                    "INTERNAL_ERROR",
                    "The run ended because an unexpected internal error occurred.",
                    elapsed_ms(),
                )

            latency_ms = self._latency_since(decision_started_at)
            try:
                action = self._normalize_action(run.id, response)
            except MalformedResponseError as exc:
                result = self._handle_malformed_response(
                    run,
                    exc.category,
                    latency_ms,
                    elapsed_ms,
                    limit_ms,
                )
                if result is not None:
                    return result
                pending_correction = exc.category
                continue

            pending_correction = None
            if isinstance(action, FinalAnswer):
                return self._complete_final_answer(
                    run,
                    action,
                    latency_ms,
                    response.attempts,
                    elapsed_ms,
                    limit_ms,
                )

            result = await self._handle_tool_action(
                run,
                action,
                response.content.strip() if response.content else None,
                latency_ms,
                response.attempts,
                elapsed_ms,
                limit_ms,
            )
            if result is not None:
                return result

    async def _compact_context(
        self,
        run: RunRecord,
        elapsed_ms: Callable[[], int],
        limit_ms: int,
    ) -> RunView | None:
        compaction_started_at = self._clock.monotonic()

        def on_summary_retry(payload: dict) -> None:
            self._repository.append_event(
                run.id,
                "retry_scheduled",
                {**payload, "operation": "context_compaction"},
                active_elapsed_ms=elapsed_ms(),
            )

        try:
            response = await self._llm.complete_summary(
                messages=self._context.build_summary_messages(
                    run,
                    self._settings.context_compact_rounds,
                ),
                timeout_budget=max(0, (limit_ms - elapsed_ms()) / 1000),
                on_retry=on_summary_retry,
            )
        except ActiveTimeBudgetExceeded as exc:
            latency_ms = self._latency_since(compaction_started_at)
            if isinstance(exc.cause, LlmError):
                return self._fail_compaction(
                    run.id,
                    exc.cause.code,
                    exc.cause.message,
                    exc.attempts,
                    latency_ms,
                    elapsed_ms,
                    limit_ms,
                    retryable=exc.cause.retryable,
                    time_limit_exceeded=True,
                )
            return self._fail_compaction(
                run.id,
                "ACTIVE_TIME_LIMIT_EXCEEDED",
                "Context compaction exhausted the active execution time budget.",
                exc.attempts,
                latency_ms,
                elapsed_ms,
                limit_ms,
                time_limit_exceeded=True,
            )
        except LlmError as exc:
            return self._fail_compaction(
                run.id,
                exc.code,
                exc.message,
                exc.attempts,
                self._latency_since(compaction_started_at),
                elapsed_ms,
                limit_ms,
                retryable=exc.retryable,
            )
        except Exception:
            return self._fail_compaction(
                run.id,
                "INTERNAL_ERROR",
                "The context summary request failed safely.",
                1,
                self._latency_since(compaction_started_at),
                elapsed_ms,
                limit_ms,
            )

        latency_ms = self._latency_since(compaction_started_at)
        if response.malformed_category is not None:
            return self._fail_compaction(
                run.id,
                "INVALID_SUMMARY_RESPONSE",
                "The model returned an invalid context summary response.",
                response.attempts,
                latency_ms,
                elapsed_ms,
                limit_ms,
            )
        summary = response.content.strip() if response.content else ""
        if not summary:
            return self._fail_compaction(
                run.id,
                "EMPTY_SUMMARY",
                "The model returned an empty context summary.",
                response.attempts,
                latency_ms,
                elapsed_ms,
                limit_ms,
            )

        try:
            compacted = self._repository.compact_context(
                run_id=run.id,
                expected_summary=run.context_summary,
                expected_recent_rounds_json=run.recent_rounds_json,
                compact_rounds=self._settings.context_compact_rounds,
                new_summary=summary,
                attempts=response.attempts,
                latency_ms=latency_ms,
                active_elapsed_ms=elapsed_ms(),
            )
        except Exception:
            return self._fail_compaction(
                run.id,
                "CONTEXT_PERSISTENCE_FAILED",
                "The compacted context could not be persisted.",
                response.attempts,
                latency_ms,
                elapsed_ms,
                limit_ms,
            )
        if not compacted:
            current = self._require_run(run.id)
            if current.status != RunStatus.RUNNING:
                return self._run_view(current)
            return self._fail_compaction(
                run.id,
                "CONTEXT_PERSISTENCE_FAILED",
                "The compacted context could not be persisted.",
                response.attempts,
                latency_ms,
                elapsed_ms,
                limit_ms,
            )

        logger.info(
            "Run context compacted.",
            extra={
                "run_id": run.id,
                "event_type": "context_compacted",
                "step_count": run.step_count,
                "attempt": response.attempts,
                "latency_ms": latency_ms,
            },
        )
        if elapsed_ms() >= limit_ms:
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        return None

    def _fail_compaction(
        self,
        run_id: str,
        error_code: str,
        error_message: str,
        attempts: int,
        latency_ms: int,
        elapsed_ms: Callable[[], int],
        limit_ms: int,
        *,
        retryable: bool = False,
        time_limit_exceeded: bool = False,
    ) -> RunView:
        self._repository.append_event(
            run_id,
            "context_compaction_failed",
            {
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "retryable": retryable,
                },
                "attempt": attempts,
                "context_unchanged": True,
            },
            latency_ms,
            active_elapsed_ms=elapsed_ms(),
        )
        logger.error(
            "Run context compaction failed.",
            extra={
                "run_id": run_id,
                "event_type": "context_compaction_failed",
                "error_code": error_code,
                "attempt": attempts,
                "latency_ms": latency_ms,
            },
        )
        if time_limit_exceeded or elapsed_ms() >= limit_ms:
            return self._finish_limit(run_id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        return self._finish_failed(
            run_id,
            "CONTEXT_COMPACTION_FAILED",
            "The run ended because context compaction failed.",
            elapsed_ms(),
        )

    def _normalize_action(self, run_id: str, response: ModelResponse) -> FinalAnswer | ToolAction:
        if response.malformed_category is not None:
            raise MalformedResponseError(response.malformed_category)
        if response.content is not None and not isinstance(response.content, str):
            raise MalformedResponseError("invalid_response_structure")
        if len(response.tool_calls) > 1:
            raise MalformedResponseError("multiple_tool_calls")
        if not response.tool_calls:
            content = response.content.strip() if response.content else ""
            if not content:
                raise MalformedResponseError("empty_response")
            return FinalAnswer(content=content)

        call = response.tool_calls[0]
        tool_call_id = call.id.strip() if isinstance(call.id, str) else ""
        tool_name = call.name.strip() if isinstance(call.name, str) else ""
        if not tool_call_id:
            raise MalformedResponseError("missing_tool_call_id")
        if not tool_name:
            raise MalformedResponseError("missing_tool_name")
        if tool_name not in self._registry.names:
            raise MalformedResponseError("unknown_tool")
        if self._repository.tool_call_id_exists(run_id, tool_call_id):
            raise MalformedResponseError("duplicate_tool_call_id")
        if call.arguments is not None and not isinstance(call.arguments, str):
            raise MalformedResponseError("invalid_arguments_json")
        try:
            arguments = json.loads(
                call.arguments or "",
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (TypeError, ValueError):
            raise MalformedResponseError("invalid_arguments_json") from None
        if not isinstance(arguments, dict):
            raise MalformedResponseError("arguments_not_object")
        return ToolAction(tool_call_id=tool_call_id, name=tool_name, arguments=arguments)

    def _complete_final_answer(
        self,
        run: RunRecord,
        action: FinalAnswer,
        latency_ms: int,
        attempts: int,
        elapsed_ms: Callable[[], int],
        limit_ms: int,
    ) -> RunView:
        step_count = run.step_count + 1
        recent_rounds = json.loads(run.recent_rounds_json)
        recent_rounds.append(
            {
                "step": step_count,
                "assistant": {"content": action.content, "tool_call": None},
                "observation": None,
            }
        )
        if not self._repository.record_llm_response(
            run_id=run.id,
            step_count=step_count,
            content=action.content,
            latency_ms=latency_ms,
            recent_rounds=recent_rounds,
            attempts=attempts,
            active_elapsed_ms=elapsed_ms(),
        ):
            return self._run_view(self._require_run(run.id))
        logger.info(
            "Model response received.",
            extra={
                "run_id": run.id,
                "event_type": "llm_response",
                "step_count": step_count,
                "attempt": attempts,
                "latency_ms": latency_ms,
            },
        )
        if elapsed_ms() >= limit_ms:
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        self._repository.finish_run(
            run.id,
            status=RunStatus.COMPLETED.value,
            active_elapsed_ms=elapsed_ms(),
            final_answer=action.content,
        )
        finished = self._require_run(run.id)
        self._log_finished(finished)
        return self._run_view(finished)

    async def _handle_tool_action(
        self,
        run: RunRecord,
        action: ToolAction,
        assistant_content: str | None,
        latency_ms: int,
        model_attempts: int,
        elapsed_ms: Callable[[], int],
        limit_ms: int,
        approved_resume: bool = False,
    ) -> RunView | None:
        step_count = run.step_count if approved_resume else run.step_count + 1
        if not approved_resume:
            if not self._repository.record_tool_action(
                run_id=run.id,
                step_count=step_count,
                assistant_content=assistant_content,
                tool_call_id=action.tool_call_id,
                tool_name=action.name,
                arguments=action.arguments,
                latency_ms=latency_ms,
                attempts=model_attempts,
                active_elapsed_ms=elapsed_ms(),
            ):
                return self._run_view(self._require_run(run.id))
            logger.info(
                "Tool action persisted.",
                extra={
                    "run_id": run.id,
                    "event_type": "tool_call",
                    "step_count": step_count,
                    "tool_name": action.name,
                    "tool_call_id": action.tool_call_id,
                },
            )
        if elapsed_ms() >= limit_ms:
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())

        try:
            typed_input = self._registry.validate_input(action.name, action.arguments)
        except ToolInputValidationError:
            observation = self._failure_observation(
                action.tool_call_id,
                "TOOL_INPUT_VALIDATION_ERROR",
                "The tool arguments failed schema validation.",
            )
            if not self._repository.record_tool_observation(
                run.id,
                action.tool_call_id,
                action.name,
                observation,
                0,
                active_elapsed_ms=elapsed_ms(),
            ):
                return self._finish_internal_error(run.id, elapsed_ms())
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            return None

        if self._registry.requires_approval(action.name) and not approved_resume:
            typed_arguments = typed_input.model_dump(mode="json")
            approval = self._repository.request_approval(
                run_id=run.id,
                tool_call_id=action.tool_call_id,
                tool_name=action.name,
                original_arguments=action.arguments,
                normalized_arguments=typed_arguments,
                active_elapsed_ms=elapsed_ms(),
            )
            if approval is None:
                current = self._require_run(run.id)
                if current.status != RunStatus.RUNNING:
                    return self._run_view(current)
                return self._finish_internal_error(run.id, elapsed_ms())
            logger.info(
                "Approval requested.",
                extra={
                    "run_id": run.id,
                    "event_type": "approval_requested",
                    "tool_name": action.name,
                    "tool_call_id": action.tool_call_id,
                },
            )
            return self._run_view(self._require_run(run.id))

        tool_started_at = self._clock.monotonic()
        remaining_seconds = max(0, (limit_ms - elapsed_ms()) / 1000)
        tool_attempts = 0

        def on_tool_attempt(attempt: int) -> None:
            nonlocal tool_attempts
            tool_attempts = attempt

        def on_tool_retry(payload: dict) -> None:
            self._repository.append_event(
                run.id,
                "retry_scheduled",
                {
                    **payload,
                    "tool_call_id": action.tool_call_id,
                    "tool_name": action.name,
                },
                active_elapsed_ms=elapsed_ms(),
            )

        context = ToolContext(
            run_id=run.id,
            tool_call_id=action.tool_call_id,
            idempotency_key=f"{run.id}:{action.tool_call_id}",
            clock=self._clock,
            repository=self._repository,
            timeout_seconds=self._settings.tool_timeout_seconds,
            max_retries=self._settings.tool_max_retries,
            retry_backoff_base_seconds=self._settings.retry_backoff_base_seconds,
            retry_backoff_max_seconds=self._settings.retry_backoff_max_seconds,
            sleeper=self._sleeper,
            random_source=self._random_source,
            on_retry=on_tool_retry,
            on_attempt=on_tool_attempt,
        )
        try:
            raw_output = await self._registry.execute(
                action.name,
                typed_input,
                context,
                remaining_seconds,
            )
            output = self._registry.validate_output(action.name, raw_output)
        except ActiveTimeBudgetExceeded as exc:
            tool_latency_ms = self._latency_since(tool_started_at)
            if isinstance(exc.cause, ToolExecutionError):
                self._append_tool_error(
                    run.id,
                    action,
                    exc.cause.code,
                    exc.cause.message,
                    exc.cause.attempts,
                    tool_latency_ms,
                    elapsed_ms(),
                )
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        except ToolOutputValidationError:
            tool_latency_ms = self._latency_since(tool_started_at)
            self._append_tool_error(
                run.id,
                action,
                "TOOL_OUTPUT_VALIDATION_ERROR",
                "The tool output failed schema validation.",
                tool_attempts or 1,
                tool_latency_ms,
                elapsed_ms(),
            )
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            return self._finish_failed(
                run.id,
                "TOOL_OUTPUT_VALIDATION_ERROR",
                "The tool returned output that failed schema validation.",
                elapsed_ms(),
            )
        except ToolExecutionError as exc:
            tool_latency_ms = self._latency_since(tool_started_at)
            if exc.code == "IDEMPOTENCY_CONFLICT":
                self._append_tool_error(
                    run.id,
                    action,
                    exc.code,
                    exc.message,
                    exc.attempts,
                    tool_latency_ms,
                    elapsed_ms(),
                )
                if elapsed_ms() >= limit_ms:
                    return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
                return self._finish_failed(
                    run.id,
                    exc.code,
                    "The incident idempotency key is bound to different arguments.",
                    elapsed_ms(),
                )
            observation = self._failure_observation(
                action.tool_call_id,
                exc.code,
                exc.message,
            )
            if not self._repository.record_tool_observation(
                run.id,
                action.tool_call_id,
                action.name,
                observation,
                tool_latency_ms,
                attempts=exc.attempts,
                active_elapsed_ms=elapsed_ms(),
            ):
                return self._finish_internal_error(run.id, elapsed_ms())
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            return None
        except Exception:
            tool_latency_ms = self._latency_since(tool_started_at)
            observation = self._failure_observation(
                action.tool_call_id,
                "TOOL_EXECUTION_FAILED",
                "The tool execution failed safely.",
            )
            if not self._repository.record_tool_observation(
                run.id,
                action.tool_call_id,
                action.name,
                observation,
                tool_latency_ms,
                attempts=tool_attempts or 1,
                active_elapsed_ms=elapsed_ms(),
            ):
                return self._finish_internal_error(run.id, elapsed_ms())
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            return None

        tool_latency_ms = self._latency_since(tool_started_at)
        observation = {
            "tool_call_id": action.tool_call_id,
            "ok": True,
            "data": output,
        }
        observation_size = len(
            json.dumps(observation, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if observation_size > self._settings.tool_result_max_bytes:
            self._append_tool_error(
                run.id,
                action,
                "TOOL_OUTPUT_TOO_LARGE",
                "The tool output exceeded the configured size limit.",
                tool_attempts or 1,
                tool_latency_ms,
                elapsed_ms(),
                size_bytes=observation_size,
            )
            if elapsed_ms() >= limit_ms:
                return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
            return self._finish_failed(
                run.id,
                "TOOL_OUTPUT_TOO_LARGE",
                "The tool returned output that exceeded the configured size limit.",
                elapsed_ms(),
            )
        if not self._repository.record_tool_observation(
            run.id,
            action.tool_call_id,
            action.name,
            observation,
            tool_latency_ms,
            attempts=tool_attempts or 1,
            active_elapsed_ms=elapsed_ms(),
        ):
            return self._finish_internal_error(run.id, elapsed_ms())
        logger.info(
            "Tool observation persisted.",
            extra={
                "run_id": run.id,
                "event_type": "tool_result",
                "step_count": step_count,
                "tool_name": action.name,
                "tool_call_id": action.tool_call_id,
                "attempt": tool_attempts or 1,
                "latency_ms": tool_latency_ms,
            },
        )
        if elapsed_ms() >= limit_ms:
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        return None

    def _handle_malformed_response(
        self,
        run: RunRecord,
        category: str,
        latency_ms: int,
        elapsed_ms: Callable[[], int],
        limit_ms: int,
    ) -> RunView | None:
        will_correct = (
            run.consecutive_malformed_responses + 1
            < self._settings.malformed_response_limit
        )
        malformed_count = self._repository.record_malformed_response(
            run.id,
            run.step_count + 1,
            category,
            latency_ms,
            will_correct=will_correct,
            active_elapsed_ms=elapsed_ms(),
        )
        if malformed_count is None:
            return self._run_view(self._require_run(run.id))
        if elapsed_ms() >= limit_ms:
            return self._finish_limit(run.id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms())
        if malformed_count >= self._settings.malformed_response_limit:
            return self._finish_failed(
                run.id,
                "MALFORMED_LLM_RESPONSE",
                "The model repeatedly returned responses that do not match the decision protocol.",
                elapsed_ms(),
            )
        return None

    def _record_llm_error(
        self,
        run_id: str,
        error: LlmError,
        latency_ms: int,
        active_elapsed_ms: int,
    ) -> None:
        self._repository.append_event(
            run_id,
            "llm_error",
            {
                "error_code": error.code,
                "message": error.message,
                "http_status": error.http_status,
                "attempt": error.attempts,
            },
            latency_ms,
            active_elapsed_ms=active_elapsed_ms,
        )
        logger.error(
            "Model decision failed.",
            extra={
                "run_id": run_id,
                "event_type": "llm_error",
                "error_code": error.code,
                "attempt": error.attempts,
                "latency_ms": latency_ms,
            },
        )

    def _append_tool_error(
        self,
        run_id: str,
        action: ToolAction,
        code: str,
        message: str,
        attempts: int,
        latency_ms: int,
        active_elapsed_ms: int,
        **extra_payload: int,
    ) -> None:
        self._repository.append_event(
            run_id,
            "tool_error",
            {
                "tool_call_id": action.tool_call_id,
                "tool_name": action.name,
                "error": {"code": code, "message": message, "retryable": False},
                "attempt": attempts,
                **extra_payload,
            },
            latency_ms,
            active_elapsed_ms=active_elapsed_ms,
        )

    @staticmethod
    def _failure_observation(tool_call_id: str, code: str, message: str) -> dict:
        return {
            "tool_call_id": tool_call_id,
            "ok": False,
            "error": {"code": code, "message": message, "retryable": False},
        }

    def _latency_since(self, started_at: float) -> int:
        return max(0, int((self._clock.monotonic() - started_at) * 1000))

    def _finish_limit(self, run_id: str, error_code: str, elapsed_ms: int) -> RunView:
        messages = {
            "MAX_STEPS_EXCEEDED": "The run reached the maximum number of decision steps.",
            "ACTIVE_TIME_LIMIT_EXCEEDED": "The run reached the active execution time limit.",
        }
        self._repository.finish_run(
            run_id,
            status=RunStatus.LIMIT_EXCEEDED.value,
            active_elapsed_ms=elapsed_ms,
            error_code=error_code,
            error_message=messages[error_code],
        )
        run = self._require_run(run_id)
        self._log_finished(run)
        return self._run_view(run)

    def _finish_internal_error(self, run_id: str, elapsed_ms: int) -> RunView:
        return self._finish_failed(
            run_id,
            "INTERNAL_ERROR",
            "The run ended because an unexpected internal error occurred.",
            elapsed_ms,
        )

    def _finish_failed(
        self,
        run_id: str,
        error_code: str,
        error_message: str,
        elapsed_ms: int,
    ) -> RunView:
        self._repository.finish_run(
            run_id,
            status=RunStatus.FAILED.value,
            active_elapsed_ms=elapsed_ms,
            error_code=error_code,
            error_message=error_message,
        )
        run = self._require_run(run_id)
        self._log_finished(run)
        return self._run_view(run)

    def _conversation_history(self, run: RunRecord) -> ConversationHistory | None:
        if run.conversation_id is None or run.turn_index is None:
            return None
        conversation = self._repository.get_conversation(run.conversation_id)
        if conversation is None:
            return None
        turns = self._repository.list_conversation_turns_before(
            run.conversation_id,
            run.turn_index,
        )
        messages: list[dict[str, str]] = []
        for turn in turns:
            if (
                turn.turn_index is None
                or turn.turn_index <= conversation.summary_through_turn_index
                or turn.status in ("RUNNING", "WAITING_APPROVAL")
            ):
                continue
            messages.append({"role": "user", "content": turn.objective})
            if turn.status == RunStatus.COMPLETED.value and turn.final_answer:
                messages.append({"role": "assistant", "content": turn.final_answer})
            elif turn.status == RunStatus.FAILED.value:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Historical turn data] Turn {turn.turn_index} ended with "
                            f"status FAILED (error code {turn.error_code or 'UNKNOWN'}). "
                            "No assistant answer was produced."
                        ),
                    }
                )
            elif turn.status == RunStatus.LIMIT_EXCEEDED.value:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Historical turn data] Turn {turn.turn_index} hit a limit "
                            f"(error code {turn.error_code or 'UNKNOWN'}). "
                            "No assistant answer was produced."
                        ),
                    }
                )
        if conversation.context_summary is None and not messages:
            return None
        return ConversationHistory(
            summary=conversation.context_summary,
            messages=messages,
        )

    def _run_view(self, record: RunRecord) -> RunView:
        pending = (
            self._repository.get_pending_approval(record.id)
            if record.status == RunStatus.WAITING_APPROVAL
            else None
        )
        return RunView.from_record(record, pending)

    def _require_run(self, run_id: str) -> RunRecord:
        run = self._repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    def _log_finished(self, run: RunRecord) -> None:
        logger.info(
            "Run finished.",
            extra={
                "run_id": run.id,
                "event_type": "run_finished",
                "status": run.status,
                "step_count": run.step_count,
                "active_elapsed_ms": run.active_elapsed_ms,
                "error_code": run.error_code,
            },
        )
