import logging

from agent_harness.agent import AgentRunner
from agent_harness.config import Settings
from agent_harness.context import ContextManager
from agent_harness.llm import DecisionClient, LlmError
from agent_harness.reliability import ActiveTimeBudgetExceeded, Clock, SystemClock
from agent_harness.schemas import RunView
from agent_harness.store import (
    ConversationNotFoundError,
    ConversationRecord,
    RunRecord,
    RunRepository,
)

logger = logging.getLogger("agent_harness.conversation")

TERMINAL_TURN_STATUSES = ("COMPLETED", "FAILED", "LIMIT_EXCEEDED")


class _CompactionFailure(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        attempts: int,
        *,
        retryable: bool = False,
        time_limit_exceeded: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.attempts = attempts
        self.retryable = retryable
        self.time_limit_exceeded = time_limit_exceeded


class ConversationService:
    def __init__(
        self,
        repository: RunRepository,
        llm: DecisionClient,
        settings: Settings,
        runner: AgentRunner,
        clock: Clock | None = None,
        context: ContextManager | None = None,
    ) -> None:
        self._repository = repository
        self._llm = llm
        self._settings = settings
        self._runner = runner
        self._clock = clock or SystemClock()
        self._context = context or ContextManager()

    def create_conversation(self, title: str | None) -> ConversationRecord:
        record = self._repository.create_conversation(title)
        logger.info(
            "Conversation created.",
            extra={
                "conversation_id": record.id,
                "event_type": "conversation_created",
                "status": record.status,
            },
        )
        return record

    def require_conversation(self, conversation_id: str) -> ConversationRecord:
        record = self._repository.get_conversation(conversation_id)
        if record is None:
            raise ConversationNotFoundError(conversation_id)
        return record

    def list_turns_page(
        self,
        conversation_id: str,
        limit: int,
        before_turn: int | None,
    ) -> list[RunRecord]:
        self.require_conversation(conversation_id)
        return self._repository.list_conversation_turns_page(
            conversation_id,
            limit,
            before_turn,
        )

    def recent_turns(self, conversation_id: str, limit: int) -> list[RunRecord]:
        return self._repository.list_conversation_turns_page(conversation_id, limit)

    async def send_message(
        self,
        conversation_id: str,
        content: str,
        client_message_id: str | None,
    ) -> tuple[RunView, bool]:
        record, replayed = self._repository.create_conversation_turn(
            conversation_id,
            content,
            client_message_id,
            limits=self._limits(),
            max_turns=self._settings.conversation_max_turns,
        )
        if replayed:
            return self._run_view(record), True

        logger.info(
            "Conversation turn started.",
            extra={
                "run_id": record.id,
                "conversation_id": conversation_id,
                "turn_index": record.turn_index,
                "event_type": "conversation_turn_started",
                "status": record.status,
            },
        )
        started_at = self._clock.monotonic()
        try:
            await self._maybe_compact(conversation_id, record, started_at)
        except _CompactionFailure as failure:
            self._finish_compaction_failure(
                record.id,
                failure,
                self._elapsed_ms(started_at),
            )
            return self._run_view(self._require_run(record.id)), False
        return await self._runner.run(record.id), False

    async def _maybe_compact(
        self,
        conversation_id: str,
        run: RunRecord,
        started_at: float,
    ) -> None:
        conversation = self.require_conversation(conversation_id)
        turns = self._repository.list_conversation_turns(conversation_id)
        turns_to_compact = self._turns_to_compact(conversation, turns)
        if turns_to_compact is None:
            return

        limit_ms = int(self._settings.agent_active_time_limit_seconds * 1000)
        remaining_seconds = max(0, (limit_ms - self._elapsed_ms(started_at)) / 1000)
        compaction_started_at = self._clock.monotonic()

        def on_summary_retry(payload: dict) -> None:
            self._repository.append_event(
                run.id,
                "retry_scheduled",
                {**payload, "operation": "conversation_compaction"},
                active_elapsed_ms=self._elapsed_ms(started_at),
            )

        try:
            response = await self._llm.complete_summary(
                messages=self._context.build_conversation_summary_messages(
                    conversation.context_summary,
                    [self._turn_summary_input(turn) for turn in turns_to_compact],
                ),
                timeout_budget=remaining_seconds,
                on_retry=on_summary_retry,
            )
        except ActiveTimeBudgetExceeded as exc:
            if isinstance(exc.cause, LlmError):
                raise _CompactionFailure(
                    exc.cause.code,
                    exc.cause.message,
                    exc.cause.attempts,
                    retryable=exc.cause.retryable,
                    time_limit_exceeded=True,
                ) from None
            raise _CompactionFailure(
                "ACTIVE_TIME_LIMIT_EXCEEDED",
                "Conversation compaction exhausted the active execution time budget.",
                exc.attempts,
                time_limit_exceeded=True,
            ) from None
        except LlmError as exc:
            raise _CompactionFailure(
                exc.code,
                exc.message,
                exc.attempts,
                retryable=exc.retryable,
            ) from None
        except Exception:
            raise _CompactionFailure(
                "INTERNAL_ERROR",
                "The conversation summary request failed safely.",
                1,
            ) from None

        latency_ms = self._latency_since(compaction_started_at)
        if response.malformed_category is not None:
            raise _CompactionFailure(
                "INVALID_SUMMARY_RESPONSE",
                "The model returned an invalid conversation summary response.",
                response.attempts,
            )
        summary = response.content.strip() if response.content else ""
        if not summary:
            raise _CompactionFailure(
                "EMPTY_SUMMARY",
                "The model returned an empty conversation summary.",
                response.attempts,
            )

        new_through_turn_index = turns_to_compact[-1].turn_index
        try:
            persisted = self._repository.compact_conversation(
                conversation_id=conversation_id,
                run_id=run.id,
                expected_summary=conversation.context_summary,
                expected_through_turn_index=conversation.summary_through_turn_index,
                new_summary=summary,
                new_through_turn_index=new_through_turn_index,
                attempts=response.attempts,
                latency_ms=latency_ms,
                active_elapsed_ms=self._elapsed_ms(started_at),
            )
        except Exception:
            raise _CompactionFailure(
                "CONTEXT_PERSISTENCE_FAILED",
                "The compacted conversation context could not be persisted.",
                response.attempts,
            ) from None
        if not persisted:
            raise _CompactionFailure(
                "CONTEXT_PERSISTENCE_FAILED",
                "The compacted conversation context could not be persisted.",
                response.attempts,
            )

        logger.info(
            "Conversation context compacted.",
            extra={
                "conversation_id": conversation_id,
                "run_id": run.id,
                "turn_index": run.turn_index,
                "event_type": "conversation_compacted",
                "attempt": response.attempts,
                "latency_ms": latency_ms,
            },
        )
        if self._elapsed_ms(started_at) >= limit_ms:
            self._finish_limit(
                run.id,
                "ACTIVE_TIME_LIMIT_EXCEEDED",
                self._elapsed_ms(started_at),
            )

    def _turns_to_compact(
        self,
        conversation: ConversationRecord,
        turns: list[RunRecord],
    ) -> list[RunRecord] | None:
        eligible = [
            turn
            for turn in turns
            if turn.turn_index is not None
            and turn.turn_index > conversation.summary_through_turn_index
            and turn.status in TERMINAL_TURN_STATUSES
        ]
        completed_count = sum(1 for turn in eligible if turn.status == "COMPLETED")
        if completed_count < self._settings.conversation_window_turns:
            return None
        return eligible[: self._settings.conversation_compact_turns]

    @staticmethod
    def _turn_summary_input(turn: RunRecord) -> dict:
        return {
            "turn_index": turn.turn_index,
            "status": turn.status,
            "user_content": turn.objective,
            "assistant_content": (
                turn.final_answer if turn.status == "COMPLETED" else None
            ),
            "error_code": turn.error_code,
        }

    def _finish_compaction_failure(
        self,
        run_id: str,
        failure: _CompactionFailure,
        elapsed_ms: int,
    ) -> None:
        self._repository.append_event(
            run_id,
            "conversation_compaction_failed",
            {
                "error": {
                    "code": failure.code,
                    "message": failure.message,
                    "retryable": failure.retryable,
                },
                "attempt": failure.attempts,
                "context_unchanged": True,
            },
            active_elapsed_ms=elapsed_ms,
        )
        logger.error(
            "Conversation context compaction failed.",
            extra={
                "run_id": run_id,
                "event_type": "conversation_compaction_failed",
                "error_code": failure.code,
                "attempt": failure.attempts,
            },
        )
        limit_ms = int(self._settings.agent_active_time_limit_seconds * 1000)
        if failure.time_limit_exceeded or elapsed_ms >= limit_ms:
            self._finish_limit(run_id, "ACTIVE_TIME_LIMIT_EXCEEDED", elapsed_ms)
            return
        self._repository.finish_run(
            run_id,
            status="FAILED",
            active_elapsed_ms=elapsed_ms,
            error_code="CONVERSATION_COMPACTION_FAILED",
            error_message="The turn ended because conversation compaction failed.",
        )

    def _finish_limit(self, run_id: str, error_code: str, elapsed_ms: int) -> None:
        messages = {
            "ACTIVE_TIME_LIMIT_EXCEEDED": "The run reached the active execution time limit.",
        }
        self._repository.finish_run(
            run_id,
            status="LIMIT_EXCEEDED",
            active_elapsed_ms=elapsed_ms,
            error_code=error_code,
            error_message=messages[error_code],
        )

    def _limits(self) -> dict:
        return {
            "agent_max_steps": self._settings.agent_max_steps,
            "agent_active_time_limit_seconds": (
                self._settings.agent_active_time_limit_seconds
            ),
            "llm_timeout_seconds": self._settings.llm_timeout_seconds,
        }

    def _elapsed_ms(self, started_at: float) -> int:
        return self._latency_since(started_at)

    def _latency_since(self, started_at: float) -> int:
        return max(0, int((self._clock.monotonic() - started_at) * 1000))

    def _require_run(self, run_id: str) -> RunRecord:
        record = self._repository.get_run(run_id)
        if record is None:
            raise LookupError(run_id)
        return record

    def _run_view(self, record: RunRecord) -> RunView:
        pending = (
            self._repository.get_pending_approval(record.id)
            if record.status == "WAITING_APPROVAL"
            else None
        )
        return RunView.from_record(record, pending)
