import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class ConversationRecord(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'ARCHIVED')", name="ck_conversations_status"),
        CheckConstraint(
            "summary_through_turn_index >= 0",
            name="ck_conversations_summary_index_nonnegative",
        ),
        Index("ix_conversations_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_through_turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunRecord(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint("step_count >= 0", name="ck_runs_step_count_nonnegative"),
        CheckConstraint("active_elapsed_ms >= 0", name="ck_runs_active_elapsed_nonnegative"),
        CheckConstraint(
            "consecutive_malformed_responses >= 0",
            name="ck_runs_malformed_count_nonnegative",
        ),
        CheckConstraint(
            "(conversation_id IS NULL AND turn_index IS NULL) OR "
            "(conversation_id IS NOT NULL AND turn_index IS NOT NULL)",
            name="ck_runs_conversation_turn_pair",
        ),
        CheckConstraint(
            "turn_index IS NULL OR turn_index >= 1",
            name="ck_runs_turn_index_positive",
        ),
        UniqueConstraint(
            "conversation_id", "turn_index", name="uq_runs_conversation_turn"
        ),
        Index(
            "uq_runs_conversation_client_message",
            "conversation_id",
            "client_message_id",
            unique=True,
            sqlite_where=text("client_message_id IS NOT NULL"),
        ),
        Index(
            "uq_runs_active_turn",
            "conversation_id",
            unique=True,
            sqlite_where=text(
                "conversation_id IS NOT NULL AND status IN ('RUNNING', 'WAITING_APPROVAL')"
            ),
        ),
        Index("ix_runs_status", "status"),
        Index("ix_runs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="RUNNING")
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    recent_rounds_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    consecutive_malformed_responses: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="RESTRICT"),
        nullable=True,
    )
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class RunEventRecord(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
        CheckConstraint("sequence >= 1", name="ck_run_events_sequence_positive"),
        CheckConstraint("latency_ms >= 0", name="ck_run_events_latency_nonnegative"),
        Index("ix_run_events_run_sequence", "run_id", "sequence"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApprovalRecord(Base):
    __tablename__ = "approvals"
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id", name="uq_approvals_run_tool_call"),
        CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED')",
            name="ck_approvals_status",
        ),
        CheckConstraint("tool_name = 'create_incident'", name="ck_approvals_tool_name"),
        CheckConstraint("length(arguments_hash) = 64", name="ck_approvals_hash_length"),
        Index(
            "uq_approvals_pending_run",
            "run_id",
            unique=True,
            sqlite_where=text("status = 'PENDING'"),
        ),
        Index("ix_approvals_run_requested", "run_id", "requested_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False
    )
    tool_call_id: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IncidentRecord(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_incidents_idempotency_key"),
        UniqueConstraint("run_id", "tool_call_id", name="uq_incidents_run_tool_call"),
        CheckConstraint("severity IN ('P1', 'P2', 'P3', 'P4')", name="ck_incidents_severity"),
        CheckConstraint("status = 'OPEN'", name="ck_incidents_status"),
        Index("ix_incidents_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False
    )
    tool_call_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(2), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApprovalDecisionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class IncidentApprovalError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class ConversationNotFoundError(LookupError):
    pass


class ConversationArchivedError(RuntimeError):
    pass


class ConversationTurnLimitError(RuntimeError):
    pass


class ConversationBusyError(RuntimeError):
    def __init__(
        self,
        active_run_id: str,
        active_status: str,
        pending_approval_id: str | None = None,
    ) -> None:
        super().__init__("CONVERSATION_BUSY")
        self.active_run_id = active_run_id
        self.active_status = active_status
        self.pending_approval_id = pending_approval_id


class Database:
    def __init__(self, database_url: str) -> None:
        engine_options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            engine_options["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **engine_options)
        self.sessions = sessionmaker(bind=self.engine, class_=Session, expire_on_commit=False)

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._migrate_runs_for_conversations()

    def _migrate_runs_for_conversations(self) -> None:
        """Add conversation columns and indexes to a pre-existing runs table.

        ``create_all`` creates new tables but never alters existing ones, so a
        database created before multi-turn support needs explicit ALTER TABLE
        statements. Every statement is idempotent.
        """
        with self.engine.begin() as connection:
            columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(runs)"))
            }
            if "conversation_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE runs ADD COLUMN conversation_id VARCHAR(36) "
                        "REFERENCES conversations(id) ON DELETE RESTRICT"
                    )
                )
            if "turn_index" not in columns:
                connection.execute(text("ALTER TABLE runs ADD COLUMN turn_index INTEGER"))
            if "client_message_id" not in columns:
                connection.execute(
                    text("ALTER TABLE runs ADD COLUMN client_message_id VARCHAR(128)")
                )

            indexes = {
                row[1] for row in connection.execute(text("PRAGMA index_list(runs)"))
            }
            if "uq_runs_conversation_turn" not in indexes:
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX uq_runs_conversation_turn "
                        "ON runs(conversation_id, turn_index)"
                    )
                )
            if "uq_runs_conversation_client_message" not in indexes:
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX uq_runs_conversation_client_message "
                        "ON runs(conversation_id, client_message_id) "
                        "WHERE client_message_id IS NOT NULL"
                    )
                )
            if "uq_runs_active_turn" not in indexes:
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX uq_runs_active_turn ON runs(conversation_id) "
                        "WHERE conversation_id IS NOT NULL "
                        "AND status IN ('RUNNING', 'WAITING_APPROVAL')"
                    )
                )

    def health(self) -> bool:
        with self.engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1

    def close(self) -> None:
        self.engine.dispose()


class RunRepository:
    def __init__(
        self,
        database: Database,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))

    def _now(self) -> datetime:
        value = self._utc_now()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def create_run(self, objective: str, limits: dict[str, Any]) -> RunRecord:
        now = self._now()
        record = RunRecord(
            id=str(uuid.uuid4()),
            objective=objective,
            status="RUNNING",
            step_count=0,
            active_elapsed_ms=0,
            context_summary=None,
            recent_rounds_json="[]",
            consecutive_malformed_responses=0,
            created_at=now,
            updated_at=now,
        )
        with self.database.sessions.begin() as session:
            session.add(record)
            session.flush()
            self._append_event(
                session,
                run_id=record.id,
                event_type="run_created",
                payload={"objective": objective, "limits": limits},
                latency_ms=None,
            )
        return record

    def create_conversation(self, title: str | None) -> ConversationRecord:
        now = self._now()
        record = ConversationRecord(
            id=str(uuid.uuid4()),
            title=title,
            status="ACTIVE",
            context_summary=None,
            summary_through_turn_index=0,
            created_at=now,
            updated_at=now,
        )
        with self.database.sessions.begin() as session:
            session.add(record)
            session.flush()
        return record

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self.database.sessions() as session:
            return session.get(ConversationRecord, conversation_id)

    def list_conversation_turns(self, conversation_id: str) -> list[RunRecord]:
        with self.database.sessions() as session:
            statement = (
                select(RunRecord)
                .where(RunRecord.conversation_id == conversation_id)
                .order_by(RunRecord.turn_index.asc())
            )
            return list(session.scalars(statement).all())

    def list_conversation_turns_before(
        self,
        conversation_id: str,
        before_turn: int,
    ) -> list[RunRecord]:
        with self.database.sessions() as session:
            statement = (
                select(RunRecord)
                .where(
                    RunRecord.conversation_id == conversation_id,
                    RunRecord.turn_index < before_turn,
                )
                .order_by(RunRecord.turn_index.asc())
            )
            return list(session.scalars(statement).all())

    def list_conversation_turns_page(
        self,
        conversation_id: str,
        limit: int,
        before_turn: int | None = None,
    ) -> list[RunRecord]:
        with self.database.sessions() as session:
            statement = select(RunRecord).where(
                RunRecord.conversation_id == conversation_id
            )
            if before_turn is not None:
                statement = statement.where(RunRecord.turn_index < before_turn)
            statement = statement.order_by(RunRecord.turn_index.desc()).limit(limit)
            return list(reversed(session.scalars(statement).all()))

    def count_conversation_turns(self, conversation_id: str) -> int:
        with self.database.sessions() as session:
            return session.scalar(
                select(func.count())
                .select_from(RunRecord)
                .where(RunRecord.conversation_id == conversation_id)
            )

    def get_conversation_turn_by_client_message_id(
        self,
        conversation_id: str,
        client_message_id: str,
    ) -> RunRecord | None:
        with self.database.sessions() as session:
            return session.scalar(
                select(RunRecord).where(
                    RunRecord.conversation_id == conversation_id,
                    RunRecord.client_message_id == client_message_id,
                )
            )

    def get_active_conversation_turn(self, conversation_id: str) -> RunRecord | None:
        with self.database.sessions() as session:
            return session.scalar(
                select(RunRecord).where(
                    RunRecord.conversation_id == conversation_id,
                    RunRecord.status.in_(("RUNNING", "WAITING_APPROVAL")),
                )
            )

    def create_conversation_turn(
        self,
        conversation_id: str,
        objective: str,
        client_message_id: str | None,
        limits: dict[str, Any],
        max_turns: int,
    ) -> tuple[RunRecord, bool]:
        """Create a Turn or return the existing Run for an idempotent replay.

        Returns ``(record, replayed)``. Raises ``ConversationNotFoundError``,
        ``ConversationArchivedError``, ``ConversationTurnLimitError`` or
        ``ConversationBusyError``.
        """
        for attempt in range(2):
            try:
                return self._create_conversation_turn_once(
                    conversation_id,
                    objective,
                    client_message_id,
                    limits,
                    max_turns,
                )
            except IntegrityError:
                existing = (
                    self.get_conversation_turn_by_client_message_id(
                        conversation_id, client_message_id
                    )
                    if client_message_id is not None
                    else None
                )
                if existing is not None:
                    return existing, True
                if attempt == 1:
                    raise self._busy_error(conversation_id) from None
        raise AssertionError("The conversation turn retry loop exited unexpectedly.")

    def _create_conversation_turn_once(
        self,
        conversation_id: str,
        objective: str,
        client_message_id: str | None,
        limits: dict[str, Any],
        max_turns: int,
    ) -> tuple[RunRecord, bool]:
        now = self._now()
        with self.database.sessions.begin() as session:
            conversation = session.get(ConversationRecord, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            if conversation.status == "ARCHIVED":
                raise ConversationArchivedError(conversation_id)
            if client_message_id is not None:
                existing = session.scalar(
                    select(RunRecord).where(
                        RunRecord.conversation_id == conversation_id,
                        RunRecord.client_message_id == client_message_id,
                    )
                )
                if existing is not None:
                    return existing, True
            active = session.scalar(
                select(RunRecord).where(
                    RunRecord.conversation_id == conversation_id,
                    RunRecord.status.in_(("RUNNING", "WAITING_APPROVAL")),
                )
            )
            if active is not None:
                raise self._busy_error_from_session(session, active)
            next_turn_index = (
                session.scalar(
                    select(func.coalesce(func.max(RunRecord.turn_index), 0)).where(
                        RunRecord.conversation_id == conversation_id
                    )
                )
                + 1
            )
            if next_turn_index > max_turns:
                raise ConversationTurnLimitError(conversation_id)

            record = RunRecord(
                id=str(uuid.uuid4()),
                objective=objective,
                status="RUNNING",
                step_count=0,
                active_elapsed_ms=0,
                context_summary=None,
                recent_rounds_json="[]",
                consecutive_malformed_responses=0,
                created_at=now,
                updated_at=now,
                conversation_id=conversation_id,
                turn_index=next_turn_index,
                client_message_id=client_message_id,
            )
            session.add(record)
            session.flush()
            conversation.updated_at = now
            self._append_event(
                session,
                run_id=record.id,
                event_type="conversation_turn_started",
                payload={
                    "conversation_id": conversation_id,
                    "turn_index": next_turn_index,
                    "objective": objective,
                    "limits": limits,
                },
                latency_ms=None,
            )
        return record, False

    def _busy_error(self, conversation_id: str) -> ConversationBusyError:
        active = self.get_active_conversation_turn(conversation_id)
        if active is None:
            return ConversationBusyError("", "UNKNOWN")
        pending = (
            self.get_pending_approval(active.id)
            if active.status == "WAITING_APPROVAL"
            else None
        )
        return ConversationBusyError(
            active.id,
            active.status,
            pending.id if pending is not None else None,
        )

    @staticmethod
    def _busy_error_from_session(
        session: Session,
        active: RunRecord,
    ) -> ConversationBusyError:
        pending = session.scalar(
            select(ApprovalRecord).where(
                ApprovalRecord.run_id == active.id,
                ApprovalRecord.status == "PENDING",
            )
        )
        pending_approval_id = pending.id if pending is not None else None
        return ConversationBusyError(active.id, active.status, pending_approval_id)

    def compact_conversation(
        self,
        conversation_id: str,
        run_id: str,
        expected_summary: str | None,
        expected_through_turn_index: int,
        new_summary: str,
        new_through_turn_index: int,
        attempts: int,
        latency_ms: int,
        active_elapsed_ms: int,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            conversation = session.get(ConversationRecord, conversation_id)
            if (
                conversation is None
                or conversation.context_summary != expected_summary
                or conversation.summary_through_turn_index != expected_through_turn_index
            ):
                return False
            updated = session.execute(
                update(ConversationRecord)
                .where(
                    ConversationRecord.id == conversation_id,
                    ConversationRecord.summary_through_turn_index
                    == expected_through_turn_index,
                )
                .values(
                    context_summary=new_summary,
                    summary_through_turn_index=new_through_turn_index,
                    updated_at=now,
                )
            )
            if updated.rowcount != 1:
                return False
            run_updated = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(active_elapsed_ms=active_elapsed_ms, updated_at=now)
            )
            if run_updated.rowcount != 1:
                return False
            self._append_event(
                session,
                run_id,
                "conversation_compacted",
                {
                    "conversation_id": conversation_id,
                    "summary_through_turn_index": new_through_turn_index,
                    "summary_length": len(new_summary),
                    "attempt": attempts,
                },
                latency_ms,
            )
        return True

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.database.sessions() as session:
            return session.get(RunRecord, run_id)

    def list_events(self, run_id: str) -> list[RunEventRecord]:
        with self.database.sessions() as session:
            statement = (
                select(RunEventRecord)
                .where(RunEventRecord.run_id == run_id)
                .order_by(RunEventRecord.sequence.asc())
            )
            return list(session.scalars(statement).all())

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        latency_ms: int | None = None,
        active_elapsed_ms: int | None = None,
    ) -> None:
        with self.database.sessions.begin() as session:
            if active_elapsed_ms is not None:
                session.execute(
                    update(RunRecord)
                    .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                    .values(
                        active_elapsed_ms=active_elapsed_ms,
                        updated_at=self._now(),
                    )
                )
            self._append_event(session, run_id, event_type, payload, latency_ms)

    def record_llm_response(
        self,
        run_id: str,
        step_count: int,
        content: str,
        latency_ms: int,
        recent_rounds: list[dict[str, Any]],
        attempts: int = 1,
        active_elapsed_ms: int | None = None,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    step_count=step_count,
                    consecutive_malformed_responses=0,
                    recent_rounds_json=_encode_json(recent_rounds),
                    updated_at=now,
                    **_active_elapsed_values(active_elapsed_ms),
                )
            )
            if result.rowcount != 1:
                return False
            self._append_event(
                session,
                run_id,
                "llm_response",
                {"step": step_count, "content": content, "attempt": attempts},
                latency_ms,
            )
        return True

    def tool_call_id_exists(self, run_id: str, tool_call_id: str) -> bool:
        with self.database.sessions() as session:
            payloads = session.scalars(
                select(RunEventRecord.payload_json).where(
                    RunEventRecord.run_id == run_id,
                    RunEventRecord.event_type == "tool_call",
                )
            ).all()
        return any(
            json.loads(payload).get("tool_call_id") == tool_call_id for payload in payloads
        )

    def record_tool_action(
        self,
        run_id: str,
        step_count: int,
        assistant_content: str | None,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        latency_ms: int,
        attempts: int = 1,
        active_elapsed_ms: int | None = None,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None or run.status != "RUNNING":
                return False
            recent_rounds = json.loads(run.recent_rounds_json)
            recent_rounds.append(
                {
                    "step": step_count,
                    "assistant": {
                        "content": assistant_content,
                        "tool_call": {
                            "id": tool_call_id,
                            "name": tool_name,
                            "arguments": arguments,
                        },
                    },
                    "observation": None,
                }
            )
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    step_count=step_count,
                    consecutive_malformed_responses=0,
                    recent_rounds_json=_encode_json(recent_rounds),
                    updated_at=now,
                    **_active_elapsed_values(active_elapsed_ms),
                )
            )
            if result.rowcount != 1:
                return False
            tool_call = {
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
            }
            self._append_event(
                session,
                run_id,
                "llm_response",
                {
                    "step": step_count,
                    "content": assistant_content,
                    "tool_call": tool_call,
                    "attempt": attempts,
                },
                latency_ms,
            )
            self._append_event(
                session,
                run_id,
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                None,
            )
        return True

    def record_tool_observation(
        self,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        observation: dict[str, Any],
        latency_ms: int,
        attempts: int = 1,
        active_elapsed_ms: int | None = None,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None or run.status != "RUNNING":
                return False
            recent_rounds = json.loads(run.recent_rounds_json)
            matching_round = None
            for round_record in reversed(recent_rounds):
                tool_call = round_record["assistant"].get("tool_call")
                if tool_call and tool_call["id"] == tool_call_id:
                    matching_round = round_record
                    break
            if matching_round is None or matching_round.get("observation") is not None:
                return False
            matching_round["observation"] = observation
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    recent_rounds_json=_encode_json(recent_rounds),
                    updated_at=now,
                    **_active_elapsed_values(active_elapsed_ms),
                )
            )
            if result.rowcount != 1:
                return False
            success = observation["ok"]
            event_type = "tool_result" if success else "tool_error"
            payload: dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "attempt": attempts,
            }
            if success:
                payload["observation"] = observation
            else:
                payload["error"] = observation["error"]
            self._append_event(
                session,
                run_id,
                event_type,
                payload,
                latency_ms,
            )
        return True

    def compact_context(
        self,
        run_id: str,
        expected_summary: str | None,
        expected_recent_rounds_json: str,
        compact_rounds: int,
        new_summary: str,
        attempts: int,
        latency_ms: int,
        active_elapsed_ms: int,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if (
                run is None
                or run.status != "RUNNING"
                or run.context_summary != expected_summary
                or run.recent_rounds_json != expected_recent_rounds_json
            ):
                return False
            recent_rounds = json.loads(expected_recent_rounds_json)
            if compact_rounds <= 0 or len(recent_rounds) < compact_rounds:
                return False
            if any(
                round_record["assistant"].get("tool_call") is not None
                and round_record.get("observation") is None
                for round_record in recent_rounds
            ):
                return False
            compacted_rounds = recent_rounds[:compact_rounds]
            retained_rounds = recent_rounds[compact_rounds:]
            result = session.execute(
                update(RunRecord)
                .where(
                    RunRecord.id == run_id,
                    RunRecord.status == "RUNNING",
                    RunRecord.context_summary == expected_summary,
                    RunRecord.recent_rounds_json == expected_recent_rounds_json,
                )
                .values(
                    context_summary=new_summary,
                    recent_rounds_json=_encode_json(retained_rounds),
                    updated_at=now,
                    **_active_elapsed_values(active_elapsed_ms),
                )
            )
            if result.rowcount != 1:
                return False
            self._append_event(
                session,
                run_id,
                "context_compacted",
                {
                    "from_step": compacted_rounds[0]["step"],
                    "to_step": compacted_rounds[-1]["step"],
                    "rounds_compacted": len(compacted_rounds),
                    "remaining_rounds": len(retained_rounds),
                    "summary_length": len(new_summary),
                    "attempt": attempts,
                },
                latency_ms,
            )
        return True

    def get_approval(self, run_id: str, approval_id: str) -> ApprovalRecord | None:
        with self.database.sessions() as session:
            return session.scalar(
                select(ApprovalRecord).where(
                    ApprovalRecord.id == approval_id,
                    ApprovalRecord.run_id == run_id,
                )
            )

    def get_pending_approval(self, run_id: str) -> ApprovalRecord | None:
        with self.database.sessions() as session:
            return session.scalar(
                select(ApprovalRecord).where(
                    ApprovalRecord.run_id == run_id,
                    ApprovalRecord.status == "PENDING",
                )
            )

    def request_approval(
        self,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        original_arguments: dict[str, Any],
        normalized_arguments: dict[str, Any],
        active_elapsed_ms: int,
    ) -> ApprovalRecord | None:
        now = self._now()
        arguments_json = canonical_json(normalized_arguments)
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None or run.status != "RUNNING":
                return None
            tool_events = session.scalars(
                select(RunEventRecord).where(
                    RunEventRecord.run_id == run_id,
                    RunEventRecord.event_type == "tool_call",
                )
            ).all()
            saved_action = next(
                (
                    json.loads(event.payload_json)
                    for event in reversed(tool_events)
                    if json.loads(event.payload_json).get("tool_call_id") == tool_call_id
                ),
                None,
            )
            if saved_action != {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": original_arguments,
            }:
                return None
            unfinished_round = next(
                (
                    round_record
                    for round_record in reversed(json.loads(run.recent_rounds_json))
                    if (round_record["assistant"].get("tool_call") or {}).get("id")
                    == tool_call_id
                ),
                None,
            )
            if (
                unfinished_round is None
                or unfinished_round.get("observation") is not None
                or unfinished_round["assistant"]["tool_call"].get("name") != tool_name
                or unfinished_round["assistant"]["tool_call"].get("arguments")
                != original_arguments
            ):
                return None
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    status="WAITING_APPROVAL",
                    active_elapsed_ms=active_elapsed_ms,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                return None
            approval = ApprovalRecord(
                id=str(uuid.uuid4()),
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_json=arguments_json,
                arguments_hash=sha256_canonical_json(arguments_json),
                status="PENDING",
                requested_at=now,
                decided_at=None,
            )
            session.add(approval)
            session.flush()
            self._append_event(
                session,
                run_id,
                "approval_requested",
                {
                    "approval_id": approval.id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments_hash": approval.arguments_hash,
                },
                None,
            )
            self._append_event(
                session,
                run_id,
                "run_state_changed",
                {
                    "from_status": "RUNNING",
                    "to_status": "WAITING_APPROVAL",
                    "reason": "approval_requested",
                },
                None,
            )
        return approval

    def decide_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
        final_answer: str | None = None,
    ) -> ApprovalRecord:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None:
                raise ApprovalDecisionError("RUN_NOT_FOUND")
            approval = session.get(ApprovalRecord, approval_id)
            if approval is None or approval.run_id != run_id:
                raise ApprovalDecisionError("APPROVAL_NOT_FOUND")
            if approval.status != "PENDING":
                raise ApprovalDecisionError("APPROVAL_ALREADY_DECIDED")
            if run.status != "WAITING_APPROVAL":
                raise ApprovalDecisionError("RUN_INVALID_STATE")
            if not approval_arguments_are_valid(approval):
                raise ApprovalDecisionError("RUN_INVALID_STATE")
            next_approval_status = "APPROVED" if decision == "approve" else "REJECTED"
            result = session.execute(
                update(ApprovalRecord)
                .where(
                    ApprovalRecord.id == approval_id,
                    ApprovalRecord.run_id == run_id,
                    ApprovalRecord.status == "PENDING",
                )
                .values(status=next_approval_status, decided_at=now)
            )
            if result.rowcount != 1:
                raise ApprovalDecisionError("APPROVAL_ALREADY_DECIDED")
            if decision == "approve":
                run_result = session.execute(
                    update(RunRecord)
                    .where(RunRecord.id == run_id, RunRecord.status == "WAITING_APPROVAL")
                    .values(status="RUNNING", updated_at=now)
                )
                if run_result.rowcount != 1:
                    raise ApprovalDecisionError("RUN_INVALID_STATE")
            else:
                run_result = session.execute(
                    update(RunRecord)
                    .where(RunRecord.id == run_id, RunRecord.status == "WAITING_APPROVAL")
                    .values(
                        status="COMPLETED",
                        final_answer=final_answer,
                        error_code=None,
                        error_message=None,
                        updated_at=now,
                        completed_at=now,
                    )
                )
                if run_result.rowcount != 1:
                    raise ApprovalDecisionError("RUN_INVALID_STATE")
            self._append_event(
                session,
                run_id,
                "approval_decided",
                {
                    "approval_id": approval_id,
                    "tool_call_id": approval.tool_call_id,
                    "decision": decision,
                    "decided_at": _format_utc(now),
                },
                None,
            )
            self._append_event(
                session,
                run_id,
                "run_state_changed",
                {
                    "from_status": "WAITING_APPROVAL",
                    "to_status": "RUNNING" if decision == "approve" else "COMPLETED",
                    "reason": (
                        "approval_approved" if decision == "approve" else "approval_rejected"
                    ),
                },
                None,
            )
            if decision == "reject":
                self._append_event(
                    session,
                    run_id,
                    "run_finished",
                    {
                        "status": "COMPLETED",
                        "step_count": run.step_count,
                        "active_elapsed_ms": run.active_elapsed_ms,
                        "error_code": None,
                    },
                    None,
                )
            session.flush()
            session.refresh(approval)
        return approval

    def find_or_create_incident(
        self,
        run_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
    ) -> IncidentRecord:
        expected_key = f"{run_id}:{tool_call_id}"
        if idempotency_key != expected_key:
            raise IncidentApprovalError("The incident idempotency key is invalid.")
        try:
            with self.database.sessions.begin() as session:
                return self._find_or_create_incident(
                    session,
                    run_id,
                    tool_call_id,
                    idempotency_key,
                    arguments,
                )
        except IntegrityError:
            with self.database.sessions.begin() as session:
                existing = session.scalar(
                    select(IncidentRecord).where(
                        IncidentRecord.idempotency_key == idempotency_key
                    )
                )
                if existing is None:
                    raise
                return self._return_matching_incident(existing, arguments)

    def _find_or_create_incident(
        self,
        session: Session,
        run_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
    ) -> IncidentRecord:
        run = session.get(RunRecord, run_id)
        approval = session.scalar(
            select(ApprovalRecord).where(
                ApprovalRecord.run_id == run_id,
                ApprovalRecord.tool_call_id == tool_call_id,
            )
        )
        if (
            run is None
            or run.status != "RUNNING"
            or approval is None
            or approval.status != "APPROVED"
            or approval.tool_name != "create_incident"
            or not approval_arguments_are_valid(approval)
        ):
            raise IncidentApprovalError("The incident action is not approved for execution.")
        existing = session.scalar(
            select(IncidentRecord).where(IncidentRecord.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return self._return_matching_incident(existing, arguments)
        if approval.arguments_json != canonical_json(arguments):
            raise IncidentApprovalError("The incident arguments do not match the approval.")
        incident = IncidentRecord(
            id=f"inc_{uuid.uuid4().hex}",
            run_id=run_id,
            tool_call_id=tool_call_id,
            title=arguments["title"],
            description=arguments["description"],
            severity=arguments["severity"],
            status="OPEN",
            idempotency_key=idempotency_key,
            created_at=self._now(),
        )
        session.add(incident)
        session.flush()
        return incident

    @staticmethod
    def _return_matching_incident(
        incident: IncidentRecord,
        arguments: dict[str, Any],
    ) -> IncidentRecord:
        if any(
            getattr(incident, field) != arguments[field]
            for field in ("title", "description", "severity")
        ):
            raise IdempotencyConflictError("The idempotency key is bound to different arguments.")
        return incident

    def record_malformed_response(
        self,
        run_id: str,
        step_count: int,
        category: str,
        latency_ms: int,
        will_correct: bool = False,
        active_elapsed_ms: int | None = None,
    ) -> int | None:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None or run.status != "RUNNING":
                return None
            malformed_count = run.consecutive_malformed_responses + 1
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    step_count=step_count,
                    consecutive_malformed_responses=malformed_count,
                    updated_at=now,
                    **_active_elapsed_values(active_elapsed_ms),
                )
            )
            if result.rowcount != 1:
                return None
            self._append_event(
                session,
                run_id,
                "malformed_llm_response",
                {
                    "category": category,
                    "consecutive_count": malformed_count,
                    "will_correct": will_correct,
                },
                latency_ms,
            )
        return malformed_count

    def finish_run(
        self,
        run_id: str,
        status: str,
        active_elapsed_ms: int,
        final_answer: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        now = self._now()
        with self.database.sessions.begin() as session:
            run = session.get(RunRecord, run_id)
            if run is None or run.status != "RUNNING":
                return False
            step_count = run.step_count
            result = session.execute(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                .values(
                    status=status,
                    active_elapsed_ms=active_elapsed_ms,
                    final_answer=final_answer,
                    error_code=error_code,
                    error_message=error_message,
                    updated_at=now,
                    completed_at=now,
                )
            )
            if result.rowcount != 1:
                return False
            self._append_event(
                session,
                run_id,
                "run_state_changed",
                {
                    "from_status": "RUNNING",
                    "to_status": status,
                    "reason": error_code or status,
                },
                None,
            )
            self._append_event(
                session,
                run_id,
                "run_finished",
                {
                    "status": status,
                    "step_count": step_count,
                    "active_elapsed_ms": active_elapsed_ms,
                    "error_code": error_code,
                },
                None,
            )
        return True

    def mark_interrupted_runs(self) -> int:
        now = self._now()
        interrupted_count = 0
        with self.database.sessions.begin() as session:
            run_ids = list(
                session.scalars(select(RunRecord.id).where(RunRecord.status == "RUNNING")).all()
            )
            for run_id in run_ids:
                result = session.execute(
                    update(RunRecord)
                    .where(RunRecord.id == run_id, RunRecord.status == "RUNNING")
                    .values(
                        status="FAILED",
                        error_code="RUN_INTERRUPTED",
                        error_message="The run was interrupted by a service restart.",
                        updated_at=now,
                        completed_at=now,
                    )
                )
                if result.rowcount != 1:
                    continue
                run = session.get(RunRecord, run_id)
                self._append_event(
                    session,
                    run_id,
                    "run_state_changed",
                    {
                        "from_status": "RUNNING",
                        "to_status": "FAILED",
                        "reason": "RUN_INTERRUPTED",
                    },
                    None,
                )
                self._append_event(
                    session,
                    run_id,
                    "run_finished",
                    {
                        "status": "FAILED",
                        "step_count": run.step_count,
                        "active_elapsed_ms": run.active_elapsed_ms,
                        "error_code": "RUN_INTERRUPTED",
                    },
                    None,
                )
                interrupted_count += 1
        return interrupted_count

    def _append_event(
        self,
        session: Session,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        latency_ms: int | None,
    ) -> RunEventRecord:
        next_sequence = session.scalar(
            select(func.coalesce(func.max(RunEventRecord.sequence), 0) + 1).where(
                RunEventRecord.run_id == run_id
            )
        )
        record = RunEventRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            sequence=next_sequence,
            event_type=event_type,
            payload_json=_encode_json(payload),
            latency_ms=latency_ms,
            created_at=self._now(),
        )
        session.add(record)
        session.flush()
        return record


def _active_elapsed_values(active_elapsed_ms: int | None) -> dict[str, int]:
    return {} if active_elapsed_ms is None else {"active_elapsed_ms": active_elapsed_ms}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_canonical_json(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def approval_arguments_are_valid(approval: ApprovalRecord) -> bool:
    try:
        arguments = json.loads(approval.arguments_json)
        return (
            isinstance(arguments, dict)
            and canonical_json(arguments) == approval.arguments_json
            and sha256_canonical_json(approval.arguments_json) == approval.arguments_hash
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
