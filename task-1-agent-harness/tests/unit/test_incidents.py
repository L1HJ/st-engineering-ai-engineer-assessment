import hashlib
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from agent_harness.schemas import format_utc
from agent_harness.store import (
    ApprovalDecisionError,
    ApprovalRecord,
    Database,
    IdempotencyConflictError,
    IncidentApprovalError,
    IncidentRecord,
    RunRecord,
    RunRepository,
    canonical_json,
)
from agent_harness.tools import ToolInputValidationError, ToolOutputValidationError


ARGUMENTS = {
    "title": "Database outage",
    "description": "Primary database connections are exhausted.",
    "severity": "P1",
}


@pytest.fixture
def repository(settings, clock):
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database, clock.utc_now)
    yield repository
    database.close()


def prepare_approval(repository, arguments=None):
    arguments = arguments or ARGUMENTS
    run = repository.create_run("Create an incident.", limits={})
    assert repository.record_tool_action(
        run_id=run.id,
        step_count=1,
        assistant_content=None,
        tool_call_id="call-incident",
        tool_name="create_incident",
        arguments=arguments,
        latency_ms=0,
        active_elapsed_ms=0,
    )
    approval = repository.request_approval(
        run_id=run.id,
        tool_call_id="call-incident",
        tool_name="create_incident",
        original_arguments=arguments,
        normalized_arguments=arguments,
        active_elapsed_ms=0,
    )
    assert approval is not None
    return run, approval


def incident_count(repository):
    with repository.database.sessions() as session:
        return session.scalar(select(func.count()).select_from(IncidentRecord))


def test_canonical_json_and_hash_bind_normalized_arguments(repository):
    run, approval = prepare_approval(
        repository,
        {"severity": "P1", "description": "Details", "title": "Title"},
    )

    expected_json = '{"description":"Details","severity":"P1","title":"Title"}'
    assert approval.arguments_json == expected_json
    assert approval.arguments_hash == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert repository.get_pending_approval(run.id).id == approval.id
    assert canonical_json({"title": "Outage café", "severity": "P2"}) == (
        '{"severity":"P2","title":"Outage café"}'
    )


def test_unapproved_incident_write_is_rejected(repository):
    run, approval = prepare_approval(repository)

    with pytest.raises(IncidentApprovalError):
        repository.find_or_create_incident(
            run.id,
            approval.tool_call_id,
            f"{run.id}:{approval.tool_call_id}",
            ARGUMENTS,
        )

    assert incident_count(repository) == 0
    assert repository.get_run(run.id).status == "WAITING_APPROVAL"


def test_approved_incident_is_idempotent_and_conflicting_arguments_preserve_original(repository):
    run, approval = prepare_approval(repository)
    repository.decide_approval(run.id, approval.id, "approve")
    key = f"{run.id}:{approval.tool_call_id}"

    first = repository.find_or_create_incident(
        run.id, approval.tool_call_id, key, ARGUMENTS
    )
    second = repository.find_or_create_incident(
        run.id, approval.tool_call_id, key, ARGUMENTS
    )

    assert first.id.startswith("inc_")
    assert second.id == first.id
    assert format_utc(second.created_at) == format_utc(first.created_at)
    assert incident_count(repository) == 1

    with pytest.raises(IdempotencyConflictError):
        repository.find_or_create_incident(
            run.id,
            approval.tool_call_id,
            key,
            {**ARGUMENTS, "title": "Replacement title"},
        )

    with repository.database.sessions() as session:
        persisted = session.scalar(select(IncidentRecord).where(IncidentRecord.id == first.id))
        assert persisted.title == ARGUMENTS["title"]
        assert persisted.status == "OPEN"
    assert incident_count(repository) == 1


def test_unique_insert_conflict_rereads_and_returns_existing_incident(
    repository, monkeypatch
):
    run, approval = prepare_approval(repository)
    repository.decide_approval(run.id, approval.id, "approve")
    key = f"{run.id}:{approval.tool_call_id}"
    existing = IncidentRecord(
        id="inc_race_winner",
        run_id=run.id,
        tool_call_id=approval.tool_call_id,
        title=ARGUMENTS["title"],
        description=ARGUMENTS["description"],
        severity=ARGUMENTS["severity"],
        status="OPEN",
        idempotency_key=key,
        created_at=repository._now(),
    )
    with repository.database.sessions.begin() as session:
        session.add(existing)

    def raise_unique_conflict(*args, **kwargs):
        raise IntegrityError("INSERT INTO incidents", {}, Exception("UNIQUE constraint failed"))

    monkeypatch.setattr(repository, "_find_or_create_incident", raise_unique_conflict)
    result = repository.find_or_create_incident(
        run.id, approval.tool_call_id, key, ARGUMENTS
    )

    assert result.id == "inc_race_winner"
    assert incident_count(repository) == 1


def test_tampered_approval_hash_blocks_decision_without_state_change(repository):
    run, approval = prepare_approval(repository)
    with repository.database.sessions.begin() as session:
        session.execute(
            update(ApprovalRecord)
            .where(ApprovalRecord.id == approval.id)
            .values(arguments_hash="0" * 64)
        )

    with pytest.raises(ApprovalDecisionError) as captured:
        repository.decide_approval(run.id, approval.id, "approve")

    assert captured.value.code == "RUN_INVALID_STATE"
    assert repository.get_run(run.id).status == "WAITING_APPROVAL"
    assert repository.get_pending_approval(run.id).status == "PENDING"
    assert incident_count(repository) == 0


def test_non_waiting_run_blocks_pending_approval_decision(repository):
    run, approval = prepare_approval(repository)
    now = datetime.now(timezone.utc)
    with repository.database.sessions.begin() as session:
        session.execute(
            update(RunRecord)
            .where(RunRecord.id == run.id)
            .values(status="COMPLETED", final_answer="Stopped.", completed_at=now)
        )

    with pytest.raises(ApprovalDecisionError) as captured:
        repository.decide_approval(run.id, approval.id, "reject", "Rejected.")

    assert captured.value.code == "RUN_INVALID_STATE"
    assert repository.get_run(run.id).final_answer == "Stopped."
    assert repository.get_pending_approval(run.id).status == "PENDING"


def test_database_allows_only_one_pending_approval_per_run(repository):
    run, approval = prepare_approval(repository)
    duplicate = ApprovalRecord(
        id="00000000-0000-0000-0000-000000000001",
        run_id=run.id,
        tool_call_id="another-call",
        tool_name="create_incident",
        arguments_json=canonical_json(ARGUMENTS),
        arguments_hash=hashlib.sha256(canonical_json(ARGUMENTS).encode()).hexdigest(),
        status="PENDING",
        requested_at=approval.requested_at,
    )

    with pytest.raises(IntegrityError):
        with repository.database.sessions.begin() as session:
            session.add(duplicate)

    assert repository.get_pending_approval(run.id).id == approval.id


@pytest.mark.parametrize(
    "arguments",
    [
        {"description": "Description", "severity": "P2"},
        {"title": " ", "description": "Description", "severity": "P2"},
        {"title": "Title", "description": "Description", "severity": "P5"},
        {"title": "Title", "description": "Description", "severity": "P2", "extra": True},
    ],
)
def test_incident_input_schema_rejects_invalid_arguments(tool_registry, arguments):
    with pytest.raises(ToolInputValidationError):
        tool_registry.validate_input("create_incident", arguments)


def test_incident_output_schema_validates_id_timestamp_and_extra_fields(tool_registry):
    valid = {
        "incident_id": "inc_123",
        "status": "OPEN",
        "severity": "P2",
        "created_at": "2026-11-21T08:00:00Z",
    }
    assert tool_registry.validate_output("create_incident", valid)["created_at"].endswith("Z")

    for raw_output in (
        {**valid, "incident_id": "other-123"},
        {**valid, "created_at": 0},
        {**valid, "extra": True},
    ):
        with pytest.raises(ToolOutputValidationError):
            tool_registry.validate_output("create_incident", raw_output)
