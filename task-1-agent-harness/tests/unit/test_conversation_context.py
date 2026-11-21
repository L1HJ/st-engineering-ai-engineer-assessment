from datetime import datetime, timezone
from types import SimpleNamespace

from agent_harness.agent import AgentRunner
from agent_harness.context import (
    CONVERSATION_DATA_NOTICE,
    ContextManager,
    ConversationHistory,
)
from agent_harness.store import ConversationRecord, RunRecord

NOW = datetime(2026, 11, 21, 8, 0, tzinfo=timezone.utc)


def make_run(**overrides) -> RunRecord:
    values = {
        "id": "run-current",
        "objective": "Also check payments.",
        "status": "RUNNING",
        "step_count": 0,
        "active_elapsed_ms": 0,
        "context_summary": None,
        "recent_rounds_json": "[]",
        "consecutive_malformed_responses": 0,
        "final_answer": None,
        "error_code": None,
        "error_message": None,
        "created_at": NOW,
        "updated_at": NOW,
        "completed_at": None,
        "conversation_id": "convo-1",
        "turn_index": 2,
        "client_message_id": None,
    }
    values.update(overrides)
    return RunRecord(**values)


def make_conversation(**overrides) -> ConversationRecord:
    values = {
        "id": "convo-1",
        "title": None,
        "status": "ACTIVE",
        "context_summary": None,
        "summary_through_turn_index": 0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return ConversationRecord(**values)


class _HistoryRepository:
    def __init__(self, conversation, turns):
        self._conversation = conversation
        self._turns = turns

    def get_conversation(self, conversation_id):
        return self._conversation

    def list_conversation_turns_before(self, conversation_id, before_turn):
        return [turn for turn in self._turns if turn.turn_index < before_turn]


def test_history_messages_follow_notice_summary_history_current_order():
    run = make_run()
    history = ConversationHistory(
        summary="The user investigated checkout.",
        messages=[
            {"role": "user", "content": "Checkout looks slow."},
            {"role": "assistant", "content": "Checkout is degraded."},
        ],
    )

    messages = ContextManager().build_messages(run, history=history)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == CONVERSATION_DATA_NOTICE
    assert messages[2]["content"].endswith("The user investigated checkout.")
    assert messages[3]["content"] == "Checkout looks slow."
    assert messages[4]["content"] == "Checkout is degraded."
    assert messages[5]["content"] == run.objective


def test_baseline_messages_are_unchanged_without_history():
    run = make_run(conversation_id=None, turn_index=None)

    messages = ContextManager().build_messages(run)

    assert messages == [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user", "content": "Also check payments."},
    ]


def test_conversation_history_excludes_tool_evidence_and_respects_summary_scope():
    turns = [
        make_run(
            id="run-1",
            turn_index=1,
            status="COMPLETED",
            final_answer="Checkout is degraded.",
            recent_rounds_json=(
                '[{"step":1,"assistant":{"content":null,"tool_call":'
                '{"id":"call-1","name":"get_service_status",'
                '"arguments":{"service_name":"checkout"}}},'
                '"observation":{"tool_call_id":"call-1","ok":true,"data":{}}}]'
            ),
        ),
        make_run(
            id="run-2",
            turn_index=2,
            status="FAILED",
            error_code="LLM_UNAVAILABLE",
        ),
    ]
    runner = AgentRunner(
        repository=_HistoryRepository(make_conversation(), turns),
        llm=None,
        settings=SimpleNamespace(),
        registry=None,
    )

    history = runner._conversation_history(
        make_run(turn_index=3, conversation_id="convo-1")
    )

    assert history is not None
    assert history.summary is None
    assert [message["role"] for message in history.messages] == [
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert history.messages[0]["content"] == "Also check payments."
    assert history.messages[1]["content"] == "Checkout is degraded."
    assert "FAILED" in history.messages[3]["content"]
    assert "LLM_UNAVAILABLE" in history.messages[3]["content"]
    assert all("tool_calls" not in message for message in history.messages)
    assert all(message["role"] != "tool" for message in history.messages)


def test_conversation_history_skips_turns_already_covered_by_summary():
    turns = [
        make_run(id="run-1", turn_index=1, status="COMPLETED", final_answer="Old."),
        make_run(id="run-2", turn_index=2, status="COMPLETED", final_answer="Recent."),
    ]
    runner = AgentRunner(
        repository=_HistoryRepository(
            make_conversation(
                context_summary="Summary of turn 1.",
                summary_through_turn_index=1,
            ),
            turns,
        ),
        llm=None,
        settings=SimpleNamespace(),
        registry=None,
    )

    history = runner._conversation_history(
        make_run(turn_index=3, conversation_id="convo-1")
    )

    assert history is not None
    assert history.summary == "Summary of turn 1."
    assert history.messages == [
        {"role": "user", "content": "Also check payments."},
        {"role": "assistant", "content": "Recent."},
    ]
