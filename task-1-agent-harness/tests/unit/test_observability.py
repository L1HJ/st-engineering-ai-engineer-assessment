import json
import logging

from agent_harness.observability import JsonFormatter, MAX_LOG_STRING_LENGTH


def test_json_formatter_redacts_secrets_and_sensitive_fields():
    secret = "fake-secret-api-key"
    formatter = JsonFormatter([secret])
    record = logging.LogRecord(
        name="agent_harness.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=f"Provider rejected {secret}",
        args=(),
        exc_info=None,
    )
    record.run_id = "run-123"
    record.event_type = "llm_error"
    record.payload = {
        "api_key": secret,
        "nested": {"Authorization": f"Bearer {secret}", "safe": "value"},
        "long_text": "x" * (MAX_LOG_STRING_LENGTH + 20),
    }

    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert secret not in formatted
    assert payload["run_id"] == "run-123"
    assert payload["event_type"] == "llm_error"
    assert payload["payload"]["api_key"] == "[REDACTED]"
    assert payload["payload"]["nested"]["Authorization"] == "[REDACTED]"
    assert payload["payload"]["nested"]["safe"] == "value"
    assert payload["payload"]["long_text"].endswith("[truncated=true]")
    assert payload["timestamp"].endswith("Z")
