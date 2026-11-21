import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable

MAX_LOG_STRING_LENGTH = 2000
SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password")
PUBLIC_FIELDS = (
    "run_id",
    "conversation_id",
    "turn_index",
    "event_type",
    "step_count",
    "tool_name",
    "tool_call_id",
    "approval_id",
    "attempt",
    "retry_count",
    "latency_ms",
    "status",
    "error_code",
    "active_elapsed_ms",
)


def _sanitize(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if any(part in normalized_key for part in SENSITIVE_KEY_PARTS):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize(item, secrets)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            result = result.replace(secret, "[REDACTED]")
        if len(result) > MAX_LOG_STRING_LENGTH:
            return result[:MAX_LOG_STRING_LENGTH] + "... [truncated=true]"
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    def __init__(self, secret_values: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secret_values if secret)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "service": getattr(record, "service", "agent-harness"),
            "message": record.getMessage(),
        }
        for field in PUBLIC_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        for field in ("payload", "details"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(
            _sanitize(payload, self._secrets),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def configure_logging(level: str, secret_values: Iterable[str] = ()) -> logging.Logger:
    logger = logging.getLogger("agent_harness")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(secret_values))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
