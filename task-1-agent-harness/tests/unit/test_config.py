import pytest
from pydantic import ValidationError

from agent_harness.config import ConfigurationError, Settings, load_settings


def test_settings_load_all_documented_defaults(tmp_path):
    settings = Settings(
        _env_file=None,
        kimi_api_key="test-key",
        kimi_model="test-model",
    )

    assert settings.kimi_base_url.unicode_string() == "https://api.moonshot.cn/v1"
    assert settings.database_url == "sqlite:///./agent_harness.db"
    assert settings.llm_timeout_seconds == 30
    assert settings.llm_max_retries == 3
    assert settings.tool_timeout_seconds == 5
    assert settings.tool_max_retries == 2
    assert settings.retry_backoff_base_seconds == 0.5
    assert settings.retry_backoff_max_seconds == 8
    assert settings.agent_max_steps == 20
    assert settings.agent_active_time_limit_seconds == 120
    assert settings.context_window_rounds == 15
    assert settings.context_compact_rounds == 10
    assert settings.malformed_response_limit == 2
    assert settings.tool_result_max_bytes == 16384
    assert settings.log_level == "INFO"


def test_missing_required_settings_have_safe_diagnostics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)

    with pytest.raises(ConfigurationError) as captured:
        load_settings()

    message = str(captured.value)
    assert "KIMI_API_KEY" in message
    assert "KIMI_MODEL" in message
    assert "Application configuration is invalid" in message


def test_configuration_error_never_echoes_api_key(monkeypatch, tmp_path):
    secret = "do-not-leak-this-key"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KIMI_API_KEY", secret)
    monkeypatch.setenv("KIMI_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/private")

    with pytest.raises(ConfigurationError) as captured:
        load_settings()

    assert secret not in str(captured.value)
    assert "DATABASE_URL" in str(captured.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_timeout_seconds": 0},
        {"llm_max_retries": -1},
        {"tool_timeout_seconds": 0},
        {"tool_max_retries": -1},
        {"retry_backoff_base_seconds": 9, "retry_backoff_max_seconds": 8},
        {"agent_max_steps": 0},
        {"agent_active_time_limit_seconds": 0},
        {"context_window_rounds": 10, "context_compact_rounds": 10},
        {"malformed_response_limit": 0},
        {"tool_result_max_bytes": 0},
        {"database_url": "postgresql://localhost/operations"},
    ],
)
def test_invalid_setting_boundaries_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            kimi_api_key="test-key",
            kimi_model="test-model",
            **overrides,
        )
