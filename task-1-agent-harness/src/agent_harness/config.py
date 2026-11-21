from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class ConfigurationError(RuntimeError):
    """Raised when application configuration is unsafe or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kimi_api_key: SecretStr
    kimi_model: str = Field(min_length=1)
    kimi_base_url: AnyHttpUrl = "https://api.moonshot.cn/v1"
    database_url: str = "sqlite:///./agent_harness.db"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    tool_timeout_seconds: float = Field(default=5, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)
    retry_backoff_base_seconds: float = Field(default=0.5, ge=0)
    retry_backoff_max_seconds: float = Field(default=8, ge=0)
    agent_max_steps: int = Field(default=20, gt=0)
    agent_active_time_limit_seconds: float = Field(default=120, gt=0)
    context_window_rounds: int = Field(default=15, gt=0)
    context_compact_rounds: int = Field(default=10, gt=0)
    conversation_window_turns: int = Field(default=12, gt=0)
    conversation_compact_turns: int = Field(default=8, gt=0)
    conversation_max_turns: int = Field(default=50, gt=0)
    malformed_response_limit: int = Field(default=2, ge=1)
    tool_result_max_bytes: int = Field(default=16384, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("kimi_model", mode="before")
    @classmethod
    def trim_model_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("kimi_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        api_key = value.get_secret_value().strip()
        if not api_key:
            raise ValueError("KIMI_API_KEY must not be empty.")
        return SecretStr(api_key)

    @field_validator("kimi_base_url", mode="before")
    @classmethod
    def trim_base_url(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        candidate = value.strip()
        try:
            url = make_url(candidate)
        except ArgumentError as exc:
            raise ValueError("DATABASE_URL must be a valid SQLite URL.") from exc
        if url.drivername not in {"sqlite", "sqlite+pysqlite"}:
            raise ValueError("DATABASE_URL must use SQLite.")
        return candidate

    @model_validator(mode="after")
    def validate_related_limits(self) -> "Settings":
        if self.retry_backoff_max_seconds < self.retry_backoff_base_seconds:
            raise ValueError(
                "RETRY_BACKOFF_MAX_SECONDS must be greater than or equal to "
                "RETRY_BACKOFF_BASE_SECONDS."
            )
        if self.context_compact_rounds >= self.context_window_rounds:
            raise ValueError("CONTEXT_COMPACT_ROUNDS must be smaller than CONTEXT_WINDOW_ROUNDS.")
        if self.conversation_compact_turns >= self.conversation_window_turns:
            raise ValueError(
                "CONVERSATION_COMPACT_TURNS must be smaller than CONVERSATION_WINDOW_TURNS."
            )
        if self.conversation_max_turns < self.conversation_window_turns:
            raise ValueError(
                "CONVERSATION_MAX_TURNS must be greater than or equal to "
                "CONVERSATION_WINDOW_TURNS."
            )
        return self


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        issues = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "settings"
            issues.append(f"{location.upper()}: {error['msg']}")
        raise ConfigurationError(
            "Application configuration is invalid. " + "; ".join(issues)
        ) from None
