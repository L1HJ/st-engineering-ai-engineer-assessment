from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAIError

from agent_harness.reliability import (
    ActiveTimeBudgetExceeded,
    Clock,
    RandomSource,
    RetryBackoff,
    RetryCallback,
    Sleeper,
    SystemClock,
    SystemSleeper,
    parse_retry_after,
)


@dataclass(frozen=True)
class RawToolCall:
    id: str | None
    name: str | None
    arguments: str | None


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[RawToolCall, ...] = ()
    attempts: int = 1
    malformed_category: str | None = None


@dataclass(frozen=True)
class SummaryResponse:
    content: str | None = None
    attempts: int = 1
    malformed_category: str | None = None


class LlmError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        http_status: int | None = None,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts


class DecisionClient(Protocol):
    async def complete_decision(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_budget: float,
        on_retry: RetryCallback | None = None,
    ) -> ModelResponse: ...

    async def complete_summary(
        self,
        messages: list[dict[str, Any]],
        timeout_budget: float,
        on_retry: RetryCallback | None = None,
    ) -> SummaryResponse: ...


class KimiAdapter:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        client: AsyncOpenAI | None = None,
        *,
        timeout_seconds: float = 30,
        max_retries: int = 3,
        retry_backoff_base_seconds: float = 0.5,
        retry_backoff_max_seconds: float = 8,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
        random_source: RandomSource | None = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._clock = clock or SystemClock()
        self._sleeper = sleeper or SystemSleeper()
        self._backoff = RetryBackoff(
            retry_backoff_base_seconds,
            retry_backoff_max_seconds,
            self._clock,
            self._sleeper,
            random_source,
        )

    async def complete_decision(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_budget: float,
        on_retry: RetryCallback | None = None,
    ) -> ModelResponse:
        completion, attempt = await self._complete_with_retries(
            messages,
            tool_schemas,
            timeout_budget,
            on_retry,
        )
        return self._normalize_response(completion, attempt)

    async def complete_summary(
        self,
        messages: list[dict[str, Any]],
        timeout_budget: float,
        on_retry: RetryCallback | None = None,
    ) -> SummaryResponse:
        completion, attempt = await self._complete_with_retries(
            messages,
            [],
            timeout_budget,
            on_retry,
        )
        return self._normalize_summary_response(completion, attempt)

    async def _complete_with_retries(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_budget: float,
        on_retry: RetryCallback | None = None,
    ) -> tuple[Any, int]:
        deadline = self._clock.monotonic() + max(0, timeout_budget)
        for attempt in range(1, self._max_retries + 2):
            remaining = (
                max(0, timeout_budget)
                if attempt == 1
                else deadline - self._clock.monotonic()
            )
            if remaining <= 0:
                raise ActiveTimeBudgetExceeded(attempts=attempt - 1)
            request: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "timeout": min(self._timeout_seconds, remaining),
            }
            if tool_schemas:
                request["tools"] = tool_schemas
            try:
                completion = await self._client.chat.completions.create(**request)
            except (
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
                OpenAIError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                error = self._classify_error(exc, attempt)
                if not error.retryable or attempt > self._max_retries:
                    raise error from exc
                await self._backoff.wait(
                    component="llm",
                    attempt=attempt,
                    retry_index=attempt - 1,
                    error_code=error.code,
                    deadline=deadline,
                    retry_after=error.retry_after,
                    on_retry=on_retry,
                    cause=error,
                )
                continue
            return completion, attempt
        raise AssertionError("The model retry loop exited unexpectedly.")

    def _classify_error(self, exc: Exception, attempt: int) -> LlmError:
        if isinstance(exc, (APITimeoutError, TimeoutError)):
            return LlmError(
                "LLM_TIMEOUT",
                "The model request exceeded the configured timeout.",
                retryable=True,
                attempts=attempt,
            )
        if isinstance(exc, (APIConnectionError, ConnectionError)):
            return LlmError(
                "LLM_UNAVAILABLE",
                "The model provider could not be reached.",
                retryable=True,
                attempts=attempt,
            )
        if isinstance(exc, APIStatusError):
            status = exc.status_code
            retry_after = parse_retry_after(
                _response_header(exc, "retry-after"),
                self._clock.utc_now(),
            )
            if status == 429:
                return LlmError(
                    "LLM_RATE_LIMITED",
                    "The model provider rate limit was reached.",
                    status,
                    retryable=True,
                    retry_after=retry_after,
                    attempts=attempt,
                )
            if status in {408, 409} or status >= 500:
                return LlmError(
                    "LLM_UNAVAILABLE",
                    "The model provider is temporarily unavailable.",
                    status,
                    retryable=True,
                    retry_after=retry_after,
                    attempts=attempt,
                )
            if status in {401, 403}:
                return LlmError(
                    "LLM_AUTHENTICATION_FAILED",
                    "The model provider rejected the configured credentials.",
                    status,
                    retry_after=retry_after,
                    attempts=attempt,
                )
            if status in {400, 404, 422}:
                return LlmError(
                    "LLM_INVALID_REQUEST",
                    "The model provider rejected the request or model configuration.",
                    status,
                    retry_after=retry_after,
                    attempts=attempt,
                )
            return LlmError(
                "LLM_REQUEST_FAILED",
                "The model provider rejected the request.",
                status,
                retry_after=retry_after,
                attempts=attempt,
            )
        return LlmError(
            "LLM_REQUEST_FAILED",
            "The model request failed safely.",
            attempts=attempt,
        )

    @staticmethod
    def _normalize_response(completion: Any, attempt: int) -> ModelResponse:
        try:
            if not completion.choices:
                return ModelResponse(attempts=attempt)
            message = completion.choices[0].message
            content = message.content
            if content is not None and not isinstance(content, str):
                raise TypeError("Model content must be a string or null.")
            raw_calls = message.tool_calls or []
            tool_calls = []
            for call in raw_calls:
                function = call.function
                values = (
                    call.id,
                    function.name if function else None,
                    function.arguments if function else None,
                )
                if any(value is not None and not isinstance(value, str) for value in values):
                    raise TypeError("Tool call fields must be strings or null.")
                tool_calls.append(RawToolCall(*values))
            return ModelResponse(
                content=content,
                tool_calls=tuple(tool_calls),
                attempts=attempt,
            )
        except (AttributeError, IndexError, TypeError):
            return ModelResponse(
                attempts=attempt,
                malformed_category="invalid_response_structure",
            )

    @staticmethod
    def _normalize_summary_response(completion: Any, attempt: int) -> SummaryResponse:
        try:
            if not completion.choices:
                return SummaryResponse(attempts=attempt)
            content = completion.choices[0].message.content
            if content is not None and not isinstance(content, str):
                raise TypeError("Summary content must be a string or null.")
            return SummaryResponse(content=content, attempts=attempt)
        except (AttributeError, IndexError, TypeError):
            return SummaryResponse(
                attempts=attempt,
                malformed_category="invalid_response_structure",
            )

    async def aclose(self) -> None:
        await self._client.close()


def _response_header(exc: APIStatusError, name: str) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return headers.get(name) if headers is not None else None
