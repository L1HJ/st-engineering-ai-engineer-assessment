import asyncio
import math
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def utc_now(self) -> datetime: ...


class Sleeper(Protocol):
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


class SystemSleeper:
    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


RetryCallback = Callable[[dict[str, Any]], None]
RandomSource = Callable[[], float]


class ActiveTimeBudgetExceeded(RuntimeError):
    def __init__(self, cause: Exception | None = None, attempts: int = 1) -> None:
        super().__init__("The active execution time budget is exhausted.")
        self.cause = cause
        self.attempts = attempts


class RetryBackoff:
    def __init__(
        self,
        base_seconds: float,
        max_seconds: float,
        clock: Clock,
        sleeper: Sleeper,
        random_source: RandomSource | None = None,
    ) -> None:
        self._base_seconds = base_seconds
        self._max_seconds = max_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._random = random_source or random.random

    def delay(self, retry_index: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return retry_after
        exponential = min(self._base_seconds * (2**retry_index), self._max_seconds)
        return self._random() * exponential

    async def wait(
        self,
        *,
        component: str,
        attempt: int,
        retry_index: int,
        error_code: str,
        deadline: float,
        retry_after: float | None = None,
        on_retry: RetryCallback | None = None,
        cause: Exception,
    ) -> None:
        remaining = deadline - self._clock.monotonic()
        if remaining <= 0:
            raise ActiveTimeBudgetExceeded(cause, attempt) from cause
        delay = self.delay(retry_index, retry_after)
        if delay > remaining:
            raise ActiveTimeBudgetExceeded(cause, attempt) from cause
        if on_retry is not None:
            on_retry(
                {
                    "component": component,
                    "attempt": attempt,
                    "retry_index": retry_index,
                    "delay_seconds": delay,
                    "error_code": error_code,
                    "remaining_budget_seconds": remaining,
                }
            )
        await self._sleeper.sleep(delay)
        if self._clock.monotonic() >= deadline:
            raise ActiveTimeBudgetExceeded(cause, attempt) from cause


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        seconds = float(candidate)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(candidate)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - now.astimezone(timezone.utc)).total_seconds())
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
