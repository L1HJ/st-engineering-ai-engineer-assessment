from datetime import datetime, timedelta, timezone


class FakeClock:
    def __init__(self) -> None:
        self._monotonic = 100.0
        self._utc = datetime(2026, 11, 21, 8, 0, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self._monotonic

    def utc_now(self) -> datetime:
        return self._utc

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._utc += timedelta(seconds=seconds)


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.calls: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock.advance(seconds)
