from copy import deepcopy
from typing import Any

from agent_harness.llm import ModelResponse, SummaryResponse
from tests.fakes.fake_clock import FakeClock


class FakeKimi:
    def __init__(
        self,
        responses: list[ModelResponse | Exception],
        clock: FakeClock | None = None,
        advance_per_call: float = 0,
        summary_responses: list[SummaryResponse | Exception] | None = None,
        advance_per_summary_call: float = 0,
    ) -> None:
        self._responses = list(responses)
        self._summary_responses = list(summary_responses or [])
        self._clock = clock
        self._advance_per_call = advance_per_call
        self._advance_per_summary_call = advance_per_summary_call
        self.calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []
        self.closed = False

    async def complete_decision(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_budget: float,
        on_retry=None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tool_schemas": deepcopy(tool_schemas),
                "timeout_budget": timeout_budget,
            }
        )
        if self._clock is not None and self._advance_per_call:
            self._clock.advance(self._advance_per_call)
        if not self._responses:
            raise AssertionError("FakeKimi received an unexpected decision request.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def complete_summary(
        self,
        messages: list[dict[str, Any]],
        timeout_budget: float,
        on_retry=None,
    ) -> SummaryResponse:
        self.summary_calls.append(
            {
                "messages": deepcopy(messages),
                "timeout_budget": timeout_budget,
            }
        )
        if self._clock is not None and self._advance_per_summary_call:
            self._clock.advance(self._advance_per_summary_call)
        if not self._summary_responses:
            raise AssertionError("FakeKimi received an unexpected summary request.")
        response = self._summary_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True
