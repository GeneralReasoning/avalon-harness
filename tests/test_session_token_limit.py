"""Host-level `max_total_tokens` guard (§4.4).

Mirrors the `max_wall_s` guard: the harness aborts the session once
cumulative *total* tokens (input + output) cross the cap. Each distinct
token is counted once — the context re-sent as input every turn is not
recounted (only the new input that turn is added), so a long conversation
with a stable context doesn't inflate the number. `None` disables the guard.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.types import EnvironmentBinding, Message

NEW_INPUT_PER_TURN = 50  # external tokens (tool results, etc.) added each turn
OUT_PER_TURN = 10


class _GrowingProvider(InferenceProvider):
    """Realistic monotonic growth: each turn's input = prior input + prior
    output + NEW_INPUT_PER_TURN. Always emits a tool call, so the session never
    stops on its own — only a host guard can end it."""

    def __init__(self):
        self._prev_in = 0
        self._prev_out = 0
        self._n = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._n += 1
        inp = self._prev_in + self._prev_out + NEW_INPUT_PER_TURN
        out = OUT_PER_TURN
        self._prev_in, self._prev_out = inp, out
        yield ToolCallChunk(id=f"c{self._n}", name="Bash", args={"command": "echo hi"})
        yield CompletedChunk(
            usage=TokenUsage(input_tokens=inp, output_tokens=out),
            stop_reason="tool_calls",
        )

    def model_ids(self):
        return ["scripted"]


class _NoopProvider(InferenceProvider):
    """Never actually invoked — used for tests that append turns by hand."""

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield ContentDeltaChunk(text="")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


def _config(tmp_path) -> SessionConfig:
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="go",
    )


@pytest.mark.asyncio
async def test_max_total_tokens_aborts_session(tmp_path):
    """Once cumulative total tokens cross the cap, the session aborts with the
    `max_total_tokens` reason. The cap (250) is many turns' worth (60/turn:
    50 new input + 10 output), so an abort proves the count is cumulative and
    includes input."""
    harness = Harness(inference=_GrowingProvider())
    session = await harness.start_session(config=_config(tmp_path))
    # max_wall_s is a generous backstop so a regression can't hang the test;
    # the token guard should fire first (turns complete in milliseconds).
    await harness.run_to_completion(session, max_wall_s=10, max_total_tokens=250)

    assert session.abort_requested
    assert session._abort_reason == "max_total_tokens"
    assert session.total_tokens > 250


@pytest.mark.asyncio
async def test_no_cap_does_not_abort(tmp_path):
    """With `max_total_tokens=None`, the guard never fires regardless of token
    volume."""
    harness = Harness(inference=_GrowingProvider())
    session = await harness.start_session(config=_config(tmp_path))
    # No cap, so cap the run by wall time instead and confirm the abort reason
    # is the wall guard, never the token guard.
    await harness.run_to_completion(session, max_wall_s=1, max_total_tokens=None)

    assert session._abort_reason != "max_total_tokens"


@pytest.mark.asyncio
async def test_new_input_delta_accounting(tmp_path):
    """The counter adds each turn's *new* input plus its output — never the
    re-sent context — and clamps to no-recount when the context shrinks
    (compaction)."""
    harness = Harness(inference=_NoopProvider())
    session = await harness.start_session(config=_config(tmp_path))
    ceo = session.ceo_name

    async def turn(inp: int, out: int) -> None:
        await session.append_message(
            ceo,
            Message(role="assistant", content="x", usage=TokenUsage(input_tokens=inp, output_tokens=out)),
        )

    # Monotonic growth: input[t] = input[t-1] + output[t-1] + 50 new tokens.
    await turn(50, 10)    # new_input 50 + out 10 -> 60
    await turn(110, 10)   # new_input 110-50-10=50 + 10 -> 120
    await turn(170, 10)   # new_input 50 + 10 -> 180
    assert session.total_tokens == 180
    # In the no-compaction case this equals final context + final output...
    assert session.total_tokens == 170 + 10
    # ...and sits strictly between output-only (30) and the naive double-counted
    # input+output sum (50+10 + 110+10 + 170+10 = 360).
    assert 30 < session.total_tokens < 360

    # Compaction: the context shrinks. The delta clamps to 0 (the summary is not
    # recounted as fresh input); only this turn's output is added.
    await turn(40, 10)
    assert session.total_tokens == 190
