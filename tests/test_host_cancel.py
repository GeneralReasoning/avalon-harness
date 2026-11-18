"""Host cancellation of `run_to_completion` (e.g. inspect's time limit).

The cancel lands on an await inside the supervisor loop. Before the fix it
skipped the supervisor's task cleanup and never aborted the session, so every
agent kept calling the model after the sample ended and `session.ended` said
"complete". A host cancel must stop all agents and end the session as aborted,
whether it is a one-shot asyncio cancel or a level-triggered anyio one.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import anyio
import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.storage import iter_jsonl
from agentorg.types import EnvironmentBinding


class _EndlessProvider(InferenceProvider):
    """Always emits a tool call, so only a host guard or cancel ends the run.
    Counts calls so the test can see whether inference continues."""

    def __init__(self):
        self.calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self.calls += 1
        await asyncio.sleep(0.01)
        yield ToolCallChunk(id=f"c{self.calls}", name="Bash", args={"command": "echo hi"})
        yield CompletedChunk(
            usage=TokenUsage(input_tokens=10, output_tokens=1), stop_reason="tool_calls"
        )

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


async def _cancel_asyncio(harness, session) -> None:
    task = asyncio.create_task(harness.run_to_completion(session, max_wall_s=60))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _cancel_anyio(harness, session) -> None:
    with anyio.move_on_after(0.3):
        await harness.run_to_completion(session, max_wall_s=60)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [_cancel_asyncio, _cancel_anyio], ids=["asyncio", "anyio"])
async def test_host_cancel_stops_agents_and_aborts(tmp_path, cancel):
    provider = _EndlessProvider()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_config(tmp_path))

    await cancel(harness, session)
    calls_at_cancel = provider.calls
    assert calls_at_cancel > 0  # the run was actually live when cancelled

    # No agent outlives the cancel, and none keeps calling the model.
    assert all(t.done() for t in session._agent_tasks.values())
    await asyncio.sleep(0.3)
    assert provider.calls == calls_at_cancel

    assert session._abort_reason == "cancelled"
    events = [(e["type"], e.get("payload")) for e in iter_jsonl(session.paths.events_jsonl)]
    tail = [ev for ev in events if ev[0] in ("session.aborted", "session.ended")]
    assert tail == [
        ("session.aborted", {"reason": "cancelled"}),
        ("session.ended", {"reason": "aborted"}),
    ]
