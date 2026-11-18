"""Env-terminal ordering: an env tool's `finished=True` must not end the
session from inside `execute()`; the dispatch loop applies terminal semantics
only after the tool result is persisted and observers have fired.
"""

from __future__ import annotations

import pytest

from agentorg.harness import Harness
from agentorg.session import SessionConfig
from agentorg.types import CEO_NAME

# Reuse the mock inference provider from the existing smoke test.
from tests.test_smoke import MockInferenceProvider


class _FakeEnvTool:
    """Stand-in for an `_OpenRewardTool` so tests don't need a live env.

    Mirrors the real adapter's post-fix contract: returns an `ok_result`
    with `{blocks, metadata, reward, finished}` and does NOT call
    `mark_terminal` / emit `environment.terminal` itself. The dispatch
    loop is responsible for terminal semantics after persistence.
    """

    def __init__(self, name: str = "answer", reward: float = 1.0, finished: bool = True):
        from agentorg.types import ToolDefinition

        self.definition = ToolDefinition(
            name=name,
            description="Submit final answer to the env.",
            input_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            read_only=False,
            scope="env",
        )
        self._reward = reward
        self._finished = finished

    async def execute(self, ctx):
        from agentorg.tools.context import ok_result

        return ok_result(
            {
                "blocks": [{"type": "text", "text": "Correct!"}],
                "metadata": {"is_correct": 1.0},
                "reward": self._reward,
                "finished": self._finished,
            }
        )


@pytest.mark.asyncio
async def test_openreward_env_tool_execute_does_not_mark_terminal(tmp_path):
    """`_OpenRewardTool.execute` must NOT mutate session terminal state or
    emit `environment.terminal` — even when the env signals `finished=True`.
    The dispatch loop applies those semantics AFTER persisting the result
    (see `agentorg.loop._apply_env_terminal_if_needed`).

    Pins the contract end of the persist-vs-stop race fix at the env-tool
    boundary so a regression in the adapter is caught here, not via a flaky
    integration symptom.
    """
    from openreward.api.environments.types import TextBlock, ToolOutput

    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.tools.context import ToolContext
    from agentorg.types import ToolDefinition

    class _StubEnvSession:
        async def call_tool(self, name, args):
            return ToolOutput(
                blocks=[TextBlock(text="Correct!")],
                metadata={"is_correct": 1.0},
                reward=1.0,
                finished=True,
            )

    class _StubAdapter:
        session = _StubEnvSession()

    defn = ToolDefinition(
        name="answer",
        description="submit",
        input_schema={"type": "object", "properties": {}},
        scope="env",
    )
    tool = _OpenRewardTool(defn, _StubAdapter())  # type: ignore[arg-type]

    cfg = SessionConfig(workspace_path=str(tmp_path), model="mock")
    harness = Harness(inference=MockInferenceProvider())
    session = await harness.start_session(config=cfg)
    agent = session.agent(CEO_NAME)
    assert agent is not None
    ctx = ToolContext(agent=agent, session=session, args={}, call_id="c1")

    result = await tool.execute(ctx)

    assert result.ok and result.output["finished"] is True
    assert result.output["reward"] == pytest.approx(1.0)
    # Session terminal state must be untouched by execute itself.
    assert session.terminal_event is None
    assert session.final_reward is None
    # And no environment.terminal event was emitted.
    from agentorg.storage import iter_jsonl

    events = list(iter_jsonl(session.paths.events_jsonl))
    assert not any(e["type"] == "environment.terminal" for e in events)


@pytest.mark.asyncio
async def test_env_terminal_persists_tool_result_before_stop(tmp_path):
    """Regression for the race where an env tool's `mark_terminal()` fired
    inside `execute()` and the supervisor cancelled the CEO before
    `_persist_tool_result` flushed the tool-result `Message` to disk and
    fired observers — leaving observers with only the assistant's tool
    call, no result and no reward.

    Post-fix invariant: when an env tool returns `finished=True`, the
    tool-result message is persisted (in-memory + on disk) and message
    observers fire BEFORE `session.mark_terminal` is called.
    """
    from agentorg.inference.base import (
        CompletedChunk,
        InferenceProvider,
        InferenceRequest,
        ToolCallChunk,
        TokenUsage,
    )
    from agentorg.storage import iter_jsonl
    from agentorg.types import EnvironmentBinding

    class _AnswerOnceProvider(InferenceProvider):
        """Turn 1: emit `answer` tool call. Subsequent turns (shouldn't fire): no-op."""

        def __init__(self):
            self._calls = 0

        async def stream(self, request: InferenceRequest):
            self._calls += 1
            if self._calls == 1:
                yield ToolCallChunk(
                    id="c1", name="answer", args={"answer": "air temperature"}
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    cfg = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="GeneralReasoning/DSBC"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    harness = Harness(inference=_AnswerOnceProvider())
    session = await harness.start_session(config=cfg)
    # Register a fake env tool in the same registry the loop will dispatch through.
    session.tools.register(_FakeEnvTool(name="answer", reward=1.0))

    result = await harness.run_to_completion(session, max_wall_s=10)

    # In-memory transcript must end with the tool-result message.
    transcript = session.transcript(CEO_NAME)
    assert transcript[-1].role == "tool", (
        f"expected tool result as last message, got {transcript[-1].role}; "
        f"tail roles: {[m.role for m in transcript[-3:]]}"
    )
    assert transcript[-1].tool_call_id == "c1"

    # Persisted transcript on disk (the spec §12.4 requirement) must match.
    on_disk = list(iter_jsonl(session.paths.messages_jsonl))
    assert on_disk[-1]["role"] == "tool", (
        f"messages.jsonl tail roles: {[m['role'] for m in on_disk[-3:]]}"
    )

    # Session-level state reflects the env reward.
    assert session.terminal_event is not None
    assert result.reward == pytest.approx(1.0)
    assert result.terminated_by_env is True

    # Event ordering: tool.completed precedes environment.terminal (the fix
    # moves the terminal emit out of the tool's execute and into the
    # dispatch loop, after persistence).
    events = list(iter_jsonl(session.paths.events_jsonl))
    tool_completed_seq = next(
        e["seq"] for e in events
        if e["type"] == "tool.completed" and e["payload"].get("call_id") == "c1"
    )
    terminal_seq = next(
        e["seq"] for e in events if e["type"] == "environment.terminal"
    )
    assert terminal_seq > tool_completed_seq, (
        "environment.terminal must follow tool.completed so observers can "
        "see the tool result before the supervisor cancels the CEO"
    )
    # And environment.terminal fires exactly once (we removed the
    # in-execute emit; only the dispatch-loop helper should emit).
    assert sum(1 for e in events if e["type"] == "environment.terminal") == 1
