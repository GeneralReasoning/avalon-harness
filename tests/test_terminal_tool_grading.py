"""Tests for openreward @terminal-tool grading + per-step reward stream.

Covers three pieces:

1. `Session.step_rewards` / `RolloutResult.step_rewards` — the ordered per-step
   env reward list, for downstream consumers that reduce it under an
   env-specific policy (e.g. mean over non-zero steps).
2. `Session.env_adapter` / `Harness.start_session` back-reference — the loop
   uses it to reach adapter-level capabilities not exposed as tools.
3. `agentorg.loop._prepare_terminal_call` + `_dispatch_terminal_tool` — routes
   the CEO's plain final message through the env's @terminal tool for grading,
   emits a synthetic ToolCall/ToolResult pair, and marks the session terminal.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
)
from agentorg.session import Session, SessionConfig
from agentorg.tools.registry import ToolRegistry
from agentorg.types import EnvironmentBinding


# ---- unit: Session.step_rewards + property ---------------------------------


def test_record_reward_derives_stream_sum_and_count(tmp_path):
    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
    )
    s = Session(config=config, inference=_Stub(), tools=ToolRegistry())

    assert s.step_rewards == []
    assert s.cumulative_reward == 0.0
    assert s.reward_step_count == 0

    s.record_reward(0.25)
    s.record_reward(0.0)
    s.record_reward(-0.1)
    s.record_reward(1.0)

    assert s.step_rewards == [0.25, 0.0, -0.1, 1.0]
    assert s.cumulative_reward == pytest.approx(1.15)
    assert s.reward_step_count == 4

    # step_rewards is a read-only copy — mutating it must not corrupt the
    # accumulator (downstream RL code sometimes edits in place).
    view = s.step_rewards
    view.append(999.0)
    assert s.step_rewards == [0.25, 0.0, -0.1, 1.0]


def test_env_adapter_starts_none_and_is_settable(tmp_path):
    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    s = Session(
        config=SessionConfig(workspace_path=str(tmp_path), model="stub", budget=0),
        inference=_Stub(),
        tools=ToolRegistry(),
    )
    assert s.env_adapter is None
    marker = object()
    s.set_env_adapter(marker)
    assert s.env_adapter is marker


# ---- integration: end-to-end terminal-tool grading through Harness ---------


class _CEOAnswerProvider(InferenceProvider):
    """Emits a single tool-less assistant turn — a Qwen-style response with a
    `<think>` block followed by the final answer. Enough to exercise both the
    strip and the synthetic-tool-call injection."""

    def __init__(self, answer_text: str = "the answer is 42",
                 reasoning: str = "let me think"):
        self._sent = False
        self._answer = answer_text
        self._reasoning = reasoning

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        # Only the CEO's first turn needs a payload; subsequent calls (there
        # shouldn't be any once the session terminates) return an empty finish.
        if not self._sent:
            self._sent = True
            yield ContentDeltaChunk(
                text=f"<think>{self._reasoning}</think>{self._answer}"
            )
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


class _FakeTerminalSpec:
    """Mirrors openreward.api.environments.types.TerminalToolSpec on
    installations too old to import it."""

    def __init__(self, name: str, arg: Optional[str]):
        self.name = name
        self.arg = arg
        self.description = ""


class _FakeToolOutput:
    """Duck-types openreward's ToolOutput enough for _dispatch_terminal_tool."""

    def __init__(self, reward: Optional[float], text: str = "graded"):
        class _Block:
            type = "text"
            def __init__(self, text): self.text = text
        self.blocks = [_Block(text)]
        self.metadata = {}
        self.reward = reward
        self.finished = True


class _FakeTerminalEnvAdapter:
    """Minimal EnvAdapter that returns no tools and exposes the terminal-tool
    surface the loop probes."""

    def __init__(self, spec: Optional[_FakeTerminalSpec], reward: Optional[float],
                 raise_on_call: bool = False):
        self._spec = spec
        self._reward = reward
        self._raise = raise_on_call
        self.terminal_calls: list[str] = []

    async def build_tools(self):
        return []

    async def terminal_tool(self):
        return self._spec

    async def call_terminal_tool(self, message: str = ""):
        self.terminal_calls.append(message)
        if self._raise:
            raise RuntimeError("grader unavailable")
        return _FakeToolOutput(self._reward)


async def _run(tmp_path, adapter: _FakeTerminalEnvAdapter,
               provider: InferenceProvider) -> "Harness.RolloutResult":  # type: ignore
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="Answer the question.",
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config, env_adapter=adapter)
    return await harness.run_to_completion(session)


@pytest.mark.asyncio
async def test_terminal_env_grades_via_call_terminal_tool_and_populates_step_rewards(tmp_path):
    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=0.7,
    )
    result = await _run(tmp_path, adapter, _CEOAnswerProvider(answer_text="42"))

    # The grader was called exactly once with the STRIPPED answer — the
    # <think>...</think> block was removed before we sent it.
    assert len(adapter.terminal_calls) == 1
    assert adapter.terminal_calls[0] == "42"
    assert "think" not in adapter.terminal_calls[0]

    # step_rewards carries the terminal reward as its sole entry; the
    # RolloutResult.step_rewards mirrors it.
    assert result.step_rewards == [pytest.approx(0.7)]
    assert result.cumulative_reward == pytest.approx(0.7)
    assert result.reward == pytest.approx(0.7)
    assert result.terminated_by_env is True
    assert (result.terminal_event or {}).get("kind") == "terminal_tool"


@pytest.mark.asyncio
async def test_terminal_reasoning_strip_falls_back_when_answer_is_inside_think(tmp_path):
    """Qwen failure mode: the model puts its entire final answer INSIDE the
    <think>...</think> block. A naive strip would send nothing to the grader.
    The harness falls back to the raw content so the grader still sees text."""
    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=0.0,
    )
    # No text outside the think block; the whole answer is inside it.
    provider = _CEOAnswerProvider(answer_text="", reasoning="the answer is 42")
    await _run(tmp_path, adapter, provider)

    assert len(adapter.terminal_calls) == 1
    # Fallback: grader gets the original content including the tags.
    assert "<think>" in adapter.terminal_calls[0]
    assert "42" in adapter.terminal_calls[0]


@pytest.mark.asyncio
async def test_terminal_env_writes_synthetic_toolcall_and_toolresult(tmp_path):
    """The CEO's transcript must end on a ToolCall(terminal) + ToolResult pair
    so a trace viewer can attach the reward badge to the ToolResult
    event (its UI doesn't render rewards on AssistantMessage events)."""
    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=1.0,
    )
    result = await _run(tmp_path, adapter, _CEOAnswerProvider(answer_text="42"))

    # Read the CEO transcript directly. RolloutResult.workspace_path + the
    # session id are all we need to reconstruct SessionPaths.
    from agentorg.storage import SessionPaths, read_jsonl

    paths = SessionPaths(result.workspace_path, result.session_id)
    # messages.jsonl mixes Message rows with inference-request telemetry rows;
    # keep only the real messages (they carry `role`).
    messages = [m for m in read_jsonl(paths.messages_jsonl) if "role" in m]

    # Find the last assistant message and the following tool result.
    last_assist_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    assistant = messages[last_assist_idx]
    assert assistant["tool_calls"], "assistant message must carry the synthetic ToolCall"
    tc = assistant["tool_calls"][0]
    assert tc["name"] == "grade"
    assert tc["args"] == {"message": "42"}

    # Next message must be the tool result matching the call_id.
    assert last_assist_idx + 1 < len(messages)
    tr = messages[last_assist_idx + 1]
    assert tr["role"] == "tool"
    assert tr["tool_call_id"] == tc["id"]
    payload = tr["tool_result"]["output"]
    assert payload["reward"] == pytest.approx(1.0)
    assert payload["finished"] is True


@pytest.mark.asyncio
async def test_non_terminal_env_leaves_pre_change_behaviour_intact(tmp_path):
    """Env with no terminal tool: the CEO's tool-less turn hits the existing
    nudge/terminate path exactly as before. step_rewards is empty (nothing was
    graded) and the terminal_event is None (no explicit termination)."""
    adapter = _FakeTerminalEnvAdapter(spec=None, reward=None)
    result = await _run(tmp_path, adapter, _CEOAnswerProvider())

    assert adapter.terminal_calls == []
    assert result.step_rewards == []
    assert result.terminal_event is None
    assert result.terminated_by_env is False


@pytest.mark.asyncio
async def test_terminal_grader_error_marks_terminal_without_a_reward(tmp_path):
    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=None,
        raise_on_call=True,
    )
    result = await _run(tmp_path, adapter, _CEOAnswerProvider(answer_text="42"))

    # Grader raised — no reward recorded, but the trace is still marked
    # terminal (the loop already exited its own graceful path).
    assert result.step_rewards == []
    assert result.reward is None
    assert result.terminated_by_env is True
    assert (result.terminal_event or {}).get("kind") == "terminal_tool"
    assert "error" in (result.terminal_event or {})


@pytest.mark.asyncio
async def test_synthetic_terminal_pair_round_trips_on_disk(tmp_path):
    """messages.jsonl keeps the synthetic ToolCall/ToolResult pair for a
    terminal-graded tool-less answer: the call is marked synthetic and the
    grader's ToolResult row is paired with it by tool_call_id."""
    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=1.0,
    )
    result = await _run(tmp_path, adapter, _CEOAnswerProvider(answer_text="42"))

    from agentorg.storage import SessionPaths, read_jsonl

    paths = SessionPaths(result.workspace_path, result.session_id)
    rows = [m for m in read_jsonl(paths.messages_jsonl) if "role" in m]

    last_assist = [m for m in rows if m["role"] == "assistant"][-1]
    assert last_assist["tool_calls"][0]["synthetic"] is True
    assert any(
        m["role"] == "tool"
        and m.get("tool_call_id") == last_assist["tool_calls"][0]["id"]
        for m in rows
    )


@pytest.mark.asyncio
async def test_terminal_grading_ends_session_and_fires_org(tmp_path):
    """§6.5: a tool-less CEO answer graded by the env's @terminal tool is
    finished=True. The episode ends there: the org agent is harness-fired
    (`reason="env-terminal"`, no exit review) and the CEO gets no further
    turn."""
    from agentorg.inference.base import ToolCallChunk

    class _CEOHireWaitAnswerProvider(InferenceProvider):
        """CEO: hire -> wait -> tool-less answer (graded). Org agents report
        once, then park."""

        def __init__(self):
            self.ceo_steps = 0
            self._org_steps = 0

        async def stream(self, request: InferenceRequest) -> AsyncIterator:
            if "You are the CEO" not in (request.messages[0].content or ""):
                self._org_steps += 1
                if self._org_steps == 1:
                    yield ToolCallChunk(
                        id="o1",
                        name="ReportToCEO",
                        args={"status": "completed", "summary": "done"},
                    )
                else:
                    yield ToolCallChunk(
                        id=f"o{self._org_steps}", name="WaitForMessages", args={}
                    )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            self.ceo_steps += 1
            if self.ceo_steps == 1:
                yield ToolCallChunk(
                    id="c1",
                    name="HireOrgAgent",
                    args={"role": "Tester", "description": "Verify it works."},
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if self.ceo_steps == 2:
                # Park so the org agent gets a chance to report.
                yield ToolCallChunk(id="c2", name="WaitForMessages", args={})
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            # Tool-less final answer -> @terminal grading fires here.
            yield ContentDeltaChunk(text="42")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    adapter = _FakeTerminalEnvAdapter(
        spec=_FakeTerminalSpec(name="grade", arg="message"),
        reward=0.7,
    )
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=2,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="Build a team, then answer.",
    )
    provider = _CEOHireWaitAnswerProvider()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config, env_adapter=adapter)
    result = await harness.run_to_completion(session, max_wall_s=15)

    # Graded exactly once, on the CEO's answer; reward locked in.
    assert adapter.terminal_calls == ["42"]
    assert result.reward == pytest.approx(0.7)
    assert result.step_rewards == [pytest.approx(0.7)]
    assert result.terminated_by_env is True
    assert session.abort_requested is False

    # The episode ended on the answer: no further CEO turn, org fired by the
    # harness with no review.
    assert provider.ceo_steps == 3
    org_agents = [a for a in session.all_agents() if a.kind == "org"]
    assert len(org_agents) == 1
    agent = org_agents[0]
    assert agent.state == "fired"
    assert agent.fired_source == "harness"
    assert agent.exit_review is None
