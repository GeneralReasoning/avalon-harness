"""Tests for per-step reward accounting and the §A.8.2 solo-thrash nudge."""

from __future__ import annotations

from typing import AsyncIterator, Optional

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
from agentorg.loop import SOLO_NUDGE_INTERVAL
from agentorg.session import SessionConfig
from agentorg.types import EnvironmentBinding


def test_session_effective_final_reward_uses_cumulative(tmp_path):
    """If the env never explicitly terminates, the effective final reward is the cumulative sum."""
    from agentorg.session import Session
    from agentorg.tools.registry import ToolRegistry

    class _Stub(InferenceProvider):
        async def stream(self, request):  # noqa: D401
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
    assert s.effective_final_reward() is None
    s.record_reward(0.3)
    s.record_reward(0.4)
    assert s.effective_final_reward() == pytest.approx(0.7)
    # Explicit terminal wins over cumulative.
    s.mark_terminal({"final": True}, reward=1.5)
    assert s.effective_final_reward() == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_env_tool_terminal_reward_lands_in_step_stream(tmp_path):
    """The terminal step's grade is a member of the per-step stream (§12.1),
    matching the @terminal-tool grading path, RL reward reducers (a
    binary-terminal env must not read as an all-zero
    stream under zero_reward_masking), and `build_trajectory`'s replay of the
    persisted payloads. `.reward` does not double-count: effective_final_reward
    prefers the explicit mark_terminal value over the cumulative sum."""
    from openreward.api.environments.types import TextBlock, ToolOutput

    from agentorg.loop import _apply_env_terminal_if_needed
    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.tools.context import ToolContext
    from agentorg.types import CEO_NAME, ToolCall, ToolDefinition
    from tests.test_smoke import MockInferenceProvider

    outputs = [
        ToolOutput(blocks=[TextBlock(text="step")], metadata={}, reward=0.3, finished=False),
        ToolOutput(blocks=[TextBlock(text="Correct!")], metadata={}, reward=1.0, finished=True),
    ]

    class _StubEnvSession:
        async def call_tool(self, name, args):
            return outputs.pop(0)

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
    session.tools.register(tool)
    agent = session.agent(CEO_NAME)
    assert agent is not None

    r1 = await tool.execute(ToolContext(agent=agent, session=session, args={}, call_id="c1"))
    r2 = await tool.execute(ToolContext(agent=agent, session=session, args={}, call_id="c2"))
    assert r1.ok and r2.ok

    # Both the intermediate delta and the terminal grade are stream members.
    assert session.step_rewards == [pytest.approx(0.3), pytest.approx(1.0)]
    assert session.cumulative_reward == pytest.approx(1.3)
    # execute still leaves terminal semantics to the dispatch loop.
    assert session.terminal_event is None

    await _apply_env_terminal_if_needed(
        session, agent, ToolCall(id="c2", name="answer"), r2
    )
    assert session.terminal_event is not None
    # Authoritative final reward is the terminal grade — not the sum.
    assert session.effective_final_reward() == pytest.approx(1.0)


class _ScriptedToolProvider(InferenceProvider):
    """Emits a sequence of single tool calls then a tool-less assistant."""

    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = list(calls)
        self._idx = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if self._idx < len(self._calls):
            name, args = self._calls[self._idx]
            self._idx += 1
            yield ToolCallChunk(id=f"c{self._idx}", name=name, args=args)
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
            return
        yield ContentDeltaChunk(text="done")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_solo_thrash_nudge_fires_after_interval(tmp_path):
    """SOLO_NUDGE_INTERVAL qualifying turns (each w/ ≥1 substantive tool call) fire a meta nudge into the CEO's inbox."""
    # Build a Bash-only solo run that the harness considers env-bound.
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="explore",
    )

    # All Bash, which is countable.
    calls = [("Bash", {"command": "echo hi"})] * (SOLO_NUDGE_INTERVAL + 2)
    harness = Harness(inference=_ScriptedToolProvider(calls))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)

    # We expect at least one nudge fired after SOLO_NUDGE_INTERVAL qualifying turns.
    assert session._solo_thrash_nudges_fired >= 1, (
        f"Expected ≥1 nudge after {SOLO_NUDGE_INTERVAL} qualifying solo turns, "
        f"got {session._solo_thrash_nudges_fired}"
    )
    # The nudge body should be in the CEO transcript.
    transcript = session.transcript(session.ceo_name)
    nudge_present = any(
        m.role == "user" and "Reminder: you have" in (m.content or "")
        for m in transcript
    )
    assert nudge_present


@pytest.mark.asyncio
async def test_solo_thrash_nudge_skipped_without_env(tmp_path):
    """No env binding = no nudge, even with budget>0 and many solo tool calls."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        headless=True,
        permission_policy="bypass",
        initial_user_message="hi",
    )
    calls = [("Bash", {"command": "echo hi"})] * (SOLO_NUDGE_INTERVAL + 4)
    harness = Harness(inference=_ScriptedToolProvider(calls))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)
    assert session._solo_thrash_nudges_fired == 0


@pytest.mark.asyncio
async def test_solo_thrash_excluded_tools_dont_count(tmp_path):
    """WaitForMessages / ListChannels etc. don't move the counter."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="hi",
    )
    # These should NOT count.
    calls = [("ListChannels", {})] * (SOLO_NUDGE_INTERVAL + 4)
    harness = Harness(inference=_ScriptedToolProvider(calls))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)
    assert session._solo_thrash_nudges_fired == 0


class _TokenReportingToolProvider(InferenceProvider):
    """Like _ScriptedToolProvider but reports a fixed token cost per turn, so a
    `max_total_tokens` cap can bound an otherwise-parking interactive run
    deterministically — the cap is crossed during the active tool turns, before
    the CEO would park on idle (no reliance on wall-clock-while-parked)."""

    def __init__(self, calls: list[tuple[str, dict]], tokens_per_turn: int):
        self._calls = list(calls)
        self._idx = 0
        self._usage = TokenUsage(input_tokens=tokens_per_turn - 10, output_tokens=10)

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if self._idx < len(self._calls):
            name, args = self._calls[self._idx]
            self._idx += 1
            yield ToolCallChunk(id=f"c{self._idx}", name=name, args=args)
            yield CompletedChunk(usage=self._usage, stop_reason="tool_calls")
            return
        yield ContentDeltaChunk(text="done")
        yield CompletedChunk(usage=self._usage, stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_solo_thrash_nudge_fires_for_interactive_envless(tmp_path):
    """The §A.8.2 nudge now fires for an ATTENDED *envless* session — no env but
    interactive (end_on_idle=False). Mirrors
    test_solo_thrash_nudge_fires_after_interval (env-bound) for the new path.

    Bounded by a token cap (not quiescence — an interactive session parks rather
    than exiting): 100 tokens/turn, cap 150 → the run aborts right after the 2nd
    qualifying turn, which is exactly when SOLO_NUDGE_INTERVAL=2 fires the nudge.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",  # the new envless default; the nudge is mode-independent
        end_on_idle=False,  # interactive — the new discriminator
        headless=True,
        permission_policy="bypass",
        initial_user_message="explore",
    )
    calls = [("Bash", {"command": "echo hi"})] * 6
    harness = Harness(inference=_TokenReportingToolProvider(calls, tokens_per_turn=100))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10, max_total_tokens=150)

    assert session._solo_thrash_nudges_fired >= 1, (
        f"expected the nudge to fire for an interactive envless CEO, got "
        f"{session._solo_thrash_nudges_fired}"
    )
    transcript = session.transcript(session.ceo_name)
    assert any(
        m.role == "user" and "Reminder: you have" in (m.content or "")
        for m in transcript
    )


class _ToollessThenToolProvider(InferenceProvider):
    """First turn: tool-less content. Subsequent turns: scripted tool calls."""

    def __init__(self, calls_after_nudge: list[tuple[str, dict]]):
        self._calls = list(calls_after_nudge)
        self._first = True
        self._idx = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if self._first:
            self._first = False
            yield ContentDeltaChunk(text="42")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        if self._idx < len(self._calls):
            name, args = self._calls[self._idx]
            self._idx += 1
            yield ToolCallChunk(id=f"c{self._idx}", name=name, args=args)
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
            return
        yield ContentDeltaChunk(text="done")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


class _AlwaysToollessProvider(InferenceProvider):
    """Every turn is tool-less content."""

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield ContentDeltaChunk(text="just-text")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_env_tool_less_nudge_recovers_on_next_turn(tmp_path):
    """§4.5.1: env-bound solo CEO emits tool-less → harness nudges, model
    then emits a real tool call → rollout continues normally and the
    streak counter is cleared."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    # First call is tool-less; after the nudge the model emits one Bash call,
    # then runs out and falls back to tool-less again to drive termination.
    harness = Harness(inference=_ToollessThenToolProvider([("Bash", {"command": "echo hi"})]))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)

    transcript = session.transcript(session.ceo_name)
    # The Bash call between the two tool-less turns must have reset the streak,
    # so the *second* tool-less turn re-armed the nudge. Two nudges proves
    # the reset worked: without it, the second tool-less would terminate
    # immediately on the leftover streak=1 instead of firing the nudge.
    n_nudges = sum(
        1 for m in transcript
        if m.role == "user" and "Bare text is not graded" in (m.content or "")
    )
    assert n_nudges == 2, f"expected 2 nudges (reset between), got {n_nudges}"


@pytest.mark.asyncio
async def test_env_tool_less_two_strikes_terminates(tmp_path):
    """Two consecutive tool-less turns end the env-bound run (§4.5.1)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    harness = Harness(inference=_AlwaysToollessProvider())
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=10)
    # Exactly two CEO assistant turns: pre-nudge + post-nudge.
    assert result.used_turns == 2
    # The nudge ran once.
    transcript = session.transcript(session.ceo_name)
    n_nudges = sum(
        1 for m in transcript
        if m.role == "user" and "Bare text is not graded" in (m.content or "")
    )
    assert n_nudges == 1


class _TokenReportingToollessProvider(InferenceProvider):
    """Every turn is tool-less content and costs a fixed token count, so a
    `max_total_tokens` cap bounds a run that would otherwise never end."""

    def __init__(self, tokens_per_turn: int):
        self._usage = TokenUsage(input_tokens=tokens_per_turn - 10, output_tokens=10)

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield ContentDeltaChunk(text="just-text")
        yield CompletedChunk(usage=self._usage, stop_reason="stop")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_submit_tool_not_offered_after_unfinished_result(tmp_path):
    """A submit-named env tool is offered as ending the rollout only until one
    of its results comes back finished=False (an env whose `submit` grades and
    continues)."""
    from agentorg.loop import _apply_env_terminal_if_needed, _env_submit_tools
    from agentorg.types import CEO_NAME, ToolCall, ToolDefinition, ToolResult
    from tests.test_smoke import MockInferenceProvider

    class _EnvTool:
        definition = ToolDefinition(
            name="submit",
            description="Grade your current work.",
            input_schema={"type": "object", "properties": {}},
            scope="env",
        )

        async def execute(self, ctx):  # pragma: no cover — never dispatched
            raise AssertionError

    cfg = SessionConfig(workspace_path=str(tmp_path), model="mock")
    session = await Harness(inference=MockInferenceProvider()).start_session(config=cfg)
    session.tools.register(_EnvTool())
    agent = session.agent(CEO_NAME)
    assert _env_submit_tools(session) == ["submit"]

    unfinished = ToolResult(
        ok=True, output={"blocks": [], "metadata": None, "reward": 0.1, "finished": False}
    )
    await _apply_env_terminal_if_needed(
        session, agent, ToolCall(id="c1", name="submit"), unfinished
    )
    assert session.terminal_event is None
    assert _env_submit_tools(session) == []


class _RecordingProvider(InferenceProvider):
    """Wraps an inner provider, recording each request's tool_choice value."""

    def __init__(self, inner: InferenceProvider):
        self._inner = inner
        self.tool_choices: list[Optional[str]] = []

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self.tool_choices.append(request.tool_choice)
        async for c in self._inner.stream(request):
            yield c

    def model_ids(self):
        return self._inner.model_ids()


@pytest.mark.asyncio
async def test_tool_choice_required_on_post_nudge_turn(tmp_path):
    """Normal turns: tool_choice is None (server default 'auto'). The turn
    immediately following a tool-less nudge is forced to 'required'."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    # Turn 1: tool-less ("42").  Turn 2 (after nudge): Bash call (recovers).
    # Turn 3: tool-less again (falls back to "done" once script exhausts).
    # Turn 4: tool-less → terminate.
    inner = _ToollessThenToolProvider([("Bash", {"command": "echo hi"})])
    recorder = _RecordingProvider(inner)
    harness = Harness(inference=recorder)
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)

    # Expected sequence of tool_choice values per inference call:
    #   turn 1 (initial)        : None   (streak=0)
    #   turn 2 (post-1st-nudge) : 'required'   (streak=1, just nudged)
    #   turn 3 (post-Bash)      : None   (Bash reset streak to 0)
    #   turn 4 (post-2nd-nudge) : 'required'   (streak re-armed)
    assert recorder.tool_choices == [None, "required", None, "required"], (
        f"unexpected tool_choice sequence: {recorder.tool_choices}"
    )


@pytest.mark.asyncio
async def test_tool_choice_unset_on_healthy_run(tmp_path):
    """A run that never triggers a tool-less nudge sees only tool_choice=None."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    # All three turns are real tool calls; the 4th falls back to tool-less,
    # which fires the first (and only) nudge — but we'll stop checking after.
    inner = _ScriptedToolProvider([("Bash", {"command": "echo a"})] * 3)
    recorder = _RecordingProvider(inner)
    harness = Harness(inference=recorder)
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=10)

    # The first three turns are all real tool calls → tool_choice stays None.
    assert recorder.tool_choices[:3] == [None, None, None]


@pytest.mark.asyncio
async def test_no_nudge_without_env_binding(tmp_path):
    """Interactive (no env) sessions still exit cleanly on a tool-less turn."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        headless=True,
        permission_policy="bypass",
        initial_user_message="solve it",
    )
    harness = Harness(inference=_AlwaysToollessProvider())
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=10)
    assert result.used_turns == 1
    assert session._env_tool_less_streak == 0


class _EndOnFinishedProvider(InferenceProvider):
    """Scripted CEO + org agents for the §6.5 end-on-finished tests.

    CEO turns come from `ceo_turn(step)` (a list of tool-call chunks, or None
    for a tool-less turn); org turns from `org_turn(step)`. Counts every turn
    so a test can assert none ran after the env finished.
    """

    def __init__(self, ceo_turn, org_turn):
        self._ceo_turn = ceo_turn
        self._org_turn = org_turn
        self.ceo_steps = 0
        self.org_steps = 0
        self.tools_by_ceo_step: dict[int, set[str]] = {}

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if "You are the CEO" in (request.messages[0].content or ""):
            self.ceo_steps += 1
            self.tools_by_ceo_step[self.ceo_steps] = {t.name for t in request.tools}
            calls = self._ceo_turn(self.ceo_steps)
        else:
            self.org_steps += 1
            calls = self._org_turn(self.org_steps)
        for call in calls or []:
            yield call
        yield CompletedChunk(
            usage=TokenUsage(), stop_reason="tool_calls" if calls else "stop"
        )

    def model_ids(self):
        return ["scripted"]


def _report_then_wait(step):
    """Org agent: report once (clearing the CEO's wait gate), then park."""
    if step == 1:
        return [ToolCallChunk(
            id="o1", name="ReportToCEO",
            args={"status": "completed", "summary": "done"},
        )]
    return [ToolCallChunk(id=f"o{step}", name="WaitForMessages", args={})]


def _end_on_finished_config(tmp_path) -> SessionConfig:
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="default",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        headless=True,
        permission_policy="bypass",
        initial_user_message="Solve the task.",
    )


def _event_rows(session) -> list[dict]:
    import json as _json

    path = session.paths.events_jsonl
    return [
        _json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_org_died_with_episode(session) -> None:
    """Every org agent was harness-fired at env-terminal, with no review, and
    the session then ended cleanly (not aborted)."""
    assert session.abort_requested is False
    org_agents = [a for a in session.all_agents() if a.kind == "org"]
    assert org_agents
    for a in org_agents:
        assert a.state == "fired"
        assert a.fired_source == "harness"
        assert a.exit_review is None
    rows = _event_rows(session)
    types = [r["type"] for r in rows]
    fired = [r for r in rows if r["type"] == "agent.fired"]
    assert {r["payload"]["name"] for r in fired} == {a.name for a in org_agents}
    assert all(r["payload"]["reason"] == "env-terminal" for r in fired)
    terminal_at = types.index("environment.terminal")
    assert all(types.index("agent.fired") > terminal_at for _ in fired)
    ended = [r for r in rows if r["type"] == "session.ended"]
    assert [r["payload"]["reason"] for r in ended] == ["complete"]


@pytest.mark.asyncio
async def test_ceo_env_terminal_fires_org_and_ends_session(tmp_path):
    """§6.5: when the CEO's env call returns finished=True the episode is
    over. The org dies with it (harness-fired, `reason="env-terminal"`, no
    exit review) and the session ends at once: the CEO gets no further turn.

    Regression for the post-terminal exit-review window, which kept the CEO
    (and an unrated roster) alive after the env had finished.
    """
    from tests.test_env_terminal_ordering import _FakeEnvTool

    def ceo_turn(step):
        if step == 1:
            return [ToolCallChunk(id="c1", name="HireOrgAgent",
                                  args={"role": "Tester", "description": "Verify it works."})]
        if step == 2:
            return [ToolCallChunk(id="c2", name="WaitForMessages", args={})]
        if step == 3:
            return [ToolCallChunk(id="c3", name="submit_answer", args={"answer": "done"})]
        # Unreachable: the episode ended on step 3.
        return [ToolCallChunk(id=f"c{step}", name="WaitForMessages", args={})]

    provider = _EndOnFinishedProvider(ceo_turn, _report_then_wait)
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_end_on_finished_config(tmp_path))
    session.tools.register(_FakeEnvTool(name="submit_answer", reward=0.5))

    result = await harness.run_to_completion(session, max_wall_s=15)

    assert result.terminated_by_env is True
    assert result.end_reason == "env-terminal"
    assert result.reward == pytest.approx(0.5)
    assert provider.ceo_steps == 3
    _assert_org_died_with_episode(session)
    # The finishing call's result is durable in the CEO transcript.
    submit = next(
        m for m in session.transcript(session.ceo_name)
        if m.role == "tool" and m.tool_call_id == "c3"
    )
    assert submit.tool_result.ok is True
    assert submit.tool_result.output["finished"] is True


@pytest.mark.asyncio
async def test_env_tool_after_finished_in_same_response_is_refused(tmp_path):
    """§6.5: once an env call returns finished=True the reward is locked in.
    A second env call later in the same response must be refused before it
    reaches the env, or a re-grade could overwrite the reward.

    Regression: if a grader's feedback on a wrong submission reveals the
    expected answer, a resubmission must not be able to overwrite the locked-in
    reward (here 0.0 -> 1.0).
    """
    from agentorg.tools.context import ok_result
    from agentorg.types import ToolDefinition

    class _LeakyGrader:
        """Grades 0.0 the first time and 1.0 on any resubmission."""

        def __init__(self):
            self.definition = ToolDefinition(
                name="submit_answer",
                description="Submit final answer to the env.",
                input_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                read_only=False,
                scope="env",
            )
            self.calls = 0

        async def execute(self, ctx):
            self.calls += 1
            return ok_result({
                "blocks": [{"type": "text", "text": "graded"}],
                "metadata": None,
                "reward": 0.0 if self.calls == 1 else 1.0,
                "finished": True,
            })

    def ceo_turn(step):
        if step == 1:
            return [
                ToolCallChunk(id="c1a", name="submit_answer", args={"answer": "wrong"}),
                ToolCallChunk(id="c1b", name="submit_answer", args={"answer": "leaked"}),
            ]
        return None

    provider = _EndOnFinishedProvider(ceo_turn, lambda step: None)
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_end_on_finished_config(tmp_path))
    grader = _LeakyGrader()
    session.tools.register(grader)

    result = await harness.run_to_completion(session, max_wall_s=15)

    # The first grade stands; the resubmission never reached the env.
    assert result.terminated_by_env is True
    assert result.reward == pytest.approx(0.0)
    assert grader.calls == 1
    assert provider.ceo_steps == 1
    resubmit = next(
        m for m in session.transcript(session.ceo_name)
        if m.role == "tool" and m.tool_call_id == "c1b"
    )
    assert resubmit.tool_result.ok is False
    assert resubmit.tool_result.error.code == "episode-finished"
    assert resubmit.tool_result.error.kind == "rejection"


@pytest.mark.asyncio
async def test_fire_parked_org_agent_unwinds_wait_loop(tmp_path):
    """Regression for a session hang: the CEO fires
    an org agent that is parked inside WaitForMessages. `fire_org_agent`
    wakes the parked task so its loop can unwind — but the WaitForMessages
    loop never checked the fired state, drained an empty inbox, and re-parked
    forever. The zombie task kept the supervisor's in-flight set non-empty,
    so the session never reached its end condition (no `session.ended`,
    workspace stuck "active" until max_wall_s).
    """

    class _FireParkedAgentProvider(InferenceProvider):
        """CEO: hire → wait (woken by report) → fire → exit.

        Org agent: report once, then park in WaitForMessages — so it is
        parked when the CEO's FireOrgAgent lands.
        """

        def __init__(self):
            self._ceo_step = 0
            self._org_step = 0

        @staticmethod
        def _extract_agent_name(request: InferenceRequest) -> Optional[str]:
            sys_prompt = request.messages[0].content or ""
            for line in sys_prompt.splitlines():
                stripped = line.strip()
                if " — " in stripped and stripped.startswith("- @"):
                    return stripped.split(" — ")[0].lstrip("- @")
            return None

        async def stream(self, request: InferenceRequest) -> AsyncIterator:
            sys_prompt = request.messages[0].content or ""
            is_ceo = "You are the CEO" in sys_prompt
            if not is_ceo:
                self._org_step += 1
                if self._org_step == 1:
                    yield ToolCallChunk(
                        id="o1",
                        name="ReportToCEO",
                        args={"status": "completed", "summary": "done"},
                    )
                    yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                    return
                yield ToolCallChunk(
                    id=f"o{self._org_step}", name="WaitForMessages", args={}
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            self._ceo_step += 1
            if self._ceo_step == 1:
                yield ToolCallChunk(
                    id="c1",
                    name="HireOrgAgent",
                    args={"role": "Tester", "description": "Verify."},
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if self._ceo_step == 2:
                yield ToolCallChunk(id="c2", name="WaitForMessages", args={})
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if self._ceo_step == 3:
                name = self._extract_agent_name(request)
                assert name, "could not find org agent name in CEO system prompt"
                yield ToolCallChunk(
                    id="c3",
                    name="FireOrgAgent",
                    args={
                        "name": name,
                        "category": "mission_complete",
                        "delivered": "Verified.",
                        "shortfalls": "",
                        "score": 8,
                    },
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            # Step 4+: tool-less — solo CEO with no env exits the loop.
            yield ContentDeltaChunk(text="")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["scripted"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="Build a test team.",
    )
    harness = Harness(inference=_FireParkedAgentProvider())
    session = await harness.start_session(config=config)

    await harness.run_to_completion(session, max_wall_s=15)

    # Pre-fix the fired agent's zombie task forced a max_wall_s abort.
    assert session.abort_requested is False, (
        "session aborted instead of completing — fired-while-parked org "
        "agent's WaitForMessages loop never unwound"
    )
    org_agents = [a for a in session.all_agents() if a.kind == "org"]
    assert len(org_agents) == 1
    assert org_agents[0].state == "fired"
    assert org_agents[0].fired_source == "ceo"
    assert org_agents[0].exit_review is not None


@pytest.mark.asyncio
async def test_org_agent_env_terminal_ends_session(tmp_path):
    """§6.5: when an ORG AGENT's env call returns finished=True, the episode
    ends for everyone: the parked CEO gets no further turn, every org agent
    (the submitter included) is harness-fired, and the session ends cleanly
    instead of hanging until max_wall_s.
    """
    from tests.test_env_terminal_ordering import _FakeEnvTool

    def ceo_turn(step):
        if step == 1:
            return [ToolCallChunk(id="c1", name="HireOrgAgent",
                                  args={"role": "Solver", "description": "Solve and submit."})]
        if step == 2:
            return [ToolCallChunk(id="c2", name="WaitForMessages", args={})]
        # Unreachable: the org agent's submit ended the episode.
        return [ToolCallChunk(id=f"c{step}", name="WaitForMessages", args={})]

    def org_turn(step):
        if step == 1:
            return [ToolCallChunk(id="o1", name="submit_answer", args={"answer": "42"})]
        return [ToolCallChunk(id=f"o{step}", name="WaitForMessages", args={})]

    provider = _EndOnFinishedProvider(ceo_turn, org_turn)
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_end_on_finished_config(tmp_path))
    session.tools.register(_FakeEnvTool(name="submit_answer", reward=0.5))

    result = await harness.run_to_completion(session, max_wall_s=15)

    assert result.terminated_by_env is True
    assert result.reward == pytest.approx(0.5)
    assert provider.ceo_steps == 2
    assert provider.org_steps == 1
    _assert_org_died_with_episode(session)


def test_reward_events_single_source_of_truth(tmp_path):
    """`reward_events` is the ONLY stored reward stream: `step_rewards`,
    `cumulative_reward`, and `reward_step_count` are derived views, entries
    carry agent + response_id attribution, the terminal grade appears exactly
    once, and `mark_terminal` never appends (double-count regression)."""
    from agentorg.session import Session
    from agentorg.tools.registry import ToolRegistry

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

    s.record_reward(0.3, agent="CEO", response_id="chatcmpl-1")
    s.record_reward(0.0, agent="Worker")
    s.record_reward(1.0, agent="CEO", response_id="chatcmpl-9", terminal=True)

    events = s.reward_events
    assert [e["kind"] for e in events] == ["step", "step", "terminal"]
    assert [e["agent"] for e in events] == ["CEO", "Worker", "CEO"]
    assert [e["response_id"] for e in events] == ["chatcmpl-1", None, "chatcmpl-9"]

    # Derived views agree with the stream by construction.
    assert s.step_rewards == [
        pytest.approx(0.3), pytest.approx(0.0), pytest.approx(1.0)
    ]
    assert s.cumulative_reward == pytest.approx(1.3)
    assert s.reward_step_count == 3
    assert sum(e["reward"] for e in events) == pytest.approx(s.cumulative_reward)

    # reward_events returns copies — callers can't corrupt the stream.
    events[0]["reward"] = 999.0
    events.append({"kind": "step", "reward": 5.0})
    assert s.cumulative_reward == pytest.approx(1.3)
    assert s.reward_step_count == 3

    # mark_terminal locks the final value but must NOT append a stream entry:
    # the terminal grade already entered via record_reward(terminal=True), so
    # a second entry would double-count the episode return.
    s.mark_terminal({"kind": "t"}, reward=1.0, stop=False)
    assert s.reward_step_count == 3
    assert sum(1 for e in s.reward_events if e["kind"] == "terminal") == 1
    assert s.effective_final_reward() == pytest.approx(1.0)
