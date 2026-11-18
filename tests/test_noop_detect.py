"""§A.8.5 no-op ("doom loop") detection, nudge, penalty and force-park.

A specialist that finishes its assignment has no way to yield, so some keep
their loop alive with shell calls that do nothing (`exit 0`, `echo done`).
These tests cover the four halves of the fix:

* the detector itself (`agentorg.noop_detect`), including the turn-level gate
  that spares a turn which did real work alongside a stray `echo`;
* the penalty stamped onto the offending ToolResult payload, and the three
  cases where it must NOT be stamped;
* the warning delivered as a `[Harness meta]` user message on the next turn;
* the force-park after `noop_park_strikes` consecutive no-op turns, and the
  removal of the grace period once an agent has been parked.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.loop import _dispatch_tools, run_agent
from agentorg.noop_detect import (
    extract_command,
    is_noop_call,
    is_noop_command,
    turn_is_noop,
)
from agentorg.org import spawn_org_agent
from agentorg.session import SessionConfig
from agentorg.tools.context import ok_result
from agentorg.types import (
    Message,
    OrgAgentSpec,
    ToolCall,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


NOOPS = [
    "exit 0",
    "exit",
    ":",
    "true",
    "exit 0;",
    " exit 0 ",
    "echo done",
    'echo "done"',
    "echo finished",
    'echo "Mission complete."',
    'echo "olive ockham signing off"',
    "echo 'bye'",
    "printf done",
    "echo",
    "echo done;",
    "ECHO Done.",
    "Exit 0",
    "echo All done.",
    "echo hello",
    "echo done\n",
]

REAL_WORK = [
    'echo "x" > file.txt',      # redirect writes a file
    "echo $(ls)",               # command substitution
    "echo a && make",           # chains real work
    "echo hi | tee log",        # pipe
    "echo a; make",             # chains real work
    "echo `date`",              # backtick substitution
    "sleep 5",                  # deliberately NOT a no-op
    "echo done\nmake build",    # the newline false positive
    "echo done\r\nmake",
    "cat << 'EOF' > f.py\nexit 0\nEOF",   # exit inside a heredoc
    "cd /x && exit 0",          # not the whole command
    "exit 0 && echo hi",
    "printf '%s' > f",
    "make build",
    "",
    "   ",
]


@pytest.mark.parametrize("command", NOOPS)
def test_noop_commands_match(command):
    assert is_noop_command(command) is True


@pytest.mark.parametrize("command", REAL_WORK)
def test_real_work_does_not_match(command):
    assert is_noop_command(command) is False


def test_extract_command_prefers_command_then_cmd_then_sole_string():
    assert extract_command({"command": "exit 0"}) == "exit 0"
    assert extract_command({"cmd": "exit 0"}) == "exit 0"
    # Env schemas may name it something else; a single string value is enough.
    assert extract_command({"script": "exit 0"}) == "exit 0"
    # Ambiguous — two strings, no known key.
    assert extract_command({"a": "exit 0", "b": "true"}) is None
    assert extract_command({"timeout": 5}) is None
    assert extract_command(None) is None


def test_is_noop_call_requires_a_shell_tool():
    assert is_noop_call(ToolCall(id="1", name="bash", args={"command": "exit 0"}))
    # Case-insensitive: the host tool is `Bash`, the env's is `bash`.
    assert is_noop_call(ToolCall(id="2", name="Bash", args={"command": "echo done"}))
    # A non-shell tool that happens to carry a matching string is not a no-op.
    assert not is_noop_call(
        ToolCall(id="3", name="PostToChannel", args={"command": "exit 0"})
    )


def test_turn_gate_spares_a_turn_that_did_real_work():
    """A stray `echo done` alongside a report is not a stall."""
    calls = [
        ToolCall(id="1", name="bash", args={"command": "echo done"}),
        ToolCall(
            id="2",
            name="ReportToCEO",
            args={"status": "completed", "summary": "s"},
        ),
    ]
    assert turn_is_noop(calls) is False
    assert turn_is_noop(calls[:1]) is True
    assert turn_is_noop([]) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _EnvShapedBash:
    """A `bash` tool returning the OpenReward env payload shape.

    `_OpenRewardTool.execute` packs `{blocks, metadata, reward, finished}`;
    the penalty is stamped into that same `reward` slot, so the tests have to
    use the real shape rather than an arbitrary dict.
    """

    def __init__(self, *, reward=None, finished=False, ok=True):
        self._reward = reward
        self._finished = finished
        self._ok = ok
        self.definition = ToolDefinition(
            name="bash",
            description="run a shell command",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            read_only=False,
            scope="env",
        )

    async def execute(self, ctx):
        return ok_result(
            {
                "blocks": [{"type": "text", "text": ""}],
                "metadata": {},
                "reward": self._reward,
                "finished": self._finished,
            }
        )


class _ScriptedProvider(InferenceProvider):
    """Yields one scripted turn per `.stream()` call, then a tool-less turn."""

    def __init__(self, script):
        self._script = list(script)

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if not self._script:
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        for chunk in self._script.pop(0):
            yield chunk
        yield CompletedChunk(
            usage=TokenUsage(), stop_reason="tool_calls", response_id="resp-1"
        )

    def model_ids(self):
        return ["mock"]


def _noop_turn(idx: int, command: str = "echo done"):
    return [ToolCallChunk(id=f"n{idx}", name="bash", args={"command": command})]


async def _session(tmp_path, provider=None, budget=2, **overrides):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=budget,
        headless=True,
        permission_policy="bypass",
        **overrides,
    )
    harness = Harness(inference=provider or _ScriptedProvider([]))
    return await harness.start_session(config=config)


def _events(session, event_type):
    out = []
    path = session.paths.events_jsonl
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == event_type:
            out.append(row)
    return out


def _meta_messages(session, name):
    return [
        m.content
        for m in session.transcript(name)
        if m.role == "user" and isinstance(m.content, str) and "[Harness meta]" in m.content
    ]


# ---------------------------------------------------------------------------
# The penalty
# ---------------------------------------------------------------------------


async def _dispatch_one(session, agent, tool, command="echo done", noop_turn=True):
    session.tools.register(tool)
    call = ToolCall(id="c1", name="bash", args={"command": command})
    assistant = Message(role="assistant", tool_calls=[call], provider_response_id="r1")
    await session.append_message(agent.name, assistant)
    await _dispatch_tools(session, agent, assistant, [call], noop_turn=noop_turn)
    result_msg = [m for m in session.transcript(agent.name) if m.role == "tool"][-1]
    return result_msg.tool_result


@pytest.mark.asyncio
async def test_penalty_stamped_on_specialist_noop(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))
    result = await _dispatch_one(session, agent, _EnvShapedBash())

    assert result.output["shaping_reward"] == pytest.approx(session.config.noop_penalty)
    assert result.output["shaping_reward"] < 0
    assert result.output["shaping_reason"] == "noop-termination-signal"
    # `reward` means "an env graded this step" and MUST stay untouched.
    assert result.output["reward"] is None
    # The env reward stream is untouched too: the penalty is a harness
    # opinion, not an env grade, so reward figures stay comparable.
    assert session.step_rewards == []
    assert session.cumulative_reward == 0.0


@pytest.mark.asyncio
async def test_penalty_sits_alongside_a_real_env_grade(tmp_path):
    """The two keys never collide, so a graded step keeps its grade."""
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))
    result = await _dispatch_one(session, agent, _EnvShapedBash(reward=1.0))

    assert result.output["reward"] == pytest.approx(1.0)
    assert result.output["shaping_reward"] == pytest.approx(session.config.noop_penalty)


@pytest.mark.asyncio
async def test_penalty_does_not_disturb_a_terminal_payload(tmp_path):
    """`_apply_env_terminal_if_needed` reads `reward`/`finished` off this
    payload as the authoritative terminal grade moments later."""
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))
    result = await _dispatch_one(session, agent, _EnvShapedBash(finished=True))

    assert result.output["reward"] is None
    assert result.output["finished"] is True


@pytest.mark.asyncio
async def test_no_penalty_when_the_turn_did_real_work(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))
    result = await _dispatch_one(session, agent, _EnvShapedBash(), noop_turn=False)

    assert "shaping_reward" not in result.output
    assert result.output["reward"] is None


# ---------------------------------------------------------------------------
# Nudge, streak, force-park
# ---------------------------------------------------------------------------


async def _drive(session, agent, timeout=5.0):
    """Run the agent's loop until it parks or exits; True iff it parked.

    Parking blocks forever by design, so the loop runs as a task and the
    waiting set is sampled while that task is still live. Cancelling first
    would be useless: `park_and_wait`'s `finally` discards the agent from
    `_waiting` on the way out, so a cancelled park looks exactly like no park.
    """
    task = asyncio.create_task(run_agent(session, agent))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    parked = False
    while loop.time() < deadline:
        await asyncio.sleep(0.01)
        if session.wakes.is_waiting(agent.name):
            parked = True
            break
        if task.done():
            break
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return parked


@pytest.mark.asyncio
async def test_first_noop_turn_warns_and_keeps_the_agent_running(tmp_path):
    provider = _ScriptedProvider([_noop_turn(1), _noop_turn(2)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    metas = _meta_messages(session, agent.name)
    assert metas, "expected a [Harness meta] warning after the first no-op turn"
    body = metas[0]
    assert "is a no-op" in body
    assert "ReportToCEO" in body and "WaitForMessages" in body
    assert "mission complete" in body
    # The specialist was not parked on the first strike.
    assert session.noop_park_count(agent.name) == 0

    detected = _events(session, "agent.noop_detected")
    assert detected
    assert detected[0]["payload"]["name"] == agent.name
    assert detected[0]["payload"]["command"] == "echo done"
    assert detected[0]["payload"]["streak"] == 1
    assert detected[0]["payload"]["parked"] is False


@pytest.mark.asyncio
async def test_repeat_warning_escalates(tmp_path):
    """A model that ignored the first warning will not read the same paragraph
    differently, so the repeat leads with the count and the consequence."""
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 4)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    metas = _meta_messages(session, agent.name)
    assert len(metas) == 2
    assert "recent turns have done nothing" not in metas[0]
    assert "2 of your recent turns have done nothing" in metas[1]
    assert "park you on your next one" in metas[1]


@pytest.mark.asyncio
async def test_three_consecutive_noop_turns_force_park(tmp_path):
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 6)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    assert await _drive(session, agent), "agent should be parked"
    assert session.noop_park_count(agent.name) == 1
    stats = session.noop_stats[agent.name]
    assert stats["turns"] == 3
    assert stats["max_run"] == 3
    assert stats["calls"] == 3

    # The final message is left PENDING on purpose: waking then parking would
    # be a no-op (WakeBroker.park_and_wait returns early if the event is set),
    # so the agent reads it on its next genuine wake.
    assert await session.inbox.has_pending(agent.name)
    pending = session.inbox.load(agent.name)
    assert any("has parked you" in (i.content or "") for i in pending)

    parked = [
        e for e in _events(session, "agent.noop_detected") if e["payload"]["parked"]
    ]
    assert len(parked) == 1
    assert parked[0]["payload"]["streak"] == 3


@pytest.mark.asyncio
async def test_real_work_resets_the_streak(tmp_path):
    """Two no-ops either side of a real command must not reach the threshold."""
    provider = _ScriptedProvider(
        [
            _noop_turn(1),
            [ToolCallChunk(id="w", name="bash", args={"command": "make build"})],
            _noop_turn(2),
        ]
    )
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    assert session.noop_park_count(agent.name) == 0
    stats = session.noop_stats[agent.name]
    assert stats["turns"] == 2
    assert stats["max_run"] == 1


@pytest.mark.asyncio
async def test_park_strikes_are_configurable(tmp_path):
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 4)])
    session = await _session(tmp_path, provider=provider, noop_park_strikes=1)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    assert await _drive(session, agent)
    assert session.noop_stats[agent.name]["turns"] == 1


@pytest.mark.asyncio
async def test_a_previously_parked_agent_gets_no_grace_period(tmp_path):
    """After one force-park the ladder collapses to a single strike, so a loop
    that resumes on wake cannot buy another three turns."""
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 8)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    task = asyncio.create_task(run_agent(session, agent))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if session.wakes.is_waiting(agent.name):
            break
    assert session.wakes.is_waiting(agent.name)
    assert session.noop_park_count(agent.name) == 1

    # Wake it as the CEO would; the very next no-op turn re-parks it.
    session.wakes.wake(agent.name)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if session.noop_park_count(agent.name) == 2:
            break
    assert session.noop_park_count(agent.name) == 2
    assert session.noop_stats[agent.name]["turns"] == 4  # 3 + 1

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_ceo_is_nudged_and_penalised_but_never_parked(tmp_path):
    """The CEO does the same thing (`exit 0`, `echo done`) and burns the same
    budget, so it gets the gate and the penalty. It cannot get the remedy:
    there is nobody for it to report to and nothing to park for, and parking
    it would strand the rollout."""
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 6)])
    session = await _session(
        tmp_path, provider=provider, budget=0, initial_user_message="do the thing"
    )
    session.tools.register(_EnvShapedBash())
    ceo = session.agent(session.ceo_name)

    parked = await _drive(session, ceo)

    assert parked is False
    assert session.noop_park_count(session.ceo_name) == 0
    assert session.noop_stats[session.ceo_name]["kind"] == "ceo"
    assert session.noop_stats[session.ceo_name]["turns"] >= 3

    detected = _events(session, "agent.noop_detected")
    assert detected and all(e["payload"]["kind"] == "ceo" for e in detected)
    assert all(e["payload"]["parked"] is False for e in detected)

    tool_msgs = [m for m in session.transcript(session.ceo_name) if m.role == "tool"]
    assert all(
        m.tool_result.output["shaping_reward"] == pytest.approx(session.config.noop_penalty)
        for m in tool_msgs
    )
    metas = _meta_messages(session, session.ceo_name)
    assert metas
    # CEO wording, not the specialist's.
    assert "ReportToCEO" not in metas[0]
    assert "The environment has not ended" in metas[0]
    assert "Keep working" in metas[0]
    # No submission out: plenty of envs expose no submission tool, and
    # offering one to a CEO that already thinks it is finished invites it to
    # stop early.
    assert "submit" not in metas[0].lower()


@pytest.mark.asyncio
async def test_rollout_metrics_report_noop_counters(tmp_path):
    from agentorg.metrics import compute_rollout_metrics

    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 6)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    metrics = compute_rollout_metrics(session)
    assert metrics["noop_turns"] == 3
    assert metrics["noop_calls"] == 3
    assert metrics["noop_max_run"] == 3
    assert metrics["noop_parks"] == 1


# ---------------------------------------------------------------------------
# Persistence and replay invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_persisted_records_carry_the_penalty(tmp_path):
    """SPEC §12.4: a ToolResult is written to events.jsonl and messages.jsonl
    verbatim, so the two must not disagree about its reward. `tool.completed`
    is emitted from inside `_execute_tool`, which is why the penalty is
    stamped there rather than back in `_dispatch_tools`."""
    provider = _ScriptedProvider([_noop_turn(1)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    from_events = [
        e["payload"]["output"]["shaping_reward"]
        for e in _events(session, "tool.completed")
        if isinstance(e["payload"].get("output"), dict)
    ]
    from_messages = [
        m.tool_result.output["shaping_reward"]
        for m in session.transcript(agent.name)
        if m.role == "tool" and isinstance(m.tool_result.output, dict)
    ]
    assert from_events == from_messages
    assert from_events == [pytest.approx(session.config.noop_penalty)]


@pytest.mark.asyncio
async def test_turn_events_stay_balanced_across_the_park(tmp_path):
    """The park branch emits its own `turn.ended` before parking, so it must
    not also fall through to the one at the bottom of the loop."""
    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 6)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    started = [e for e in _events(session, "turn.started") if e["name"] == agent.name]
    ended = [e for e in _events(session, "turn.ended") if e["name"] == agent.name]
    assert len(started) == len(ended) == 3
    assert [e["payload"]["turn_index"] for e in ended] == [0, 1, 2]


@pytest.mark.asyncio
async def test_penalty_is_stamped_on_the_parallel_read_only_path(tmp_path):
    """`bash` fails the env adapter's read-only heuristic today, so it always
    dispatches sequentially. An env is free to name its shell something that
    does look read-only, and the penalty must survive that."""
    tool = _EnvShapedBash()
    tool.definition = tool.definition.model_copy(update={"read_only": True})
    provider = _ScriptedProvider([_noop_turn(1)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(tool)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    tool_msg = [m for m in session.transcript(agent.name) if m.role == "tool"][-1]
    assert tool_msg.tool_result.output["shaping_reward"] == pytest.approx(
        session.config.noop_penalty
    )


@pytest.mark.asyncio
async def test_a_failed_shell_call_is_still_a_noop_turn_but_is_not_stamped(tmp_path):
    """An `exit 0` against a shell tool that is not registered still signals
    "I am done" and still warrants the nudge, but there is no successful
    payload to stamp."""
    provider = _ScriptedProvider(
        [[ToolCallChunk(id="n1", name="sh", args={"command": "exit 0"})]]
    )
    session = await _session(tmp_path, provider=provider)
    # `sh` is in SHELL_TOOL_NAMES but nothing registers it, so the call fails.
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    tool_msg = [m for m in session.transcript(agent.name) if m.role == "tool"][-1]
    assert tool_msg.tool_result.ok is False
    assert session.noop_stats[agent.name]["turns"] == 1
    assert _meta_messages(session, agent.name)


@pytest.mark.asyncio
async def test_penalty_stays_out_of_every_env_reward_consumer(tmp_path):
    """The regression this key choice exists to prevent.

    `trajectory.build_trajectory` and downstream consumers that sum `reward`
    treat each `output["reward"]` as an env grade, where a -0.05 is
    indistinguishable from a real one. None of them may see the penalty.
    """
    from agentorg.trajectory import build_trajectory

    provider = _ScriptedProvider([_noop_turn(1)])
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)
    await session.persist_state()

    # build_trajectory, off disk.
    traj = build_trajectory(session.paths.root)
    assert traj["reward_steps"] == 0
    assert traj["cumulative_reward"] is None

    # The extractors' exact read: payload.output.reward on tool.completed.
    assert [
        e["payload"]["output"].get("reward")
        for e in _events(session, "tool.completed")
        if isinstance(e["payload"].get("output"), dict)
    ] == [None]

    # And the env reward stream.
    assert session.step_rewards == []


# ---------------------------------------------------------------------------
# Interleaving: the evasion strict runs let through
# ---------------------------------------------------------------------------


def _work_turn(idx: int):
    return [ToolCallChunk(id=f"w{idx}", name="bash", args={"command": "make build"})]


@pytest.mark.asyncio
async def test_noops_interleaved_with_real_work_still_trip_the_window(tmp_path):
    """The reported CEO shape: `echo done` / real call / `exit 0` / real call.

    A strict consecutive run resets on every real call and never reaches the
    threshold, which is how three CEO rollouts slipped past the first version.
    """
    provider = _ScriptedProvider(
        [
            _noop_turn(1, "echo finished"),
            _work_turn(1),
            _noop_turn(2, "exit 0"),
            _work_turn(2),
            _noop_turn(3, 'echo "Final PnL: -75.83"'),
            _work_turn(3),
            _noop_turn(4, "exit 0"),
        ]
    )
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    assert await _drive(session, agent), "interleaved no-ops should still park"
    stats = session.noop_stats[agent.name]
    assert stats["parks"] == 1
    # The consecutive run never got past 1 — which is exactly the point.
    assert stats["max_run"] == 1


@pytest.mark.asyncio
async def test_the_window_ages_out_so_sparse_noops_never_trip(tmp_path):
    """One `echo hello` probe every few turns is not a doom loop. With a
    window of 4 and 3 strikes, a no-op every third turn can never accumulate."""
    script = []
    for i in range(1, 5):
        script += [_noop_turn(i, "echo hello"), _work_turn(i), _work_turn(100 + i)]
    provider = _ScriptedProvider(script)
    session = await _session(tmp_path, provider=provider, noop_window=4)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    assert session.noop_park_count(agent.name) == 0
    assert session.noop_stats[agent.name]["turns"] == 4


@pytest.mark.asyncio
async def test_rollout_metrics_split_noops_by_agent_kind(tmp_path):
    """CEO and specialist no-ops have different causes and remedies; the
    metrics should report them separately without trace parsing."""
    from agentorg.metrics import compute_rollout_metrics

    provider = _ScriptedProvider([_noop_turn(i) for i in range(1, 4)])
    session = await _session(
        tmp_path, provider=provider, budget=0, initial_user_message="go"
    )
    session.tools.register(_EnvShapedBash())

    await _drive(session, session.agent(session.ceo_name))

    metrics = compute_rollout_metrics(session)
    assert metrics["noop_turns_ceo"] >= 3
    assert metrics["noop_turns_org"] == 0
    assert metrics["noop_parks"] == 0


# ---------------------------------------------------------------------------
# `exit 0` is unconditional: no legitimate use, in any turn
# ---------------------------------------------------------------------------


ALWAYS = ["exit 0", "exit", ":", "true", "exit 0;", "Exit 0", " exit  12 "]
INERT_ONLY = ["echo done", 'echo "done"', "echo hello", "printf done", "echo"]


@pytest.mark.parametrize("command", ALWAYS)
def test_unconditional_noops(command):
    from agentorg.noop_detect import is_always_noop_command

    assert is_noop_command(command) is True
    assert is_always_noop_command(command) is True


@pytest.mark.parametrize("command", INERT_ONLY)
def test_message_carrying_noops_are_not_unconditional(command):
    """`echo` does nothing either, but one on its own can be a probe or a
    marker. `exit` cannot be anything."""
    from agentorg.noop_detect import is_always_noop_command

    assert is_noop_command(command) is True
    assert is_always_noop_command(command) is False


@pytest.mark.parametrize("command", REAL_WORK)
def test_real_work_is_never_unconditional(command):
    from agentorg.noop_detect import is_always_noop_command

    assert is_always_noop_command(command) is False


@pytest.mark.asyncio
async def test_a_stray_exit_beside_real_work_is_penalised_but_not_a_stall(tmp_path):
    """Pairing `exit 0` with a real call is the obvious way to slip a
    termination signal past a whole-turn gate. The call is penalised and
    counted; the turn is not treated as a stall."""
    provider = _ScriptedProvider(
        [
            [
                ToolCallChunk(id="a", name="bash", args={"command": "exit 0"}),
                ToolCallChunk(id="b", name="bash", args={"command": "make build"}),
            ]
        ]
    )
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    results = {
        m.tool_call_id: m.tool_result.output
        for m in session.transcript(agent.name)
        if m.role == "tool"
    }
    assert results["a"]["shaping_reward"] == pytest.approx(session.config.noop_penalty)
    assert "shaping_reward" not in results["b"]

    stats = session.noop_stats[agent.name]
    assert stats["calls"] == 1
    assert stats["turns"] == 0     # the turn did real work
    assert stats["max_run"] == 0
    assert session.noop_park_count(agent.name) == 0

    # No nudge for a productive turn, but the event is there to measure.
    assert _meta_messages(session, agent.name) == []
    detected = _events(session, "agent.noop_detected")
    assert [e["payload"]["turn_noop"] for e in detected] == [False]


@pytest.mark.asyncio
async def test_a_stray_echo_beside_real_work_is_left_alone(tmp_path):
    """The other half of the same rule: `echo done` next to a ReportToCEO is a
    stray line in a working turn, and penalising it would be wrong."""
    provider = _ScriptedProvider(
        [
            [
                ToolCallChunk(id="a", name="bash", args={"command": "echo done"}),
                ToolCallChunk(id="b", name="bash", args={"command": "make build"}),
            ]
        ]
    )
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    await _drive(session, agent)

    assert all(
        "shaping_reward" not in m.tool_result.output
        for m in session.transcript(agent.name)
        if m.role == "tool"
    )
    assert session.noop_stats == {}
    assert _events(session, "agent.noop_detected") == []


# ---------------------------------------------------------------------------
# The CEO nudge has no park to bound it, so it is rationed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ceo_nudge_is_rationed_but_the_penalty_is_not(tmp_path):
    """A specialist's ladder ends at the park; the CEO has no terminal state,
    so without rationing the same ~1 KB paragraph lands on every no-op turn:
    over 100 such turns is ~100 KB of harness meta, into a transcript that is
    already going wrong."""
    n = 111
    provider = _ScriptedProvider([_noop_turn(i) for i in range(n)])
    session = await _session(
        tmp_path, provider=provider, budget=0, initial_user_message="go"
    )
    session.tools.register(_EnvShapedBash())
    ceo = session.agent(session.ceo_name)

    await _drive(session, ceo, timeout=25)

    turns = session.noop_stats[session.ceo_name]["turns"]
    assert turns == n
    metas = _meta_messages(session, session.ceo_name)
    # One per window, not one per turn.
    assert len(metas) <= n // session.config.noop_window + 1
    assert sum(len(m) for m in metas) < 10_000

    # Rationing the text must not ration the signal.
    assert len(_events(session, "agent.noop_detected")) == n
    penalised = [
        m for m in session.transcript(session.ceo_name)
        if m.role == "tool" and "shaping_reward" in (m.tool_result.output or {})
    ]
    assert len(penalised) == n

    # First is the full warning; repeats are terse.
    assert "The environment has not ended" in metas[0]
    assert all("Still no-ops:" in m for m in metas[1:])
    assert all(len(m) < len(metas[0]) for m in metas[1:])


@pytest.mark.asyncio
async def test_a_real_call_in_between_does_not_lift_the_rationing(tmp_path):
    """The regression that made the first rationing attempt useless.

    Resetting on any productive turn sounds right — a fresh spell deserves the
    full warning — but an agent alternating `echo done` with a real call is
    never more than one turn from a reset, so it drew a full-length warning on
    every single no-op turn. That is precisely the interleaved shape this
    feature exists to catch, so the rationing was off exactly where it was
    needed.
    """
    provider = _ScriptedProvider(
        [
            _noop_turn(1), _work_turn(1), _noop_turn(2), _work_turn(2),
            _noop_turn(3), _work_turn(3), _noop_turn(4),
        ]
    )
    session = await _session(
        tmp_path, provider=provider, budget=0, initial_user_message="go"
    )
    session.tools.register(_EnvShapedBash())

    await _drive(session, session.agent(session.ceo_name))

    metas = _meta_messages(session, session.ceo_name)
    # Four no-op turns, but the interleaved real calls buy no extra warnings.
    assert session.noop_stats[session.ceo_name]["turns"] == 4
    assert len(metas) == 1
    assert "The environment has not ended" in metas[0]
    # Every no-op call is still penalised and still emits an event.
    assert len(_events(session, "agent.noop_detected")) == 4


@pytest.mark.asyncio
async def test_a_tool_less_turn_never_reaches_the_window(tmp_path):
    """A turn with no tool calls is not productive, and must not age a slot off
    the window. It never gets the chance: every branch of the §4.5.1 tool-less
    path returns or continues before the no-op block, so `turn_is_noop([])`
    returning False is unreachable rather than merely harmless."""
    provider = _ScriptedProvider(
        [_noop_turn(1), [], _noop_turn(2), [], _noop_turn(3)]
    )
    session = await _session(tmp_path, provider=provider)
    session.tools.register(_EnvShapedBash())
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Worker", description="x"))

    # A tool-less turn parks an org agent, so the run stops at the first one.
    await _drive(session, agent)

    # The window was never aged by the tool-less turn: the single no-op turn
    # before it is still the only entry, and no productive turn was recorded.
    assert session.noop_stats[agent.name]["turns"] == 1
    assert list(session._noop_window[agent.name]) == [True]


# ---------------------------------------------------------------------------
# The standing prompt rule
# ---------------------------------------------------------------------------


def _ceo_prompt(**kw):
    from agentorg.prompts import build_ceo_system_prompt

    base = dict(
        mode="default", env=None, toolset_name=None, has_filesystem_tools=False,
        headless=True, workspace_agents_md=None, date_str=None,
        budget=2, live_org_agents=[],
    )
    base.update(kw)
    return build_ceo_system_prompt(**base)


def test_the_wind_down_rule_reaches_a_solo_ceo():
    """It used to live in COORDINATION_PRINCIPLES, which only renders once the
    CEO has live specialists — so the shape most likely to decide it is
    finished never saw the rule at all."""
    for kw in ({"budget": 0}, {"budget": 2}, {"mode": "org-plan", "budget": 2}):
        prompt = _ceo_prompt(**kw)
        assert "# Ending the rollout" in prompt, kw
        assert "Never use the shell to wind down" in prompt, kw
        assert "recorded as no-ops" in prompt, kw


def test_the_wind_down_rule_never_makes_submission_unconditional():
    """Many envs expose no submission tool, so an unqualified "submit when
    done" is an instruction the CEO cannot follow — and reads as permission to
    stop, to one that has already decided it is finished."""
    for budget in (0, 2):
        prompt = _ceo_prompt(budget=budget)
        line = next(
            x for x in prompt.split("\n") if "submission tool" in x and "genuinely" in x
        )
        assert line.startswith("- If you have a submission tool")
        assert "assume the environment has not ended" in line
        assert "default to continuing" in line


def test_the_wind_down_rule_only_offers_agents_when_there_are_agents():
    assert "direct an agent" not in _ceo_prompt(budget=0)
    assert "direct an agent" in _ceo_prompt(budget=2)


# ---------------------------------------------------------------------------
# The §4.5.1 tool-less nudge must not contradict §A.8.5
# ---------------------------------------------------------------------------


class _NamedEnvTool(_EnvShapedBash):
    def __init__(self, name):
        super().__init__()
        self.definition = self.definition.model_copy(update={"name": name})


async def _tool_less_nudge(tmp_path, tool_names, truncated=False):
    from agentorg.loop import _inject_env_tool_less_nudge

    session = await _session(tmp_path, budget=0)
    for n in tool_names:
        session.tools.register(_NamedEnvTool(n))
    await _inject_env_tool_less_nudge(session, truncated=truncated)
    return session.inbox.load(session.ceo_name)[-1].content


@pytest.mark.asyncio
async def test_tool_less_nudge_never_offers_a_shell_as_the_way_to_submit(tmp_path):
    """The contradiction this closes: §4.5.1 listed every `scope="env"` tool as
    a way to "submit your answer and end the rollout" — `bash`, `ls`, `grep`
    included. An agent that took it at its word reached for `bash`, put the
    cheapest thing it could in it, and would be penalised by §A.8.5 for a call
    the harness had just asked for."""
    body = await _tool_less_nudge(tmp_path, ["bash", "ls", "grep", "submit_answer"])

    assert "call `submit_answer` with your final answer" in body
    # Gated on wanting to stop, not offered unconditionally.
    assert "If you want to finish now and stop working" in body
    for shell in ("`bash`", "`ls`", "`grep`"):
        assert shell not in body
    assert "Do not reach for a no-op" in body


@pytest.mark.asyncio
async def test_tool_less_nudge_hedges_when_no_tool_looks_like_a_submitter(tmp_path):
    """Recognition is a name heuristic, so absence is not something the harness
    knows. Asserting "this environment has no submission tool" would be a
    confident falsehood the moment an env names its one something the verb list
    misses — so state the uncertainty and give the default."""
    body = await _tool_less_nudge(tmp_path, ["bash", "ls", "grep"])

    assert "your default assumption should be that there is more to do" in body
    assert "and you want to finish now" in body
    # No claim about what the environment does or does not have.
    assert "has no submission tool" not in body
    assert "no submission tool" not in body
    assert "Do not reach for a no-op" in body


@pytest.mark.asyncio
async def test_tool_less_nudge_hedge_survives_an_unrecognised_submit_tool(tmp_path):
    """The case the hedge exists for: a real submission tool the verb list does
    not recognise. The nudge must not tell the agent it isn't there."""
    body = await _tool_less_nudge(tmp_path, ["bash", "hand_in_solution"])

    assert "no submission tool" not in body
    assert "your default assumption should be that there is more to do" in body


@pytest.mark.asyncio
async def test_tool_less_nudge_lists_several_submit_tools(tmp_path):
    body = await _tool_less_nudge(tmp_path, ["bash", "submit_answer", "answer"])
    assert "`answer`, `submit_answer`" in body
    assert "`bash`" not in body


@pytest.mark.asyncio
async def test_the_truncated_variant_gets_the_same_routing(tmp_path):
    body = await _tool_less_nudge(tmp_path, ["bash"], truncated=True)
    assert "cut off at the per-turn output limit" in body
    assert "your default assumption should be that there is more to do" in body
    assert "Do not reach for a no-op" in body
