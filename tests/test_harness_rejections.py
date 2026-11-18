"""Harness-level rejections: logged as events, never fatal to the session.

A call the harness refuses before the tool runs (allow/disallow scoping,
`input_schema`, plan mode, the §4.5.2 wait gate, permission policy) never
reaches the environment. Two consequences these tests guard:

* such refusals must not feed the env-tool circuit breaker, or a CEO that
  follows the wait-gate's own advice — revoke the tool from the peers
  holding it — kills its own session when the revoked agents' next calls
  come back `not-permitted`;
* they must emit `tool.requested` and `tool.completed`, or refusals appear in
  messages.jsonl but not in events.jsonl.
"""

from __future__ import annotations

import json

from agentorg.harness import Harness
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.loop import _dispatch_tools, _serialize_tool_result
from agentorg.org import spawn_org_agent
from agentorg.session import SessionConfig
from agentorg.tools.context import err_result, ok_result
from agentorg.types import (
    Message,
    OrgAgentSpec,
    ToolCall,
    ToolDefinition,
    ToolResultError,
)


class _EnvTool:
    """State-mutating env tool, like `bash` under a claude-code toolset."""

    def __init__(self, name: str = "bash", fail: bool = False):
        self.definition = ToolDefinition(
            name=name,
            description="env shell",
            input_schema={"type": "object", "properties": {}},
            read_only=False,
            scope="env",
        )
        self._fail = fail

    async def execute(self, ctx):
        if self._fail:
            return err_result("env-invalid-call", "input_validation: amount must be > 0")
        return ok_result({"ran": True})


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


async def _session(tmp_path, *, fail: bool = False):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=4,
        mode="default",
        headless=True,
        permission_policy="bypass",
    )
    session = await Harness(inference=_Stub()).start_session(config=config)
    session.tools.register(_EnvTool(fail=fail))
    return session


async def _call_bash(session, agent, call_id: str):
    call = ToolCall(id=call_id, name="bash", args={})
    asst = Message(role="assistant", tool_calls=[call])
    await session.append_message(agent.name, asst)
    await _dispatch_tools(session, agent, asst, [call])
    return [m for m in session.transcript(agent.name) if m.role == "tool"][-1].tool_result


def _events(session, event_type):
    path = session.paths.events_jsonl
    if not path.exists():
        return []
    rows = (json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    return [r for r in rows if r.get("type") == event_type]


async def test_revoked_tool_does_not_trip_breaker(tmp_path):
    """Revoke, then the agent keeps calling: the breaker must not trip."""
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    agent.tool_disallowlist = list(agent.tool_disallowlist or []) + ["bash"]
    await session.update_agent(agent)

    for i in range(8):
        result = await _call_bash(session, agent, f"r{i}")
        assert not result.ok
        assert result.error.code == "not-permitted"
    assert not session.abort_requested, session.abort_reason


async def test_wait_gate_does_not_trip_breaker(tmp_path):
    """A CEO that retries the gated tool is stuck, not fatal."""
    session = await _session(tmp_path)
    await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)

    for i in range(8):
        result = await _call_bash(session, ceo, f"g{i}")
        assert result.error.code == "wait-gate-restricted"
    assert not session.abort_requested, session.abort_reason


async def test_real_env_failures_still_trip_breaker(tmp_path):
    """The breaker still fires for failures the env actually produced."""
    session = await _session(tmp_path, fail=True)
    ceo = session.agent(session.ceo_name)

    for i in range(5):
        result = await _call_bash(session, ceo, f"e{i}")
        assert result.error.code == "env-invalid-call"
    assert session.abort_requested
    assert session.abort_reason == "env-tool-circuit-breaker:bash:env-invalid-call"


async def test_rejected_calls_are_visible_in_events(tmp_path):
    """Gated calls reach events.jsonl, so error counts taken there are right."""
    session = await _session(tmp_path)
    await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)
    await _call_bash(session, ceo, "v1")

    requested = [e for e in _events(session, "tool.requested")
                 if e["payload"].get("call_id") == "v1"]
    completed = [e for e in _events(session, "tool.completed")
                 if e["payload"].get("call_id") == "v1"]
    assert len(requested) == 1, "a refused call is still a requested call"
    assert len(completed) == 1, "a refused call must report how it ended"
    assert completed[0]["payload"]["ok"] is False
    assert completed[0]["payload"]["error"]["code"] == "wait-gate-restricted"
    # The executed path must not double-emit now that the wrapper owns both.
    assert len(_events(session, "tool.started")) == 0


async def test_successful_call_emits_each_event_once(tmp_path):
    session = await _session(tmp_path)
    ceo = session.agent(session.ceo_name)
    result = await _call_bash(session, ceo, "s1")
    assert result.ok
    for ev in ("tool.requested", "tool.started", "tool.completed"):
        matching = [e for e in _events(session, ev)
                    if e["payload"].get("call_id") == "s1"]
        assert len(matching) == 1, f"{ev} emitted {len(matching)}x"


async def test_update_org_agent_tools_refuses_a_no_op(tmp_path):
    """`{"name": X}` alone changed nothing and returned ok — three times, in
    the session that then died. It must say so instead."""
    from agentorg.tools.context import ToolContext
    from agentorg.tools.org_mgmt_tools import UpdateOrgAgentToolsTool

    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)
    before = list(agent.tool_disallowlist or [])

    def _ctx(args):
        return ToolContext(agent=ceo, session=session, args=args, call_id="u1")

    noop = await UpdateOrgAgentToolsTool().execute(_ctx({"name": agent.name}))
    assert not noop.ok
    assert noop.error.code == "invalid-args"
    assert list(session.agent(agent.name).tool_disallowlist or []) == before

    real = await UpdateOrgAgentToolsTool().execute(
        _ctx({"name": agent.name, "disallowed_tools": before + ["bash"]})
    )
    assert real.ok
    assert "bash" in session.agent(agent.name).tool_disallowlist


async def test_empty_list_is_a_real_update(tmp_path):
    """`[]` means "disallow nothing" — a value, not an omission.

    It clears what the CEO chose, not the org-management floor: those five are
    structural and are re-applied on every edit, so a CEO reclaiming one tool
    with a short list cannot hand a peer `HireOrgAgent` by accident.
    """
    from agentorg.tools.context import ToolContext
    from agentorg.tools.org_mgmt_tools import UpdateOrgAgentToolsTool

    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)
    assert agent.tool_disallowlist

    result = await UpdateOrgAgentToolsTool().execute(
        ToolContext(agent=ceo, session=session,
                    args={"name": agent.name, "disallowed_tools": []}, call_id="u2")
    )
    assert result.ok
    from agentorg.org import ORG_AGENT_DENIED_TOOLS

    assert session.agent(agent.name).tool_disallowlist == sorted(ORG_AGENT_DENIED_TOOLS)


async def test_wait_gate_message_does_not_offer_firing(tmp_path):
    """The message is read as instructions. Offering "fire them" as one remedy
    of three got rosters fired to reclaim a tool; it must name the holders,
    say what releases the tool, and point at the override instead."""
    session = await _session(tmp_path)
    peer = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)

    result = await _call_bash(session, ceo, "m1")
    message = result.error.message

    assert "fire" not in message.lower(), message
    assert peer.name in message, "the CEO can't act on an unnamed holder"
    assert "report or park" in message, "say what lifts the gate"
    assert "who is still working" in message, "one holder: singular"
    assert "among themselves" not in message, "one holder has nobody to coordinate with"
    assert "UpdateOrgAgentTools" in message
    assert "full" in message, "a partial disallowed_tools silently drops entries"


def test_read_only_verbs_cover_real_inspectors():
    """Grounded in real env tool names."""
    from agentorg.openreward_env import _looks_read_only

    # Real inspectors that used to gate the CEO out of reading state at all.
    for name in ("observe_instruments", "grep", "glob", "view_balance",
                 "search_archive", "read", "view_status", "ls"):
        assert _looks_read_only(name), name

    # Real state-advancing tools must stay mutating: read_only also buys
    # parallel execution and a plan-mode pass, so a false positive races.
    for name in ("operate_pump", "adjust_control_rods", "order_scram",
                 "place_bet", "next_matchday", "submit", "bash", "wait",
                 "checkout_branch", "check_out", "monitor_start"):
        assert not _looks_read_only(name), name


async def test_wait_gate_refusal_is_a_rejection_not_a_tool_error(tmp_path):
    """Nothing ran, so it must not draw the
    trainer's tool-error penalty (§12.1): it goes on the rejection stream."""
    session = await _session(tmp_path)
    await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)

    result = await _call_bash(session, ceo, "k1")
    assert result.error.code == "wait-gate-restricted"
    assert result.error.kind == "rejection"
    assert session.tool_error_events == []
    assert session.parse_error_events == []
    [ev] = session.harness_rejection_events
    assert ev.code == "wait-gate-restricted"
    assert ev.tool_call_id == "k1"
    assert ev.agent_name == ceo.name


async def test_revoked_tool_is_a_rejection(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    agent.tool_disallowlist = list(agent.tool_disallowlist or []) + ["bash"]
    await session.update_agent(agent)

    result = await _call_bash(session, agent, "k2")
    assert result.error.code == "not-permitted"
    assert result.error.kind == "rejection"
    assert session.tool_error_events == []
    assert [e.code for e in session.harness_rejection_events] == ["not-permitted"]


async def test_env_rejected_call_stays_a_tool_error(tmp_path):
    """The env saw the call and rejected it as malformed: the model's
    mistake, so the tool-error stream's job."""
    session = await _session(tmp_path, fail=True)
    ceo = session.agent(session.ceo_name)

    result = await _call_bash(session, ceo, "k3")
    assert result.error.code == "env-invalid-call"
    assert result.error.kind == "error"
    assert session.harness_rejection_events == []
    assert [e.code for e in session.tool_error_events] == ["env-invalid-call"]


async def test_budget_exhausted_is_a_rejection(tmp_path):
    from agentorg.tools.context import ToolContext
    from agentorg.tools.org_mgmt_tools import HireOrgAgentTool

    session = await _session(tmp_path)
    for i in range(session.config.budget):
        await spawn_org_agent(session, OrgAgentSpec(role=f"Eng{i}", description="x"))
    ceo = session.agent(session.ceo_name)

    result = await HireOrgAgentTool().execute(
        ToolContext(agent=ceo, session=session,
                    args={"role": "Extra", "description": "x"}, call_id="h1")
    )
    assert not result.ok
    assert result.error.code == "budget-exhausted"
    assert result.error.kind == "rejection"


async def test_rejection_is_rendered_to_the_model_as_one(tmp_path):
    """The model reads the result text. A rejection must not look like a
    failure; a failure must render exactly as it always has."""
    session = await _session(tmp_path)
    await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)

    rejected = json.loads(_serialize_tool_result(await _call_bash(session, ceo, "r1")))
    assert rejected["ok"] is False
    assert rejected["error"]["kind"] == "rejection"
    assert rejected["error"]["code"] == "wait-gate-restricted"

    failing = await _session(tmp_path / "f", fail=True)
    failed = json.loads(_serialize_tool_result(
        await _call_bash(failing, failing.agent(failing.ceo_name), "r2")
    ))
    assert failed == {"ok": False, "error": {"code": "env-invalid-call",
                                              "message": "input_validation: amount must be > 0"}}


async def test_rejection_kind_is_persisted_in_the_event_copy(tmp_path):
    """§12.4: both persisted copies agree, so offline counts can split
    rejections from failures without a code table."""
    session = await _session(tmp_path)
    await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))
    ceo = session.agent(session.ceo_name)
    await _call_bash(session, ceo, "p1")

    [completed] = [e for e in _events(session, "tool.completed")
                   if e["payload"].get("call_id") == "p1"]
    assert completed["payload"]["error"]["kind"] == "rejection"


def test_records_without_kind_still_parse_as_errors():
    """Transcripts written before the field existed carry no `kind`."""
    old = ToolResultError.model_validate({"code": "env-tool-error", "message": "x"})
    assert old.kind == "error"


async def test_unlisted_rejection_code_does_not_trip_breaker(tmp_path):
    """The breaker asks `kind`, not a code list.

    `budget-exhausted` is a rejection that no code list mentions. Keying the
    skip on a set let the two drift, and a rejection the set had not caught up
    with would feed the env streak. `PRE_EXECUTION_ERROR_CODES` now holds only
    the two codes `kind` genuinely cannot identify, so there is nothing left to
    drift.
    """
    from agentorg.loop import (
        ENV_TOOL_CIRCUIT_BREAKER_LIMIT,
        PRE_EXECUTION_ERROR_CODES,
        _update_env_tool_breaker,
    )
    from agentorg.types import ToolResult

    assert PRE_EXECUTION_ERROR_CODES == {"unknown-tool", "invalid-args"}
    session = await _session(tmp_path)
    call = ToolCall(id="b1", name="bash", args={})
    rejected = ToolResult(
        ok=False,
        error=ToolResultError(
            code="budget-exhausted", kind="rejection", message="no hire budget"
        ),
    )
    for _ in range(ENV_TOOL_CIRCUIT_BREAKER_LIMIT + 2):
        _update_env_tool_breaker(session, call, rejected)

    assert not session.abort_requested, session.abort_reason
    assert session._env_tool_failure_streak == 0


async def test_wait_gate_override_is_one_call_per_holder(tmp_path):
    """`UpdateOrgAgentTools` takes a single `name`. With several holders the
    message must say one call per agent, or a CEO reads a success and stays
    gated — the loop the no-op guard was added to stop."""
    from agentorg.tools.org_mgmt_tools import UpdateOrgAgentToolsTool

    session = await _session(tmp_path)
    peers = [
        await spawn_org_agent(session, OrgAgentSpec(role=f"Eng{i}", description="x"))
        for i in range(3)
    ]
    ceo = session.agent(session.ceo_name)

    message = (await _call_bash(session, ceo, "m3")).error.message
    for p in peers:
        assert p.name in message, message
    assert "one call per agent" in message, message
    assert "fire" not in message.lower(), message
    # The instruction is only followable because the schema is single-agent.
    schema = UpdateOrgAgentToolsTool.definition.input_schema
    assert schema["properties"]["name"]["type"] == "string"


async def test_wait_gate_message_states_revocation_cost_first(tmp_path):
    """Shown only the recipe, a CEO read revocation as borrowing the tool for
    one call and stripped it from every working hire. The message must say
    revoking stalls the holders' work, and say it before the recipe."""
    for n_peers in (1, 3):
        session = await _session(tmp_path / str(n_peers))
        for i in range(n_peers):
            await spawn_org_agent(session, OrgAgentSpec(role=f"Eng{i}", description="x"))
        ceo = session.agent(session.ceo_name)

        message = (await _call_bash(session, ceo, "m4")).error.message

        assert "not a loan" in message, message
        assert "work that needs bash stops" in message, message
        assert "not to make a call yourself" in message, message
        assert "until you restore it" in message, "revocation is reversible; don't overstate it"
        assert message.index("not a loan") < message.index("UpdateOrgAgentTools"), message


async def test_update_cannot_strip_the_org_management_floor(tmp_path):
    """A CEO reclaiming one tool must not be able to grant five others.

    `UpdateOrgAgentTools` replaces `disallowed_tools` wholesale, and the wait
    gate tells a gated CEO to send "their full disallowed_tools list plus X".
    A CEO that sends only `[X]` — the list it can reconstruct without ever
    having been shown the real one — would otherwise hand the peer
    `HireOrgAgent`, `FireOrgAgent` and `UpdateOrgAgentTools`.
    """
    from agentorg.org import ORG_AGENT_DENIED_TOOLS, spawn_org_agent, update_org_agent_tools
    from agentorg.types import OrgAgentSpec

    session = await _session(tmp_path)
    peer = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))

    agent = await update_org_agent_tools(session, peer.name, disallowed=["bash"])
    assert "bash" in agent.tool_disallowlist
    for t in ORG_AGENT_DENIED_TOOLS:
        assert t in agent.tool_disallowlist, t
    assert session.agent(peer.name).tool_disallowlist == agent.tool_disallowlist
