"""Smoke-test the org spawn / report / fire lifecycle via mocked inference."""

from __future__ import annotations

import time
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
from agentorg.storage import iter_jsonl
from agentorg.tools.context import ToolContext
from agentorg.tools.org_tools import WaitForMessagesTool


class ScriptedProvider(InferenceProvider):
    """Deterministic scripted provider.

    Returns a list of `chunks` lists in order, advancing one per `.stream()` call.
    Once exhausted, returns a tool-less assistant message saying "done".
    """

    def __init__(self, script):
        self._script = list(script)

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if not self._script:
            yield ContentDeltaChunk(text="done")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        chunks = self._script.pop(0)
        for c in chunks:
            yield c
        yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_hire_and_fire_with_exit_review(tmp_path):
    """CEO hires an agent, agent reports, CEO fires it with a review."""
    # We need to fill the agent name in turn 4; we'll patch at runtime using a
    # callable script wrapper.
    class ResolvingProvider(InferenceProvider):
        def __init__(self):
            self._step = 0
            self._cached_agent_name = None

        async def stream(self, request):
            self._step += 1
            if self._step == 1:
                yield ToolCallChunk(
                    id="c1",
                    name="HireOrgAgent",
                    args={"role": "Tester", "description": "Verify the system works."},
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if self._step == 2:
                # this is an org agent's first turn — it reports
                # but ScriptedProvider can't distinguish caller; assume agent
                yield ToolCallChunk(
                    id="a1",
                    name="ReportToCEO",
                    args={"status": "completed", "summary": "All good."},
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if self._step == 3:
                # CEO sees the report; fire the agent.
                # We need the actual name — pull from the request's system prompt or args.
                # Simpler: pick any name from the system prompt (which has the roster).
                sys = request.messages[0].content or ""
                name = ""
                for line in sys.splitlines():
                    if " — " in line and line.strip().startswith("- @"):
                        name = line.strip().split(" — ")[0].lstrip("- @")
                        break
                yield ToolCallChunk(
                    id="c3",
                    name="FireOrgAgent",
                    args={
                        "name": name,
                        "category": "mission_complete",
                        "delivered": "Verified end-to-end functioning.",
                        "shortfalls": "—",
                        "score": 8,
                    },
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            # Everyone else: done.
            yield ContentDeltaChunk(text="done")
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
    harness = Harness(inference=ResolvingProvider())
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=30)

    # We should have one org agent, fired with an exit review.
    org_agents = [a for a in session.all_agents() if a.kind == "org"]
    assert len(org_agents) == 1
    agent = org_agents[0]
    assert agent.state == "fired"
    assert agent.exit_review is not None
    assert agent.exit_review.category == "mission_complete"
    assert agent.exit_review.score == 8


@pytest.mark.asyncio
async def test_wait_for_messages_short_circuits_with_no_live_org_agents(tmp_path):
    """Regression: the CEO must not park in WaitForMessages
    when there are no live org agents — nothing can wake the wait. The tool
    returns immediately with `no_live_peers` and a hint to take action."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
    )

    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    assert session.live_org_agents() == []

    tool = WaitForMessagesTool()
    ctx = ToolContext(
        agent=session.agent(session.ceo_name),
        session=session,
        args={},
        call_id="c1",
    )

    t0 = time.monotonic()
    result = await tool.execute(ctx)
    elapsed = time.monotonic() - t0

    assert result.ok is True
    assert isinstance(result.output, dict)
    assert result.output.get("no_live_peers") is True
    assert result.output.get("live_peers") == []
    assert any("No org agents are live" in s for s in result.output.get("suggestions", []))
    # Short-circuit must be essentially instantaneous.
    assert elapsed < 1.0, f"WaitForMessages parked for {elapsed:.2f}s when it should have short-circuited"


@pytest.mark.asyncio
async def test_wait_for_messages_parks_when_org_agents_live_and_active(tmp_path):
    """SPEC §4.5.1: when a live org agent is not yet parked in the wait, the
    CEO's WaitForMessages must park (not short-circuit). We verify parking by
    racing the tool against a short `asyncio.wait_for` and expecting
    `TimeoutError` (the tool is blocked on the wake event, no quiescence)."""
    import asyncio
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
    )

    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    await spawn_org_agent(
        session,
        OrgAgentSpec(role="Tester", description="x"),
    )
    assert len(session.live_org_agents()) == 1
    # spawned agent has no coroutine running here, so it is NOT in the
    # WakeBroker waiting set — the org is not quiescent.
    assert not session.is_org_quiescent()

    tool = WaitForMessagesTool()
    ctx = ToolContext(
        agent=session.agent(session.ceo_name),
        session=session,
        args={},
        call_id="c1",
    )

    # The tool should park indefinitely (no timeout, no quiescence). Race
    # against a 0.3s wait_for and expect TimeoutError as proof of parking.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tool.execute(ctx), timeout=0.3)


@pytest.mark.asyncio
async def test_wait_for_messages_returns_quiescent_marker_when_all_org_parked(tmp_path):
    """SPEC §4.5.1 checkpoint 1: when every live org agent is itself parked
    in the wait, the CEO's WaitForMessages returns the quiescent marker
    immediately instead of parking."""
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
    )

    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    org_agent = await spawn_org_agent(
        session,
        OrgAgentSpec(role="Tester", description="x"),
    )
    # Simulate the org agent having entered its own wait by registering it
    # in the WakeBroker's waiting set directly. (Without an event loop driver
    # to run its coroutine, we have to fake the waiting-set membership.)
    session.wakes._waiting.add(org_agent.name)
    assert session.is_org_quiescent()

    tool = WaitForMessagesTool()
    ctx = ToolContext(
        agent=session.agent(session.ceo_name),
        session=session,
        args={},
        call_id="c1",
    )

    t0 = time.monotonic()
    result = await tool.execute(ctx)
    elapsed = time.monotonic() - t0

    assert result.ok is True
    assert isinstance(result.output, dict)
    assert result.output.get("quiescent") is True
    assert "no_live_peers" not in result.output
    # Activity card must carry only time-free fields.
    peers = result.output.get("live_peers", [])
    assert len(peers) == 1
    card = peers[0]
    assert card["name"] == org_agent.name
    assert card["runtime"] == "idle"
    assert "turn_count" in card
    assert "recent_tool_calls" in card
    # No wall-clock fields.
    for forbidden in (
        "last_assistant_ts",
        "inflight_since_ts",
        "inflight_last_chunk_ts",
        "inflight_output_tokens_approx",
    ):
        assert forbidden not in card, f"activity card must not expose {forbidden}"
    assert "state" not in card
    # Short-circuit must be essentially instantaneous.
    assert elapsed < 1.0, f"WaitForMessages parked {elapsed:.2f}s when it should short-circuit"


@pytest.mark.asyncio
async def test_spawn_injects_archive_attachments_into_mandate(tmp_path):
    """OrgAgentSpec.archive_attachments must surface in the spawned agent's
    first user message as a `<attachments: …>` suffix the worker can recognise."""
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
    )

    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    agent = await spawn_org_agent(
        session,
        OrgAgentSpec(
            role="Verifier",
            description="Verify the derivation handed off by the CEO.",
            archive_attachments=["ceo-derivation.md", "notes.md"],
        ),
    )
    transcript = session.transcript(agent.name)
    user_msgs = [m for m in transcript if m.role == "user" and m.source == "env"]
    assert user_msgs, "expected at least one mandate user message"
    mandate = user_msgs[0].content
    assert isinstance(mandate, str)
    assert "Begin work on your mission now." in mandate
    assert "<attachments: ceo-derivation.md, notes.md>" in mandate, mandate


@pytest.mark.asyncio
async def test_wait_gate_gates_only_tools_a_peer_also_holds(tmp_path):
    """SPEC §4.5.2: a non-readonly tool gates the CEO only if some
    unreported, non-parked peer also holds it. A tool the CEO reserved
    (disallowed on every running peer) must remain callable.
    """
    from agentorg.loop import _dispatch_tools
    from agentorg.org import spawn_org_agent
    from agentorg.tools.context import ok_result
    from agentorg.types import (
        Message,
        OrgAgentSpec,
        ToolCall,
        ToolDefinition,
    )

    class _MutEnvTool:
        def __init__(self, name):
            self.definition = ToolDefinition(
                name=name,
                description=f"mutates state via {name}",
                input_schema={"type": "object", "properties": {}},
                read_only=False,
                scope="env",
            )

        async def execute(self, ctx):
            return ok_result({"called": self.definition.name})

    class _Stub(InferenceProvider):
        async def stream(self, request):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    # Register two env-like state-mutating tools.
    shared = _MutEnvTool("step_shared")
    reserved = _MutEnvTool("step_reserved")
    session.tools.register(shared)
    session.tools.register(reserved)

    # One peer holds `step_shared` but has `step_reserved` disallowed.
    await spawn_org_agent(
        session,
        OrgAgentSpec(
            role="Operator",
            description="x",
            disallowed_tools=["step_reserved"],
        ),
    )
    assert len(session.live_org_agents()) == 1
    assert not session.is_org_quiescent()  # live + unreported

    ceo = session.agent(session.ceo_name)
    shared_call = ToolCall(id="c1", name="step_shared", args={})
    reserved_call = ToolCall(id="c2", name="step_reserved", args={})
    asst = Message(role="assistant", tool_calls=[shared_call, reserved_call])
    await session.append_message(ceo.name, asst)
    await _dispatch_tools(session, ceo, asst, [shared_call, reserved_call])

    by_id = {m.tool_call_id: m for m in session.transcript(ceo.name) if m.role == "tool"}
    # Shared tool: peer holds it → wait-gate kicks in.
    assert not by_id["c1"].tool_result.ok
    assert by_id["c1"].tool_result.error.code == "wait-gate-restricted"
    # Reserved tool: no peer holds it → CEO call goes through.
    assert by_id["c2"].tool_result.ok, by_id["c2"].tool_result.error


@pytest.mark.asyncio
async def test_ceo_tool_restriction_off_lets_ceo_share_tools(tmp_path):
    """`ceo_tool_restriction=False`: a tool an unreported, working peer also
    holds stays visible to the CEO and its call runs."""
    from agentorg.loop import _dispatch_tools, _stream_turn
    from agentorg.org import spawn_org_agent
    from agentorg.tools.context import ok_result
    from agentorg.types import Message, OrgAgentSpec, ToolCall, ToolDefinition

    class _MutEnvTool:
        definition = ToolDefinition(
            name="step_shared",
            description="mutates state",
            input_schema={"type": "object", "properties": {}},
            read_only=False,
            scope="env",
        )

        async def execute(self, ctx):
            return ok_result({"called": True})

    class _Stub(InferenceProvider):
        def __init__(self):
            self.tool_names = []

        async def stream(self, request):
            self.tool_names.append({t.name for t in request.tools})
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    async def _run(restrict: bool):
        provider = _Stub()
        session = await Harness(inference=provider).start_session(config=SessionConfig(
            workspace_path=str(tmp_path / str(restrict)),
            model="mock",
            budget=2,
            mode="default",
            headless=True,
            permission_policy="bypass",
            ceo_tool_restriction=restrict,
        ))
        session.tools.register(_MutEnvTool())
        await spawn_org_agent(session, OrgAgentSpec(role="Operator", description="x"))
        assert not session.is_org_quiescent()  # live + unreported + not parked

        ceo = session.agent(session.ceo_name)
        await _stream_turn(session, ceo)
        call = ToolCall(id="c1", name="step_shared", args={})
        asst = Message(role="assistant", tool_calls=[call])
        await session.append_message(ceo.name, asst)
        await _dispatch_tools(session, ceo, asst, [call])
        result = next(
            m for m in session.transcript(ceo.name) if m.tool_call_id == "c1"
        ).tool_result
        return provider.tool_names[-1], result

    visible, result = await _run(restrict=True)
    assert "step_shared" not in visible
    assert result.error.code == "wait-gate-restricted"

    visible, result = await _run(restrict=False)
    assert "step_shared" in visible
    assert result.ok, result.error


def test_plan_parser_reads_archive_attachments():
    """Both snake_case and camelCase forms parse into archive_attachments."""
    from agentorg.plan import extract_org_block

    md = """## Org design

```agents-org
{
  "agents": [
    {"role": "A", "description": "x", "archive_attachments": ["foo.md"]},
    {"role": "B", "description": "y", "archiveAttachments": ["bar.md", "baz.md"]}
  ]
}
```
"""
    block = extract_org_block(md)
    assert block is not None
    assert block.agents[0].archive_attachments == ["foo.md"]
    assert block.agents[1].archive_attachments == ["bar.md", "baz.md"]


class _CrashOrgProvider(InferenceProvider):
    """CEO hires one agent, then waits; the org agent's model call raises.

    Callers are told apart by their toolset: only the CEO holds HireOrgAgent.
    """

    def __init__(self):
        self.ceo_requests: list[InferenceRequest] = []

    async def stream(self, request):
        if not any(t.name == "HireOrgAgent" for t in request.tools):
            raise RuntimeError("upstream 500")
            yield  # pragma: no cover
        self.ceo_requests.append(request)
        step = len(self.ceo_requests)
        if step == 1:
            yield ToolCallChunk(
                id="c1", name="HireOrgAgent", args={"role": "Tester", "description": "x"}
            )
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
        elif step == 2:
            yield ToolCallChunk(id="c2", name="WaitForMessages", args={})
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
        else:
            yield ContentDeltaChunk(text="done")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


def _crash_config(tmp_path) -> SessionConfig:
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="Build a team.",
    )


async def _run_crash_session(tmp_path, provider):
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_crash_config(tmp_path))
    t0 = time.monotonic()
    await harness.run_to_completion(session, max_wall_s=10)
    elapsed = time.monotonic() - t0
    (org,) = [a for a in session.all_agents() if a.kind == "org"]
    fired_events = [
        e for e in iter_jsonl(session.paths.events_jsonl)
        if e.get("type") == "agent.fired"
    ]
    return session, org, elapsed, fired_events


def _ceo_saw_crash_notice(provider: _CrashOrgProvider, name: str) -> bool:
    return any(
        f"@{name} stopped" in (m.content or "")
        for r in provider.ceo_requests
        for m in r.messages
        if isinstance(m.content, str)
    )


@pytest.mark.asyncio
async def test_org_agent_inference_crash_wakes_parked_ceo(tmp_path):
    """A crashed org agent never parks, so without retirement the CEO's
    WaitForMessages (released only by quiescence or a message) never returns
    and the run idles until max_wall_s."""
    provider = _CrashOrgProvider()
    session, org, elapsed, fired_events = await _run_crash_session(tmp_path, provider)

    assert session.abort_reason is None, f"session aborted: {session.abort_reason}"
    assert elapsed < 5.0
    assert org.state == "fired"
    assert org.fired_source == "harness"
    assert org.exit_review is None
    assert _ceo_saw_crash_notice(provider, org.name)
    assert [e["payload"]["reason"] for e in fired_events] == ["crashed"]


@pytest.mark.asyncio
async def test_org_agent_non_inference_crash_wakes_parked_ceo(tmp_path, monkeypatch):
    """Same as above, but the org agent's task dies with an exception raised
    outside the inference try block (here: tool dispatch)."""
    import agentorg.loop as loop_mod

    real_dispatch = loop_mod._dispatch_tools

    async def _dispatch(session, agent, *args, **kwargs):
        if agent.kind == "org":
            raise OSError("disk full")
        return await real_dispatch(session, agent, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "_dispatch_tools", _dispatch)

    class _OrgCallsTool(_CrashOrgProvider):
        async def stream(self, request):
            if not any(t.name == "HireOrgAgent" for t in request.tools):
                yield ToolCallChunk(id="a1", name="WaitForMessages", args={})
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            async for c in super().stream(request):
                yield c

    provider = _OrgCallsTool()
    session, org, elapsed, fired_events = await _run_crash_session(tmp_path, provider)

    assert session.abort_reason is None, f"session aborted: {session.abort_reason}"
    assert elapsed < 5.0
    assert org.state == "fired"
    assert org.fired_source == "harness"
    assert _ceo_saw_crash_notice(provider, org.name)
    assert any(
        "OSError: disk full" in (m.content or "")
        for r in provider.ceo_requests
        for m in r.messages
        if isinstance(m.content, str)
    )
    assert len(fired_events) == 1


@pytest.mark.asyncio
async def test_ceo_fire_during_org_inference_is_not_overwritten_by_crash(tmp_path):
    """The CEO fires an agent while its model call is in flight, then the call
    fails. The CEO's fire stands: no harness re-fire, no crash notice."""
    import asyncio

    gate = asyncio.Event()
    org_in_flight = asyncio.Event()

    class _FireThenCrash(InferenceProvider):
        def __init__(self):
            self.session = None
            self.ceo_requests: list[InferenceRequest] = []

        async def stream(self, request):
            if not any(t.name == "HireOrgAgent" for t in request.tools):
                org_in_flight.set()
                await gate.wait()
                raise RuntimeError("upstream 500")
            self.ceo_requests.append(request)
            step = len(self.ceo_requests)
            if step == 1:
                yield ToolCallChunk(
                    id="c1", name="HireOrgAgent", args={"role": "Tester", "description": "x"}
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            if step == 2:
                await org_in_flight.wait()
                (org,) = self.session.live_org_agents()
                yield ToolCallChunk(
                    id="c2",
                    name="FireOrgAgent",
                    args={
                        "name": org.name,
                        "category": "mission_complete",
                        "delivered": "x",
                        "shortfalls": "—",
                        "score": 5,
                    },
                )
                yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
                return
            # The supervisor keeps running until the org task (which fails
            # once released) is reaped, so the crash path is exercised.
            gate.set()
            yield ContentDeltaChunk(text="done")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["mock"]

    provider = _FireThenCrash()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=_crash_config(tmp_path))
    provider.session = session
    await harness.run_to_completion(session, max_wall_s=10)

    (org,) = [a for a in session.all_agents() if a.kind == "org"]
    fired_events = [
        e for e in iter_jsonl(session.paths.events_jsonl)
        if e.get("type") == "agent.fired"
    ]
    assert gate.is_set()
    assert org.state == "fired"
    assert org.fired_source == "ceo"
    assert org.exit_review is not None and org.exit_review.score == 5
    assert [e["payload"]["source"] for e in fired_events] == ["ceo"]
    assert not any(
        i.kind == "meta" and "stopped" in i.content
        for i in await session.inbox.drain(session.ceo_name)
    )
