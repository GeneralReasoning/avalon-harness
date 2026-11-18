"""Tests for the WritePlan tool and the org-plan permission gate.

WritePlan was introduced to fix a long-standing bug where, in env-bound
sessions, the model could write the plan via the env's lowercase `write`
tool (sandbox-side) — landing the file inside the sandbox where
ExitOrgPlanMode (host-side) could never find it. WritePlan owns the
plan path internally so the model cannot fumble it.
"""

from __future__ import annotations

from pathlib import Path
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
from agentorg.tools.context import ToolContext
from agentorg.tools.org_tools import WritePlanTool


class _ScriptedProvider(InferenceProvider):
    """Yields scripted tool calls then a tool-less 'done'."""

    def __init__(self, scripts: list[list[ToolCallChunk]]):
        self._scripts = list(scripts)

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if not self._scripts:
            yield ContentDeltaChunk(text="done")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        for chunk in self._scripts.pop(0):
            yield chunk
        yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_write_plan_writes_to_session_plan_path(tmp_path):
    """WritePlan should land at the harness's `session.plan_path`, not anywhere
    the model passes — there's no path arg.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)

    tool = WritePlanTool()
    ctx = ToolContext(
        session=session,
        agent=session.agent(session.ceo_name),
        args={"content": "## Problem shape\n\n```agents-org\n{\"agents\":[]}\n```\n"},
        call_id="call_test",
    )
    result = await tool.execute(ctx)
    assert result.ok, f"WritePlan failed: {result.error}"
    assert Path(session.plan_path).exists()
    assert "agents-org" in Path(session.plan_path).read_text()


@pytest.mark.asyncio
async def test_write_plan_rejects_non_string_content(tmp_path):
    """`content` is required and must be a string — guard against the model
    accidentally calling WritePlan() with no body, which would otherwise
    create an empty plan file ExitOrgPlanMode could then mis-parse.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)

    tool = WritePlanTool()
    ctx = ToolContext(
        session=session,
        agent=session.agent(session.ceo_name),
        args={},  # missing content
        call_id="call_test",
    )
    result = await tool.execute(ctx)
    assert not result.ok
    assert result.error.code == "invalid-args"


@pytest.mark.asyncio
async def test_write_plan_rejects_unterminated_agents_org_fence(tmp_path):
    """If the model emits ```agents-org ... { ... } without the closing ```,
    WritePlan should reject in the same turn rather than silently storing a
    plan that ExitOrgPlanMode will then quietly spawn nothing from.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)

    malformed = (
        "## Org design\n"
        "```agents-org\n"
        '{"agents": [{"role": "Specialist", "description": "x"}]}\n'
        # NB: closing ``` deliberately missing
    )
    tool = WritePlanTool()
    ctx = ToolContext(
        session=session,
        agent=session.agent(session.ceo_name),
        args={"content": malformed},
        call_id="call_test",
    )
    result = await tool.execute(ctx)
    assert not result.ok
    assert result.error.code == "plan-parse-error"
    assert "unterminated" in result.error.message
    # And the plan file must not have been written.
    assert not Path(session.plan_path).exists()


@pytest.mark.asyncio
async def test_write_plan_rejects_missing_block_in_multi_agent_mode(tmp_path):
    """In multi-agent org-plan mode (budget > 0) the plan MUST include an
    agents-org block (ORG_PLAN_MULTI contract). A plan that omits the block
    entirely would silently spawn nothing at ExitOrgPlanMode — reject it
    at write time instead.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)

    no_block = "## Problem shape\n- Boundaries: test\n\n(no fence at all)\n"
    result = await WritePlanTool().execute(
        ToolContext(
            session=session,
            agent=session.agent(session.ceo_name),
            args={"content": no_block},
            call_id="call_test",
        )
    )
    assert not result.ok
    assert result.error.code == "plan-missing-org-block"
    assert "budget=2" in result.error.message
    assert not Path(session.plan_path).exists()


@pytest.mark.asyncio
async def test_write_plan_accepts_no_block_in_solo_mode(tmp_path):
    """Solo mode (budget == 0, ORG_PLAN_SOLO contract) explicitly tells the
    CEO NOT to include an agents-org block. WritePlan must accept a
    block-less plan in that mode.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)

    no_block = "## Problem shape\n- Boundaries: solo work\n\n(no fence; solo)\n"
    result = await WritePlanTool().execute(
        ToolContext(
            session=session,
            agent=session.agent(session.ceo_name),
            args={"content": no_block},
            call_id="call_test",
        )
    )
    assert result.ok, f"WritePlan failed: {result.error}"
    assert Path(session.plan_path).exists()


@pytest.mark.asyncio
async def test_write_plan_then_exit_org_plan_mode(tmp_path):
    """Invoke WritePlan then ExitOrgPlanMode directly via ToolContext —
    asserts the write lands where Exit reads from, so the previously-bug
    of write/Write divergence cannot occur.
    """
    from agentorg.tools.org_tools import ExitOrgPlanModeTool

    plan_markdown = (
        "## Problem shape\n"
        "- Boundaries: test\n"
        "## Org design\n"
        "```agents-org\n"
        '{"agents": [{"role": "Specialist", "description": "Do the work."}]}\n'
        "```\n"
    )

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)

    # 1) WritePlan
    wp_result = await WritePlanTool().execute(
        ToolContext(session=session, agent=ceo, args={"content": plan_markdown}, call_id="c1")
    )
    assert wp_result.ok
    assert Path(session.plan_path).exists()

    # 2) ExitOrgPlanMode reads that same path and spawns the roster.
    exit_result = await ExitOrgPlanModeTool().execute(
        ToolContext(session=session, agent=ceo, args={}, call_id="c2")
    )
    assert exit_result.ok, f"ExitOrgPlanMode failed: {exit_result.error}"
    org_agents = [a for a in session.all_agents() if a.kind == "org"]
    assert len(org_agents) == 1, f"expected 1 org agent, got {[a.name for a in org_agents]}"
    assert org_agents[0].role == "Specialist"


@pytest.mark.asyncio
async def test_exit_org_plan_mode_pre_creates_plan_channels(tmp_path):
    """When the agents-org block declares `channels`, ExitOrgPlanMode pre-creates
    them BEFORE spawning specialists. This prevents the CEO/specialists from
    racing on `CreateChannels` (otherwise multiple agents can try to create the
    same channel and the all-or-nothing semantics kill the batch).
    """
    from agentorg.tools.org_tools import ExitOrgPlanModeTool

    plan_markdown = (
        "## Problem shape\n"
        "- Boundaries: test\n"
        "## Org design\n"
        "```agents-org\n"
        "{"
        '"agents": [{"role": "Specialist", "description": "Do the work."}],'
        '"channels": ['
        '{"name": "strategy", "default_subscription": "light"},'
        '{"name": "verification", "default_subscription": "full"}'
        ']'
        "}\n"
        "```\n"
    )

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="design",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)

    await WritePlanTool().execute(
        ToolContext(session=session, agent=ceo, args={"content": plan_markdown}, call_id="c1")
    )
    exit_result = await ExitOrgPlanModeTool().execute(
        ToolContext(session=session, agent=ceo, args={}, call_id="c2")
    )
    assert exit_result.ok, f"ExitOrgPlanMode failed: {exit_result.error}"
    assert sorted(exit_result.output["channels_created"]) == ["strategy", "verification"]

    await session.channels.load()
    names = {ch.name for ch in session.channels.list_regular()}
    assert {"announcements", "random", "strategy", "verification"}.issubset(names)
    assert session.channels.resolve("verification").default_subscription == "full"


@pytest.mark.asyncio
async def test_create_channels_idempotent_on_existing_name(tmp_path):
    """Re-creating an existing channel must not error — the bug class this
    fixes is the CEO and freshly spawned specialists racing on the same
    channel name. Existing names come back in `existing`, not as a fatal
    `invalid-args`.
    """
    from agentorg.tools.comms_tools import CreateChannelsTool

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)

    # First create — both new.
    r1 = await CreateChannelsTool().execute(
        ToolContext(
            session=session,
            agent=ceo,
            args={"channels": [{"name": "alpha"}, {"name": "beta"}]},
            call_id="c1",
        )
    )
    assert r1.ok
    assert sorted(r1.output["created"]) == ["alpha", "beta"]
    assert r1.output["existing"] == []

    # Second create — overlapping batch with `alpha` already present + one new.
    r2 = await CreateChannelsTool().execute(
        ToolContext(
            session=session,
            agent=ceo,
            args={"channels": [{"name": "alpha"}, {"name": "gamma"}]},
            call_id="c2",
        )
    )
    assert r2.ok, f"expected idempotent success, got {r2.error}"
    assert r2.output["created"] == ["gamma"]
    assert r2.output["existing"] == ["alpha"]

    # In-batch duplicates remain a hard error.
    r3 = await CreateChannelsTool().execute(
        ToolContext(
            session=session,
            agent=ceo,
            args={"channels": [{"name": "delta"}, {"name": "delta"}]},
            call_id="c3",
        )
    )
    assert not r3.ok
    assert r3.error.code == "invalid-args"


@pytest.mark.asyncio
async def test_plan_mode_gate_allows_write_plan_blocks_write_to_archive(tmp_path):
    """In non-permissive plan mode (no env, or budget=0), the gate should
    allow WritePlan/EnterOrgPlanMode/ExitOrgPlanMode and reject the
    state-mutating WriteToArchive.

    Drives `_dispatch_tools` directly so we don't need a full loop driver
    here — we're testing the gate, not the loop.
    """
    from agentorg.loop import _dispatch_tools
    from agentorg.types import Message, ToolCall

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,  # budget=0 → plan_mode_permissive=False
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="plan solo",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)

    # Pretend the CEO emitted these tool calls in plan mode.
    plan_call = ToolCall(id="c1", name="WritePlan", args={"content": "## Problem shape\n"})
    archive_call = ToolCall(
        id="c2", name="WriteToArchive", args={"path": "scratch.md", "content": "blocked"}
    )
    asst = Message(role="assistant", tool_calls=[plan_call, archive_call])
    await session.append_message(ceo.name, asst)

    await _dispatch_tools(session, ceo, asst, [plan_call, archive_call])

    transcript = session.transcript(ceo.name)
    by_id = {m.tool_call_id: m for m in transcript if m.role == "tool"}
    # WritePlan should have been allowed (succeeded).
    assert by_id["c1"].tool_result.ok, "WritePlan must be allowed in plan mode"
    # WriteToArchive should have been blocked.
    assert not by_id["c2"].tool_result.ok
    assert by_id["c2"].tool_result.error.code == "plan-mode-restricted"


@pytest.mark.asyncio
async def test_plan_mode_permissive_for_interactive_envless(tmp_path):
    """An ATTENDED envless session (no env, end_on_idle=False) with budget>0 is
    permissive in org-plan mode: the plan-mode gate is lifted so the CEO may act
    directly (here, WriteToArchive) rather than being force-locked into
    WritePlan — the env-bound permissive path, extended to interactive envless
    sessions. The
    contrast case (end_on_idle=True: headless, no human, no env) stays strict
    and blocks the same call, proving end_on_idle is the discriminator.

    Drives `_dispatch_tools` directly (no loop) — we're testing the gate.
    """
    from agentorg.loop import _dispatch_tools
    from agentorg.types import Message, ToolCall

    async def _archive_result(end_on_idle: bool, sub: str):
        ws = tmp_path / sub
        ws.mkdir()
        config = SessionConfig(
            workspace_path=str(ws),
            model="scripted",
            budget=2,
            mode="org-plan",
            end_on_idle=end_on_idle,
            headless=True,
            permission_policy="bypass",
            initial_user_message="go",
        )
        harness = Harness(inference=_ScriptedProvider([]))
        session = await harness.start_session(config=config)
        ceo = session.agent(session.ceo_name)
        call = ToolCall(id="c1", name="WriteToArchive", args={"path": "s.md", "content": "x"})
        asst = Message(role="assistant", tool_calls=[call])
        await session.append_message(ceo.name, asst)
        await _dispatch_tools(session, ceo, asst, [call])
        return next(
            m.tool_result
            for m in session.transcript(ceo.name)
            if m.role == "tool" and m.tool_call_id == "c1"
        )

    interactive = await _archive_result(end_on_idle=False, sub="interactive")
    headless = await _archive_result(end_on_idle=True, sub="headless")

    # Headless-envless (no human, no env) stays strict: the gate blocks it.
    assert not headless.ok
    assert headless.error.code == "plan-mode-restricted"
    # Interactive-envless is permissive: the SAME call is not gate-blocked.
    assert interactive.ok or interactive.error.code != "plan-mode-restricted", (
        f"interactive envless org-plan must be permissive; got {interactive.error}"
    )


@pytest.mark.asyncio
async def test_case_insensitive_tool_resolution(tmp_path):
    """A miscased tool name (`writeplan` vs `WritePlan`) should still resolve.

    This is the second half of the same bug: even with WritePlan owning the
    plan path, the model still reflexively types `Bash` instead of `bash` (or
    vice versa) due to training-data priors. The harness should rescue these
    via case-insensitive lookup instead of returning unknown-tool.
    """
    from agentorg.loop import _dispatch_tools
    from agentorg.types import Message, ToolCall

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        mode="org-plan",
        headless=True,
        permission_policy="bypass",
        initial_user_message="case test",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)

    # Sanity-check the resolver directly.
    assert session.tools.resolve("WritePlan")[0] == "WritePlan"
    assert session.tools.resolve("writeplan")[0] == "WritePlan"
    assert session.tools.resolve("WRITEPLAN")[0] == "WritePlan"
    assert session.tools.resolve("DefinitelyNotARealTool") is None

    # End-to-end: dispatch a miscased call and assert it ran on the canonical tool.
    miscased = ToolCall(
        id="c1",
        name="writeplan",
        args={"content": "## Problem shape\n\n```agents-org\n{\"agents\":[]}\n```\n"},
    )
    asst = Message(role="assistant", tool_calls=[miscased])
    await session.append_message(ceo.name, asst)
    await _dispatch_tools(session, ceo, asst, [miscased])

    transcript = session.transcript(ceo.name)
    tool_msg = next(m for m in transcript if m.role == "tool" and m.tool_call_id == "c1")
    assert tool_msg.tool_result.ok, f"miscased call should be rescued, got {tool_msg.tool_result.error}"
    # The dispatch path rewrote the canonical name onto the call.
    assistant_msg = next(m for m in transcript if m.role == "assistant")
    assert assistant_msg.tool_calls[0].name == "WritePlan", (
        "dispatch should rewrite call.name to canonical for downstream consumers"
    )
    assert Path(session.plan_path).exists()


@pytest.mark.asyncio
async def test_case_insensitive_disallowlist(tmp_path):
    """A CEO-typed mis-cased entry in disallowed_tools still gates the tool."""
    from agentorg.loop import _dispatch_tools
    from agentorg.types import Message, ToolCall

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=2,
        headless=True,
        permission_policy="bypass",
        initial_user_message="case test",
    )
    harness = Harness(inference=_ScriptedProvider([]))
    session = await harness.start_session(config=config)
    ceo = session.agent(session.ceo_name)
    # Simulate the CEO typing "writeplan" (lowercase) in disallowed_tools.
    ceo.tool_disallowlist = ["writeplan"]

    call = ToolCall(id="c1", name="WritePlan", args={"content": "## blocked"})
    asst = Message(role="assistant", tool_calls=[call])
    await session.append_message(ceo.name, asst)
    await _dispatch_tools(session, ceo, asst, [call])

    tool_msg = next(m for m in session.transcript(ceo.name) if m.role == "tool")
    assert not tool_msg.tool_result.ok
    assert tool_msg.tool_result.error.code == "not-permitted"


@pytest.mark.asyncio
async def test_plan_path_is_per_session_in_a_shared_workspace(tmp_path):
    """Two sessions in one workspace must not share a plan file. An evaluation run may
    use one workspace for every trial; with the plan at the workspace root, one CEO's
    WritePlan would overwrite another's and its ExitOrgPlanMode spawn the wrong roster.
    """
    def cfg():
        return SessionConfig(
            workspace_path=str(tmp_path), model="scripted", budget=2, mode="org-plan",
            headless=True, permission_policy="bypass", initial_user_message="design",
        )
    a = await Harness(inference=_ScriptedProvider([])).start_session(config=cfg())
    b = await Harness(inference=_ScriptedProvider([])).start_session(config=cfg())
    assert a.plan_path != b.plan_path
    assert Path(a.plan_path).is_relative_to(a.paths.root)
    assert Path(b.plan_path).is_relative_to(b.paths.root)

    plan_a = "## A\n\n```agents-org\n{\"agents\":[]}\n```\n"
    plan_b = "## B\n\n```agents-org\n{\"agents\":[]}\n```\n"
    for session, content in ((a, plan_a), (b, plan_b)):
        ctx = ToolContext(session=session, agent=session.agent(session.ceo_name),
                          args={"content": content}, call_id="call_test")
        assert (await WritePlanTool().execute(ctx)).ok
    assert Path(a.plan_path).read_text() == plan_a
    assert Path(b.plan_path).read_text() == plan_b
