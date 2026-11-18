"""Code fixes from the paper/SPEC audit: the budget default the spec documents,
WritePlan hidden from org agents (it always rejects them), and all-channel
search reporting each of the caller's own trace hits once."""

from __future__ import annotations

from agentorg.harness import Harness
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.loop import visible_tool_defs
from agentorg.org import spawn_org_agent
from agentorg.session import SessionConfig
from agentorg.tools.comms_tools import SearchChannelMessagesTool
from agentorg.tools.context import ToolContext
from agentorg.types import Message, OrgAgentSpec


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


async def _session(tmp_path, *, budget=2):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=budget,
        headless=True,
        permission_policy="bypass",
    )
    return await Harness(inference=_Stub()).start_session(config=config)


def test_session_budget_defaults_to_two(tmp_path):
    assert SessionConfig(workspace_path=str(tmp_path), model="stub").budget == 2


async def test_org_agent_does_not_see_write_plan(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="tester", description="t"))
    org_tools = {d.name for d in visible_tool_defs(session, agent)}
    ceo_tools = {d.name for d in visible_tool_defs(session, session.agent(session.ceo_name))}
    assert "WritePlan" not in org_tools
    assert "WritePlan" in ceo_tools


async def test_all_channel_search_reports_own_trace_hits_once(tmp_path):
    session = await _session(tmp_path)
    ceo = session.agent(session.ceo_name)
    await session.append_message(ceo.name, Message(role="assistant", content="needle-7f3"))
    r = await SearchChannelMessagesTool().execute(ToolContext(
        agent=ceo, session=session, args={"pattern": "needle-7f3"}, call_id="c",
    ))
    assert r.ok, r
    hits = r.output["hits"]
    own = [h for h in hits if h["channel"] in (ceo.name, "self")]
    assert len(own) == 1, hits


async def test_env_bound_claude_code_toolset_gets_no_host_tools(tmp_path):
    # Host-side Bash/Read/... would run outside the env's sandbox and sit
    # beside the env's own lowercase bash/read/... under case-distinct names.
    from agentorg.types import EnvironmentBinding

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=2,
        headless=True,
        permission_policy="bypass",
        environment=EnvironmentBinding(kind="openreward", task_id="x/y", split="train"),
        toolset_name="claude-code",
    )
    session = await Harness(inference=_Stub()).start_session(config=config)
    names = set(session.tools.names())
    assert not names & {"Bash", "Read", "Write", "Edit", "Glob", "Grep", "TaskRead", "TaskKill", "WebFetch"}
