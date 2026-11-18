"""Model-facing text fixes found while exporting every prompt for the public release.

Each test pins what the model now reads: no Python repr quotes around error
messages, no host paths, one attachment-marker spelling (the one the prompts
document), and the documented argument names actually working.
"""

from __future__ import annotations

import json

from agentorg.compaction import _render_conversation_for_compaction
from agentorg.harness import Harness
from agentorg.inbox import render_inbox
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.loop import _to_provider_message
from agentorg.org import spawn_org_agent
from agentorg.session import SessionConfig
from agentorg.tools.comms_tools import (
    SearchChannelMessagesTool,
    SetChannelSubscriptionTool,
    _stringify_msg_content,
)
from agentorg.tools.context import ToolContext
from agentorg.tools.org_mgmt_tools import (
    ExitOrgPlanModeTool,
    FireOrgAgentTool,
    UpdateOrgAgentToolsTool,
)
from agentorg.tools.org_tools import MAX_MESSAGE_BYTES, PostToChannelTool
from agentorg.tools.wait_tools import ReportToCEOTool
from agentorg.types import Message, OrgAgentSpec, ToolResult, UserSegment


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


async def _session(tmp_path, *, budget=2, mode="default"):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=budget,
        mode=mode,
        headless=True,
        permission_policy="bypass",
    )
    session = await Harness(inference=_Stub()).start_session(config=config)
    await session.channels.create_channels(
        [{"name": "strategy"}],
        live_agent_names=session.live_agent_names(),
        external_labels=set(),
    )
    return session


def _ctx(session, args, agent=None):
    return ToolContext(
        agent=agent or session.agent(session.ceo_name), session=session, args=args, call_id="c"
    )


async def test_post_to_channel_accepts_a_leading_hash(tmp_path):
    session = await _session(tmp_path)
    r = await PostToChannelTool().execute(_ctx(session, {"channel": "#strategy", "content": "hi"}))
    assert r.ok, r


async def test_unknown_channel_errors_share_one_wording(tmp_path):
    session = await _session(tmp_path)
    results = [
        await PostToChannelTool().execute(_ctx(session, {"channel": "#nope", "content": "hi"})),
        await SearchChannelMessagesTool().execute(_ctx(session, {"channel": "nope", "pattern": "x"})),
        await SetChannelSubscriptionTool().execute(_ctx(session, {"channel": "nope", "level": "full"})),
    ]
    for r in results:
        assert r.error.code == "unknown-channel"
        assert r.error.message == "No channel named 'nope'"


async def test_unknown_agent_error_has_no_repr_quotes(tmp_path):
    session = await _session(tmp_path)
    fire = await FireOrgAgentTool().execute(_ctx(session, {
        "name": "Ghost", "category": "redundant", "delivered": "-", "shortfalls": "-", "score": 5,
    }))
    retool = await UpdateOrgAgentToolsTool().execute(_ctx(session, {"name": "Ghost", "disallowed_tools": ["bash"]}))
    for r in (fire, retool):
        assert r.error.code == "unknown-agent"
        assert r.error.message == "No such agent: Ghost"


async def test_exit_plan_mode_without_plan_hides_host_path(tmp_path):
    session = await _session(tmp_path, mode="org-plan")
    r = await ExitOrgPlanModeTool().execute(_ctx(session, {}))
    assert r.error.code == "plan-missing"
    assert str(tmp_path) not in r.error.message
    assert "WritePlan" in r.error.message


async def test_report_to_ceo_accepts_documented_attachments_arg(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="tester", description="t"))
    await session.inbox.drain(session.ceo_name)
    r = await ReportToCEOTool().execute(_ctx(session, {
        "status": "completed", "summary": "done", "attachments": ["report.md"],
    }, agent=agent))
    assert r.ok, r
    [item] = await session.inbox.drain(session.ceo_name)
    assert [a.archive_path for a in item.attachments] == ["report.md"]
    assert "<attachments: report.md>" in render_inbox([item], recipient=session.ceo_name)


async def test_report_to_ceo_size_error_names_summary(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="tester", description="t"))
    r = await ReportToCEOTool().execute(_ctx(session, {
        "status": "completed", "summary": "x" * (MAX_MESSAGE_BYTES + 1),
    }, agent=agent))
    assert r.error.code == "content-too-large"
    assert r.error.message.startswith("ReportToCEO summary is ")


def test_rich_user_message_uses_documented_attachment_marker():
    msg = Message(role="user", content=[
        UserSegment(type="text", text="See these."),
        UserSegment(type="attachment", archive_path="a.md"),
        UserSegment(type="attachment", archive_path="b.md"),
    ])
    assert _to_provider_message(msg).content == "See these.\n<attachments: a.md, b.md>"


def test_compaction_render_shows_tool_results_as_json():
    msg = Message(role="tool", tool_call_id="t1",
                  tool_result=ToolResult(ok=True, output={"ok": True, "note": None, "name": "é"}))
    rendered = _render_conversation_for_compaction([msg])
    assert json.dumps({"ok": True, "note": None, "name": "é"}, ensure_ascii=False) in rendered
    assert "True" not in rendered


async def test_report_to_ceo_rejects_malformed_attachments(tmp_path):
    session = await _session(tmp_path)
    agent = await spawn_org_agent(session, OrgAgentSpec(role="tester", description="t"))
    for bad in ("report.md", [{"path": "x"}]):
        r = await ReportToCEOTool().execute(_ctx(session, {
            "status": "completed", "summary": "done", "attachments": bad,
        }, agent=agent))
        assert r.error.code == "invalid-args", bad


async def test_post_to_self_error_has_single_hash(tmp_path):
    session = await _session(tmp_path)
    r = await PostToChannelTool().execute(_ctx(session, {"channel": "#self", "content": "hi"}))
    assert r.error.code == "read-only-channel"
    assert "`#self`" in r.error.message and "##" not in r.error.message


def test_trace_view_uses_documented_attachment_marker():
    msg = Message(role="user", content=[
        UserSegment(type="text", text="See these."),
        UserSegment(type="attachment", archive_path="a.md"),
    ])
    assert _stringify_msg_content(msg) == "See these.\n<attachments: a.md>"
