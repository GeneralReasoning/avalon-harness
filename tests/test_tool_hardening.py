"""Hardening regressions for tool changes — input caps, role uniqueness,
AskUserQuestion non-headless, ReadTool large-file streaming, BashTool zombie reap.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agentorg.harness import Harness, org_level_tools
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.tools.context import MAX_TOOL_OUTPUT_BYTES, ToolContext
from agentorg.tools.fs_tools import MAX_READ_OUTPUT_BYTES, ReadTool
from agentorg.tools.org_tools import (
    AskUserQuestionTool,
    MAX_MESSAGE_BYTES,
    PostToChannelTool,
    ReportToCEOTool,
    SendMessageTool,
    ViewArchiveFileTool,
    ViewChannelMessagesTool,
)
from agentorg.types import Agent


class _Stub(InferenceProvider):
    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


def _ceo_ctx(session, args, *, call_id="c1") -> ToolContext:
    return ToolContext(
        agent=session.agent(session.ceo_name),
        session=session,
        args=args,
        call_id=call_id,
    )


@pytest.mark.asyncio
async def test_post_to_channel_rejects_oversized_body(tmp_path):
    """PostToChannel bodies exceeding MAX_MESSAGE_BYTES → content-too-large.
    Source-side cap so a single post can't pin the next-turn context for every
    subscriber; if the agent needs to share more it should WriteToArchive."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    await session.channels.create_channels(
        [{"name": "general"}], live_agent_names=session.live_agent_names(),
        external_labels=set(),
    )

    tool = PostToChannelTool()
    big = "x" * (MAX_MESSAGE_BYTES + 1)
    r = await tool.execute(_ceo_ctx(session, {"channel": "general", "content": big}))
    assert r.ok is False
    assert r.error.code == "content-too-large"


@pytest.mark.asyncio
async def test_send_message_rejects_oversized_body(tmp_path):
    """Same 16 KiB cap on SendMessage. Previously uncapped — a 10 MB DM would
    inflate the recipient's WaitForMessages output unbounded."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        headless=True,
        permission_policy="bypass",
        external_labels=["operator"],
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    tool = SendMessageTool()
    big = "x" * (MAX_MESSAGE_BYTES + 1)
    r = await tool.execute(_ceo_ctx(session, {"to": "operator", "content": big}))
    assert r.ok is False
    assert r.error.code == "content-too-large"


@pytest.mark.asyncio
async def test_report_to_ceo_rejects_oversized_summary(tmp_path):
    """Same cap on ReportToCEO. Long reports should archive the body and pass
    the path in `artifacts`, not inline it into the CEO's next-turn inbox."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=1,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    # ReportToCEO is org-only; spawn one via the same path the CEO would use.
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec
    agent = await spawn_org_agent(
        session,
        OrgAgentSpec(role="tester", description="t"),
    )
    big = "x" * (MAX_MESSAGE_BYTES + 1)
    ctx = ToolContext(
        agent=agent, session=session,
        args={"status": "completed", "summary": big}, call_id="r1",
    )
    r = await ReportToCEOTool().execute(ctx)
    assert r.ok is False
    assert r.error.code == "content-too-large"


@pytest.mark.asyncio
async def test_ask_user_question_non_headless_returns_error(tmp_path):
    """Non-headless mode → headless-only error (was: silent ok with answer=None)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        headless=False,  # ← interactive
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    tool = AskUserQuestionTool()
    r = await tool.execute(_ceo_ctx(session, {"question": "?", "options": ["a", "b"]}))
    assert r.ok is False
    assert r.error.code == "headless-only"


@pytest.mark.asyncio
async def test_ask_user_question_headless_auto_picks_first(tmp_path):
    """Headless mode keeps existing auto-pick behavior."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    tool = AskUserQuestionTool()
    r = await tool.execute(_ceo_ctx(session, {"question": "?", "options": ["yes", "no"]}))
    assert r.ok is True
    assert r.output["answer"] == "yes"
    assert r.output["auto_chosen"] is True


@pytest.mark.parametrize("solo", [True, False])
def test_ask_user_question_not_registered(solo):
    """No session gets AskUserQuestion: headless answered it with the agent's own first option."""
    names = {t.definition.name for t in org_level_tools(solo=solo)}
    assert "AskUserQuestion" not in names


@pytest.mark.asyncio
async def test_read_tool_streams_with_islice(tmp_path):
    """ReadTool must NOT readlines() the whole file; offset+limit window is bounded."""
    # Synthesize a 100k-line file.
    big = tmp_path / "huge.txt"
    big.write_text("\n".join(f"line-{i:06d}" for i in range(100_000)) + "\n")

    tool = ReadTool()
    agent = Agent(
        name="CEO", session_id="s", kind="ceo", role="CEO",
        work_dir=str(tmp_path), system_prompt="",
    )
    # Read a small window deep into the file.
    ctx = ToolContext(
        agent=agent,
        session=None,  # not touched by ReadTool
        args={"file_path": "huge.txt", "offset": 50_000, "limit": 5},
        call_id="r1",
    )
    r = await tool.execute(ctx)
    assert r.ok is True
    lines = r.output["lines"]
    # 5 rendered lines.
    assert len(lines.splitlines()) == 5
    # First line is at offset 50_001 (1-indexed).
    assert "50001\tline-050000" in lines
    assert r.output["viewed_lines"] == 5
    assert r.output["offset"] == 50_000


@pytest.mark.asyncio
async def test_read_tool_rejects_binary_files(tmp_path):
    """A NUL byte in the head MUST short-circuit to a `binary-file` error so the
    agent doesn't ingest megabytes of replacement-char garbage that pin context
    above the model's limit. Repro: sessions died after `Read /usr/bin/python3`."""
    binary = tmp_path / "fake.bin"
    # ELF-ish header: enough NUL bytes that any random open() would hit one.
    binary.write_bytes(b"\x7fELF" + b"\x00" * 1024 + b"more bytes")

    agent = Agent(
        name="CEO", session_id="s", kind="ceo", role="CEO",
        work_dir=str(tmp_path), system_prompt="",
    )
    ctx = ToolContext(
        agent=agent, session=None,
        args={"file_path": "fake.bin"}, call_id="r-bin",
    )
    r = await ReadTool().execute(ctx)
    assert r.ok is False
    assert r.error is not None
    assert r.error.code == "binary-file"


@pytest.mark.asyncio
async def test_read_tool_caps_output_bytes(tmp_path):
    """Output is hard-capped at MAX_READ_OUTPUT_BYTES; the result must carry
    `truncated=True` and a usable `next_offset` so the agent can resume paging.
    Without this cap a 2000-line `limit` on a file with long lines (or one
    binary-ish read that slipped past detection) could still push hundreds of
    KB into the transcript."""
    fat = tmp_path / "fat.txt"
    # 500 lines × 200 chars = 100 KB of content → must hit the 30 KB cap.
    fat.write_text("\n".join("x" * 200 for _ in range(500)) + "\n")

    agent = Agent(
        name="CEO", session_id="s", kind="ceo", role="CEO",
        work_dir=str(tmp_path), system_prompt="",
    )
    ctx = ToolContext(
        agent=agent, session=None,
        args={"file_path": "fat.txt", "limit": 500}, call_id="r-fat",
    )
    r = await ReadTool().execute(ctx)
    assert r.ok is True
    assert r.output["truncated"] is True
    assert r.output["next_offset"] > 0
    assert len(r.output["lines"].encode("utf-8")) <= MAX_READ_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_read_tool_truncates_overly_long_lines(tmp_path):
    """A single 1 MB line shouldn't blow the cap on one line — truncate it
    in-place and signal via `line_truncations`."""
    blob = tmp_path / "minified.js"
    blob.write_text("var x = " + ("a" * 1_000_000) + ";\n")

    agent = Agent(
        name="CEO", session_id="s", kind="ceo", role="CEO",
        work_dir=str(tmp_path), system_prompt="",
    )
    ctx = ToolContext(
        agent=agent, session=None,
        args={"file_path": "minified.js"}, call_id="r-min",
    )
    r = await ReadTool().execute(ctx)
    assert r.ok is True
    assert r.output["line_truncations"] == 1
    assert "[line truncated]" in r.output["lines"]
    assert len(r.output["lines"].encode("utf-8")) <= MAX_READ_OUTPUT_BYTES


async def _session_with_channel(tmp_path, *, budget: int = 0):
    """Spin up a session with a single channel `general` for view/search tests."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=budget,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    await session.channels.create_channels(
        [{"name": "general"}], live_agent_names=session.live_agent_names(),
        external_labels=set(),
    )
    return session


@pytest.mark.asyncio
async def test_view_channel_messages_caps_output_bytes(tmp_path):
    """ViewChannelMessages MUST stop adding messages once the rendered window
    exceeds MAX_TOOL_OUTPUT_BYTES. Without this, the 200-msg `limit` ceiling
    × 16 KiB MAX_MESSAGE_BYTES could push 3 MB per call."""
    session = await _session_with_channel(tmp_path)
    # 30 posts of ~4 KB each = ~120 KB — should be trimmed to ~32 KB worth.
    body = "x" * 4_000
    for _ in range(30):
        r = await PostToChannelTool().execute(
            _ceo_ctx(session, {"channel": "general", "content": body})
        )
        assert r.ok is True

    r = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "limit": 200})
    )
    assert r.ok is True
    # Output trimmed: count cap was 200 but byte cap kicks in earlier.
    assert r.output["older_not_shown"] > 0
    rendered_bytes = sum(
        len(str(m).encode("utf-8")) for m in r.output["messages"]
    )
    # Allow a small overshoot from the "always keep at least one entry" rule
    # plus per-entry framing overhead.
    assert rendered_bytes <= MAX_TOOL_OUTPUT_BYTES + 8_000


@pytest.mark.asyncio
async def test_view_archive_file_pages_when_oversize(tmp_path):
    """ViewArchiveFile windows on byte offset when the file exceeds the cap.
    Archive max is 256 KiB; without paging the model gets the whole thing."""
    session = await _session_with_channel(tmp_path)
    big = "y" * (MAX_TOOL_OUTPUT_BYTES * 2 + 100)
    await session.archive.write("notes/big.md", big)

    r = await ViewArchiveFileTool().execute(
        _ceo_ctx(session, {"path": "notes/big.md"})
    )
    assert r.ok is True
    assert r.output["truncated"] is True
    assert r.output["next_offset"] == MAX_TOOL_OUTPUT_BYTES
    assert len(r.output["content"].encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES

    # Second page picks up where the first stopped.
    r2 = await ViewArchiveFileTool().execute(
        _ceo_ctx(session, {"path": "notes/big.md", "offset": MAX_TOOL_OUTPUT_BYTES})
    )
    assert r2.ok is True
    assert r2.output["offset"] == MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_openreward_env_tool_caps_oversize_output(tmp_path):
    """Env-side tools (`scope="env"`) bypass the per-tool cap that local
    agentorg tools apply, so the adapter MUST truncate at the boundary.
    Without this, a single `bash`/`view`/`read` env call can return
    megabytes (real cases: a 1 MB nested `tool_result` mirrored into a
    virtual channel, a 5.6 MB env `read`)."""
    import json as _json
    from openreward.api.environments.types import TextBlock, ToolOutput

    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.types import ToolDefinition

    big_text = "x" * (MAX_TOOL_OUTPUT_BYTES * 2 + 100)

    class _StubEnvSession:
        async def call_tool(self, name, args):
            return ToolOutput(
                blocks=[TextBlock(text=big_text)],
                metadata={"path": "/big/file"},
                reward=None,
                finished=False,
            )

    class _StubAdapter:
        session = _StubEnvSession()

    defn = ToolDefinition(
        name="read", description="env read",
        input_schema={"type": "object", "properties": {}},
        scope="env",
    )
    tool = _OpenRewardTool(defn, _StubAdapter())  # type: ignore[arg-type]
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    ctx = _ceo_ctx(session, {})

    r = await tool.execute(ctx)
    assert r.ok is True
    assert r.output["truncated"] is True
    assert r.output["dropped_bytes"] > 0
    # Hard guarantee: serialized payload fits the cap.
    serialized = _json.dumps(r.output, ensure_ascii=False).encode("utf-8")
    assert len(serialized) <= MAX_TOOL_OUTPUT_BYTES
    # Preserved fields still surface so the dispatch loop can read them.
    assert r.output["reward"] is None
    assert r.output["finished"] is False


@pytest.mark.asyncio
async def test_openreward_env_tool_maps_session_death_to_abort_code(tmp_path):
    """A `call_tool` exception that signals an unrecoverable env-session death
    (SDK retry-exhaustion) maps to the generic `ENV_SESSION_DEAD_CODE` so the
    loop aborts; any other exception is `env-failure` (recoverable)."""
    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.env_adapter import ENV_SESSION_DEAD_CODE
    from agentorg.types import ToolDefinition

    class MaxRetriesError(Exception):
        """Mimics the OpenReward SDK exception name (which is what the adapter
        renders into the error message and classifies on)."""

    class _Adapter:
        def __init__(self, exc):
            class _Sess:
                async def call_tool(self, name, args):
                    raise exc
            self.session = _Sess()

    defn = ToolDefinition(
        name="place_bet", description="bet",
        input_schema={"type": "object", "properties": {}}, scope="env",
    )
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    ctx = _ceo_ctx(session, {})

    dead = MaxRetriesError("(environment session sid=abc) Exceeded 5 retries for /call")
    r_dead = await _OpenRewardTool(defn, _Adapter(dead)).execute(ctx)  # type: ignore[arg-type]
    assert r_dead.ok is False
    assert r_dead.error.code == ENV_SESSION_DEAD_CODE

    r_other = await _OpenRewardTool(defn, _Adapter(ValueError("transient hiccup"))).execute(ctx)  # type: ignore[arg-type]
    assert r_other.ok is False
    assert r_other.error.code == "env-failure"


@pytest.mark.asyncio
async def test_openreward_env_tool_preserves_small_outputs(tmp_path):
    """Small env outputs must round-trip unchanged — no spurious `truncated`
    flag, no extra fields. Guards against regressions where the cap helper
    adds noise to every result."""
    from openreward.api.environments.types import TextBlock, ToolOutput

    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.types import ToolDefinition

    class _StubEnvSession:
        async def call_tool(self, name, args):
            return ToolOutput(
                blocks=[TextBlock(text="hello")],
                metadata={"k": "v"},
                reward=0.5,
                finished=False,
            )

    class _StubAdapter:
        session = _StubEnvSession()

    defn = ToolDefinition(
        name="bash", description="env bash",
        input_schema={"type": "object", "properties": {}},
        scope="env",
    )
    tool = _OpenRewardTool(defn, _StubAdapter())  # type: ignore[arg-type]
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)

    r = await tool.execute(_ceo_ctx(session, {}))
    assert r.ok is True
    assert "truncated" not in r.output
    assert "dropped_bytes" not in r.output
    assert r.output["blocks"] == [{"type": "text", "text": "hello"}]
    assert r.output["metadata"] == {"k": "v"}
    assert r.output["reward"] == 0.5


@pytest.mark.asyncio
async def test_openreward_env_tool_caps_multi_block_output(tmp_path):
    """When multiple text blocks together exceed the cap, the head blocks
    survive intact, the spilling block is truncated in-place, and trailing
    blocks are dropped. Mirrors how `view`/`bash` env tools can emit
    several text chunks per call."""
    import json as _json
    from openreward.api.environments.types import TextBlock, ToolOutput

    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.types import ToolDefinition

    chunk = "y" * (MAX_TOOL_OUTPUT_BYTES // 2)  # 16 KiB each at default cap

    class _StubEnvSession:
        async def call_tool(self, name, args):
            return ToolOutput(
                blocks=[TextBlock(text=chunk) for _ in range(4)],
                metadata={},
                reward=None,
                finished=False,
            )

    class _StubAdapter:
        session = _StubEnvSession()

    defn = ToolDefinition(
        name="view", description="env view",
        input_schema={"type": "object", "properties": {}},
        scope="env",
    )
    tool = _OpenRewardTool(defn, _StubAdapter())  # type: ignore[arg-type]
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)

    r = await tool.execute(_ceo_ctx(session, {}))
    assert r.ok is True
    assert r.output["truncated"] is True
    serialized = _json.dumps(r.output, ensure_ascii=False).encode("utf-8")
    assert len(serialized) <= MAX_TOOL_OUTPUT_BYTES
    # At least the head block fits whole, and the spilling block was either
    # dropped or truncated in-place — at least one block was dropped.
    assert r.output.get("dropped_blocks", 0) + (1 if r.output.get("dropped_bytes", 0) > 0 else 0) >= 1


# ---------------------------------------------------------------------------
# ViewChannelMessages cursor semantics: `since_idx` is INCLUSIVE (idx >= N),
# `before_idx` is EXCLUSIVE (idx < N). Regression for models reading a channel
# "from the start" via since_idx=0 and silently missing the first message.
# ---------------------------------------------------------------------------


async def _post(session, channel, n):
    """Post `n` messages to `channel`; they get idx 0..n-1 in order."""
    for i in range(n):
        r = await PostToChannelTool().execute(
            _ceo_ctx(session, {"channel": channel, "content": f"msg-{i}"})
        )
        assert r.ok is True


@pytest.mark.asyncio
async def test_since_idx_zero_returns_first_message(tmp_path):
    """since_idx=0 must include the first message (idx=0), not skip it."""
    session = await _session_with_channel(tmp_path)
    await _post(session, "general", 1)
    r = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "since_idx": 0})
    )
    assert r.ok is True
    assert [m["idx"] for m in r.output["messages"]] == [0]
    assert r.output["older_not_shown"] == 0


@pytest.mark.asyncio
async def test_since_idx_is_inclusive(tmp_path):
    """since_idx=N returns idx >= N (inclusive)."""
    session = await _session_with_channel(tmp_path)
    await _post(session, "general", 4)  # idx 0,1,2,3
    r = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "since_idx": 2})
    )
    assert r.ok is True
    assert [m["idx"] for m in r.output["messages"]] == [2, 3]
    assert r.output["older_not_shown"] == 2


@pytest.mark.asyncio
async def test_before_idx_stays_exclusive(tmp_path):
    """before_idx=N returns idx < N (exclusive); before_idx=0 is empty."""
    session = await _session_with_channel(tmp_path)
    await _post(session, "general", 3)  # idx 0,1,2
    r = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "before_idx": 2})
    )
    assert r.ok is True
    assert [m["idx"] for m in r.output["messages"]] == [0, 1]

    r0 = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "before_idx": 0})
    )
    assert r0.ok is True
    assert r0.output["messages"] == []


@pytest.mark.asyncio
async def test_resume_cursor_returns_only_new(tmp_path):
    """No-cursor resume must NOT re-show the last-seen message even though
    since_idx is now inclusive (resume reads from cursor_was + 1)."""
    session = await _session_with_channel(tmp_path)
    await _post(session, "general", 3)  # idx 0,1,2
    first = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general"})
    )
    assert [m["idx"] for m in first.output["messages"]] == [0, 1, 2]
    await _post(session, "general", 2)  # idx 3,4
    second = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general"})
    )
    assert second.output["resumed_from_cursor"] is True
    assert second.output["cursor_was"] == 2
    assert [m["idx"] for m in second.output["messages"]] == [3, 4]


@pytest.mark.asyncio
async def test_newer_not_shown_counts_strictly_newer(tmp_path):
    """newer_not_shown counts messages strictly after the shown window —
    must compensate for the now-inclusive since_idx."""
    session = await _session_with_channel(tmp_path)
    await _post(session, "general", 3)  # idx 0,1,2
    r = await ViewChannelMessagesTool().execute(
        _ceo_ctx(session, {"channel": "general", "before_idx": 2})  # shows idx 0,1
    )
    assert [m["idx"] for m in r.output["messages"]] == [0, 1]
    assert r.output["latest_idx"] == 1
    assert r.output["newer_not_shown"] == 1  # only idx 2, not idx 1 itself


@pytest.mark.asyncio
async def test_view_virtual_channel_chunks_oversized_messages(tmp_path):
    """Virtual (trace) channels: an oversized message is paged in 32 KB chunks,
    one per call, so no call exceeds the cap. The everything-fits case keeps the
    classic message-paged shape (chunk fields appear ONLY on a split message)."""
    from agentorg.types import Message

    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    ceo = session.ceo_name

    # Deterministic transcript: two small messages, then one ~4-chunk monster.
    # Append via the real path so they're persisted — the #self virtual view
    # reads the full on-disk history (full_transcript), not the in-memory list.
    for _m in (
        Message(role="user", content="go"),
        Message(role="assistant", content="ok small"),
        Message(role="assistant", content="z" * (MAX_TOOL_OUTPUT_BYTES * 3 + 5_000)),
        Message(role="assistant", content="after"),
    ):
        await session.append_message(ceo, _m)

    tool = ViewChannelMessagesTool()
    def page_bytes(o):
        return sum(len(str(m).encode("utf-8")) for m in o["messages"])

    # From the start: packs the two small messages, stops before the monster.
    # Everything fits → CLASSIC message-paged shape, no chunk fields/churn.
    out = (await tool.execute(_ceo_ctx(session, {"channel": "self", "since_idx": 0}))).output
    assert out["is_virtual"] is True
    assert out["total_messages"] == 4
    assert [m["idx"] for m in out["messages"]] == [0, 1]
    assert out["older_not_shown"] == 0 and out["newer_not_shown"] == 2
    assert "older_not_shown" in out and "next_cursor" not in out
    assert "total_chunks" not in out and "of_chunks" not in out["messages"][0]
    assert page_bytes(out) <= MAX_TOOL_OUTPUT_BYTES + 8_000

    # Resume → monster chunk 0, alone, bounded. Chunk fields appear ONLY now.
    out = (await tool.execute(_ceo_ctx(session, {"channel": "self"}))).output
    assert len(out["messages"]) == 1
    e = out["messages"][0]
    assert e["idx"] == 2 and e["chunk"] == 0 and e["of_chunks"] >= 4 and "text" in e
    assert page_bytes(out) <= MAX_TOOL_OUTPUT_BYTES + 8_000
    assert out["next_cursor"] == {"idx": 2, "chunk": 1}
    assert out["older_not_shown"] == 2 and out["newer_not_shown"] == 1

    # Resume again → next chunk of the same message.
    out = (await tool.execute(_ceo_ctx(session, {"channel": "self"}))).output
    assert out["messages"][0]["idx"] == 2 and out["messages"][0]["chunk"] == 1

    # Jump within a message: since_idx + chunk.
    out = (await tool.execute(_ceo_ctx(session, {"channel": "self", "since_idx": 2, "chunk": 3}))).output
    e = out["messages"][0]
    assert e["idx"] == 2 and e["chunk"] == 3


@pytest.mark.asyncio
async def test_openreward_env_tool_maps_session_not_found_to_abort_code(tmp_path):
    """A 400 `Session not found.` from the env proxy means the session id is
    gone: no retry can succeed, so it must map to `ENV_SESSION_DEAD_CODE` and
    abort rather than be misread as a recoverable `env-failure`.

    Built from a real `SessionResponseError` (not a hand-written string) so the
    test pins the ACTUAL rendering the adapter classifies on.
    """
    import aiohttp
    from multidict import CIMultiDict
    from yarl import URL

    from agentorg.openreward_env import _OpenRewardTool
    from agentorg.env_adapter import ENV_SESSION_DEAD_CODE
    from agentorg.types import ToolDefinition

    try:
        from openreward.api._session.http import SessionResponseError
    except ImportError:  # pragma: no cover - SDK not installed in this env
        pytest.skip("openreward SDK not installed")

    url = URL("https://api.openreward.ai/v1/sessions/sid-abc/call")
    request_info = aiohttp.RequestInfo(
        url=url, method="POST", headers=CIMultiDict(), real_url=url,
    )
    exc = SessionResponseError(
        request_info, (), status=400, message="Session not found.",
    )
    # Guard the substring match against an aiohttp rendering change.
    assert "Session not found" in f"{exc.__class__.__name__}: {exc}"

    class _Adapter:
        def __init__(self, e):
            class _Sess:
                async def call_tool(self, name, args):
                    raise e
            self.session = _Sess()

    defn = ToolDefinition(
        name="place_bet", description="bet",
        input_schema={"type": "object", "properties": {}}, scope="env",
    )
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=0,
        headless=True, permission_policy="bypass",
    )
    harness = Harness(inference=_Stub())
    session = await harness.start_session(config=config)
    ctx = _ceo_ctx(session, {})

    r = await _OpenRewardTool(defn, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]
    assert r.ok is False
    assert r.error.code == ENV_SESSION_DEAD_CODE

    # A 500 carries the env server's own `detail` text, which is arbitrary user
    # code output — it must NOT abort the session even if it says the same thing.
    user_exc = SessionResponseError(
        request_info, (), status=500, message="ValueError: Session not found in my_env.py",
    )
    r_user = await _OpenRewardTool(defn, _Adapter(user_exc)).execute(ctx)  # type: ignore[arg-type]
    assert r_user.error.code == "env-failure"
