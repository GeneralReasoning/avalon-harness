"""Compaction tests (§11.4)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from agentorg.compaction import (
    AUTO_COMPACT_BUFFER_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    MAX_PROMPT_TOO_LONG_RETRIES,
    MICROCOMPACT_BUFFER_TOKENS,
    MIN_THRESHOLD_TOKENS,
    OLD_RESULT_SENTINEL,
    RESERVED_SUMMARY_OUTPUT_TOKENS,
    SUFFIX_BUDGET_WINDOW_DIVISOR,
    SUMMARY_ERROR_MAX_CHARS,
    _MESSAGE_OVERHEAD_TOKENS,
    CompactionConfig,
    _CHAT_TEMPLATE_TOKEN_RE,
    _BadSummaryReason,
    _build_compact_prompt,
    _extract_response_summary,
    _render_conversation_for_compaction,
    _select_compact_boundary,
    _summarize_prefix,
    _validate_summary,
    auto_compact_if_needed,
    estimate_tokens,
    get_auto_compact_threshold,
    get_microcompact_threshold,
    get_model_context_window,
    get_suffix_token_budget,
    microcompact_if_needed,
)
from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.storage import iter_jsonl
from agentorg.tools.context import ToolContext
from agentorg.tools.org_tools import (
    SearchMyHistoryTool,
    ViewChannelMessagesTool,
    ViewMyHistoryTool,
)
from agentorg.types import (
    Agent,
    Message,
    MessageUsage,
    ToolCall,
    ToolResult,
    ToolResultError,
)
from tests.test_smoke import MockInferenceProvider


def test_estimate_counts_latest_assistant_output():
    """A turn that emits a large output must be visible to the estimator on the
    very next check, not one turn later. Otherwise a very large turn stays
    invisible (estimate anchored only on the turn's *input_tokens*) until the
    next inference reports it via input_tokens — by which point the prompt has
    already overflowed the window and no between-turns compaction can fire."""
    transcript = [
        Message(role="user", content="solve it"),
        Message(
            role="assistant",
            content="x" * 40,
            usage=MessageUsage(input_tokens=11_842, output_tokens=53_694),
        ),
    ]
    # prompt that produced the turn (input_tokens) + the turn's own re-sent output.
    assert estimate_tokens(transcript) == 11_842 + 53_694


def test_estimate_excludes_reasoning_from_latest_output():
    """Reasoning is not re-sent to the next prompt (loop._to_provider_message),
    so only the non-reasoning portion of the output should count."""
    transcript = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="answer",
            usage=MessageUsage(
                input_tokens=1_000, output_tokens=50_000, reasoning_tokens=49_000
            ),
        ),
    ]
    assert estimate_tokens(transcript) == 1_000 + (50_000 - 49_000)


def _bulky_tool_call_pair(payload: str, *, tool: str = "Read", call_id: str | None = None) -> list[Message]:
    """A matched assistant tool_call + tool result, with the tool named so the
    microcompactor can identify it as bulky (per SPEC §11.4.1).

    The microcompactor walks back from each tool-result message to find the
    assistant message that issued the corresponding tool_call; without the
    assistant pair the tool name is unknown and the result is left alone.
    """
    cid = call_id or "c1"
    return [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=cid, name=tool, args={})],
        ),
        Message(
            role="tool",
            tool_call_id=cid,
            tool_result=ToolResult(ok=True, output={"data": payload}),
        ),
    ]


@pytest.mark.asyncio
async def test_microcompaction_fires_at_most_once_per_agent(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    # Pump the CEO transcript with bulky tool results (Read is in BULKY_TOOLS).
    transcript = session.transcript(session.ceo_name)
    payload = "x" * 4000
    for i in range(20):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))

    first = await microcompact_if_needed(session, session.ceo_name)
    assert first is True
    assert session.agent(session.ceo_name).microcompacted is True

    # Append more bulky tool results — second call for the same agent must NOT
    # fire again (prefix cache was already invalidated once).
    for i in range(20, 40):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))
    second = await microcompact_if_needed(session, session.ceo_name)
    assert second is False


@pytest.mark.asyncio
async def test_microcompaction_is_per_agent_independent(tmp_path):
    """Per-agent cap (§11.4.1): one agent firing must not lock out a peer.

    A specialist that microcompacts early must not block the CEO from
    microcompacting later when the CEO's transcript climbs past the threshold.
    The CEO's prefix cache is independent of the specialist's, so a
    per-session cap would be wrong.
    """
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    # Spawn a specialist and mark its microcompaction slot as already burned.
    peer = await spawn_org_agent(
        session, OrgAgentSpec(role="Specialist", description="x")
    )
    peer.microcompacted = True
    await session.update_agent(peer)

    # The CEO's own slot must still be available.
    transcript = session.transcript(session.ceo_name)
    payload = "x" * 4000
    for i in range(20):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True, "CEO should be able to microcompact even when a peer already did"
    assert session.agent(session.ceo_name).microcompacted is True
    assert session.agent(peer.name).microcompacted is True


@pytest.mark.asyncio
async def test_microcompaction_only_clears_bulky_tools(tmp_path):
    """SPEC §11.4.1: only Bash/Edit/Read/Glob/Grep/Write/WebFetch/WebSearch
    + env-side tools are eligible. Coordination, channel, archive, and report
    tool results carry persistent state the agent needs to keep referring to
    and MUST NOT be cleared, even when older than the preserve-recent window.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    transcript = session.transcript(session.ceo_name)
    payload = "x" * 4000

    # 10 channel-scrollback results (NOT bulky — must survive).
    for i in range(10):
        transcript.extend(
            _bulky_tool_call_pair(
                payload, tool="ViewChannelMessages", call_id=f"chan-{i}"
            )
        )
    # 10 Read results (bulky — most should be cleared, last 5 preserved).
    for i in range(10):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"read-{i}"))

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True

    # Channel results must be untouched.
    channel_results = [
        m for m in transcript
        if m.role == "tool" and (m.tool_call_id or "").startswith("chan-")
    ]
    assert len(channel_results) == 10
    for m in channel_results:
        assert m.tool_result is not None
        out = m.tool_result.output
        assert not (isinstance(out, dict) and out.get("_compacted")), (
            f"channel tool result {m.tool_call_id} was compacted despite not being bulky"
        )

    # Read results: 5 most recent preserved, 5 older cleared.
    read_results = [
        m for m in transcript
        if m.role == "tool" and (m.tool_call_id or "").startswith("read-")
    ]
    assert len(read_results) == 10
    compacted = [
        m for m in read_results
        if isinstance(m.tool_result.output, dict) and m.tool_result.output.get("_compacted")
    ]
    preserved = [m for m in read_results if m not in compacted]
    assert len(compacted) == 5
    assert len(preserved) == 5
    # The preserved ones must be the *last* 5 (most recent) by call_id ordering.
    assert [m.tool_call_id for m in preserved] == [f"read-{i}" for i in range(5, 10)]


@pytest.mark.asyncio
async def test_microcompaction_slot_resets_after_full_compaction(tmp_path):
    """SPEC §11.4.1 + §11.4.2: the microcompaction cap is per-agent per
    full-compaction window. Cadence: micro → full → micro → full → …

    After auto_compact_if_needed fires for an agent, that agent's
    `microcompacted` bool MUST be reset to False so the next window opens.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            micro_compact_threshold=1, auto_compact_threshold=1
        ),
    )
    harness = Harness(inference=MockInferenceProvider("a fresh summary"))
    session = await harness.start_session(config=config)

    transcript = session.transcript(session.ceo_name)
    payload = "x" * 4000

    # Window 1: build up bulky tool results, fire microcompaction.
    for i in range(20):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"w1-{i}"))
    first_micro = await microcompact_if_needed(session, session.ceo_name)
    assert first_micro is True
    assert session.agent(session.ceo_name).microcompacted is True

    # A second microcompaction in the same window must be blocked.
    for i in range(20):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"w1b-{i}"))
    blocked = await microcompact_if_needed(session, session.ceo_name)
    assert blocked is False, "microcompaction must not fire twice in one window"

    # Full compaction fires — replaces prefix with summary and MUST reset the slot.
    fired_full = await auto_compact_if_needed(
        session,
        session.ceo_name,
        preserve_recent_turns=4,
    )
    assert fired_full is True
    assert session.agent(session.ceo_name).microcompacted is False, (
        "full compaction MUST reset the agent's microcompaction slot"
    )

    # Window 2: rebuild bulky tool results past the threshold. Now
    # microcompaction is permitted again.
    transcript = session.transcript(session.ceo_name)
    for i in range(20):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"w2-{i}"))
    second_micro = await microcompact_if_needed(session, session.ceo_name)
    assert second_micro is True, "microcompaction must be allowed again after a full compaction"
    assert session.agent(session.ceo_name).microcompacted is True


@pytest.mark.asyncio
async def test_microcompaction_treats_env_tools_as_bulky(tmp_path):
    """Env-scoped tools count as bulky per SPEC §11.4.1 ('equivalent env-side
    tools'). The eligible set is computed at fire time from the session's
    tool registry, so any tool registered with scope='env' is included."""
    from agentorg.types import ToolDefinition

    class _FakeEnvTool:
        def __init__(self, name: str):
            self.definition = ToolDefinition(
                name=name,
                description="env",
                input_schema={"type": "object", "properties": {}},
                read_only=False,
                scope="env",
            )

        async def execute(self, ctx):  # pragma: no cover — not invoked
            from agentorg.tools.context import ok_result
            return ok_result({})

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)
    session.tools.register(_FakeEnvTool("observe_instruments"))

    transcript = session.transcript(session.ceo_name)
    payload = "x" * 4000
    for i in range(10):
        transcript.extend(
            _bulky_tool_call_pair(
                payload, tool="observe_instruments", call_id=f"env-{i}"
            )
        )

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True

    env_results = [m for m in transcript if m.role == "tool"]
    compacted = sum(
        1 for m in env_results
        if isinstance(m.tool_result.output, dict)
        and m.tool_result.output.get("_compacted")
    )
    assert compacted == 5  # 10 total, 5 most recent preserved, 5 cleared


@pytest.mark.asyncio
async def test_auto_compaction_preserves_pre_compaction_messages_on_disk(tmp_path):
    """§11.1 + §12.4: messages.jsonl is append-only. Auto-compaction MUST NOT
    drop the pre-compaction prefix from disk — it is the source of truth for
    SFT/RL training. The live in-memory transcript still shrinks (summary +
    last K turns); the file just keeps the original records as well, behind
    a compaction-boundary marker."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)

    n_messages = 24
    for i in range(n_messages):
        await session.append_message(
            session.ceo_name,
            Message(
                role="assistant",
                content=f"msg-{i:02d}",
                tool_calls=None,
            ),
        )

    transcript_before = list(session.transcript(session.ceo_name))
    assert len(transcript_before) == n_messages

    fired = await auto_compact_if_needed(
        session,
        session.ceo_name,
        preserve_recent_turns=4,
    )
    assert fired is True

    live = session.transcript(session.ceo_name)
    assert live[0].compaction is True
    assert live[0].role == "user"
    suffix = live[1:]
    # One preserved "turn" is one API round; this transcript is assistant-only
    # messages, so 4 preserved rounds == the last 4 messages.
    assert len(suffix) == 4
    assert [m.content for m in suffix] == [m.content for m in transcript_before[-4:]]

    raw_records = list(iter_jsonl(session.transcript_path(session.ceo_name)))
    boundaries = [r for r in raw_records if r.get("type") == "compaction-boundary"]
    assert len(boundaries) == 1
    assert boundaries[0]["suffix_count"] == 4
    # MockInferenceProvider returns an untagged summary → unusable. We still log
    # the attempt (for a future rebuild) but tag it so the builder skips it.
    assert boundaries[0]["summary_training"]["outcome"] == "bad-summary"

    message_records = [r for r in raw_records if r.get("type") != "compaction-boundary"]
    original_contents = [m.content for m in transcript_before]
    for original in original_contents:
        assert any(r.get("content") == original for r in message_records), (
            f"pre-compaction message {original!r} missing from messages.jsonl"
        )

    reloaded = session._load_transcript(session.ceo_name)
    assert [m.content for m in reloaded] == [m.content for m in live]
    assert reloaded[0].compaction is True


@pytest.mark.asyncio
async def test_full_transcript_and_self_views_recover_pre_compaction_turns(tmp_path):
    """After a compaction the live transcript shrinks to summary + suffix, but
    full_transcript() recovers the original pre-compaction turns from disk — and
    so do the virtual-channel read paths that now use it: the #self trace
    (ViewChannelMessages) and the solo recovery tools (ViewMyHistory /
    SearchMyHistory)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)
    ceo = session.ceo_name
    agent = session.agent(ceo)

    N = 24
    for i in range(N):
        await session.append_message(
            ceo, Message(role="assistant", content=f"msg-{i:02d}", tool_calls=None)
        )

    fired = await auto_compact_if_needed(session, ceo, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(ceo)
    full = session.full_transcript(ceo)

    # Live view shrank and lost the oldest turns; full view kept everything,
    # in order, with no injected summary message.
    assert len(live) == 5  # summary + 4 preserved rounds (assistant-only)
    assert "msg-00" not in [m.content for m in live]
    assert [m.content for m in full] == [f"msg-{i:02d}" for i in range(N)]
    assert not any(getattr(m, "compaction", False) for m in full)

    # Multi-agent #self read path now surfaces the dropped prefix.
    res_self = await ViewChannelMessagesTool().execute(
        ToolContext(
            agent=agent, session=session,
            args={"channel": "self", "since_idx": 0, "limit": 200}, call_id="c1",
        )
    )
    assert res_self.output["total_messages"] == N
    assert "msg-00" in json.dumps(res_self.output)

    # Solo recovery tools (channel-free names) reach the same full history.
    res_view = await ViewMyHistoryTool().execute(
        ToolContext(
            agent=agent, session=session,
            args={"since_idx": 0, "limit": 200}, call_id="c2",
        )
    )
    assert res_view.ok and res_view.output["total_messages"] == N
    assert "msg-00" in json.dumps(res_view.output)

    res_search = await SearchMyHistoryTool().execute(
        ToolContext(
            agent=agent, session=session,
            args={"pattern": "msg-00", "output_mode": "count"}, call_id="c3",
        )
    )
    total = sum(c.get("count", 0) for c in res_search.output.get("counts", {}).values())
    assert total >= 1


@pytest.mark.asyncio
async def test_teammate_trace_recovers_pre_compaction_turns(tmp_path):
    """Multi-agent: a CEO viewing a teammate's trace channel (#<name>) after
    that teammate compacted must see the full pre-compaction history, not just
    the shrunken live transcript. Same shared virtual-channel read path as
    #self, with target resolving to the peer."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)
    ceo_agent = session.agent(session.ceo_name)

    mate = Agent(
        name="RedAristotle", session_id=session.session_id, kind="org",
        role="Analyst", description="d", work_dir=str(tmp_path), system_prompt="",
    )
    await session.register_agent(mate)

    N = 24
    for i in range(N):
        await session.append_message(
            "RedAristotle", Message(role="assistant", content=f"mate-{i:02d}", tool_calls=None)
        )
    fired = await auto_compact_if_needed(session, "RedAristotle", preserve_recent_turns=4)
    assert fired is True

    # Teammate's own live context shrank (lost the oldest turns)...
    assert "mate-00" not in [m.content for m in session.transcript("RedAristotle")]

    # ...but the CEO's view of #RedAristotle recovers the full history.
    res = await ViewChannelMessagesTool().execute(
        ToolContext(
            agent=ceo_agent, session=session,
            args={"channel": "RedAristotle", "since_idx": 0, "limit": 200}, call_id="c1",
        )
    )
    assert res.output["total_messages"] == N
    assert "mate-00" in json.dumps(res.output)


@pytest.mark.asyncio
async def test_auto_compaction_records_summary_training_block(tmp_path):
    """The boundary record persists the FULL summarization call — system +
    rendered prompt + raw completion + provenance — so a training-data builder
    can train the model to PRODUCE compaction summaries, not just consume them.
    (Consume side stays the extracted <summary> body injected as new context.)"""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>why this matters</analysis><summary>the durable summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    for i in range(24):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"msg-{i:02d}")
        )

    fired = await auto_compact_if_needed(
        session, session.ceo_name, preserve_recent_turns=4
    )
    assert fired is True

    raw_records = list(iter_jsonl(session.transcript_path(session.ceo_name)))
    boundaries = [r for r in raw_records if r.get("type") == "compaction-boundary"]
    assert len(boundaries) == 1
    training = boundaries[0]["summary_training"]
    assert training["outcome"] == "accepted"
    assert training["attempts"] == 1
    assert training["stop_reason"] == "stop"
    assert "usage" in training
    assert "Primary Request and Intent" in training["system"]
    assert training["prompt"].startswith("Here is the conversation to summarize:")
    # Full raw turn (analysis + summary), not just the extracted <summary> body.
    assert training["completion"]["role"] == "assistant"
    assert training["completion"]["content"] == (
        "<analysis>why this matters</analysis><summary>the durable summary</summary>"
    )
    # The consume-side message still carries the extracted body, prefixed.
    assert boundaries[0]["summary"]["content"] == "[compaction]\nthe durable summary"


@pytest.mark.asyncio
async def test_auto_compaction_emits_compaction_fired_event(tmp_path):
    """Dashboards / downstream listeners need to know when compaction fires.
    A `compaction.fired` event is emitted after the boundary is durable on
    disk, carrying enough shape + provenance to surface the moment without
    re-parsing messages.jsonl."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>the summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    for i in range(24):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"msg-{i:02d}")
        )

    fired = await auto_compact_if_needed(
        session, session.ceo_name, preserve_recent_turns=4
    )
    assert fired is True

    events = list(iter_jsonl(session.paths.events_jsonl))
    compaction_events = [e for e in events if e.get("type") == "compaction.fired"]
    assert len(compaction_events) == 1
    ev = compaction_events[0]
    assert ev["name"] == session.ceo_name
    payload = ev["payload"]
    assert payload["name"] == session.ceo_name
    assert payload["suffix_count"] == 4  # 4 preserved rounds, assistant-only
    assert payload["prefix_count"] == 24 - 4
    assert payload["outcome"] == "accepted"
    assert payload["attempts"] == 1
    # Provenance of the trigger: fired through the estimate gate, not the
    # loop's prompt-too-long recovery; no assistant carried usage, so the
    # estimate was the pure chars/4 fallback with no density calibration.
    assert payload["forced"] is False
    assert payload["anchor_valid"] is False
    assert payload["calibration"] == 1.0


@pytest.mark.asyncio
async def test_summary_training_prompt_reflects_microcompaction(tmp_path):
    """Distribution match for RL: microcompaction rewrites old bulky tool
    results to a sentinel in the LIVE transcript only, and the summarizer sees
    that view. The stored training prompt must therefore carry the sentinel —
    reconstructing from the (un-microcompacted) on-disk records would not."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="kimi-k2.6",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            micro_compact_threshold=1, auto_compact_threshold=1
        ),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>compacted-history summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    # Enough bulky Read results that microcompaction clears the older ones.
    transcript = session.transcript(session.ceo_name)
    for i in range(12):
        transcript.extend(_bulky_tool_call_pair("z" * 2000, tool="Read", call_id=f"c{i}"))

    assert await microcompact_if_needed(session, session.ceo_name) is True
    # Now full-compact; the summarizer renders the microcompacted live prefix.
    assert await auto_compact_if_needed(
        session, session.ceo_name, preserve_recent_turns=2
    ) is True

    raw_records = list(iter_jsonl(session.transcript_path(session.ceo_name)))
    boundary = next(r for r in raw_records if r.get("type") == "compaction-boundary")
    training = boundary["summary_training"]
    assert training["outcome"] == "accepted"
    # The sentinel proves the prompt is the microcompacted view the model saw.
    assert OLD_RESULT_SENTINEL in training["prompt"]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def test_extract_response_summary_with_tag():
    body = "preamble <analysis>thinking</analysis> <summary>the real summary</summary> trailing"
    summary, had_tag = _extract_response_summary(body)
    assert had_tag is True
    assert summary == "the real summary"


def test_extract_response_summary_without_tag_strips_analysis_block():
    body = "<analysis>thinking</analysis>\n\nfallback content"
    summary, had_tag = _extract_response_summary(body)
    assert had_tag is False
    assert summary == "fallback content"


def test_validate_summary_happy():
    assert _validate_summary("ok prose", had_summary_tag=True) is None


def test_validate_summary_rejects_empty():
    issue = _validate_summary("   ", had_summary_tag=True)
    assert issue is not None
    assert issue.kind == "empty"


def test_validate_summary_rejects_template_leak():
    issue = _validate_summary(
        "summary <|tool_calls_section_begin|> garbage", had_summary_tag=True
    )
    assert issue is not None
    assert issue.kind == "tool-template-leak"
    # Named without delimiters: the detail is echoed into the retry prompt and
    # the placeholder, where a literal <|...|> could tokenize as the real token.
    assert "tool_calls_section_begin" in issue.detail
    assert not _CHAT_TEMPLATE_TOKEN_RE.search(issue.detail)


def test_validate_summary_rejects_missing_tag():
    issue = _validate_summary("raw content with no tags", had_summary_tag=False)
    assert issue is not None
    assert issue.kind == "format-violation"


def test_render_conversation_for_compaction_labels_roles_and_tool_calls():
    msgs = [
        Message(role="user", content="please refactor foo.py"),
        Message(
            role="assistant",
            content="will do",
            tool_calls=[ToolCall(id="c1", name="Read", args={"path": "foo.py"})],
        ),
        Message(
            role="tool",
            tool_call_id="c1",
            tool_result=ToolResult(ok=True, output="def foo(): pass"),
        ),
        Message(
            role="tool",
            tool_call_id="c2",
            tool_result=ToolResult(
                ok=False, error=ToolResultError(code="boom", message="no")
            ),
        ),
        Message(role="user", source="meta", content="[nudge]"),
    ]
    rendered = _render_conversation_for_compaction(msgs)
    assert "USER:\nplease refactor foo.py" in rendered
    assert "ASSISTANT:\nwill do\n(tool call) Read(" in rendered
    assert "TOOL RESULT (c1):" in rendered
    assert "def foo(): pass" in rendered
    assert "TOOL ERROR (c2):\nboom: no" in rendered
    assert "USER (meta):\n[nudge]" in rendered
    # Blocks are joined by "---" separators in a single user-facing prose body.
    assert "\n\n---\n\n" in rendered


def test_build_compact_prompt_includes_retry_section_when_given():
    plain = _build_compact_prompt(None)
    retry = _build_compact_prompt(
        _BadSummaryReason("tool-template-leak", "saw <|tool_call_begin|>")
    )
    assert "Previous Attempt Was Rejected" not in plain
    assert "Previous Attempt Was Rejected" in retry
    assert "saw <|tool_call_begin|>" in retry
    # The prompt's own text names no chat-template token: a server that
    # tokenizes prompt text with special tokens live would splice a real one
    # (e.g. <|im_start|>) into the system prompt.
    for prompt in (plain, _build_compact_prompt(_BadSummaryReason("empty", "x"))):
        assert not _CHAT_TEMPLATE_TOKEN_RE.search(prompt), prompt


# ---------------------------------------------------------------------------
# End-to-end summary path with controllable providers
# ---------------------------------------------------------------------------


class _ScriptedSummaryProvider(InferenceProvider):
    """Yields a pre-scripted content string per call. One call = one entry."""

    def __init__(self, scripted_contents: list[str]):
        self._contents = list(scripted_contents)
        self.calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        idx = min(self.calls, len(self._contents) - 1)
        self.calls += 1
        yield ContentDeltaChunk(text=self._contents[idx])
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


def _prefix_messages() -> list[Message]:
    return [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]


@pytest.mark.asyncio
async def test_summary_happy_path_extracts_summary_tag(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>brief</analysis><summary>the kept summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    result = await _summarize_prefix(session, session.ceo_name, _prefix_messages())
    assert result.summary == "the kept summary"
    assert provider.calls == 1
    # The full request as sent + the raw completion are captured for SFT.
    assert result.outcome == "accepted"
    assert result.attempts == 1
    assert result.ptl_truncations == 0
    assert result.completion == "<analysis>brief</analysis><summary>the kept summary</summary>"
    assert "Primary Request and Intent" in result.system_prompt
    assert result.prompt_body.startswith("Here is the conversation to summarize:")
    # Completion metadata captured for the quality gate.
    assert result.stop_reason == "stop"
    assert result.usage is not None


@pytest.mark.asyncio
async def test_summary_retries_once_on_format_violation_and_recovers(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _ScriptedSummaryProvider(
        [
            "no tags at all just prose",
            "<analysis>x</analysis><summary>recovered on retry</summary>",
        ]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    result = await _summarize_prefix(session, session.ceo_name, _prefix_messages())
    assert result.summary == "recovered on retry"
    assert provider.calls == 2
    # A recovered retry is still a good summary — captured (not dropped), with
    # provenance so the builder can filter on it if it wants.
    assert result.outcome == "accepted"
    assert result.attempts == 2
    assert result.completion == "<analysis>x</analysis><summary>recovered on retry</summary>"


@pytest.mark.asyncio
async def test_summary_gives_up_after_one_retry(tmp_path):
    """MAX_BAD_SUMMARY_RETRIES = 1 → 2 attempts total, then a structured error."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _ScriptedSummaryProvider(
        [
            "<summary><|tool_call_begin|>functions.step(...)</summary>",
            "<summary><|tool_calls_section_end|> still leaking</summary>",
        ]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    result = await _summarize_prefix(session, session.ceo_name, _prefix_messages())
    assert provider.calls == 2  # initial + one retry, no more
    assert result.summary.startswith("[summary unavailable:")
    assert "tool-template-leak" in result.summary
    # Even a failed summary is logged (for a future rebuild) — but tagged so
    # the builder skips it as a training target by default.
    assert result.outcome == "bad-summary"
    assert result.issue_kind == "tool-template-leak"
    assert result.completion is not None


@pytest.mark.asyncio
async def test_microcompaction_uses_derived_threshold_when_none(tmp_path):
    """With `threshold=None` the function derives from the session's model
    instead of using a hardcoded number. Verify by sizing the transcript so
    the derived Kimi threshold is exactly crossed."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="kimi-k2.6",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    micro_threshold = get_microcompact_threshold("kimi-k2.6")
    # Each bulky pair adds ≈ payload/4 tokens. Build enough to land well above
    # the derived threshold so the derivation path actually fires.
    transcript = session.transcript(session.ceo_name)
    payload = "x" * 16_000  # ~4k tokens per pair
    pair_count = (micro_threshold // 4_000) + 4
    for i in range(pair_count):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))

    # No explicit threshold — must come from `get_microcompact_threshold("kimi-k2.6")`.
    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True
    assert session.agent(session.ceo_name).microcompacted is True


class _PTLOnceThenOKProvider(InferenceProvider):
    """Raises a prompt-too-long-style error on call 1, then returns content."""

    def __init__(self, content: str):
        self._content = content
        self.calls = 0
        self.user_bodies: list[str] = []

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self.calls += 1
        for m in request.messages:
            if m.role == "user":
                self.user_bodies.append(str(m.content))
        if self.calls == 1:
            raise ValueError("prompt too long: input too large for context window")
        yield ContentDeltaChunk(text=self._content)
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


@pytest.mark.asyncio
async def test_summary_retries_on_prompt_too_long_with_truncated_head(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _PTLOnceThenOKProvider(
        "<analysis>brief</analysis><summary>recovered after head drop</summary>"
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    # Build a multi-round prefix so head truncation has rounds to drop.
    prefix: list[Message] = []
    for i in range(10):
        prefix.append(Message(role="user", content=f"q-{i}"))
        prefix.append(
            Message(
                role="assistant",
                content=f"a-{i}",
                tool_calls=[ToolCall(id=f"c-{i}", name="Read", args={})],
            )
        )
        prefix.append(
            Message(
                role="tool",
                tool_call_id=f"c-{i}",
                tool_result=ToolResult(ok=True, output=f"r-{i}"),
            )
        )

    result = await _summarize_prefix(session, session.ceo_name, prefix)
    assert result.summary == "recovered after head drop"
    assert provider.calls == 2
    # The completion is captured with the TRUNCATED prompt that produced it —
    # faithful precisely because we store the request as sent, not reconstruct.
    assert result.outcome == "accepted"
    assert result.ptl_truncations == 1
    assert result.completion == "<analysis>brief</analysis><summary>recovered after head drop</summary>"
    assert "[earlier conversation truncated for compaction retry]" in result.prompt_body
    # The second call's rendered body must be shorter than the first — the
    # head got dropped.
    assert len(provider.user_bodies[1]) < len(provider.user_bodies[0])
    # The truncated body must contain the PTL marker so the model knows
    # earlier context was dropped. The opening user round (q-0) is the
    # pinned mandate and survives; ten droppable rounds lose 10//5 = 2 from
    # the head (a-0 and a-1 with their results), and the marker sits after
    # the pin because the next survivor is the assistant a-2.
    assert "[earlier conversation truncated for compaction retry]" in provider.user_bodies[1]
    assert "q-0" in provider.user_bodies[1]
    assert "a-0" not in provider.user_bodies[1]
    assert "a-1" not in provider.user_bodies[1]
    assert "a-2" in provider.user_bodies[1]


@pytest.mark.asyncio
async def test_summary_ptl_gives_up_when_head_cannot_be_truncated(tmp_path):
    """A single API round (one group) can't be truncated — `_summarize_prefix`
    must surface the error instead of retrying uselessly."""

    class _AlwaysPTL(InferenceProvider):
        def __init__(self):
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            raise ValueError("prompt too long")
            yield  # unreachable, marks this as async generator

        def model_ids(self):
            return ["mock"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _AlwaysPTL()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    # Single round, no leading user → exactly one group → truncation returns None.
    prefix = [
        Message(role="assistant", content="just one round"),
        Message(
            role="tool",
            tool_call_id="c0",
            tool_result=ToolResult(ok=True, output="r"),
        ),
    ]
    result = await _summarize_prefix(session, session.ceo_name, prefix)
    assert result.summary.startswith("[summary unavailable:")
    assert result.outcome == "ptl-failed"
    assert provider.calls == 1  # No retry possible — truncate returned None.


def test_truncate_head_drops_first_group_and_adds_marker():
    """Round grouping: leading user → group 0; each assistant starts a new
    group with its tool/user follow-ups. 5 rounds with leading users produces
    6 groups (the trailing assistant's pair is its own final group). Group 0
    opens with a user message, so it is the pinned mandate and survives;
    drop_count = max(1, 5//5) = 1 removes the first droppable group
    (a-0, r-0, u-1). First survivor is a-1 (assistant) → marker inserted
    between the pinned round and the survivors.
    """
    from agentorg.compaction import (
        _PTL_RETRY_MARKER,
        _truncate_head_for_ptl_retry,
    )

    rounds: list[Message] = []
    for i in range(5):
        rounds.append(Message(role="user", content=f"u-{i}"))
        rounds.append(Message(role="assistant", content=f"a-{i}"))
        rounds.append(
            Message(
                role="tool",
                tool_call_id=f"c-{i}",
                tool_result=ToolResult(ok=True, output=f"r-{i}"),
            )
        )

    truncated = _truncate_head_for_ptl_retry(rounds)
    assert truncated is not None
    contents = [m.content for m in truncated]
    assert contents[:3] == ["u-0", _PTL_RETRY_MARKER, "a-1"]
    assert "a-0" not in contents and "u-1" not in contents
    # Three messages dropped, one marker inserted.
    assert len(truncated) == len(rounds) - 2


def test_truncate_head_pins_mandate_and_previous_summary():
    """After the first compaction the summariser's prefix opens with the
    pinned mandate and the previous summary (both user messages, one API
    round). Head truncation must never drop that round: a summary written
    without it describes a segment with no task in it. The marker marks
    where the drop happened, after the pinned round; a second retry does not
    stack markers and keeps the pin."""
    from agentorg.compaction import (
        _PTL_RETRY_MARKER,
        _truncate_head_for_ptl_retry,
    )

    prefix: list[Message] = [
        Message(role="user", source="human", content="mandate"),
        Message(role="user", source="meta", content="[compaction]\nsummary", compaction=True),
    ]
    for i in range(6):
        prefix.append(Message(role="assistant", content=f"a-{i}"))
        prefix.append(
            Message(
                role="tool",
                tool_call_id=f"c-{i}",
                tool_result=ToolResult(ok=True, output=f"r-{i}"),
            )
        )

    once = _truncate_head_for_ptl_retry(prefix)
    assert once is not None
    contents = [m.content for m in once]
    assert contents[:4] == ["mandate", "[compaction]\nsummary", _PTL_RETRY_MARKER, "a-1"]
    assert "a-0" not in contents

    twice = _truncate_head_for_ptl_retry(once)
    assert twice is not None
    contents = [m.content for m in twice]
    assert contents[:4] == ["mandate", "[compaction]\nsummary", _PTL_RETRY_MARKER, "a-2"]
    assert contents.count(_PTL_RETRY_MARKER) == 1


def test_truncate_head_returns_none_when_only_pinned_round_would_remain():
    """Pinned round plus a single droppable round: dropping it would leave the
    summariser nothing to summarise, so truncation cannot help."""
    from agentorg.compaction import _truncate_head_for_ptl_retry

    prefix = [
        Message(role="user", content="mandate"),
        Message(role="assistant", content="a-0"),
        Message(
            role="tool",
            tool_call_id="c0",
            tool_result=ToolResult(ok=True, output="r"),
        ),
    ]
    assert _truncate_head_for_ptl_retry(prefix) is None


def test_truncate_head_returns_none_when_single_group():
    """When the prefix is a single API round (assistant starts the messages),
    grouping produces 1 group — nothing to drop."""
    from agentorg.compaction import _truncate_head_for_ptl_retry

    one_group = [
        Message(role="assistant", content="lonely"),
        Message(
            role="tool",
            tool_call_id="c0",
            tool_result=ToolResult(ok=True, output="r"),
        ),
    ]
    assert _truncate_head_for_ptl_retry(one_group) is None


def test_truncate_head_strips_existing_marker_before_retrying():
    """If a PTL retry already prepended a marker, the next retry must strip
    and re-prepend a single marker — not stack them."""
    from agentorg.compaction import (
        _PTL_RETRY_MARKER,
        _truncate_head_for_ptl_retry,
    )

    with_marker: list[Message] = [
        Message(role="user", source="meta", content=_PTL_RETRY_MARKER),
    ]
    for i in range(5):
        with_marker.append(Message(role="assistant", content=f"a-{i}"))
        with_marker.append(Message(role="user", content=f"u-{i}"))

    truncated = _truncate_head_for_ptl_retry(with_marker)
    assert truncated is not None
    marker_count = sum(
        1 for m in truncated
        if m.source == "meta" and m.content == _PTL_RETRY_MARKER
    )
    assert marker_count <= 1


def test_is_prompt_too_long_detects_common_messages():
    from agentorg.compaction import _is_prompt_too_long_error

    assert _is_prompt_too_long_error(ValueError("prompt too long"))
    assert _is_prompt_too_long_error(RuntimeError("input too large for model"))
    assert _is_prompt_too_long_error(RuntimeError("exceeded maximum context length"))
    assert _is_prompt_too_long_error(Exception("too many tokens in request"))
    # Status code attribute path.
    err = RuntimeError("server says no")
    setattr(err, "status_code", 413)
    assert _is_prompt_too_long_error(err)
    # Negative case.
    assert not _is_prompt_too_long_error(ValueError("rate limited"))


@pytest.mark.asyncio
async def test_auto_compaction_walks_past_leading_tool_messages(tmp_path):
    """Regression: §11.4.2 must not split a tool_call/tool_result pair.

    If the boundary lands on a tool message whose paired assistant
    tool_calls is in the prefix, the suffix would open with a dangling
    tool message and providers would reject the next inference call.
    Boundary must walk forward past leading tool messages.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("summary"))
    session = await harness.start_session(config=config)

    # Build a transcript whose naive boundary lands on a tool message.
    # preserve_recent_turns=4 → suffix size = 8. We craft the tail so the
    # 8th-from-last message is a `tool` whose paired assistant is in the
    # prefix.
    for i in range(10):
        await session.append_message(
            session.ceo_name,
            Message(role="assistant", content=f"early-{i}"),
        )
    # Insert a tool_call followed by enough tool results that the
    # boundary cleaves between them.
    await session.append_message(
        session.ceo_name,
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="cA", name="Read", args={})],
        ),
    )
    for tail in range(8):
        await session.append_message(
            session.ceo_name,
            Message(
                role="tool",
                tool_call_id=f"cT{tail}",
                tool_result=ToolResult(ok=True, output={"i": tail}),
            ),
        )
    # Final tail: a few "normal" assistant turns.
    for i in range(4):
        await session.append_message(
            session.ceo_name,
            Message(role="assistant", content=f"late-{i}"),
        )

    fired = await auto_compact_if_needed(
        session,
        session.ceo_name,
        preserve_recent_turns=4,
    )
    assert fired is True

    live = session.transcript(session.ceo_name)
    # Summary comes first, then non-tool message.
    assert live[0].compaction is True
    assert live[1].role != "tool", (
        f"suffix must not open with a tool message; got {live[1].role}"
    )


def test_load_transcript_handles_new_combined_boundary(tmp_path):
    """Verify the new combined boundary+summary record reloads cleanly."""
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
    )
    session = Session(config=config, inference=_Stub(), tools=ToolRegistry())

    from agentorg.storage import append_jsonl

    path = session.transcript_path(session.ceo_name)
    prefix = [
        Message(role="user", content="prefix-1"),
        Message(role="assistant", content="prefix-2"),
        Message(role="user", content="prefix-3"),
    ]
    suffix = [
        Message(role="assistant", content="suffix-1"),
        Message(role="user", content="suffix-2"),
    ]
    for m in prefix + suffix:
        append_jsonl(path, m)
    summary = Message(
        role="user", source="meta", content="[compaction]\nthe summary",
        compaction=True,
    )
    append_jsonl(
        path,
        {
            "type": "compaction-boundary",
            "seq": 7,
            "suffix_count": 2,
            "summary": summary.model_dump(by_alias=True),
        },
    )

    reloaded = session._load_transcript(session.ceo_name)
    # [summary, suffix-1, suffix-2]
    assert len(reloaded) == 3
    assert reloaded[0].compaction is True
    assert reloaded[0].content == "[compaction]\nthe summary"
    assert reloaded[1].content == "suffix-1"
    assert reloaded[2].content == "suffix-2"


# ---------------------------------------------------------------------------
# CompactionConfig — user-tunable context window / thresholds (§11.4)
# ---------------------------------------------------------------------------


def test_compaction_config_defaults_reproduce_legacy_thresholds():
    """A bare CompactionConfig must reproduce the previously-hardcoded numbers,
    so adding the config is purely additive (no behavior change by default)."""
    cfg = CompactionConfig()
    for model in ("kimi-k2.6", "claude-opus-4-8", "gpt-5-mini", "gemini-2.5-pro", "mystery"):
        # The `config=None` path and an explicit default config must agree.
        assert get_model_context_window(model) == get_model_context_window(model, cfg)
        assert get_auto_compact_threshold(model) == get_auto_compact_threshold(model, cfg)
        assert get_microcompact_threshold(model) == get_microcompact_threshold(model, cfg)
    # Spot-check the arithmetic against the module constants.
    window = get_model_context_window("claude-opus-4-8")
    assert window == 200_000
    assert get_auto_compact_threshold("claude-opus-4-8") == (
        window - RESERVED_SUMMARY_OUTPUT_TOKENS - AUTO_COMPACT_BUFFER_TOKENS
    )
    assert get_microcompact_threshold("claude-opus-4-8") == (
        get_auto_compact_threshold("claude-opus-4-8") - MICROCOMPACT_BUFFER_TOKENS
    )


def test_context_window_hard_override_applies_to_every_model():
    cfg = CompactionConfig(context_window_tokens=256_000)
    # Wins regardless of model name — even known ones.
    assert get_model_context_window("claude-opus-4-8", cfg) == 256_000
    assert get_model_context_window("anything-at-all", cfg) == 256_000
    # Thresholds derive from the overridden window.
    assert get_auto_compact_threshold("claude-opus-4-8", cfg) == (
        256_000 - RESERVED_SUMMARY_OUTPUT_TOKENS - AUTO_COMPACT_BUFFER_TOKENS
    )


def test_context_window_override_falls_back_to_builtin_table_when_unset():
    # With no override, known models resolve via the built-in table and unknown
    # ones fall through to DEFAULT_CONTEXT_WINDOW_TOKENS.
    cfg = CompactionConfig()
    assert get_model_context_window("gemini-2.5-pro", cfg) == 1_000_000
    assert get_model_context_window("who-knows", cfg) == DEFAULT_CONTEXT_WINDOW_TOKENS
    # An explicit window corrects an unknown/custom model.
    cfg2 = CompactionConfig(context_window_tokens=131_072)
    assert get_model_context_window("nemotron", cfg2) == 131_072


def test_buffers_and_summary_cap_are_configurable():
    cfg = CompactionConfig(
        context_window_tokens=100_000,
        reserved_summary_output_tokens=5_000,
        auto_compact_buffer_tokens=5_000,
        microcompact_buffer_tokens=10_000,
        summary_max_tokens=4_000,
    )
    assert get_auto_compact_threshold("mock", cfg) == 100_000 - 5_000 - 5_000
    assert get_microcompact_threshold("mock", cfg) == 90_000 - 10_000
    assert cfg.summary_max_tokens == 4_000


def test_min_threshold_floor_clamps_tiny_windows():
    # A window smaller than the buffers would push the derived threshold to
    # zero/negative; the floor keeps it sane.
    cfg = CompactionConfig(context_window_tokens=1_000)
    assert get_auto_compact_threshold("mock", cfg) == MIN_THRESHOLD_TOKENS
    assert get_microcompact_threshold("mock", cfg) == MIN_THRESHOLD_TOKENS
    # The floor itself is tunable.
    cfg2 = CompactionConfig(context_window_tokens=1_000, min_threshold_tokens=500)
    assert get_auto_compact_threshold("mock", cfg2) == 500
    assert get_microcompact_threshold("mock", cfg2) == 500


class _MaxTokensCapturingProvider(InferenceProvider):
    """Records the output cap of every request so a test can assert that the
    configured summary cap reaches the inference call — and that it rides on
    `max_completion_tokens`, never `max_tokens`."""

    def __init__(self, content: str):
        self._content = content
        self.completion_caps_seen: list[int | None] = []
        self.max_tokens_seen: list[int | None] = []

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self.completion_caps_seen.append(request.max_completion_tokens)
        self.max_tokens_seen.append(request.max_tokens)
        yield ContentDeltaChunk(text=self._content)
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


@pytest.mark.asyncio
async def test_session_compaction_config_sets_summary_max_tokens(tmp_path):
    """`SessionConfig.compaction.summary_max_tokens` must flow into the
    summarization request — the lib user's knob actually reaches the call.

    It must arrive as `max_completion_tokens` and leave `max_tokens` unset: some
    servers read `max_tokens` as a *total* input+output budget, so a large
    rendered prefix can leave zero generation budget and the summary dies as a
    zero-output `length` stop."""
    provider = _MaxTokensCapturingProvider(
        "<analysis>a</analysis><summary>s</summary>"
    )
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(summary_max_tokens=4_321),
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    await _summarize_prefix(session, session.ceo_name, _prefix_messages())
    assert provider.completion_caps_seen == [4_321]
    assert provider.max_tokens_seen == [None]


@pytest.mark.asyncio
async def test_session_compaction_config_microcompact_preserve_recent(tmp_path):
    """`microcompact_preserve_recent` set on the session config changes how many
    recent bulky results survive — proving SessionConfig.compaction reaches the
    runtime (default preserve is 5; here we set 2)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            micro_compact_threshold=1, microcompact_preserve_recent=2
        ),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    transcript = session.transcript(session.ceo_name)
    for i in range(6):
        transcript.extend(_bulky_tool_call_pair("data", tool="Read", call_id=f"c{i}"))

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True

    def _is_cleared(m: Message) -> bool:
        out = m.tool_result.output
        return isinstance(out, dict) and bool(out.get("_compacted"))

    tool_msgs = [m for m in transcript if m.role == "tool"]
    cleared = [m for m in tool_msgs if _is_cleared(m)]
    intact = [m for m in tool_msgs if not _is_cleared(m)]
    # 6 bulky results, preserve_recent=2 → 4 cleared, 2 kept. (The default of 5
    # would have cleared only 1.)
    assert len(cleared) == 4
    assert len(intact) == 2


@pytest.mark.asyncio
async def test_session_compaction_config_drives_derived_microcompact_threshold(tmp_path):
    """With threshold=None, microcompaction derives from session.config.compaction,
    so shrinking the configured context window lowers the trigger point."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            context_window_tokens=70_000, min_threshold_tokens=10_000
        ),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)

    derived = get_microcompact_threshold(session.state.model, config.compaction)
    transcript = session.transcript(session.ceo_name)
    payload = "x" * 16_000  # ≈ 4k tokens per bulky pair
    pair_count = (derived // 4_000) + 4
    for i in range(pair_count):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))

    # No explicit threshold — must come from the session's compaction config.
    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True
    assert session.agent(session.ceo_name).microcompacted is True


@pytest.mark.asyncio
async def test_compaction_config_absolute_auto_threshold_fires(tmp_path):
    """An absolute `auto_compact_threshold` on the config triggers full
    compaction without the caller passing `threshold=` — this is the
    consolidated home of the old `SessionConfig.auto_compact_threshold`."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("summary"))
    session = await harness.start_session(config=config)
    for i in range(20):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"m-{i}")
        )

    # No threshold= argument: the trigger must come from the config override.
    fired = await auto_compact_if_needed(session, session.ceo_name, preserve_recent_turns=4)
    assert fired is True


@pytest.mark.asyncio
async def test_compaction_config_absolute_micro_threshold_fires(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=1),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)
    transcript = session.transcript(session.ceo_name)
    for i in range(8):
        transcript.extend(_bulky_tool_call_pair("data", tool="Read", call_id=f"c{i}"))

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is True


@pytest.mark.asyncio
async def test_compaction_config_huge_absolute_threshold_suppresses_firing(tmp_path):
    """A large absolute threshold suppresses compaction even on a big transcript
    that would cross the derived point — proving the override is consulted (not
    merely additive)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(micro_compact_threshold=10_000_000),
    )
    harness = Harness(inference=MockInferenceProvider("done"))
    session = await harness.start_session(config=config)
    transcript = session.transcript(session.ceo_name)
    payload = "x" * 16_000  # ≈ 4k tokens per pair; 40 pairs ≈ 160k tokens
    for i in range(40):
        transcript.extend(_bulky_tool_call_pair(payload, tool="Read", call_id=f"c{i}"))

    fired = await microcompact_if_needed(session, session.ceo_name)
    assert fired is False


# ---------------------------------------------------------------------------
# Round-based boundary selection, suffix budget, mandate pinning (§11.4.2)
# ---------------------------------------------------------------------------


def _round(content_chars: int, idx: int, *, reasoning_chars: int = 0) -> list[Message]:
    """One API round: an assistant message + its paired tool result."""
    return [
        Message(
            role="assistant",
            content="a" * content_chars,
            reasoning=("r" * reasoning_chars) or None,
            tool_calls=[ToolCall(id=f"rc{idx}", name="Read", args={})],
        ),
        Message(
            role="tool",
            tool_call_id=f"rc{idx}",
            tool_result=ToolResult(ok=True, output={"i": idx}),
        ),
    ]


def test_select_boundary_keeps_requested_rounds_when_under_budget():
    transcript = [Message(role="user", content="mandate")]
    for i in range(10):
        transcript.extend(_round(40, i))
    boundary, strip_before = _select_compact_boundary(transcript, 4, 100_000)
    # 4 preserved rounds x 2 messages each; suffix opens on the assistant.
    assert strip_before == boundary  # nothing stripped when everything fits
    assert boundary == len(transcript) - 8
    assert transcript[boundary].role == "assistant"


def test_select_boundary_never_opens_suffix_on_tool_message():
    """Rounds group an assistant with ALL its trailing tool results (parallel
    tool calls), so no candidate boundary can split a pair."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(6):
        transcript.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id=f"p{i}-{j}", name="Read", args={}) for j in range(3)],
            )
        )
        for j in range(3):
            transcript.append(
                Message(
                    role="tool",
                    tool_call_id=f"p{i}-{j}",
                    tool_result=ToolResult(ok=True, output={"j": j}),
                )
            )
    for keep in range(1, 5):
        boundary, _ = _select_compact_boundary(transcript, keep, 100_000)
        assert transcript[boundary].role == "assistant"
        # Whole rounds only: 1 assistant + 3 tool results per kept round.
        assert len(transcript) - boundary == keep * 4


def test_select_boundary_shrinks_keep_until_suffix_fits_budget():
    transcript = [Message(role="user", content="mandate")]
    for i in range(6):
        transcript.extend(_round(4000, i))  # ~1000 tokens per round
    boundary, strip_before = _select_compact_boundary(transcript, 4, 2_500)
    assert strip_before == boundary
    # 4 and 3 rounds (~4k / ~3k tokens) blow the budget; 2 rounds fit.
    assert boundary == len(transcript) - 4
    assert transcript[boundary].role == "assistant"


def test_select_boundary_empty_suffix_when_one_round_over_budget():
    transcript = [Message(role="user", content="mandate")]
    for i in range(6):
        transcript.extend(_round(4000, i))
    boundary, strip_before = _select_compact_boundary(transcript, 4, 10)
    assert boundary == len(transcript)
    assert strip_before == boundary


def test_select_boundary_returns_none_when_transcript_too_short():
    # A single round (plus mandate): the only plans are a no-op and a
    # zero-round wipe. Waiting one round strictly improves the menu — the
    # next attempt keeps this round and summarizes it instead.
    transcript = [Message(role="user", content="mandate")] + _round(4000, 0)
    assert _select_compact_boundary(transcript, 4, 100_000) is None
    assert _select_compact_boundary([Message(role="user", content="m")], 4, 100_000) is None


def test_select_boundary_two_rounds_keeps_the_newest():
    """Two rounds must yield keep=1, never a zero-round wipe: the old
    MIN_PREFIX_ROUNDS = 2 floor made max_keep = 0 here, paying the summary
    call anyway and throwing away the whole tool chain with it."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(2):
        transcript.extend(_round(4000, i))
    boundary, strip_before = _select_compact_boundary(transcript, 4, 100_000)
    assert boundary == len(transcript) - 2
    assert strip_before == boundary
    assert transcript[boundary].role == "assistant"


def test_select_boundary_min_prefix_rounds_waits_instead_of_wiping():
    """A raised floor DEFERS compaction (None) until the prefix can span it —
    it never buys the old hardcoded-2 behavior of firing a zero-round wipe."""
    two = [Message(role="user", content="mandate")]
    for i in range(2):
        two.extend(_round(4000, i))
    assert _select_compact_boundary(two, 4, 100_000, min_prefix_rounds=2) is None
    three = two + _round(4000, 2)
    boundary, strip_before = _select_compact_boundary(
        three, 4, 100_000, min_prefix_rounds=2
    )
    # Fires once 3 rounds exist: 2 summarized, the newest kept.
    assert boundary == len(three) - 2
    assert strip_before == boundary


def test_select_boundary_min_prefix_rounds_clamps_to_one():
    """0 would allow an empty summarized prefix — a no-op that pays a summary
    call, recovers nothing, and re-fires forever. Clamped to the structural
    floor of 1."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(2):
        transcript.extend(_round(4000, i))
    boundary, _ = _select_compact_boundary(transcript, 4, 100_000, min_prefix_rounds=0)
    assert boundary == len(transcript) - 2  # same plan as the default floor


def test_select_boundary_two_giant_rounds_ladder_keeps_bare_newest():
    """A common 64k-window shape: mandate + exactly two reasoning rounds whose
    mass alone crosses the trigger threshold. With the ladder the newest
    round survives bare; without it the walk still degrades to the loud
    zero-round last resort."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(2):
        transcript.extend(_round(40, i, reasoning_chars=40_000))  # ~10k tokens each
    budget = 2_000  # far below one round's reasoning, far above a bare round
    b, s = _select_compact_boundary(transcript, 4, budget)
    assert b == len(transcript) and s == b  # flag off: last resort unchanged
    b, s = _select_compact_boundary(transcript, 4, budget, strip_reasoning=True)
    assert b == len(transcript) - 2  # round 2 survives
    assert s == len(transcript)  # bare
    assert transcript[b].role == "assistant"


def test_select_boundary_prices_resent_reasoning():
    """The suffix budget must price the RENDERED suffix — reasoning is
    preserved thinking (re-sent in full), so every kept round's reasoning
    counts against the budget and shrinks the keep."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(5):
        transcript.extend(_round(40, i, reasoning_chars=8_000))  # ~2k reasoning tokens each
    budget = 2_600  # fits 3 light rounds but only ONE turn's reasoning
    b, _ = _select_compact_boundary(transcript, 3, budget)
    assert len(transcript) - b == 1 * 2


def test_strip_ladder_prefers_bare_tool_chain_over_empty_suffix():
    """One round's re-sent reasoning alone exceeds the whole suffix budget:
    without the ladder the keep-walk degrades to an EMPTY suffix (the
    zero-round wall — the agent resumes from mandate + summary with no tool
    chain); with it, the full tool chain survives bare."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(6):
        transcript.extend(_round(40, i, reasoning_chars=40_000))  # ~10k tokens
    budget = 2_000  # far below one round's reasoning, far above 4 bare rounds
    b, s = _select_compact_boundary(transcript, 4, budget)
    assert b == len(transcript) and s == b  # flag off: today's zero-round wall
    b, s = _select_compact_boundary(transcript, 4, budget, strip_reasoning=True)
    assert len(transcript) - b == 4 * 2  # max keep survives
    assert s == len(transcript)  # every kept round bare
    assert transcript[b].role == "assistant"


def test_strip_ladder_least_stripping_that_fits_wins():
    """The ladder walks oldest-first: reasoning survives on the most recent
    kept rounds, and only as many older rounds are stripped as the budget
    demands. Round count dominates: keep=3 with two stripped is chosen even
    though keep=1 with reasoning intact would also fit."""
    transcript = [Message(role="user", content="mandate")]
    for i in range(6):
        transcript.extend(_round(40, i, reasoning_chars=4_000))  # ~1k tokens
    b, s = _select_compact_boundary(transcript, 3, 1_500, strip_reasoning=True)
    assert len(transcript) - b == 3 * 2
    # Two older kept rounds stripped; the newest keeps its reasoning.
    assert s == len(transcript) - 2


@pytest.mark.asyncio
async def test_strip_ladder_splice_strips_reasoning_and_clears_usage(tmp_path):
    """End-to-end through auto_compact_if_needed: the spliced transcript
    carries the strip plan (older kept rounds bare, newest keeps reasoning),
    every suffix assistant's stale `usage` is cleared (it measured the
    PRE-compaction rendering and would poison the estimate_tokens anchor,
    re-firing compaction into a zero-round wipe on the next turn), the
    boundary record persists the plan, and a reloaded transcript renders
    identically to the live one."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            auto_compact_threshold=1,
            auto_compact_preserve_suffix_max_tokens=2_500,
            strip_reasoning_to_fit=True,
        ),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)
    ceo = session.ceo_name

    await session.append_message(ceo, Message(role="user", content="mandate"))
    for i in range(10):
        await session.append_message(
            ceo,
            Message(
                role="assistant",
                content=f"msg-{i:02d}",
                reasoning="r" * 8_000,  # ~2k tokens: one round busts 2.5k alone
                tool_calls=[ToolCall(id=f"c{i}", name="Read", args={})],
                usage=MessageUsage(
                    input_tokens=50_000, output_tokens=2_000, reasoning_tokens=2_000
                ),
            ),
        )
        await session.append_message(
            ceo,
            Message(
                role="tool",
                tool_call_id=f"c{i}",
                tool_result=ToolResult(ok=True, output={"i": i}),
            ),
        )

    fired = await auto_compact_if_needed(session, ceo, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(ceo)
    # pinned mandate, then the summary, then 4 rounds x 2 messages.
    assert live[0].content == "mandate"
    assert live[1].compaction is True
    suffix = live[2:]
    assert len(suffix) == 8
    assistants = [m for m in suffix if m.role == "assistant"]
    assert [m.reasoning is None for m in assistants] == [True, True, True, False]
    assert all(m.usage is None for m in assistants)
    # Tool chain intact on every kept round, stripped or not.
    assert all(m.tool_calls for m in assistants)

    outcome = session.compaction_outcomes[-1]
    assert outcome["suffix_count"] == 8
    assert outcome["stripped_rounds"] == 3

    records = list(iter_jsonl(session.transcript_path(ceo)))
    boundary = [r for r in records if r.get("type") == "compaction-boundary"][0]
    assert boundary["suffix_count"] == 8
    assert boundary["strip_prefix_count"] == 6  # 3 stripped rounds x 2 messages

    reloaded = session._load_transcript(ceo)
    assert [m.content for m in reloaded] == [m.content for m in live]
    assert [m.reasoning for m in reloaded] == [m.reasoning for m in live]
    assert [m.usage for m in reloaded] == [m.usage for m in live]


@pytest.mark.asyncio
async def test_two_round_transcript_compacts_to_bare_tool_chain(tmp_path):
    """End-to-end for the residual 64k zero-round shape: mandate + exactly
    two reasoning rounds over the trigger threshold. The splice must keep the
    newest round (bare) behind the summary — not wipe the tool chain."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            auto_compact_threshold=1,
            auto_compact_preserve_suffix_max_tokens=2_500,
            strip_reasoning_to_fit=True,
        ),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)
    ceo = session.ceo_name

    await session.append_message(ceo, Message(role="user", content="mandate"))
    for i in range(2):
        await session.append_message(
            ceo,
            Message(
                role="assistant",
                content=f"msg-{i:02d}",
                reasoning="r" * 16_000,  # ~4k tokens: one round busts 2.5k alone
                tool_calls=[ToolCall(id=f"c{i}", name="Read", args={})],
            ),
        )
        await session.append_message(
            ceo,
            Message(
                role="tool",
                tool_call_id=f"c{i}",
                tool_result=ToolResult(ok=True, output={"i": i}),
            ),
        )

    fired = await auto_compact_if_needed(session, ceo, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(ceo)
    assert [m.role for m in live] == ["user", "user", "assistant", "tool"]
    assert live[0].content == "mandate"
    assert live[1].compaction is True
    assert live[2].content == "msg-01"  # the NEWEST round survived
    assert live[2].reasoning is None  # bare, via the ladder
    assert live[2].tool_calls  # tool chain intact
    outcome = session.compaction_outcomes[-1]
    assert outcome["suffix_count"] == 2
    assert outcome["stripped_rounds"] == 1


@pytest.mark.asyncio
async def test_session_compaction_config_min_prefix_rounds(tmp_path):
    """The knob defaults to 1 and plumbs through auto_compact_if_needed: with
    min_prefix_rounds=2 a 2-round over-threshold transcript WAITS, then fires
    once a third round lands."""
    assert CompactionConfig().min_prefix_rounds == 1
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            auto_compact_threshold=1,
            min_prefix_rounds=2,
        ),
    )
    harness = Harness(inference=MockInferenceProvider("summary text"))
    session = await harness.start_session(config=config)
    ceo = session.ceo_name

    await session.append_message(ceo, Message(role="user", content="mandate"))
    for i in range(2):
        await session.append_message(
            ceo,
            Message(
                role="assistant",
                content=f"msg-{i:02d}",
                tool_calls=[ToolCall(id=f"c{i}", name="Read", args={})],
            ),
        )
        await session.append_message(
            ceo,
            Message(
                role="tool",
                tool_call_id=f"c{i}",
                tool_result=ToolResult(ok=True, output={"i": i}),
            ),
        )
    assert await auto_compact_if_needed(session, ceo, preserve_recent_turns=4) is False

    await session.append_message(
        ceo,
        Message(
            role="assistant",
            content="msg-02",
            tool_calls=[ToolCall(id="c2", name="Read", args={})],
        ),
    )
    await session.append_message(
        ceo,
        Message(
            role="tool",
            tool_call_id="c2",
            tool_result=ToolResult(ok=True, output={"i": 2}),
        ),
    )
    assert await auto_compact_if_needed(session, ceo, preserve_recent_turns=4) is True
    live = session.transcript(ceo)
    # mandate, summary, newest round kept — rounds 0-1 summarized.
    assert [m.role for m in live] == ["user", "user", "assistant", "tool"]
    assert live[2].content == "msg-02"


def test_estimate_counts_anchor_reasoning():
    """The anchor turn's reasoning is re-sent (preserved thinking), so the
    whole output counts. Reasoning the provider counted but never returned
    as text (`reasoning=None`) can't be re-sent and is subtracted."""
    def _transcript(reasoning):
        return [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="answer",
                reasoning=reasoning,
                usage=MessageUsage(
                    input_tokens=1_000, output_tokens=50_000, reasoning_tokens=49_000
                ),
            ),
        ]
    assert estimate_tokens(_transcript("R" * 400)) == 1_000 + 50_000
    assert estimate_tokens(_transcript(None)) == 1_000 + 1_000


def test_estimate_prices_trailing_reasoning():
    anchor = Message(
        role="assistant",
        content="x",
        usage=MessageUsage(input_tokens=1_000, output_tokens=100),
    )
    def _est(trailing):
        return estimate_tokens([Message(role="user", content="hi"), anchor, trailing])
    with_r = _est(Message(role="assistant", content="", reasoning="r" * 4_000))
    without_r = _est(Message(role="assistant", content=""))
    # The trailing turn's reasoning (~1000 tokens) is re-sent, so it's priced.
    assert with_r - without_r == 4_000 // 4


def test_estimate_fallback_includes_system_prompt():
    transcript = [Message(role="user", content="u" * 4_000)]
    est = estimate_tokens(transcript, system_prompt="s" * 4_000)
    assert est == 1_000 + _MESSAGE_OVERHEAD_TOKENS + 1_000


def test_get_suffix_token_budget_derives_from_window():
    assert get_suffix_token_budget("claude-opus-4-8") == 200_000 // SUFFIX_BUDGET_WINDOW_DIVISOR
    assert get_suffix_token_budget("mock") == DEFAULT_CONTEXT_WINDOW_TOKENS // SUFFIX_BUDGET_WINDOW_DIVISOR
    # Explicit config value wins outright.
    cfg = CompactionConfig(auto_compact_preserve_suffix_max_tokens=1_234)
    assert get_suffix_token_budget("claude-opus-4-8", cfg) == 1_234
    # An absolute trigger threshold must NOT collapse the derived budget.
    cfg = CompactionConfig(auto_compact_threshold=1)
    assert get_suffix_token_budget("mock", cfg) == DEFAULT_CONTEXT_WINDOW_TOKENS // SUFFIX_BUDGET_WINDOW_DIVISOR


def test_get_suffix_token_budget_small_window_capped_by_threshold():
    """On small windows the fixed reservations dominate: window//4 alone
    would put the post-compaction restart point AT the trigger threshold
    (65k: threshold 32k, restart ~33k) and compaction would thrash. The
    threshold-aware cap must bind instead."""
    cfg = CompactionConfig(context_window_tokens=65_000)
    threshold = get_auto_compact_threshold("nemotron", cfg)  # 65k - 33k = 32k
    budget = get_suffix_token_budget("nemotron", cfg)
    assert budget == threshold // 2 - cfg.summary_max_tokens == 4_000
    assert budget < 65_000 // SUFFIX_BUDGET_WINDOW_DIVISOR

    # Windows so small nothing fits clamp to zero (always-empty suffix +
    # loud warning) rather than going negative.
    cfg = CompactionConfig(context_window_tokens=40_000)
    assert get_suffix_token_budget("nemotron", cfg) == 0


def test_suffix_budget_restart_invariant_across_windows():
    """The reason the derivation exists: after a compaction, the prompt
    (summary output cap + suffix budget) must restart with real headroom
    below the derived trigger threshold, on EVERY window size — otherwise
    compaction re-fires immediately."""
    for window in (65_000, 128_000, 200_000, 1_000_000):
        cfg = CompactionConfig(context_window_tokens=window)
        threshold = get_auto_compact_threshold("any-model", cfg)
        budget = get_suffix_token_budget("any-model", cfg)
        restart = budget + cfg.summary_max_tokens
        assert restart <= 0.75 * threshold, (
            f"window={window}: restart point {restart} leaves <25% headroom "
            f"below threshold {threshold}"
        )


async def test_compaction_pins_mandate_and_reloads(tmp_path):
    """§11.4.2: the opening user message (the mandate) survives compaction
    verbatim ahead of the summary — in memory AND across a disk reload."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>the summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    mandate = "You are CEO. Build the thing.\n\nBegin work on your mission now."
    await session.append_message(
        session.ceo_name, Message(role="user", source="env", content=mandate)
    )
    for i in range(20):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"msg-{i:02d}")
        )

    fired = await auto_compact_if_needed(session, session.ceo_name, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(session.ceo_name)
    assert live[0].role == "user"
    assert live[0].content == mandate
    assert live[0].compaction is False
    assert live[1].compaction is True
    assert [m.content for m in live[2:]] == [f"msg-{i:02d}" for i in range(16, 20)]

    records = list(iter_jsonl(session.transcript_path(session.ceo_name)))
    boundary = next(r for r in records if r.get("type") == "compaction-boundary")
    assert [p["content"] for p in boundary["pinned"]] == [mandate]

    reloaded = session._load_transcript(session.ceo_name)
    assert [m.content for m in reloaded] == [m.content for m in live]


async def test_compaction_empty_suffix_keeps_mandate_and_reloads(tmp_path):
    """A suffix budget no round can fit -> keep zero rounds. The live
    transcript becomes [mandate, summary] and reloads identically
    (suffix_count == 0 must not resurrect the whole file)."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(
            auto_compact_threshold=1,
            auto_compact_preserve_suffix_max_tokens=0,
        ),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>everything summarized</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    await session.append_message(
        session.ceo_name, Message(role="user", source="env", content="the mandate")
    )
    for i in range(10):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"msg-{i}")
        )

    fired = await auto_compact_if_needed(session, session.ceo_name, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(session.ceo_name)
    assert [m.content for m in live] == [
        "the mandate",
        "[compaction]\neverything summarized",
    ]
    records = list(iter_jsonl(session.transcript_path(session.ceo_name)))
    boundary = next(r for r in records if r.get("type") == "compaction-boundary")
    assert boundary["suffix_count"] == 0

    reloaded = session._load_transcript(session.ceo_name)
    assert [m.content for m in reloaded] == [m.content for m in live]


async def test_short_reasoning_heavy_transcript_still_compacts(tmp_path):
    """Regression: the old `len(transcript) <= preserve_recent_turns` guard
    blocked compaction on short-but-huge transcripts — exactly the
    preserved-thinking scenario the dynamic suffix was built for."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>s</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    await session.append_message(
        session.ceo_name, Message(role="user", source="env", content="mandate")
    )
    for i in range(3):
        await session.append_message(
            session.ceo_name,
            Message(role="assistant", content=f"turn-{i}", reasoning="r" * 200),
        )

    # 4 messages total, preserve_recent_turns=8: the old guard returned False.
    fired = await auto_compact_if_needed(session, session.ceo_name, preserve_recent_turns=8)
    assert fired is True
    live = session.transcript(session.ceo_name)
    assert live[0].content == "mandate"
    assert live[1].compaction is True


class _AppendDuringSummaryProvider(InferenceProvider):
    """Simulates a message landing (e.g. externally injected user input) while the
    summary call is in flight."""

    def __init__(self):
        self.session = None
        self.agent_name = None

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        await self.session.append_message(
            self.agent_name, Message(role="user", content="landed-mid-summary")
        )
        yield ContentDeltaChunk(text="<analysis>a</analysis><summary>s</summary>")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


async def test_message_appended_during_summary_survives_compaction(tmp_path):
    """Regression: the splice must re-slice the suffix from the live list
    under the transcript lock. A snapshot taken before the (long) summary
    await silently dropped anything appended during the call."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=1),
    )
    provider = _AppendDuringSummaryProvider()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)
    provider.session = session
    provider.agent_name = session.ceo_name

    for i in range(20):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"msg-{i:02d}")
        )

    fired = await auto_compact_if_needed(session, session.ceo_name, preserve_recent_turns=4)
    assert fired is True

    live = session.transcript(session.ceo_name)
    assert live[-1].content == "landed-mid-summary"
    # And the on-disk replay agrees (the record landed before the boundary
    # line, inside suffix_count).
    reloaded = session._load_transcript(session.ceo_name)
    assert [m.content for m in reloaded] == [m.content for m in live]


# ---------------------------------------------------------------------------
# Forced compaction + main-call prompt-too-long recovery (§11.4.2)
# ---------------------------------------------------------------------------


async def test_force_compaction_bypasses_threshold_gate(tmp_path):
    """`force=True` must compact even when the estimate sits far below the
    trigger — it exists for the loop's PTL recovery, where the provider has
    already proven the estimate wrong."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
        compaction=CompactionConfig(auto_compact_threshold=10_000_000),
    )
    provider = _ScriptedSummaryProvider(
        ["<analysis>a</analysis><summary>forced summary</summary>"]
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)
    for i in range(12):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"m-{i}")
        )

    assert await auto_compact_if_needed(session, session.ceo_name) is False
    assert await auto_compact_if_needed(session, session.ceo_name, force=True) is True
    live = session.transcript(session.ceo_name)
    assert any(m.compaction for m in live)


class _PtlUntilCompactedProvider(InferenceProvider):
    """Main-turn calls raise prompt-too-long while the prompt is still big;
    once compaction shrinks it they succeed. Summary calls (recognized by the
    compaction system prompt) always return a valid summary."""

    def __init__(self, big_prompt_messages: int):
        self._big = big_prompt_messages
        self.main_calls = 0
        self.ptl_raised = 0
        self.summary_calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        system = request.messages[0].content or ""
        if "detailed summary" in system:
            self.summary_calls += 1
            yield ContentDeltaChunk(
                text="<analysis>a</analysis><summary>the recovery summary</summary>"
            )
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        self.main_calls += 1
        if len(request.messages) >= self._big:
            self.ptl_raised += 1
            raise RuntimeError(
                "prompt too long: 141000 tokens > maximum context length"
            )
        yield ContentDeltaChunk(text="recovered-answer")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


async def test_main_call_ptl_forces_compaction_and_retries_turn(tmp_path):
    """Regression for the worst failure mode: a provider 413 on the MAIN
    inference call used to kill the agent's run loop permanently. The loop
    must treat it as ground truth, force a compaction (bypassing the
    estimate gate, which by construction said everything was fine), and
    retry the turn once against the shrunken prompt."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="the original mandate",
        # ~100-token rounds against a 250-token suffix budget -> compaction
        # keeps ~2 rounds, dropping the main prompt well below the provider's
        # simulated PTL trip point.
        compaction=CompactionConfig(auto_compact_preserve_suffix_max_tokens=250),
    )
    provider = _PtlUntilCompactedProvider(big_prompt_messages=8)
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    for i in range(10):
        await session.append_message(
            session.ceo_name, Message(role="assistant", content=f"turn-{i} " + "x" * 400)
        )
    await session.append_message(
        session.ceo_name, Message(role="user", content="continue")
    )

    await harness.run_to_completion(session, max_wall_s=30)

    live = session.transcript(session.ceo_name)
    # The PTL fired once, forced exactly one compaction, and the retried
    # turn produced a real assistant answer — the agent did NOT die.
    assert provider.ptl_raised == 1
    assert provider.summary_calls == 1
    assert provider.main_calls == 2
    assert any(m.compaction for m in live)
    assert any(
        m.role == "assistant" and (m.content or "").strip() == "recovered-answer"
        for m in live
    )
    # The mandate was pinned through the forced compaction.
    assert live[0].role == "user"
    assert live[0].content == "the original mandate"


def _rendered_tokens(*messages: Message) -> int:
    from agentorg.compaction import _rendered_message_cost, _rendered_view

    return sum(
        _rendered_message_cost(m, pm)
        for m, pm in zip(messages, _rendered_view(list(messages)))
    )


def test_estimate_skips_anchor_reported_net_of_cache():
    """A server that reports `input_tokens` net of prefix-cache hits
    (cached_tokens left at 0) produces a report that is ~1% of the real
    prompt on every warm turn. Such a report must not anchor the estimate;
    the walk continues to the older, cache-cold turn whose count is real.
    E.g. a turn reporting a few hundred tokens against a ~98k prompt."""
    from agentorg.compaction import estimate_tokens_detail

    mandate = Message(role="user", content="m" * 4_000)
    cold = Message(
        role="assistant",
        content="c" * 400,
        tool_calls=[ToolCall(id="t1", name="bash", args={"command": "ls"})],
        usage=MessageUsage(input_tokens=1_500, output_tokens=120),
    )
    result_1 = Message(
        role="tool", tool_call_id="t1", tool_result=ToolResult(ok=True, output="r" * 8_000)
    )
    warm = Message(
        role="assistant",
        content="w" * 400,
        tool_calls=[ToolCall(id="t2", name="bash", args={"command": "ls"})],
        usage=MessageUsage(input_tokens=40, output_tokens=120),
    )
    result_2 = Message(
        role="tool", tool_call_id="t2", tool_result=ToolResult(ok=True, output="r" * 8_000)
    )
    transcript = [mandate, cold, result_1, warm, result_2]

    detail = estimate_tokens_detail(transcript)
    assert detail.anchor_index == 1  # the cache-cold turn, not the last assistant
    ratio = 1_500 / _rendered_tokens(mandate)
    assert detail.calibration == pytest.approx(ratio)
    trailing = _rendered_tokens(result_1, warm, result_2)
    assert detail.tokens == 1_500 + 120 + int(trailing * ratio)
    # The old estimator anchored on the warm turn's 40 tokens and saw only
    # the one result appended after it.
    old_estimate = 40 + 120 + _rendered_tokens(result_2)
    assert detail.tokens > 3 * old_estimate


def test_estimate_calibrates_trailing_by_anchor_density():
    """A trusted anchor measures tokens per four characters for this model on
    this content. Within [CALIBRATION_MIN, CALIBRATION_MAX] that density is
    applied to the chars/4 part of everything after the anchor; a ratio
    outside the band (a tiny transcript behind a large tool schema) is not a
    density and leaves the trailing cost unscaled."""
    from agentorg.compaction import CALIBRATION_MAX, estimate_tokens_detail

    mandate = Message(role="user", content="m" * 4_000)
    rendered_prompt = _rendered_tokens(mandate)
    trailing_msg = Message(
        role="tool", tool_call_id="t1", tool_result=ToolResult(ok=True, output="r" * 8_000)
    )
    trailing = _rendered_tokens(trailing_msg)

    def _detail(input_tokens: int):
        anchor = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="t1", name="bash", args={"command": "ls"})],
            usage=MessageUsage(input_tokens=input_tokens, output_tokens=10),
        )
        return estimate_tokens_detail([mandate, anchor, trailing_msg])

    dense = _detail(int(rendered_prompt * 1.5))
    assert dense.calibration == pytest.approx(1.5, abs=1e-3)
    assert dense.tokens == int(rendered_prompt * 1.5) + 10 + int(trailing * dense.calibration)

    implausible = _detail(int(rendered_prompt * (CALIBRATION_MAX + 3)))
    assert implausible.calibration == 1.0
    assert implausible.tokens == int(rendered_prompt * (CALIBRATION_MAX + 3)) + 10 + trailing


def test_estimate_falls_back_to_rendered_when_no_report_is_trusted():
    """Every usage report below the gate → no anchor; the estimate is the
    pure rendered cost plus the system prompt, exactly the first-turn path."""
    from agentorg.compaction import estimate_tokens_detail

    mandate = Message(role="user", content="m" * 4_000)
    warm = Message(
        role="assistant",
        content="w" * 400,
        usage=MessageUsage(input_tokens=40, output_tokens=120),
    )
    detail = estimate_tokens_detail([mandate, warm], system_prompt="s" * 4_000)
    assert detail.anchor_index is None
    assert detail.calibration == 1.0
    assert detail.tokens == 1_000 + _rendered_tokens(mandate, warm)


def test_estimate_density_ignores_exact_reasoning_in_the_denominator():
    """The anchor's density is tokens-per-four-chars on the chars/4 part only.

    Provider-counted reasoning tokens in the prompt are already tokens, so
    leaving them in the denominator drags the measured density toward 1.0 and
    under-scales the trailing segment on reasoning-heavy transcripts.
    """
    from agentorg.compaction import estimate_tokens_detail

    # 4000 chars of content (→ 1000 chars/4) with reasoning the provider
    # counted exactly at 1000 tokens, then a truthful report at density 1.4:
    # 1.4 * 1000 + 1000 = 2400.
    prefix = Message(
        role="assistant",
        content="x" * 4000,
        reasoning="r" * 10,
        usage=MessageUsage(input_tokens=10, output_tokens=0, reasoning_tokens=1000),
    )
    anchor = Message(
        role="assistant",
        content="y" * 400,
        usage=MessageUsage(input_tokens=2400 + _MESSAGE_OVERHEAD_TOKENS * 1,
                           output_tokens=0, reasoning_tokens=0),
    )
    detail = estimate_tokens_detail([prefix, anchor])
    # (2404 - 1000) / (1000 + 4) == 1.398…, not (2404) / (2004) == 1.19…
    assert 1.39 <= detail.calibration <= 1.41, detail.calibration


@pytest.mark.asyncio
async def test_summary_unavailable_does_not_embed_provider_error_text(tmp_path):
    """The unavailable-summary placeholder replaces the agent's context, so the
    exception text it carries is capped. Providers can serialize the whole
    request into the error, and a several-hundred-KB "summary" would make the
    post-compaction prompt larger than the one it replaced. The tail is kept:
    the head of such an error is the request, the cause is at the end."""

    request_dump = (
        "\nRequest:\nCRITICAL: Respond with TEXT ONLY." + "x" * 50_000
        + "\nBadRequestError(maximum context length)"
    )

    class _AlwaysPTLWithRequestDump(InferenceProvider):
        def __init__(self):
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            raise ValueError(request_dump)
            yield  # unreachable, marks this as async generator

        def model_ids(self):
            return ["mock"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    provider = _AlwaysPTLWithRequestDump()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)

    prefix: list[Message] = []
    for i in range(20):
        prefix.append(Message(role="user", content=f"q-{i}"))
        prefix.append(Message(role="assistant", content=f"a-{i}"))

    result = await _summarize_prefix(session, session.ceo_name, prefix)
    # PTL retries exhaust, then the error falls through to inference-error.
    assert result.outcome == "inference-error"
    assert result.ptl_truncations == MAX_PROMPT_TOO_LONG_RETRIES
    single_round = [Message(role="assistant", content="just one round")]
    ptl_failed = await _summarize_prefix(session, session.ceo_name, single_round)
    assert ptl_failed.outcome == "ptl-failed"

    for r in (result, ptl_failed):
        assert r.summary.startswith("[summary unavailable: ValueError: ...")
        assert r.summary.endswith("\nBadRequestError(maximum context length)]")
        assert "CRITICAL" not in r.summary
        assert len(r.summary) < SUMMARY_ERROR_MAX_CHARS + 100
