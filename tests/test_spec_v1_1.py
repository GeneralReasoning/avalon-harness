"""Tests for SPEC v1.1 additions:

- §4.4 session.aborted event with reason
- §A.4/§A.5 tagged-but-not-running surfacing
- §A.4 WaitForMessages activity signal in timeout marker
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from agentorg.events import WakeBroker
from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.tools.context import ToolContext
from agentorg.tools.org_tools import (
    PostToChannelTool,
    WaitForMessagesTool,
)


class _StubProvider(InferenceProvider):
    """No-op provider — every stream call yields a tool-less assistant turn."""

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield ContentDeltaChunk(text="ok")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


def _make_config(tmp_path, **kwargs):
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=2,
        mode="default",
        headless=True,
        permission_policy="bypass",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# §4.4 session.aborted event with reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_abort_emits_event_once_with_reason(tmp_path):
    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))

    session.abort("test-reason")
    await session.emit_aborted_if_needed()
    # Idempotent: second call is a no-op.
    await session.emit_aborted_if_needed()

    # Pull events off disk to assert
    from agentorg.storage import iter_jsonl

    records = list(iter_jsonl(session.paths.events_jsonl))
    abort_events = [r for r in records if r.get("type") == "session.aborted"]
    assert len(abort_events) == 1
    assert abort_events[0]["payload"]["reason"] == "test-reason"


@pytest.mark.asyncio
async def test_session_abort_reason_is_first_wins(tmp_path):
    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))

    session.abort("first")
    session.abort("second")  # late call must not overwrite
    await session.emit_aborted_if_needed()

    from agentorg.storage import iter_jsonl

    records = list(iter_jsonl(session.paths.events_jsonl))
    abort_events = [r for r in records if r.get("type") == "session.aborted"]
    assert len(abort_events) == 1
    assert abort_events[0]["payload"]["reason"] == "first"


# ---------------------------------------------------------------------------
# §A.4 / §A.5 tagged-but-not-running surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_to_channel_surfaces_tagged_but_not_running(tmp_path):
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))
    await session.channels.load()
    await session.channels.create_channels(
        [{"name": "general", "default_subscription": "light"}],
        live_agent_names=session.live_agent_names(),
        external_labels=set(),
    )
    real = await spawn_org_agent(
        session, OrgAgentSpec(role="Tester", description="x")
    )

    ceo = session.agent(session.ceo_name)
    tool = PostToChannelTool()
    ctx = ToolContext(
        agent=ceo,
        session=session,
        args={
            "channel": "general",
            "content": f"hi @{real.name} and @GhostAgent please verify",
        },
        call_id="c1",
    )
    result = await tool.execute(ctx)
    assert result.ok
    out = result.output
    assert real.name in out.get("tagged", [])
    assert "GhostAgent" in out.get("tagged_but_not_running", [])


# ---------------------------------------------------------------------------
# §4.5.1 — time-free wait gate: WakeBroker.park_and_wait + quiescent marker
# ---------------------------------------------------------------------------


def test_wake_broker_park_and_wait_tracks_waiting_set():
    """An agent inside `park_and_wait` shows up in `waiting_agents()` and
    is removed on wake. No timeout argument."""

    async def driver():
        broker = WakeBroker()
        assert broker.waiting_agents() == set()

        async def waiter():
            await broker.park_and_wait("Alice")

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert broker.is_waiting("Alice")
        broker.wake("Alice")
        await task
        assert not broker.is_waiting("Alice")

    asyncio.run(driver())


def test_wake_broker_on_parked_callback_runs_under_lock():
    """`park_and_wait(on_parked=…)` runs the callback synchronously after
    registering the agent in the waiting set — the callback can observe
    its own presence."""

    async def driver():
        broker = WakeBroker()
        observed: dict[str, set[str]] = {}

        def _cb():
            observed["snapshot"] = broker.waiting_agents()

        async def waiter():
            await broker.park_and_wait("Alice", on_parked=_cb)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        # Callback observed Alice already in the waiting set.
        assert observed.get("snapshot") == {"Alice"}
        broker.wake("Alice")
        await task

    asyncio.run(driver())


@pytest.mark.asyncio
async def test_activity_card_runtime_is_the_only_busy_signal(tmp_path):
    """A peer mid-turn (not parked) reads `runtime: active` and the card has
    no `state` field that could contradict it (it would say "idle")."""
    from agentorg.org import spawn_org_agent
    from agentorg.tools.wait_tools import _build_peer_cards
    from agentorg.types import OrgAgentSpec

    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))
    busy = await spawn_org_agent(session, OrgAgentSpec(role="Busy", description="x"))
    parked = await spawn_org_agent(session, OrgAgentSpec(role="Parked", description="x"))
    session.wakes._waiting.add(parked.name)

    cards = {c["name"]: c for c in _build_peer_cards(session, session.ceo_name)}
    assert cards[busy.name]["runtime"] == "active"
    assert cards[parked.name]["runtime"] == "idle"
    for card in cards.values():
        assert "state" not in card


@pytest.mark.asyncio
async def test_list_channels_marks_fired_traces_without_agent_state(tmp_path):
    """ListChannels virtual entries expose `fired: bool`, not the agent
    record's `state` (which only ever reads idle / fired)."""
    from agentorg.org import spawn_org_agent
    from agentorg.tools.comms_tools import ListChannelsTool
    from agentorg.types import OrgAgentSpec

    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))
    live = await spawn_org_agent(session, OrgAgentSpec(role="Live", description="x"))
    gone = await spawn_org_agent(session, OrgAgentSpec(role="Gone", description="x"))
    gone.state = "fired"

    ctx = ToolContext(
        agent=session.agent(session.ceo_name), session=session, args={}, call_id="c1"
    )
    result = await ListChannelsTool().execute(ctx)
    assert result.ok
    virtual = {e["name"]: e for e in result.output["channels"] if e["is_virtual"]}
    assert virtual[live.name]["fired"] is False
    assert virtual[gone.name]["fired"] is True
    assert virtual[session.ceo_name]["fired"] is False
    assert virtual["self"]["alias_for"] == session.ceo_name
    for entry in virtual.values():
        assert "agent_state" not in entry


@pytest.mark.asyncio
async def test_waitformessages_quiescent_marker_exposes_activity_signal(tmp_path):
    """SPEC §4.5.1 checkpoint 1: when every live org agent is parked, the
    CEO's WaitForMessages returns `quiescent: true` with a per-peer activity
    card carrying only time-free fields (no inflight_*, no last_assistant_ts).
    """
    from agentorg.org import spawn_org_agent
    from agentorg.types import OrgAgentSpec

    harness = Harness(inference=_StubProvider())
    session = await harness.start_session(config=_make_config(tmp_path))
    agent = await spawn_org_agent(
        session, OrgAgentSpec(role="Tester", description="x")
    )

    # Simulate the agent being parked in its wait.
    session.wakes._waiting.add(agent.name)
    assert session.is_org_quiescent()

    tool = WaitForMessagesTool()
    ctx = ToolContext(
        agent=session.agent(session.ceo_name),
        session=session,
        args={},
        call_id="c1",
    )
    result = await tool.execute(ctx)
    assert result.ok
    out = result.output
    assert out.get("quiescent") is True
    assert "timeout" not in out
    assert "poked" not in out  # harness no longer pokes idle org agents

    # Activity card present and carries only time-free fields.
    assert isinstance(out["live_peers"], list)
    assert len(out["live_peers"]) == 1
    peer = out["live_peers"][0]
    assert peer["name"] == agent.name
    assert peer["runtime"] == "idle"
    assert "recent_tool_calls" in peer
    assert "turn_count" in peer
    for forbidden in (
        "last_assistant_ts",
        "inflight_output_tokens_approx",
        "inflight_since_ts",
        "inflight_last_chunk_ts",
    ):
        assert forbidden not in peer, f"activity card must not expose {forbidden}"
    # The agent record's `state` is always "idle" on a live peer and models
    # read it over `runtime` — the card must not carry it.
    assert "state" not in peer

    # The harness no longer DMs idle peers — driving the org is the CEO's job.
    pending = await session.inbox.peek_count(agent.name)
    assert pending == 0
