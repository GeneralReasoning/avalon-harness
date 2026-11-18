"""Wait / Report tools: park-for-activity and structured CEO reports (§4.5)."""

from __future__ import annotations

import asyncio
from typing import Any

from agentorg.tools.context import (
    ToolContext,
    err_result,
    ok_result,
    rejected_result,
)
from agentorg.tools.org_tools import MAX_MESSAGE_BYTES, _too_long_error
from agentorg.types import (
    Attachment,
    InboxItem,
    ReportPayload,
    ToolDefinition,
    ToolResult,
)


class WaitForMessagesTool:
    definition = ToolDefinition(
        name="WaitForMessages",
        description=(
            "Park until new activity arrives — a DM, a channel notification "
            "you're subscribed to, a tag, a reaction on your post, or (CEO "
            "only) an org-agent report. The tool takes no arguments and has "
            "no timeout: it returns when something actually happens. "
            "Special CEO-only return: if every live org agent is itself "
            "parked when you call (or while you're parked), the tool returns "
            "a quiescent marker (`quiescent: true`) immediately — no peer can "
            "wake you and you must act."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        name = ctx.agent.name
        is_ceo = ctx.agent.kind == "ceo"

        while True:
            # A fired agent must never (re-)park. `fire_org_agent` wakes the
            # parked task precisely so this loop can unwind — without this
            # check the woken loop drains an empty inbox and parks again
            # forever, leaving a zombie task that keeps the supervisor's
            # in-flight set non-empty (session can never end).
            if ctx.agent.state == "fired":
                return ok_result({"fired": True})

            # Drain inbox first. WaitForMessages deliberately is NOT
            # budget-capped (unlike the per-tool-output-cap in §5.8): the
            # loop's top-of-iteration drain would re-feed any deferred items
            # as a user message on the very next turn, so capping here would
            # just split the same content across two messages without
            # bounding what the agent ultimately sees. Inbox-driven content
            # is bounded at the source by MAX_MESSAGE_BYTES (§7.9).
            items = await session.inbox.drain(name)
            if items:
                return ok_result(_serialize_drained(session, items, name))

            if session.stop_requested():
                return ok_result({"aborted": True})

            # SPEC §4.5.1 checkpoint 1 — CEO short-circuit.
            if is_ceo:
                if not session.live_org_agents():
                    return ok_result(_no_live_peers_marker())
                if session.is_org_quiescent():
                    await session.events.emit(
                        "agent.quiesced",
                        {"ceo": name, "source": "ceo-entry"},
                        name=name,
                    )
                    return ok_result(_quiescent_marker(session, name))

            # SPEC §4.5.1 checkpoint 2 — org-agent quiescence trigger,
            # composed atomically with park registration. The `on_parked`
            # callback runs synchronously under WakeBroker's lock with
            # `name` already in the waiting set, so the issubset check
            # is race-free against this same agent's registration.
            if is_ceo:
                await session.wakes.park_and_wait(name)
            else:
                ceo_name = session.ceo_name

                def _on_parked() -> None:
                    live = {a.name for a in session.live_agents()}
                    if ceo_name not in live:
                        return
                    if live.issubset(session.wakes.waiting_agents()):
                        asyncio.create_task(
                            session.signal_quiescence_to_ceo(source="org-trigger")
                        )

                await session.wakes.park_and_wait(name, on_parked=_on_parked)

            if session.stop_requested():
                drained = await session.inbox.drain(name)
                if drained:
                    return ok_result(_serialize_drained(session, drained, name))
                return ok_result({"aborted": True})
            # Spurious wake or quiescence wake → loop. Next iteration
            # drains the inbox (catching the org-trigger meta item) or
            # short-circuits via checkpoint 1.


def _no_live_peers_marker() -> dict[str, Any]:
    return {
        "no_live_peers": True,
        "live_peers": [],
        "suggestions": [
            "No org agents are live — waiting has nothing to wake it.",
            "If in org-plan mode: write the plan and call ExitOrgPlanMode.",
            "If running solo: take the next action directly, or submit.",
            "Use HireOrgAgent to spawn a specialist if more help is needed.",
        ],
    }


def _quiescent_marker(session, viewer_name: str) -> dict[str, Any]:
    """Build the SPEC §4.5.1 quiescent marker — the time-free analogue
    of the legacy timeout marker. Carries `quiescent: true`, the
    per-peer activity card for each live peer, and the structural
    action menu."""
    live_peers = _build_peer_cards(session, viewer_name)
    return {
        "quiescent": True,
        "live_peers": live_peers,
        "suggestions": [
            "Every live org agent is currently parked — no peer can wake you. Act:",
            "PostToChannel an update, decision, or question to the team.",
            "SendMessage a teammate with new direction or context they need.",
            "FireOrgAgent on a peer whose work is done or clearly stuck.",
            "Take a direct action yourself (read, search, write to the archive, submit).",
            "Re-calling WaitForMessages now would short-circuit immediately on the same rule.",
            "Read each peer's `turn_delta` (turns since you last looked) — positive but no inbound posts means a silent worker. `runtime` is the busy/idle signal: `turn_delta` 0 with `runtime: active` means mid-generation or mid-tool (leave them); only `runtime: idle` with `turn_delta` 0 means genuinely idle.",
        ],
    }


def _serialize_drained(session, items, recipient: str) -> dict[str, Any]:
    from agentorg.inbox import render_inbox

    rendered = render_inbox(items, recipient=recipient)
    return {
        # `seq` is the inbox-arrival counter — internal ordering metadata
        # that the model conflates with `msg_idx` (per-channel post index).
        # Items are already in seq order; the actual post idx lives in
        # `msg_key` ("<channel>:<idx>") and the rendered header surfaces it.
        "items": [i.model_dump(by_alias=True, exclude={"seq"}) for i in items],
        "rendered": rendered,
        # Always include peer cards so the recipient (esp. the CEO) sees the
        # `turn_delta` for each live peer alongside the delivered messages.
        # Cheap (~one short dict per peer) and the highest-value signal for
        # "what has everyone been doing while I waited".
        "live_peers": _build_peer_cards(session, recipient),
    }


def _build_peer_cards(session, viewer_name: str) -> list[dict[str, Any]]:
    """Build per-peer activity cards and snapshot each peer's turn_count
    into the viewer's `_peer_turn_count_seen` map so the next call's
    `turn_delta` is computed against this view."""
    cards = [
        _peer_activity(session, a, viewer_name)
        for a in session.live_agents()
        if a.name != viewer_name
    ]
    for card in cards:
        session._peer_turn_count_seen[(viewer_name, card["name"])] = card["turn_count"]
    return cards


# Per-peer activity card (SPEC §4.5.1). Time-free: the only "how busy?"
# signal is `recent_tool_calls` over a fixed-size sliding window of
# transcript messages, plus the lifetime `turn_count` the viewer can
# diff across snapshots. The card MUST NOT carry any wall-clock
# timestamps or derived "time since" fields.
_ACTIVITY_LOOKBACK_MESSAGES = 20


def _peer_activity(session, agent, viewer_name: str) -> dict[str, Any]:
    """Compact activity card for `agent`, intended for a peer's eyes.

    `runtime` is one of:
      - "idle":   parked in a wait (not mid-turn)
      - "active": mid-turn (generating tokens / running a tool / between
                  turns with pending input)
      - "fired":  reaped — included so the CEO doesn't keep referencing
                  someone who's already gone; the loop filters fired
                  agents elsewhere so this branch is mostly defensive.

    The card deliberately omits the agent record's `state`: it is never
    "running" in practice, so on a live peer it always reads "idle" —
    models read that over `runtime: active` and poke busy peers.

    `turn_count` is the lifetime number of assistant messages on the
    peer's transcript.

    `turn_delta` is the number of new assistant turns the peer has run
    since the viewer last saw an activity card for them via
    `WaitForMessages`. Positive with no inbound message ⇒ silent worker.
    Zero means idle only together with `runtime: idle` — a peer in one
    long generation or tool call stays at zero while `runtime: active`.
    This pre-computes the diff so the viewer doesn't have to remember
    prior `turn_count`s.
    """
    transcript = session.transcript(agent.name)
    recent = transcript[-_ACTIVITY_LOOKBACK_MESSAGES:]
    recent_tool_calls = sum(
        len(m.tool_calls or []) for m in recent if m.role == "assistant"
    )
    if agent.state == "fired":
        runtime_state = "fired"
    elif session.wakes.is_waiting(agent.name):
        runtime_state = "idle"
    else:
        runtime_state = "active"
    turn_count = session.assistant_turn_count(agent.name)
    prior = session._peer_turn_count_seen.get((viewer_name, agent.name))
    turn_delta = turn_count - prior if prior is not None else None
    return {
        "name": agent.name,
        "role": agent.role,
        "runtime": runtime_state,
        "recent_tool_calls": recent_tool_calls,
        "turn_count": turn_count,
        "turn_delta": turn_delta,
    }


class ReportToCEOTool:
    definition = ToolDefinition(
        name="ReportToCEO",
        description=(
            "Send a structured report to the CEO: status is one of 'completed' | 'blocked' | 'failed'. "
            "The summary is the body. Attachments are archive paths."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
                "summary": {"type": "string"},
                "artifacts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "summary"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if ctx.agent.kind != "org":
            return rejected_result("not-permitted", "ReportToCEO is only available to org agents.")
        status = ctx.args.get("status")
        summary = ctx.args.get("summary") or ""
        summary_bytes = len(summary.encode("utf-8"))
        if summary_bytes > MAX_MESSAGE_BYTES:
            return _too_long_error("ReportToCEO", summary_bytes, field="summary")
        # The schema calls the field `artifacts`, but the prompts tell agents that
        # ReportToCEO takes `attachments` like SendMessage / PostToChannel. Accept
        # both so a report sent the documented way doesn't silently lose its files.
        paths = ctx.args.get("artifacts") or ctx.args.get("attachments") or []
        if not (isinstance(paths, list) and all(isinstance(p, str) for p in paths)):
            return err_result("invalid-args", "attachments must be a list of archive paths.")
        artifacts = [Attachment(archive_path=p) for p in paths]
        payload = ReportPayload(status=status, summary=summary, artifacts=artifacts).model_dump()
        session = ctx.session
        item = InboxItem(
            kind="report",
            **{"from": ctx.agent.name},
            payload=payload,
            content=summary,
            attachments=artifacts,
        )
        await session.inbox.append(session.ceo_name, item)
        if ctx.agent.name not in session.state.reported_agent_names:
            session.state.reported_agent_names.append(ctx.agent.name)
            await session.persist_state()
        await session.events.emit(
            "report.delivered",
            {"from": ctx.agent.name, "payload": payload},
            name=ctx.agent.name,
        )
        session.wakes.wake(session.ceo_name)
        return ok_result({"delivered": True})
