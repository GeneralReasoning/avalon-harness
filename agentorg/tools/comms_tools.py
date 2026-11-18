"""Communications tools: DMs, channels, reactions, subscriptions (§7)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any, Optional

from agentorg.channels import light_snippet
from agentorg.inbox import render_attachment_marker
from agentorg.tools.context import (
    MAX_TOOL_OUTPUT_BYTES,
    ToolContext,
    err_result,
    ok_result,
)
from agentorg.tools.org_tools import MAX_MESSAGE_BYTES, _too_long_error
from agentorg.types import (
    Attachment,
    InboxItem,
    ToolDefinition,
    ToolResult,
)

LIGHT_LEVEL_LITERAL = ["full", "light", "muted"]

# Trace (virtual) channels are paged in chunks of this size: an oversized
# message spans ceil(size / VIRTUAL_CHUNK_BYTES) of them, one per call. Defaults
# to the per-call tool-output cap so a chunk is "one tool result's worth".
VIRTUAL_CHUNK_BYTES = MAX_TOOL_OUTPUT_BYTES


# ---------------------------------------------------------------------------
# DMs / channels
# ---------------------------------------------------------------------------


class SendMessageTool:
    definition = ToolDefinition(
        name="SendMessage",
        description=(
            "Send a private direct message to a single agent (CEO or org agent) or external sender. "
            "Pass the bare name (no @ prefix). Attachments are archive paths."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient agent name or external label."},
                "content": {"type": "string"},
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Archive paths to attach.",
                },
            },
            "required": ["to", "content"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        to = (ctx.args.get("to") or "").lstrip("@").strip()
        content = ctx.args.get("content") or ""
        attachments = [Attachment(archive_path=p) for p in ctx.args.get("attachments") or []]
        if not to or not content:
            return err_result("invalid-args", "Both `to` and `content` are required.")
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_MESSAGE_BYTES:
            return _too_long_error("SendMessage", content_bytes)
        session = ctx.session
        recipient = _resolve_recipient(session, to)
        if recipient is None:
            return err_result("unknown-recipient", f"No such agent or external sender: {to}")
        item = InboxItem(
            kind="dm",
            **{"from": ctx.agent.name},
            content=content,
            attachments=attachments,
        )
        await session.inbox.append(recipient, item)
        session.wakes.wake(recipient)
        await session.events.emit(
            "dm.delivered",
            {"from": ctx.agent.name, "to": recipient},
            name=ctx.agent.name,
        )
        return ok_result({"delivered_to": recipient})


def _resolve_recipient(session, token: str) -> Optional[str]:
    token_lower = token.lower()
    for name in session.all_agent_names():
        if name.lower() == token_lower:
            agent = session.agent(name)
            if agent and agent.state != "fired":
                return name
            return None
    for label in session.external_labels:
        if label.lower() == token_lower:
            return label
    return None


class PostToChannelTool:
    definition = ToolDefinition(
        name="PostToChannel",
        description="Post a markdown message to a channel. Mentions are @<Name> tokens in the body.",
        input_schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "content": {"type": "string"},
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["channel", "content"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        channel = (ctx.args.get("channel") or "").lstrip("#")
        content = ctx.args.get("content") or ""
        attachments = [Attachment(archive_path=p) for p in ctx.args.get("attachments") or []]
        if not channel or not content:
            return err_result("invalid-args", "Both `channel` and `content` are required.")
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_MESSAGE_BYTES:
            return _too_long_error("PostToChannel", content_bytes)
        session = ctx.session
        await session.channels.load()
        # Reject writes to virtual channels.
        if _is_virtual_channel_name(session, channel):
            return err_result(
                "read-only-channel",
                f"`#{channel}` is a virtual (read-only) channel — use SendMessage for direct comms.",
            )

        # External label resolution requires the author's external labels too.
        live = session.live_agent_names()
        externals = set(session.external_labels)
        author_is_external = ctx.agent.name in externals

        try:
            message, counts = await session.channels.post(
                channel=channel,
                author=ctx.agent.name,
                is_external=author_is_external,
                content=content,
                attachments=attachments,
                live_agents=live,
                external_labels=externals,
            )
        except KeyError:
            return err_result("unknown-channel", f"No channel named {channel!r}")

        # Fan-out notifications.
        await _fanout(session, message)
        await session.events.emit(
            "channel.posted",
            {
                "channel": message.channel,
                "msg_key": message.msg_key,
                "author": message.author,
                "mentions": [m.model_dump() for m in message.mentions],
            },
            name=ctx.agent.name,
        )
        return ok_result(counts.model_dump())


def _entry_byte_size(entry: dict[str, Any]) -> int:
    """Approximate serialized size of one rendered message/inbox entry.
    JSON-encoded byte length is a faithful proxy for the prompt-side cost
    once `_serialize_tool_result` (loop.py) re-serializes the dict for the
    provider."""
    try:
        return len(json.dumps(entry, default=str, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(entry).encode("utf-8"))


def _cap_entry_text(entry: dict[str, Any], budget: int) -> dict[str, Any]:
    """Backstop: if a SINGLE entry exceeds `budget`, truncate its largest text
    field (with a marker) so no rendered page can exceed the cap. Real/search
    entries are already bounded (16 KB post cap / snippets) and trace channels
    chunk instead — this just makes the ceiling a hard guarantee everywhere."""
    if _entry_byte_size(entry) <= budget:
        return entry
    capped = dict(entry)
    candidates = [(k, v) for k, v in capped.items() if isinstance(v, str) and len(v) > 256]
    if not candidates:
        return capped
    key, val = max(candidates, key=lambda kv: len(kv[1]))
    vb = val.encode("utf-8")
    overshoot = _entry_byte_size(capped) - budget
    keep = max(256, len(vb) - overshoot - 256)
    capped[key] = vb[:keep].decode("utf-8", "ignore") + f"\n[+truncated to fit the {budget}-byte cap]"
    return capped


def _trim_entries_by_budget(
    entries: list[dict[str, Any]],
    *,
    budget: int,
    drop_from: str,
) -> tuple[list[dict[str, Any]], int]:
    """Trim a rendered-entry list down to a byte budget.

    `drop_from` is `"head"` (drop oldest, keep tail — natural for the
    no-cursor / `before_ts`-only window where the agent wants the latest
    messages) or `"tail"` (drop newest, keep head — for the `since_ts`
    forward-page case where the agent is catching up from a cursor).

    Returns `(trimmed, dropped_count)`. Always keeps at least one element
    so a single oversize entry can't make the result empty — the agent
    can still see it and decide to paginate.
    """
    if not entries:
        return entries, 0
    entries = [_cap_entry_text(e, budget) for e in entries]
    sizes = [_entry_byte_size(e) for e in entries]
    total = sum(sizes)
    if total <= budget:
        return entries, 0
    if drop_from == "head":
        kept_sizes = 0
        kept_from = len(entries)  # index of first kept entry, counted from start
        for i in range(len(entries) - 1, -1, -1):
            if kept_sizes + sizes[i] > budget and kept_from < len(entries):
                break
            kept_sizes += sizes[i]
            kept_from = i
        return entries[kept_from:], kept_from
    # drop_from == "tail"
    kept_sizes = 0
    kept_until = 0  # exclusive end of kept slice
    for i in range(len(entries)):
        if kept_sizes + sizes[i] > budget and kept_until > 0:
            break
        kept_sizes += sizes[i]
        kept_until = i + 1
    return entries[:kept_until], len(entries) - kept_until


def _is_virtual_channel_name(session, name: str) -> bool:
    lower = name.lower().lstrip("#")
    if lower in ("self", "ceo"):
        return True
    return lower in {a.lower() for a in session.all_agent_names()}


async def _fanout(session, message):
    """Push channel notifications into subscriber inboxes per §7.2/§7.4."""
    channel = session.channels.resolve(message.channel)
    if channel is None:
        return
    mentioned_names = {m.name.lower() for m in message.mentions if m.kind == "agent" and m.name}
    live_agents = session.live_agent_names()
    for agent in live_agents:
        if agent == message.author and not message.is_external:
            continue
        level = session.channels.effective_subscription(channel, agent)
        tagged = agent.lower() in mentioned_names
        if not tagged and level == "muted":
            continue
        if not tagged and level == "light":
            item = InboxItem(
                kind="channel_preview",
                channel=message.channel,
                author=message.author,
                preview=light_snippet(message.content),
                mentions=message.mentions,
                payload={"is_external": message.is_external},
                msg_key=message.msg_key,
            )
        else:
            item = InboxItem(
                kind="channel_post",
                channel=message.channel,
                author=message.author,
                content=message.content,
                attachments=message.attachments,
                mentions=message.mentions,
                payload={"is_external": message.is_external},
                msg_key=message.msg_key,
            )
        await session.inbox.append(agent, item)
        session.wakes.wake(agent)


class CreateChannelsTool:
    definition = ToolDefinition(
        name="CreateChannels",
        description=(
            "Batch-create channels. Each entry takes a `name`, optional "
            "`default_subscription` ('full'|'light'|'muted'), and optional "
            "`subscriptions` map of agent_name -> level for per-agent overrides."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "default_subscription": {
                                "type": "string",
                                "enum": LIGHT_LEVEL_LITERAL,
                            },
                            "subscriptions": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": LIGHT_LEVEL_LITERAL,
                                },
                            },
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["channels"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        specs = ctx.args.get("channels") or []
        if not isinstance(specs, list) or not specs:
            return err_result("invalid-args", "channels must be a non-empty list")
        session = ctx.session
        try:
            created, existing = await session.channels.create_channels(
                specs,
                live_agent_names=session.live_agent_names(),
                external_labels=set(session.external_labels),
            )
        except ValueError as e:
            return err_result("invalid-args", str(e))
        for ch in created:
            await session.events.emit("channel.created", {"channel": ch.name})
        return ok_result(
            {
                "created": [c.name for c in created],
                "existing": [c.name for c in existing],
            }
        )


class ListChannelsTool:
    definition = ToolDefinition(
        name="ListChannels",
        description=(
            "List channels in this session. Regular channels show your subscription level; "
            "virtual channels (agent traces) show is_virtual=true plus agent role and whether it was fired."
        ),
        input_schema={"type": "object", "properties": {}},
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        await session.channels.load()
        entries: list[dict[str, Any]] = []
        for ch in session.channels.list_regular():
            entries.append(
                {
                    "name": ch.name,
                    "message_count": session.channels.total_messages(ch.name),
                    "subscription": session.channels.effective_subscription(ch, ctx.agent.name),
                    "is_virtual": False,
                }
            )
        # Agent-trace virtual channels. Counts reflect the full history
        # (across compaction boundaries) to match what ViewChannelMessages shows.
        for agent in session.all_agents():
            entries.append(
                {
                    "name": agent.name,
                    "message_count": len(session.full_transcript(agent.name)),
                    "is_virtual": True,
                    "agent_role": agent.role,
                    "fired": agent.state == "fired",
                }
            )
        entries.append(
            {
                "name": "self",
                "message_count": len(session.full_transcript(ctx.agent.name)),
                "is_virtual": True,
                "agent_role": ctx.agent.role,
                "alias_for": ctx.agent.name,
            }
        )
        return ok_result({"channels": entries})


class ViewChannelMessagesTool:
    definition = ToolDefinition(
        name="ViewChannelMessages",
        description=(
            "Return a window of channel messages, lowest `idx` first. "
            "No-cursor calls resume from your last view per (viewer, channel) "
            "and return only what's new. Defaults: `limit=10` on real "
            "channels, `5` on trace channels (`#<agent>`, `#self`); max 200. "
            "`since_idx=N` reads forward, inclusive (returns `idx >= N`, so "
            "`since_idx=0` starts at the first message); `before_idx=N` "
            "reads backward, exclusive (returns `idx < N`). Response carries "
            "`total_messages` / `older_not_shown` / `newer_not_shown` / `cursor_was`. "
            "On a virtual channel an oversized message is paged in "
            f"~{VIRTUAL_CHUNK_BYTES // 1024} KB chunks: that entry carries "
            "`chunk`/`of_chunks`/`text` and the response adds `next_cursor` — pass "
            "`chunk=j` with `since_idx` to jump within one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "since_idx": {"type": "integer"},
                "before_idx": {"type": "integer"},
                "chunk": {"type": "integer", "minimum": 0},
            },
            "required": ["channel"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        channel = (ctx.args.get("channel") or "").lstrip("#")
        since_idx = ctx.args.get("since_idx")
        before_idx = ctx.args.get("before_idx")
        session = ctx.session
        caller = ctx.agent.name

        is_virtual = _is_virtual_channel_name(session, channel)
        default_limit = 5 if is_virtual else 10
        limit = int(ctx.args.get("limit") or default_limit)
        limit = max(1, min(200, limit))

        # Resume from per-(viewer, channel) cursor when the model didn't
        # specify a window. This makes repeated peeks return only NEW
        # messages, the common case for monitoring a teammate's trace.
        # `since_idx` is inclusive, and `cursor_was` is the last idx already
        # shown, so resume from `cursor_was + 1` to skip what's been seen.
        cursor_key = (caller, channel.lower())
        cursor_was = session._vcm_cursors.get(cursor_key)

        if is_virtual:
            # Trace channels carry a (msg_idx, chunk) cursor and page oversized
            # messages in VIRTUAL_CHUNK_BYTES chunks — _view_virtual owns resume.
            return _view_virtual(
                session, caller, channel, limit, since_idx, before_idx,
                chunk_arg=ctx.args.get("chunk"),
                cursor_key=cursor_key, cursor_was=cursor_was,
            )

        # Real channels resume from an integer last-idx cursor.
        resumed = False
        if since_idx is None and before_idx is None and cursor_was is not None:
            since_idx = cursor_was + 1
            resumed = True

        await session.channels.load()
        ch = session.channels.resolve(channel)
        if ch is None:
            return err_result("unknown-channel", f"No channel named {channel!r}")

        # Total channel size, for navigation hints in the response.
        total_messages = sum(1 for _ in session.channels.iter_messages(ch.name))
        all_msgs = list(
            session.channels.iter_messages(
                ch.name, since_idx=since_idx, before_idx=before_idx
            )
        )
        older_not_shown = 0
        newer_not_shown = 0
        # `drop_direction` is the side we'd drop FURTHER messages from if the
        # window exceeds the byte budget — same end the count cap already
        # drops from, so size trimming is just a continuation.
        if len(all_msgs) > limit:
            if since_idx is None and before_idx is None:
                window = all_msgs[-limit:]
                drop_direction = "head"
            elif before_idx is not None and since_idx is None:
                window = all_msgs[-limit:]
                drop_direction = "head"
            else:
                window = all_msgs[:limit]
                drop_direction = "tail"
        else:
            window = all_msgs
            drop_direction = "head"
        reaction_folds = session.channels.fold_reactions(ch.name)
        rendered: list[dict[str, Any]] = []
        for m in window:
            entry = m.model_dump()
            groups = reaction_folds.get(m.msg_key, [])
            if groups:
                entry["reactions"] = [g.model_dump() for g in groups]
                entry["reactions_summary"] = "  ".join(
                    f"{g.emoji} {g.count}" for g in groups
                )
            rendered.append(entry)
        rendered, _dropped = _trim_entries_by_budget(
            rendered, budget=MAX_TOOL_OUTPUT_BYTES, drop_from=drop_direction,
        )
        oldest_idx = rendered[0].get("idx") if rendered else None
        latest_idx = rendered[-1].get("idx") if rendered else None
        # `older_not_shown` / `newer_not_shown` are relative to the channel,
        # not just the requested slice: count whatever sits outside the
        # returned window, so the model can see at a glance whether there's
        # more in either direction (incl. history they've already seen but
        # may want to revisit after a compaction).
        if oldest_idx is not None:
            older_not_shown = sum(
                1 for _ in session.channels.iter_messages(ch.name, before_idx=oldest_idx)
            )
            # `since_idx` is inclusive, so count strictly-newer with +1.
            newer_not_shown = sum(
                1 for _ in session.channels.iter_messages(ch.name, since_idx=latest_idx + 1)
            )
        else:
            # Nothing came back, so the counts are relative to the point the
            # caller asked about rather than to a window that does not exist.
            # An exhausted `before_idx` page has nothing older and the whole
            # channel ahead of it; a forward view that came back empty has the
            # whole channel behind it. Reporting the forward answer for both
            # told a backward pager that history it had already run out of was
            # still there, and it kept paging.
            older_not_shown = (
                sum(
                    1
                    for _ in session.channels.iter_messages(
                        ch.name, before_idx=before_idx
                    )
                )
                if before_idx is not None
                else total_messages
            )
            newer_not_shown = total_messages - older_not_shown
        # Update the cursor to the latest idx we've shown this viewer.
        if latest_idx is not None:
            session._vcm_cursors[cursor_key] = max(
                session._vcm_cursors.get(cursor_key, -1), latest_idx
            )
        out: dict[str, Any] = {
            "channel": ch.name,
            "messages": rendered,
            "oldest_idx": oldest_idx,
            "latest_idx": latest_idx,
            "total_messages": total_messages,
            "older_not_shown": older_not_shown,
            "newer_not_shown": newer_not_shown,
            "cursor_was": cursor_was,
            "resumed_from_cursor": resumed,
        }
        # An empty window on a channel the caller is subscribed to at `full`
        # is a poll that could never pay off: every post there is already
        # delivered into the caller's next turn (or wakes it from the wait).
        # Only for a forward-looking view: the default window, a resumed
        # cursor, or an explicit `since_idx` all ask "anything new?", so
        # nothing back means caught up. `before_idx` asks what came BEFORE a
        # point, and nothing back there means the history ran out — telling
        # that caller never to look at the channel again is a non sequitur.
        polled = before_idx is None
        if (
            polled
            and not rendered
            and session.channels.effective_subscription(ch, caller) == "full"
        ):
            out["note"] = (
                f"No new messages. You are subscribed to #{ch.name} at 'full', so "
                "every new post is delivered into your next turn automatically "
                "(and wakes you if you are parked in WaitForMessages). There is "
                "no need to check this channel again."
            )
        return ok_result(out)


def _structured_virtual_entry(target: str, i: int, msg) -> dict[str, Any]:
    """Full structured trace entry (used for messages that fit a single chunk)."""
    return {
        "msg_key": f"{target}:{i}:{msg.role}",
        "author": target,
        "idx": i,
        "role": msg.role,
        "content": _stringify_msg_content(msg),
        "reasoning": msg.reasoning,
        "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
        "tool_result": msg.tool_result.model_dump() if msg.tool_result else None,
    }


def _flat_render_msg(target: str, i: int, msg) -> str:
    """Flat-text rendering of one trace message, for byte-chunking an oversized
    one (a teammate's big tool_result, etc.) into UTF-8-safe VIRTUAL_CHUNK_BYTES windows."""
    parts = [f"[{target} #{i} {msg.role}]"]
    content = _stringify_msg_content(msg)
    if content:
        parts.append(content)
    if getattr(msg, "reasoning", None):
        parts.append("<reasoning>\n" + msg.reasoning + "\n</reasoning>")
    if getattr(msg, "tool_calls", None):
        parts.append(
            "<tool_calls>\n"
            + json.dumps([tc.model_dump() for tc in msg.tool_calls], default=str, ensure_ascii=False)
            + "\n</tool_calls>"
        )
    if getattr(msg, "tool_result", None):
        parts.append(
            "<tool_result>\n"
            + json.dumps(msg.tool_result.model_dump(), default=str, ensure_ascii=False)
            + "\n</tool_result>"
        )
    return "\n".join(parts)


def _virtual_msg_size(msg) -> int:
    """Cheap byte estimate of a trace message's renderable payload — decides
    small-vs-oversized. Exact slicing of an oversized one uses _flat_render_msg."""
    n = len(_stringify_msg_content(msg) or "")
    if getattr(msg, "reasoning", None):
        n += len(msg.reasoning)
    if getattr(msg, "tool_calls", None):
        n += sum(
            len(json.dumps(tc.model_dump(), default=str, ensure_ascii=False))
            for tc in msg.tool_calls
        )
    if getattr(msg, "tool_result", None):
        n += len(json.dumps(msg.tool_result.model_dump(), default=str, ensure_ascii=False))
    return n


def _virtual_tail_start(
    sizes: list[int], chunks_of: list[int], limit: int, end: int
) -> tuple[int, int]:
    """Start (msg_idx, chunk) so a forward-pack fills the LAST page before `end`
    — the default 'show me the latest activity' view."""
    if end <= 0:
        return (0, 0)
    last = end - 1
    if chunks_of[last] > 1:
        return (last, chunks_of[last] - 1)  # latest chunk of the latest (big) message
    bytes_used = 0
    count = 0
    start = last
    i = last
    while i >= 0 and chunks_of[i] <= 1 and count < limit and bytes_used + sizes[i] <= VIRTUAL_CHUNK_BYTES:
        bytes_used += sizes[i]
        count += 1
        start = i
        i -= 1
    return (start, 0)


def _virtual_pack(
    target: str,
    transcript,
    sizes: list[int],
    chunks_of: list[int],
    start: tuple[int, int],
    limit: int,
    stop_before: int,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """Forward-pack a single ≤VIRTUAL_CHUNK_BYTES page from `start`, returning
    (entries, next_pos). Small messages pack several per page (full, structured
    shape); an oversized message returns exactly ONE chunk on its own page."""
    entries: list[dict[str, Any]] = []
    bytes_used = 0
    i, start_chunk = start
    end = min(stop_before, len(transcript))
    while i < end and len(entries) < limit:
        if chunks_of[i] <= 1:  # fits one chunk → show whole; pack with neighbours
            if entries and bytes_used + sizes[i] > VIRTUAL_CHUNK_BYTES:
                return entries, (i, 0)
            entries.append(_structured_virtual_entry(target, i, transcript[i]))
            bytes_used += sizes[i]
            i += 1
            start_chunk = 0
        else:  # oversized → one chunk, alone
            if entries:
                return entries, (i, 0)
            flat = _flat_render_msg(target, i, transcript[i]).encode("utf-8")
            total = max(1, math.ceil(len(flat) / VIRTUAL_CHUNK_BYTES))
            j = min(max(0, start_chunk), total - 1)
            window = flat[j * VIRTUAL_CHUNK_BYTES:(j + 1) * VIRTUAL_CHUNK_BYTES].decode(
                "utf-8", "replace"
            )
            entries.append(
                {
                    "msg_key": f"{target}:{i}:{transcript[i].role}",
                    "author": target,
                    "idx": i,
                    "role": transcript[i].role,
                    "chunk": j,
                    "of_chunks": total,
                    "truncated": True,
                    "text": window,
                }
            )
            return entries, ((i, j + 1) if j + 1 < total else (i + 1, 0))
    return entries, (i, 0)


def _view_virtual(
    session, caller: str, name: str, limit: int, since_idx, before_idx,
    *, chunk_arg, cursor_key: tuple[str, str], cursor_was,
) -> ToolResult:
    target = name
    lower = name.lower()
    if lower == "self":
        target = caller
    else:
        for cand in session.all_agent_names():  # preserve case
            if cand.lower() == lower:
                target = cand
                break

    # Full history (across compaction boundaries) so a trace view surfaces the
    # original reasoning/tool I/O that compaction summarised away — not just the
    # live post-compaction tail.
    transcript = session.full_transcript(target)
    total_messages = len(transcript)

    # Cheap per-message size → chunk counts. A normal message is 1 chunk; only
    # an oversized one (e.g. a teammate's big tool_result) is split. The
    # everything-fits case stays the classic message-paged view (below).
    sizes = [_virtual_msg_size(m) for m in transcript]
    chunks_of = [max(1, math.ceil(s / VIRTUAL_CHUNK_BYTES)) for s in sizes]

    # `idx` is the message's transcript position. `since_idx` inclusive (>=),
    # `before_idx` exclusive (<) — mirrors real-channel semantics.
    resumed = False
    if since_idx is not None:
        start = (max(0, int(since_idx)), max(0, int(chunk_arg or 0)))
        stop_before = total_messages
    elif before_idx is not None:
        # before_idx is model-supplied — clamp to the transcript so a
        # too-large value can't index past the end in _virtual_tail_start.
        stop_before = min(max(0, int(before_idx)), total_messages)
        start = _virtual_tail_start(sizes, chunks_of, limit, end=stop_before)
    elif cursor_was is not None:
        # Cursor = NEXT position to read: an int msg_idx in the common case, or
        # (msg_idx, chunk) when we paused mid-oversized-message.
        start = cursor_was if isinstance(cursor_was, tuple) else (int(cursor_was), 0)
        stop_before = total_messages
        resumed = True
    else:
        start = _virtual_tail_start(sizes, chunks_of, limit, end=total_messages)
        stop_before = total_messages

    entries, next_pos = _virtual_pack(
        target, transcript, sizes, chunks_of, start, limit, stop_before
    )
    next_msg, next_chunk = next_pos

    if entries:
        oldest_idx = entries[0]["idx"]
        latest_idx = entries[-1]["idx"]
        older_not_shown = oldest_idx
        newer_not_shown = max(0, total_messages - 1 - latest_idx)
    else:
        oldest_idx = latest_idx = None
        older_not_shown = total_messages
        newer_not_shown = 0

    # Store NEXT to read: a plain int (classic, unchanged) when chunk==0, a
    # (msg_idx, chunk) tuple only when paused mid-oversized-message.
    session._vcm_cursors[cursor_key] = next_pos if next_chunk else next_msg

    result = {
        "channel": target,
        "is_virtual": True,
        "messages": entries,
        "oldest_idx": oldest_idx,
        "latest_idx": latest_idx,
        "total_messages": total_messages,
        "older_not_shown": older_not_shown,
        "newer_not_shown": newer_not_shown,
        "cursor_was": cursor_was[0] if isinstance(cursor_was, tuple) else cursor_was,
        "resumed_from_cursor": resumed,
    }
    # Chunk metadata appears ONLY when an oversized message was actually split,
    # so an everything-fits page is identical to the classic message-paged view.
    if any(e.get("of_chunks", 1) > 1 for e in entries):
        result["next_cursor"] = {"idx": next_msg, "chunk": next_chunk}
    return ok_result(result)


def _stringify_msg_content(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        chunks = []
        paths = []
        for seg in msg.content:
            if hasattr(seg, "text") and seg.text:
                chunks.append(seg.text)
            elif hasattr(seg, "archive_path") and seg.archive_path:
                paths.append(seg.archive_path)
        if paths:
            chunks.append(render_attachment_marker(paths))
        return "\n".join(chunks)
    return ""


class SearchChannelMessagesTool:
    definition = ToolDefinition(
        name="SearchChannelMessages",
        description=(
            "Regex search over channel message content. Omit `channel` to search all channels."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "channel": {"type": "string"},
                "ignore_case": {"type": "boolean"},
                "before_lines": {"type": "integer", "minimum": 0, "maximum": 10},
                "after_lines": {"type": "integer", "minimum": 0, "maximum": 10},
                "multiline": {"type": "boolean"},
                "since_idx": {"type": "integer"},
                "author": {"type": "string"},
                "output_mode": {"type": "string", "enum": ["content", "count"]},
                "head_limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["pattern"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        await session.channels.load()
        pat = ctx.args.get("pattern")
        if not pat:
            return err_result("invalid-args", "pattern is required")
        flags = re.MULTILINE
        if ctx.args.get("ignore_case"):
            flags |= re.IGNORECASE
        if ctx.args.get("multiline"):
            flags |= re.DOTALL
        try:
            rx = re.compile(pat, flags)
        except re.error as e:
            return err_result("invalid-regex", str(e))
        channel = (ctx.args.get("channel") or "").lstrip("#")
        targets: list[str] = []
        if channel:
            ch = session.channels.resolve(channel)
            if ch is None and not _is_virtual_channel_name(session, channel):
                return err_result("unknown-channel", f"No channel named {channel!r}")
            targets = [channel if _is_virtual_channel_name(session, channel) else ch.name]
        else:
            targets = [c.name for c in session.channels.list_regular()]
            # Every agent's trace, the caller's own included (once).
            targets += [a.name for a in session.all_agents()]
        author = ctx.args.get("author")
        since_idx = ctx.args.get("since_idx")
        head = int(ctx.args.get("head_limit") or 200)
        output_mode = ctx.args.get("output_mode") or "content"

        # Hard bound on bodies scanned per call (defensive against a
        # pathological regex over a huge channel — the regex itself can't
        # be timed out from Python's `re`, but capping iterations keeps
        # the loop bounded). Applies to both output modes.
        MAX_BODIES_PER_SEARCH = 5000
        MAX_BODY_BYTES = 256 * 1024  # skip extremely long bodies

        if output_mode == "count":
            counts: dict[str, dict[str, Any]] = {}
            scanned = 0
            hit_cap = False
            for ch_name in targets:
                count = 0
                latest = None
                msgs = _iter_for_search(session, ch_name, since_idx, caller=ctx.agent.name)
                for m in msgs:
                    if author and (m.get("author") or "").lower() != author.lower():
                        continue
                    body = m.get("content") or ""
                    if len(body) > MAX_BODY_BYTES:
                        continue
                    if scanned >= MAX_BODIES_PER_SEARCH:
                        hit_cap = True
                        break
                    scanned += 1
                    if rx.search(body):
                        count += 1
                        latest = max(latest or 0, m.get("idx") or 0)
                if count:
                    counts[ch_name] = {"count": count, "latest_idx": latest}
                if hit_cap:
                    break
            out: dict[str, Any] = {"counts": counts}
            if hit_cap:
                # Counts were exact before the cap existed; say when they aren't.
                out["truncated"] = True
            return ok_result(out)

        # Collect ALL matching messages first, sort by ts desc, THEN
        # truncate to `head_limit`.
        hits: list[dict[str, Any]] = []
        bodies_scanned = 0
        for ch_name in targets:
            msgs = list(_iter_for_search(session, ch_name, since_idx, caller=ctx.agent.name))
            for pos, m in enumerate(msgs):
                if bodies_scanned >= MAX_BODIES_PER_SEARCH:
                    break
                if author and (m.get("author") or "").lower() != author.lower():
                    continue
                body = m.get("content") or ""
                if len(body) > MAX_BODY_BYTES:
                    continue
                bodies_scanned += 1
                if rx.search(body):
                    before = []
                    after = []
                    b = int(ctx.args.get("before_lines") or 0)
                    a = int(ctx.args.get("after_lines") or 0)
                    if b > 0:
                        before = [_msg_summary(x) for x in msgs[max(0, pos - b): pos]]
                    if a > 0:
                        after = [_msg_summary(x) for x in msgs[pos + 1: pos + 1 + a]]
                    hits.append(
                        {
                            "channel": ch_name,
                            "match": _msg_summary(m),
                            "before": before,
                            "after": after,
                        }
                    )
            if bodies_scanned >= MAX_BODIES_PER_SEARCH:
                break
        # Order matches by descending `idx` so newest first.
        hits.sort(key=lambda h: h["match"]["idx"], reverse=True)
        hits = hits[:head]
        # Size cap is in addition to `head_limit`: an agent searching with
        # `before_lines=10, after_lines=10` against a channel of large posts
        # can saturate the model's context with one call even at head_limit=20.
        # We sorted newest-first, so dropping from the tail keeps the most
        # recent matches.
        hits, dropped_by_size = _trim_entries_by_budget(
            hits, budget=MAX_TOOL_OUTPUT_BYTES, drop_from="tail",
        )
        result: dict[str, Any] = {"hits": hits}
        if dropped_by_size:
            result["older_hits_not_shown"] = dropped_by_size
        return ok_result(result)


def _iter_for_search(session, ch_name: str, since_idx, *, caller: str):
    if _is_virtual_channel_name(session, ch_name):
        target = ch_name if ch_name != "self" else caller
        for cand in session.all_agent_names():
            if cand.lower() == ch_name.lower():
                target = cand
                break
        for i, msg in enumerate(session.full_transcript(target)):
            if since_idx is not None and i < since_idx:  # inclusive, mirrors real channels
                continue
            yield {
                "author": target,
                "idx": i,
                "role": msg.role,
                "content": _stringify_msg_content(msg),
            }
        return
    folds = session.channels.fold_reactions(ch_name)
    for m in session.channels.iter_messages(ch_name, since_idx=since_idx):
        entry = m.model_dump()
        groups = folds.get(m.msg_key, [])
        if groups:
            entry["reactions_summary"] = "  ".join(
                f"{g.emoji} {g.count}" for g in groups
            )
        yield entry


class ViewMyHistoryTool:
    """Solo-mode view of the agent's own trace (#self), presented without any
    channel vocabulary. Thin alias over ViewChannelMessages on `#self`; the
    virtual read path returns the full history across compaction boundaries, so
    this surfaces turns that compaction summarised out of the live context."""

    definition = ToolDefinition(
        name="ViewMyHistory",
        description=(
            "Return a window of your own past turns — your reasoning and tool "
            "calls/results — lowest `idx` first, INCLUDING earlier turns that were "
            "summarised away when your context filled (compaction). No-cursor calls "
            "resume from your last view and return only what's new. Default `limit=5`, "
            "max 200. `since_idx=N` reads forward, inclusive (returns `idx >= N`, so "
            "`since_idx=0` starts at your first turn); `before_idx=N` reads backward, "
            "exclusive (returns `idx < N`). Response carries `total_messages` / "
            "`older_not_shown` / `newer_not_shown` / `cursor_was`. An oversized turn is "
            f"paged in ~{VIRTUAL_CHUNK_BYTES // 1024} KB chunks (that entry carries "
            "`chunk`/`of_chunks`/`text`, response adds `next_cursor`; pass `chunk=j` "
            "with `since_idx` to jump within one)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "since_idx": {"type": "integer"},
                "before_idx": {"type": "integer"},
                "chunk": {"type": "integer", "minimum": 0},
            },
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        self_ctx = replace(ctx, args={**ctx.args, "channel": "self"})
        return await ViewChannelMessagesTool().execute(self_ctx)


class SearchMyHistoryTool:
    """Solo-mode search over the agent's own trace (#self). Thin alias over
    SearchChannelMessages locked to `#self`; reaches the full history across
    compaction boundaries."""

    definition = ToolDefinition(
        name="SearchMyHistory",
        description=(
            "Regex/substring search over your own past turns, including ones "
            "summarised away by compaction. `output_mode=count` gives a cheap "
            "did-I-already-cover-X signal; `output_mode=content` (default) returns "
            "matches newest first with optional `before_lines`/`after_lines` context."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "ignore_case": {"type": "boolean"},
                "before_lines": {"type": "integer", "minimum": 0, "maximum": 10},
                "after_lines": {"type": "integer", "minimum": 0, "maximum": 10},
                "multiline": {"type": "boolean"},
                "since_idx": {"type": "integer"},
                "output_mode": {"type": "string", "enum": ["content", "count"]},
                "head_limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["pattern"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        self_ctx = replace(ctx, args={**ctx.args, "channel": "self"})
        return await SearchChannelMessagesTool().execute(self_ctx)


def _msg_summary(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "author": m.get("author"),
        "idx": m.get("idx"),
        "content": m.get("content"),
    }


EMOJI_MIN_CHARS = 1
EMOJI_MAX_CHARS = 32


class ReactToMessageTool:
    definition = ToolDefinition(
        name="ReactToMessage",
        description=(
            "Toggle an emoji reaction on a channel post (§7.5). Calling with the same "
            "emoji twice removes it. Bypasses subscription levels for the target's author. "
            "Cheaper than a full reply — use it for acknowledgement ('👍 agree', '✅ on it', "
            "'👀 looking', '💭 thinking'). Reactions are an acknowledgement "
            "primitive, not a content channel: no body beyond the emoji. Pass the target "
            "post's `idx` value from ViewChannelMessages."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "msg_idx": {
                    "type": "integer",
                    "description": "The per-channel post `idx` of the target ChannelMessage.",
                },
                "emoji": {"type": "string", "minLength": 1, "maxLength": EMOJI_MAX_CHARS},
            },
            "required": ["channel", "msg_idx", "emoji"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        channel = (ctx.args.get("channel") or "").lstrip("#")
        emoji = ctx.args.get("emoji") or ""
        msg_idx = ctx.args.get("msg_idx")
        if not (EMOJI_MIN_CHARS <= len(emoji) <= EMOJI_MAX_CHARS):
            return err_result(
                "invalid-args",
                f"emoji must be {EMOJI_MIN_CHARS}-{EMOJI_MAX_CHARS} characters.",
            )
        if not isinstance(msg_idx, int):
            return err_result("invalid-args", "msg_idx must be an integer (the post's `idx`).")
        if _is_virtual_channel_name(session, channel):
            return err_result(
                "read-only-channel",
                f"`#{channel}` is a virtual (read-only) channel — reactions don't apply.",
            )
        await session.channels.load()
        ch = session.channels.resolve(channel)
        if ch is None:
            return err_result("unknown-channel", f"No channel named {channel!r}")
        target = session.channels.find_message_by_idx(ch.name, msg_idx)
        if target is None:
            return err_result("unknown-message", f"No message at idx={msg_idx} in #{ch.name}")

        ev, now_reacting = await session.channels.toggle_reaction(
            channel=ch.name,
            msg_key=target.msg_key,
            emoji=emoji,
            reactor=ctx.agent.name,
        )

        # Emit event.
        await session.events.emit(
            "channel.reaction.added" if ev.op == "add" else "channel.reaction.removed",
            {
                "channel": ch.name,
                "msg_key": target.msg_key,
                "reactor": ctx.agent.name,
                "emoji": emoji,
            },
            name=ctx.agent.name,
        )

        # Notify post author on add only (§7.5).
        if ev.op == "add":
            author = target.author
            same_reactor = (author == ctx.agent.name) and not target.is_external
            author_is_agent = author in session.all_agent_names()
            if author_is_agent and not same_reactor:
                preview = light_snippet(target.content)
                item = InboxItem(
                    kind="reaction",
                    channel=ch.name,
                    msg_key=target.msg_key,
                    reactor=ctx.agent.name,
                    emoji=emoji,
                    target_content_preview=preview,
                )
                await session.inbox.append(author, item)
                session.wakes.wake(author)

        return ok_result(
            {
                "ok": True,
                "now_reacting": now_reacting,
                "confirmation": (
                    f"{'Added' if now_reacting else 'Removed'} {emoji} "
                    f"on #{ch.name} msg_idx={msg_idx}"
                ),
            }
        )


class SetChannelSubscriptionTool:
    definition = ToolDefinition(
        name="SetChannelSubscription",
        description="Set YOUR OWN subscription level on a channel: 'full' | 'light' | 'muted'.",
        input_schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "level": {"type": "string", "enum": LIGHT_LEVEL_LITERAL},
            },
            "required": ["channel", "level"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        ch = (ctx.args.get("channel") or "").lstrip("#")
        level = ctx.args.get("level")
        if _is_virtual_channel_name(session, ch):
            return err_result(
                "read-only-channel",
                f"`#{ch}` is virtual — subscriptions don't apply.",
            )
        ok = await session.channels.set_subscription(ch, ctx.agent.name, level)
        if not ok:
            return err_result("unknown-channel", f"No channel named {ch!r}")
        return ok_result({"channel": ch, "level": level})
