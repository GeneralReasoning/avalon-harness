"""Per-agent inbox primitives (§3.7, §4.3).

The inbox is the queue of items awaiting delivery on the agent's next turn.
On disk it's `pending.jsonl`, written append-only and truncated by rewrite
on drain. In memory it's a simple list guarded by an asyncio lock.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from agentorg.storage import append_jsonl_async, read_jsonl
from agentorg.types import AgentName, InboxItem


class InboxStore:
    """Manages every agent's inbox in the session.

    `pending_path(name)` is a JSONL file that holds the *current* pending
    items. On drain we rewrite it atomically.
    """

    def __init__(self, ceo_name: AgentName, paths_factory):
        # paths_factory: name -> Path
        self._paths_factory = paths_factory
        self._ceo_name = ceo_name
        self._locks: dict[str, asyncio.Lock] = {}
        self._counter = 0
        self._counter_lock = asyncio.Lock()

    def _lock(self, name: AgentName) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def _next_seq(self) -> int:
        async with self._counter_lock:
            self._counter += 1
            return self._counter

    def init_counter_from_disk(self, agent_names: list[AgentName]) -> None:
        """Bump `_counter` past the max seq across every loaded inbox.

        Called from `Session.resume` so post-resume items don't sort
        before pre-resume items in `render_inbox` (which orders by
        `(seq, ts)`). Safe to call before the event loop is running —
        no async needed for the read-only scan.
        """
        max_seq = self._counter
        for name in agent_names:
            path = self._paths_factory(name)
            for raw in read_jsonl(path):
                seq = raw.get("seq")
                if isinstance(seq, int) and seq > max_seq:
                    max_seq = seq
        self._counter = max_seq

    def load(self, name: AgentName) -> list[InboxItem]:
        path = self._paths_factory(name)
        items: list[InboxItem] = []
        for raw in read_jsonl(path):
            try:
                items.append(InboxItem.model_validate(raw))
            except Exception:
                continue
        return items

    async def append(self, name: AgentName, item: InboxItem) -> None:
        if item.seq == 0:
            item.seq = await self._next_seq()
        async with self._lock(name):
            await append_jsonl_async(self._paths_factory(name), item)

    async def drain(self, name: AgentName) -> list[InboxItem]:
        """Read all pending items and reset the file."""
        async with self._lock(name):
            path = self._paths_factory(name)
            items = self.load(name)
            if items:
                # Truncate the file in-place.
                await asyncio.to_thread(_truncate_file, path)
            return items

    async def peek_count(self, name: AgentName) -> int:
        async with self._lock(name):
            items = self.load(name)
            return len(items)

    async def has_pending(self, name: AgentName) -> bool:
        return (await self.peek_count(name)) > 0


def _truncate_file(path: Path) -> None:
    if path.exists():
        path.write_bytes(b"")


def render_inbox(items: list[InboxItem], *, recipient: Optional[AgentName] = None) -> str:
    """Render a collapsed inbox into the body of a `turn_user_message` (§4.3).

    Each item gets a header naming its kind, sender/author/channel, plus any
    attachments and (for the recipient) any mentions.
    """
    if not items:
        return ""
    blocks: list[str] = []
    for item in sorted(items, key=lambda i: i.seq):
        blocks.append(_render_item(item, recipient=recipient))
    return "\n\n".join(blocks)


def _render_item(item: InboxItem, *, recipient: Optional[AgentName]) -> str:
    if item.kind == "dm":
        header = f"[DM from @{item.from_}]"
        body = item.content or ""
        atts = _render_attachments(item)
        return _join_lines(header, body, atts)

    if item.kind == "channel_post":
        author_tag = "(Human) " if item.payload and item.payload.get("is_external") else ""
        idx_tag = _idx_tag(item)
        header = f"[#{item.channel}{idx_tag} — {author_tag}@{item.author} posted]"
        mention_line = _mention_banner(item, recipient)
        body = item.content or ""
        atts = _render_attachments(item)
        return _join_lines(header, mention_line, body, atts)

    if item.kind == "channel_preview":
        idx_tag = _idx_tag(item)
        header = f"[#{item.channel}{idx_tag} — @{item.author} posted (preview)]"
        snippet = item.preview or ""
        suffix = f"(use ViewChannelMessages(channel={item.channel!r}) for the full post)"
        return _join_lines(header, snippet, suffix)

    if item.kind == "reaction":
        preview = (item.target_content_preview or "").strip()
        suffix = f': "{preview}"' if preview else ""
        return (
            f"🔔 @{item.reactor} reacted {item.emoji} to your post in "
            f"#{item.channel}{suffix}"
        )

    if item.kind == "report":
        payload = item.payload or {}
        status = payload.get("status", "completed")
        header = f"[Report from @{item.from_} — {status}]"
        body = payload.get("summary") or item.content or ""
        atts = _render_attachments(item)
        return _join_lines(header, body, atts)

    if item.kind == "user":
        header = "[External user input]"
        body = item.content or ""
        return _join_lines(header, body)

    if item.kind == "meta":
        header = "[Harness meta]"
        body = item.content or ""
        return _join_lines(header, body)

    if item.kind == "tool_update":
        header = "[Tool registry updated]"
        body = item.content or ""
        return _join_lines(header, body)

    if item.kind == "task_update":
        header = "[Background task update]"
        body = item.content or ""
        return _join_lines(header, body)

    if item.kind == "predecessor_handoff":
        header = "[Predecessor handoff]"
        body = item.content or ""
        return _join_lines(header, body)

    return item.content or ""


def _idx_tag(item: InboxItem) -> str:
    """Render ` idx=<n>` from `msg_key` ("<channel>:<idx>") for channel notifications.
    Empty when the item carries no msg_key. The agent uses this idx with
    `ReactToMessage(channel, msg_idx, emoji)`.
    """
    if not item.msg_key:
        return ""
    _, _, tail = item.msg_key.rpartition(":")
    return f" idx={tail}" if tail.isdigit() else ""


def _mention_banner(item: InboxItem, recipient: Optional[AgentName]) -> str:
    if not recipient or not item.mentions:
        return ""
    for m in item.mentions:
        target = m.name or m.label
        if target and target.lower() == recipient.lower():
            return "🔔 You were tagged"
    return ""


def _render_attachments(item: InboxItem) -> str:
    return render_attachment_marker([a.archive_path for a in item.attachments or []])


def render_attachment_marker(paths: list[str]) -> str:
    """The `<attachments: a.md, b.md>` marker the prompts tell agents to look for."""
    if not paths:
        return ""
    return f"<attachments: {', '.join(paths)}>"


def _join_lines(*lines: str) -> str:
    return "\n".join(ln for ln in lines if ln)
