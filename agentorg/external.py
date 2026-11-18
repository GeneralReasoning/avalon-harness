"""External interface — operations exposed to humans, frontends and RL runners (§10)."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from agentorg.session import Session
from agentorg.storage import iter_jsonl
from agentorg.types import (
    Attachment,
    Event,
    InboxItem,
    NotificationCounts,
)


class ExternalInterface:
    """Single-trust, in-process external interface (§10.2)."""

    def __init__(self, session: Session):
        self.session = session

    async def subscribe_events(self) -> AsyncIterator[Event]:
        async for ev in self.session.events.stream():
            yield ev

    def replay_events(self, from_seq: int = 0) -> list[Event]:
        out: list[Event] = []
        for raw in iter_jsonl(self.session.paths.events_jsonl):
            try:
                ev = Event.model_validate(raw)
            except Exception:
                continue
            if ev.seq >= from_seq:
                out.append(ev)
        return out

    async def send_user_input(
        self,
        content: str,
        *,
        sender: str = "human",
        attachments: Optional[list[str]] = None,
    ) -> None:
        item = InboxItem(
            kind="user",
            content=content,
            **{"from": sender},
            attachments=[Attachment(archive_path=p) for p in (attachments or [])],
        )
        await self.session.inbox.append(self.session.ceo_name, item)
        self.session.wakes.wake(self.session.ceo_name)
        await self.session.events.emit(
            "external.input.received", {"sender": sender, "kind": "user"}
        )

    async def post_to_channel(
        self,
        channel: str,
        content: str,
        *,
        sender: str,
        attachments: Optional[list[str]] = None,
    ) -> NotificationCounts:
        session = self.session
        if sender not in session.external_labels:
            session._external_labels.add(sender)
        await session.channels.load()
        message, counts = await session.channels.post(
            channel=channel,
            author=sender,
            is_external=True,
            content=content,
            attachments=[Attachment(archive_path=p) for p in (attachments or [])],
            live_agents=session.live_agent_names(),
            external_labels=session.external_labels,
        )
        await session.events.emit(
            "external.input.received", {"sender": sender, "kind": "channel"}
        )
        from agentorg.tools.org_tools import _fanout

        await _fanout(session, message)
        return counts

    async def dm_agent(
        self,
        name: str,
        content: str,
        *,
        sender: str,
        attachments: Optional[list[str]] = None,
    ) -> None:
        session = self.session
        if sender not in session.external_labels:
            session._external_labels.add(sender)
        item = InboxItem(
            kind="dm",
            **{"from": sender},
            content=content,
            attachments=[Attachment(archive_path=p) for p in (attachments or [])],
        )
        await session.inbox.append(name, item)
        session.wakes.wake(name)
        await session.events.emit(
            "external.input.received", {"sender": sender, "kind": "dm"}
        )

    async def react_to_message(
        self,
        channel: str,
        msg_idx: int,
        emoji: str,
        *,
        sender: str,
    ) -> dict[str, object]:
        """Toggle a reaction as an external sender (§10.1, §7.5)."""
        session = self.session
        if sender not in session.external_labels:
            session._external_labels.add(sender)
        await session.channels.load()
        ch = session.channels.resolve(channel)
        if ch is None:
            raise KeyError(f"No channel named {channel!r}")
        target = session.channels.find_message_by_idx(ch.name, msg_idx)
        if target is None:
            raise KeyError(f"No message at idx={msg_idx} in #{ch.name}")
        ev, now_reacting = await session.channels.toggle_reaction(
            channel=ch.name,
            msg_key=target.msg_key,
            emoji=emoji,
            reactor=sender,
        )
        await session.events.emit(
            "channel.reaction.added" if ev.op == "add" else "channel.reaction.removed",
            {
                "channel": ch.name,
                "msg_key": target.msg_key,
                "reactor": sender,
                "emoji": emoji,
            },
        )
        if ev.op == "add" and target.author in session.all_agent_names() and not target.is_external:
            from agentorg.channels import light_snippet

            preview = light_snippet(target.content)
            item = InboxItem(
                kind="reaction",
                channel=ch.name,
                msg_key=target.msg_key,
                reactor=sender,
                emoji=emoji,
                target_content_preview=preview,
            )
            await session.inbox.append(target.author, item)
            session.wakes.wake(target.author)
        await session.events.emit(
            "external.input.received", {"sender": sender, "kind": "reaction"}
        )
        return {"now_reacting": now_reacting, "ts": ev.ts}

    async def decide_permission(
        self,
        call_id: str,
        decision: str,
        *,
        decided_by: str = "human",
    ) -> None:
        """Resolve a pending `tool.permission.asked` (§10.1). `decided_by` is for
        the caller's audit log, not the spec-exact `tool.permission.decided`
        event (`{call_id, decision}`) — the loop emits that when it resumes."""
        if decision not in ("allow", "deny"):
            raise ValueError(f"decision must be 'allow' or 'deny', got {decision!r}")
        self.session.permissions.resolve(call_id, decision)

    async def abort_session(self, reason: str = "external") -> None:
        self.session.abort(reason)
        await self.session.emit_aborted_if_needed()
