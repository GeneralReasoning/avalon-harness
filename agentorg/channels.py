"""Channel storage, subscriptions, virtual channels (§7)."""

from __future__ import annotations

import asyncio
import re
from typing import Iterable, Optional

from agentorg.storage import (
    SessionPaths,
    append_jsonl_async,
    iter_jsonl,
    read_json,
    to_jsonable,
    write_json_async,
)
from agentorg.types import (
    Attachment,
    Channel,
    ChannelMessage,
    Mention,
    NotificationCounts,
    ReactionEvent,
    ReactionGroup,
    SubscriptionLevel,
)

LIGHT_SNIPPET_MAX_CHARS = 200
# Reserved virtual-channel names per §7.8 — the literal "self" alias and the
# CEO's name literal "CEO" (which resolves to the CEO's agent-trace channel).
RESERVED_VIRTUAL_NAMES = {"self", "ceo"}
CHANNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
MENTION_RE = re.compile(r"@([A-Za-z0-9_-]{1,50})")


class ChannelStore:
    """Filesystem-backed channel registry."""

    def __init__(self, paths: SessionPaths) -> None:
        self._paths = paths
        self._lock = asyncio.Lock()
        # Per-channel locks guarding read-modify-write paths (post
        # idx assignment, toggle_reaction). Created lazily on first
        # use; identity stable across the channel's life.
        self._channel_locks: dict[str, asyncio.Lock] = {}
        # Per-channel monotonic post-index counter (SPEC §3.4). Lazy-
        # initialised on first post from `total_messages(channel)` so a
        # resumed session continues from the right idx. Guarded by the
        # per-channel lock.
        self._post_counters: dict[str, int] = {}
        self._registry: dict[str, Channel] = {}
        self._loaded = False

        # Incremental reaction-fold cache (§7.5). Each channel's reaction
        # log is replayed once on first access into `_reaction_latest` and
        # `_reaction_earliest_add`; subsequent `toggle_reaction` calls
        # update these maps in-place so `fold_reactions` is O(latest)
        # instead of O(log size).
        self._reaction_latest: dict[str, dict[tuple[str, str, str], ReactionEvent]] = {}
        self._reaction_earliest_add: dict[str, dict[tuple[str, str, str], int]] = {}
        self._reaction_event_count: dict[str, int] = {}
        self._reaction_cache_loaded: set[str] = set()

    # ---- registry IO -------------------------------------------------------

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            meta = read_json(self._paths.channels_meta) or {"channels": {}}
            for name, raw in meta.get("channels", {}).items():
                try:
                    self._registry[name] = Channel.model_validate(raw)
                except Exception:
                    continue
            self._loaded = True

    def _channel_lock(self, name: str) -> asyncio.Lock:
        key = name.lower()
        if key not in self._channel_locks:
            self._channel_locks[key] = asyncio.Lock()
        return self._channel_locks[key]

    async def _persist(self) -> None:
        meta = {
            "channels": {n: to_jsonable(c) for n, c in self._registry.items()},
        }
        await write_json_async(self._paths.channels_meta, meta)

    def list_regular(self) -> list[Channel]:
        return list(self._registry.values())

    def exists_case_sensitive(self, name: str) -> bool:
        return name in self._registry

    def resolve(self, name: str) -> Optional[Channel]:
        # Case-insensitive resolution. All channel lookups should go through
        # this method; the registry stores channels under their original-case
        # `name` key, so a plain `dict.get` would miss case mismatches.
        for cname, ch in self._registry.items():
            if cname.lower() == name.lower():
                return ch
        return None

    # ---- create -----------------------------------------------------------

    async def create_channels(
        self,
        specs: list[dict],
        *,
        live_agent_names: set[str],
        external_labels: set[str],
    ) -> tuple[list[Channel], list[Channel]]:
        """Batch-create channels — name-idempotent (§7.1).

        Names that already exist in the registry are skipped and returned in
        the second list, so concurrent callers (CEO + freshly spawned org
        agents) don't race on `CreateChannels`. Validation errors (bad name,
        reserved virtual name, in-batch duplicate) still raise.

        Holds the store-level `_lock` across validation + mutation + persist
        so two concurrent calls can't both pass uniqueness checks against an
        identical registry snapshot.

        Returns (created, existing) — both lists of `Channel` instances.
        """
        await self.load()

        async with self._lock:
            new_channels: list[Channel] = []
            existing_channels: list[Channel] = []
            seen_local: set[str] = set()
            existing_lookup = {n.lower(): n for n in self._registry.keys()}
            reserved = RESERVED_VIRTUAL_NAMES | {n.lower() for n in live_agent_names}

            for spec in specs:
                name = spec.get("name", "")
                if not CHANNEL_NAME_RE.match(name):
                    raise ValueError(
                        f"Invalid channel name {name!r} — must match [A-Za-z0-9_-]{{1,50}}"
                    )
                lower = name.lower()
                if lower in reserved:
                    raise ValueError(
                        f"Channel name {name!r} collides with a reserved virtual channel (agent name / 'self')."
                    )
                if lower in seen_local:
                    raise ValueError(f"Channel {name!r} is duplicated in batch.")
                seen_local.add(lower)
                if lower in existing_lookup:
                    existing_channels.append(self._registry[existing_lookup[lower]])
                    continue
                default_sub = spec.get("default_subscription", spec.get("default_level", "light"))
                subs = dict(spec.get("subscriptions", {}))
                # Validate per-agent subs
                for k in list(subs.keys()):
                    if k not in live_agent_names and k not in external_labels:
                        # Keep — agent may join later; spec says subscriptions are per-agent override.
                        pass
                channel = Channel(
                    name=name,
                    session_id=self._paths.session_id,
                    default_subscription=default_sub,
                    subscriptions=subs,
                )
                new_channels.append(channel)

            for ch in new_channels:
                self._registry[ch.name] = ch
            if new_channels:
                await self._persist()
            return new_channels, existing_channels

    async def set_subscription(
        self, channel: str, agent: str, level: SubscriptionLevel
    ) -> bool:
        await self.load()
        ch = self.resolve(channel)
        if ch is None:
            return False
        ch.subscriptions[agent] = level
        await self._persist()
        return True

    def effective_subscription(self, channel: Channel, agent: str) -> SubscriptionLevel:
        if agent in channel.subscriptions:
            return channel.subscriptions[agent]
        return channel.default_subscription

    # ---- post -------------------------------------------------------------

    async def post(
        self,
        *,
        channel: str,
        author: str,
        is_external: bool,
        content: str,
        attachments: Optional[list[Attachment]] = None,
        live_agents: set[str],
        external_labels: set[str],
    ) -> tuple[ChannelMessage, NotificationCounts]:
        await self.load()
        ch = self.resolve(channel)
        if ch is None:
            raise KeyError(f"Channel {channel!r} does not exist")

        mentions = parse_mentions(
            content,
            live_agents=live_agents,
            external_labels=external_labels,
        )

        # Per-channel idx is the sole ordering primitive (SPEC §3.4).
        # Assign + append under the per-channel lock so two concurrent
        # posts get strictly monotonic, non-colliding idx values.
        async with self._channel_lock(ch.name):
            if ch.name not in self._post_counters:
                self._post_counters[ch.name] = self.total_messages(ch.name)
            idx = self._post_counters[ch.name]
            self._post_counters[ch.name] = idx + 1
            msg_key = f"{ch.name}:{idx}"
            message = ChannelMessage(
                channel=ch.name,
                idx=idx,
                msg_key=msg_key,
                author=author,
                is_external=is_external,
                content=content,
                attachments=attachments or [],
                mentions=mentions,
            )
            await append_jsonl_async(self._paths.channel_path(ch.name), message)

        counts = NotificationCounts()
        mentioned_names = {m.name.lower() for m in mentions if m.kind == "agent" and m.name}
        mentioned_labels = {m.label.lower() for m in mentions if m.kind == "external" and m.label}
        for agent in live_agents:
            if agent == author and not is_external:
                continue  # author doesn't notify itself
            level = self.effective_subscription(ch, agent)
            tagged = agent.lower() in mentioned_names
            if tagged:
                counts.full += 1
                counts.tagged.append(agent)
            else:
                if level == "full":
                    counts.full += 1
                elif level == "light":
                    counts.light += 1
                else:
                    counts.muted += 1
        # External-label mentions are recorded too.
        for lbl in mentioned_labels:
            counts.tagged.append(lbl)
        # §A.4 / §A.5 — surface tags that didn't land on a live teammate
        # (fired agent or typo) so the poster can retarget.
        counts.tagged_but_not_running = [
            m.name for m in mentions if m.kind == "stale" and m.name
        ]
        return message, counts

    def iter_messages(
        self,
        channel: str,
        *,
        since_idx: Optional[int] = None,
        before_idx: Optional[int] = None,
    ) -> Iterable[ChannelMessage]:
        path = self._paths.channel_path(channel)
        for raw in iter_jsonl(path):
            try:
                m = ChannelMessage.model_validate(raw)
            except Exception:
                continue
            # `since_idx` is INCLUSIVE (returns idx >= since_idx) so that
            # `since_idx=0` reads from the very first message; `before_idx`
            # stays EXCLUSIVE (returns idx < before_idx) for scrollback.
            if since_idx is not None and m.idx < since_idx:
                continue
            if before_idx is not None and m.idx >= before_idx:
                continue
            yield m

    def total_messages(self, channel: str) -> int:
        path = self._paths.channel_path(channel)
        return sum(1 for _ in iter_jsonl(path))

    # ---- reactions  (§7.5)  ------------------------------------------------

    def find_message_by_idx(self, channel: str, msg_idx: int) -> Optional[ChannelMessage]:
        """Locate a channel message by its per-channel `idx`. None if absent."""
        for msg in self.iter_messages(channel):
            if msg.idx == msg_idx:
                return msg
        return None

    def iter_reactions(self, channel: str) -> Iterable[ReactionEvent]:
        path = self._paths.channel_reactions_path(channel)
        for raw in iter_jsonl(path):
            try:
                yield ReactionEvent.model_validate(raw)
            except Exception:
                continue

    def _ensure_reaction_cache(self, channel: str) -> None:
        if channel in self._reaction_cache_loaded:
            return
        latest = self._reaction_latest.setdefault(channel, {})
        earliest = self._reaction_earliest_add.setdefault(channel, {})
        count = 0
        for ev in self.iter_reactions(channel):
            key = (ev.msg_key, ev.emoji, ev.reactor)
            latest[key] = ev
            if ev.op == "add" and key not in earliest:
                earliest[key] = count
            count += 1
        self._reaction_event_count[channel] = count
        self._reaction_cache_loaded.add(channel)

    def _apply_reaction_to_cache(self, channel: str, ev: ReactionEvent) -> None:
        """Update in-memory fold state with a freshly-appended event.

        Called under the channel lock after the disk append succeeds.
        Cache and disk briefly disagree during the await on disk write;
        callers of `fold_reactions` outside the lock may see stale state
        for that window. Same eventual-consistency guarantee as before.
        """
        latest = self._reaction_latest.setdefault(channel, {})
        earliest = self._reaction_earliest_add.setdefault(channel, {})
        key = (ev.msg_key, ev.emoji, ev.reactor)
        idx = self._reaction_event_count.get(channel, 0)
        latest[key] = ev
        if ev.op == "add" and key not in earliest:
            earliest[key] = idx
        self._reaction_event_count[channel] = idx + 1

    def fold_reactions(
        self, channel: str, msg_key: Optional[str] = None
    ) -> dict[str, list[ReactionGroup]]:
        """Fold the reaction log into per-message ReactionGroup lists.

        Returns `{msg_key: [ReactionGroup, ...]}`. If `msg_key` is given, only
        that message's groups are folded.

        Per SPEC §3.4: for each `(msg_key, emoji, reactor)` triple, the
        latest event in **append order** decides whether that reactor
        currently has the emoji on. UI orders groups by `first_reaction_idx`
        — the append-order index of the earliest still-in-effect add for
        that group — so chip ordering is stable across toggles.
        """
        self._ensure_reaction_cache(channel)
        latest = self._reaction_latest.get(channel, {})
        earliest_add = self._reaction_earliest_add.get(channel, {})

        groups: dict[str, dict[str, ReactionGroup]] = {}
        for (mk, emoji, reactor), ev in latest.items():
            if msg_key is not None and mk != msg_key:
                continue
            if ev.op != "add":
                continue
            per_msg = groups.setdefault(mk, {})
            grp = per_msg.get(emoji)
            first_idx = earliest_add.get((mk, emoji, reactor), 0)
            if grp is None:
                grp = ReactionGroup(
                    emoji=emoji,
                    count=0,
                    reactors=[],
                    first_reaction_idx=first_idx,
                )
                per_msg[emoji] = grp
            grp.count += 1
            grp.reactors.append(reactor)
            grp.first_reaction_idx = min(grp.first_reaction_idx, first_idx)

        out: dict[str, list[ReactionGroup]] = {}
        for mk, per_msg in groups.items():
            ordered = sorted(per_msg.values(), key=lambda g: g.first_reaction_idx)
            out[mk] = ordered
        return out

    async def toggle_reaction(
        self,
        *,
        channel: str,
        msg_key: str,
        emoji: str,
        reactor: str,
    ) -> tuple[ReactionEvent, bool]:
        """Append an add/remove event based on current fold; return (event, now_reacting).

        Holds a per-channel lock across fold + append so two concurrent
        toggles by the same (reactor, emoji) can't both observe
        active=False and both write `op=add` — the fold would still be
        correct (latest event wins), but the on-disk log would gain a
        spurious duplicate.
        """
        async with self._channel_lock(channel):
            groups = self.fold_reactions(channel, msg_key=msg_key)
            active = False
            for grp in groups.get(msg_key, []):
                if grp.emoji == emoji and reactor in grp.reactors:
                    active = True
                    break
            op = "remove" if active else "add"
            ev = ReactionEvent(
                channel=channel,
                msg_key=msg_key,
                emoji=emoji,
                reactor=reactor,
                op=op,
            )
            await append_jsonl_async(self._paths.channel_reactions_path(channel), ev)
            self._apply_reaction_to_cache(channel, ev)
            return ev, (op == "add")


# ---------------------------------------------------------------------------
# Mention parsing  (§7.4)
# ---------------------------------------------------------------------------


def parse_mentions(
    content: str,
    *,
    live_agents: set[str],
    external_labels: set[str],
) -> list[Mention]:
    """Extract @name tokens from a message body and resolve them."""
    found: list[Mention] = []
    seen: set[str] = set()
    for match in MENTION_RE.finditer(content):
        token = match.group(1)
        if token.lower() in seen:
            continue
        seen.add(token.lower())
        resolved = _resolve(token, live_agents, external_labels)
        if resolved is not None:
            found.append(resolved)
        else:
            found.append(Mention(kind="stale", name=token))
    return found


def _resolve(token: str, live_agents: set[str], external_labels: set[str]) -> Optional[Mention]:
    for agent in live_agents:
        if agent.lower() == token.lower():
            return Mention(kind="agent", name=agent)
    for label in external_labels:
        if label.lower() == token.lower():
            return Mention(kind="external", label=label)
    return None


def light_snippet(content: str, *, limit: int = LIGHT_SNIPPET_MAX_CHARS) -> str:
    body = content.strip().replace("\n", " ")
    if len(body) <= limit:
        return body
    return body[: limit - 1].rstrip() + "…"
