"""Tests for the reaction primitive (§7.5)."""

from __future__ import annotations

import pytest

from agentorg.channels import ChannelStore
from agentorg.storage import SessionPaths


@pytest.mark.asyncio
async def test_reaction_toggle_add_then_remove(tmp_path):
    paths = SessionPaths(str(tmp_path), "sess")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    await store.create_channels(
        [{"name": "team"}],
        live_agent_names={"Alice", "Bob"},
        external_labels=set(),
    )
    msg, _ = await store.post(
        channel="team",
        author="Alice",
        is_external=False,
        content="design doc",
        live_agents={"Alice", "Bob"},
        external_labels=set(),
    )
    # Bob adds 👍.
    ev1, active = await store.toggle_reaction(
        channel="team", msg_key=msg.msg_key, emoji="👍", reactor="Bob"
    )
    assert active is True and ev1.op == "add"
    folds = store.fold_reactions("team", msg_key=msg.msg_key)
    groups = folds[msg.msg_key]
    assert len(groups) == 1 and groups[0].emoji == "👍" and groups[0].count == 1
    # Bob toggles again → remove.
    ev2, active2 = await store.toggle_reaction(
        channel="team", msg_key=msg.msg_key, emoji="👍", reactor="Bob"
    )
    assert active2 is False and ev2.op == "remove"
    folds = store.fold_reactions("team", msg_key=msg.msg_key)
    assert folds.get(msg.msg_key, []) == []


@pytest.mark.asyncio
async def test_reaction_groups_stable_chip_order(tmp_path):
    paths = SessionPaths(str(tmp_path), "sess")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    await store.create_channels(
        [{"name": "team"}], live_agent_names={"A", "B", "C"}, external_labels=set()
    )
    msg, _ = await store.post(
        channel="team",
        author="A",
        is_external=False,
        content="hi",
        live_agents={"A", "B", "C"},
        external_labels=set(),
    )
    # 👀 first, then 👍 — first_reacted_at order should hold.
    await store.toggle_reaction(
        channel="team", msg_key=msg.msg_key, emoji="👀", reactor="B"
    )
    await store.toggle_reaction(
        channel="team", msg_key=msg.msg_key, emoji="👍", reactor="C"
    )
    folds = store.fold_reactions("team", msg_key=msg.msg_key)
    emojis = [g.emoji for g in folds[msg.msg_key]]
    assert emojis == ["👀", "👍"]


@pytest.mark.asyncio
async def test_concurrent_toggle_reaction_no_duplicate_events(tmp_path):
    """Two concurrent toggles by the same (reactor, emoji) on the same
    message must serialize via the per-channel lock and produce exactly one
    add + one remove (not two duplicate adds).
    """
    import asyncio

    paths = SessionPaths(str(tmp_path), "sess")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    await store.create_channels(
        [{"name": "team"}], live_agent_names={"A", "B"}, external_labels=set()
    )
    msg, _ = await store.post(
        channel="team",
        author="A",
        is_external=False,
        content="hi",
        live_agents={"A", "B"},
        external_labels=set(),
    )

    # Fire two toggles simultaneously. With the per-channel lock, they
    # serialize: one observes inactive→adds, the other observes
    # active→removes. Net: zero active reactions, 2 events on disk.
    results = await asyncio.gather(
        store.toggle_reaction(channel="team", msg_key=msg.msg_key, emoji="👍", reactor="B"),
        store.toggle_reaction(channel="team", msg_key=msg.msg_key, emoji="👍", reactor="B"),
    )
    ops = sorted(ev.op for ev, _ in results)
    assert ops == ["add", "remove"], f"expected one add + one remove, got {ops}"

    folds = store.fold_reactions("team", msg_key=msg.msg_key)
    assert folds.get(msg.msg_key, []) == []


