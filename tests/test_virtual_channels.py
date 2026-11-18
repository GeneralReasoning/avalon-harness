"""CEO is a reserved virtual-channel name; rejection works on CreateChannels."""

from __future__ import annotations

import pytest

from agentorg.channels import ChannelStore
from agentorg.storage import SessionPaths


@pytest.mark.asyncio
async def test_create_channel_rejects_ceo_and_self(tmp_path):
    paths = SessionPaths(str(tmp_path), "sess")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    with pytest.raises(ValueError):
        await store.create_channels(
            [{"name": "CEO"}], live_agent_names=set(), external_labels=set()
        )
    with pytest.raises(ValueError):
        await store.create_channels(
            [{"name": "self"}], live_agent_names=set(), external_labels=set()
        )


@pytest.mark.asyncio
async def test_create_channel_rejects_agent_name(tmp_path):
    paths = SessionPaths(str(tmp_path), "sess")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    with pytest.raises(ValueError):
        await store.create_channels(
            [{"name": "RedKant"}],
            live_agent_names={"RedKant"},
            external_labels=set(),
        )
