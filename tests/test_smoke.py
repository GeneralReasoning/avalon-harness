"""Offline smoke test — no network, mock inference."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.session import SessionConfig


class MockInferenceProvider(InferenceProvider):
    """Deterministic provider: emits a single tool-less assistant message."""

    def __init__(self, content: str = "ok"):
        self._content = content

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield ContentDeltaChunk(text=self._content)
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


class ToolCallProvider(InferenceProvider):
    """First turn: emit tool call. Subsequent turns: tool-less assistant (done)."""

    def __init__(self, tool_name: str, args: dict, content: str = ""):
        self._tool_name = tool_name
        self._args = args
        self._content = content
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            if self._content:
                yield ContentDeltaChunk(text=self._content)
            yield ToolCallChunk(id="call_0", name=self._tool_name, args=self._args)
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
        else:
            yield ContentDeltaChunk(text="done")
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


async def _run_session(harness, config):
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=30)
    return session, result


@pytest.mark.asyncio
async def test_solo_ceo_no_tools(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="What's 1+1?",
    )
    harness = Harness(inference=MockInferenceProvider("It's 2."))
    session, result = await _run_session(harness, config)
    transcript = session.transcript(session.ceo_name)
    assert any(m.role == "assistant" and (m.content or "").strip() == "It's 2." for m in transcript)
    assert result.used_turns >= 1


@pytest.mark.asyncio
async def test_post_to_channel(tmp_path):
    # Channels are a multi-agent feature: the channel tools are only registered
    # when budget > 0 (a solo budget=0 session has no peers to talk to).
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=1,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="Just verify the channels.",
    )
    harness = Harness(
        inference=ToolCallProvider("CreateChannels", {"channels": [{"name": "general"}]}, ""),
    )
    session, result = await _run_session(harness, config)
    await session.channels.load()
    assert session.channels.resolve("general") is not None


@pytest.mark.asyncio
async def test_archive_roundtrip(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="Write a doc.",
    )
    harness = Harness(
        inference=ToolCallProvider(
            "WriteToArchive", {"path": "notes/foo.md", "content": "hello"}, ""
        ),
    )
    session, result = await _run_session(harness, config)
    assert (Path(session.paths.archive_dir) / "notes/foo.md").exists()
