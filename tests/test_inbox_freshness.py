"""Freshness invariant (§4.3): the inbox is drained as late as possible —
after compaction, immediately before the inference request is assembled — so a
message delivered during a slow/queued compaction (or the previous turn's long
inference) is folded into *this* turn's request rather than deferred a whole
turn.

Regression for the ordering bug where the implicit top-of-loop drain ran
*before* compaction + a (potentially minutes-long, queued) inference. A peer's
report delivered during that window only surfaced on the next turn; on a
terminal turn there is no next turn, so the report was persisted *after* the
final submission and never seen by the model that submitted.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

import agentorg.compaction as compaction_mod
from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    TokenUsage,
)
from agentorg.session import SessionConfig
from agentorg.types import InboxItem


class RecordingProvider(InferenceProvider):
    """Records the message array of every inference request, then emits a
    tool-less assistant message (which ends a solo, env-less CEO run)."""

    def __init__(self) -> None:
        self.requests: list[list] = []

    async def stream(self, request) -> AsyncIterator:
        self.requests.append(list(request.messages))
        yield ContentDeltaChunk(text="done")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


@pytest.mark.asyncio
async def test_message_delivered_during_compaction_is_seen_same_turn(tmp_path, monkeypatch):
    MARKER = "FRESHNESS_MARKER_REPORT_BODY"
    injected = {"done": False}

    async def injecting_microcompact(session, name, *args, **kwargs):
        # Simulate a peer's report landing in the inbox *during* this turn's
        # compaction step — i.e. after the loop's top-of-iteration park check
        # but before the post-compaction drain. Inject exactly once.
        if not injected["done"]:
            injected["done"] = True
            await session.inbox.append(
                name,
                InboxItem(
                    kind="report",
                    from_="ProofWriter",
                    payload={"status": "completed", "summary": MARKER},
                ),
            )

    async def noop_autocompact(session, name, *args, **kwargs):
        return None

    monkeypatch.setattr(compaction_mod, "microcompact_if_needed", injecting_microcompact)
    monkeypatch.setattr(compaction_mod, "auto_compact_if_needed", noop_autocompact)

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="Solve it.",
    )
    provider = RecordingProvider()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=30)

    assert provider.requests, "no inference request was captured"
    # Exactly one inference ran: the report was delivered during that turn's
    # compaction, so it must appear in that same request.
    first_request = provider.requests[0]
    blob = "\n".join(
        m.content for m in first_request if isinstance(getattr(m, "content", None), str)
    )
    assert MARKER in blob, (
        "a report delivered during compaction was not folded into the same "
        "turn's inference request — freshness invariant (§4.3) violated"
    )
