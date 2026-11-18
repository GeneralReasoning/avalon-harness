"""SPEC §4.5.1: a CEO whose model call failed is restarted only by new input.
When every live org agent is parked, none comes, so the harness restarts the
CEO once with the quiescent marker; if the CEO fails again with the org still
quiescent, the session ends as `inference-error` instead of idling until the
wall-clock cap."""

from __future__ import annotations

import asyncio

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    TokenUsage,
    ToolCallChunk,
)
from agentorg.session import SessionConfig


class _SilentSpecialist(InferenceProvider):
    """The CEO hires one specialist, then its calls fail per `ceo_fails`. The
    specialist ends its only turn without a tool call or a message, so it
    parks without telling the CEO anything."""

    def __init__(self, ceo_fails: set[int]):
        self.ceo_fails = ceo_fails
        self.ceo_calls = 0
        self.org_name = None

    async def stream(self, request):
        who = (request.extra_headers or {}).get("X-Agent-Id")
        if who in (None, "CEO"):
            self.ceo_calls += 1
            n = self.ceo_calls
            if n in self.ceo_fails:
                raise ConnectionError("server unreachable")
            if n == 1:
                yield ToolCallChunk(id="c1", name="HireOrgAgent", args={"role": "Tester", "description": "Do it."})
            elif self.org_name is not None and n == 3:
                yield ToolCallChunk(id="c3", name="FireOrgAgent", args={
                    "name": self.org_name, "category": "mission_complete",
                    "delivered": "x", "shortfalls": "-", "score": 8,
                })
            else:
                yield ContentDeltaChunk(text="done")
                yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
                return
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
            return
        self.org_name = who
        await asyncio.sleep(0.2)
        yield ContentDeltaChunk(text="finished, waiting")
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["silent"]


async def _run(tmp_path, provider):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="silent", budget=2, mode="default",
        headless=True, permission_policy="bypass", initial_user_message="go",
    )
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=20)
    return session, result


async def test_quiescent_org_restarts_a_ceo_after_a_transient_failure(tmp_path):
    provider = _SilentSpecialist(ceo_fails={2})
    session, result = await _run(tmp_path, provider)
    assert result.end_reason == "complete"
    assert provider.ceo_calls >= 3
    assert all(a.state == "fired" for a in session.all_agents() if a.kind == "org")
    assert result.duration_s < 10


async def test_ceo_that_keeps_failing_ends_the_session_instead_of_idling(tmp_path):
    provider = _SilentSpecialist(ceo_fails=set(range(2, 100)))
    session, result = await _run(tmp_path, provider)
    assert result.end_reason == "inference-error:ConnectionError"
    assert provider.ceo_calls == 3  # the hire, the failure, one restart
    assert result.duration_s < 10
