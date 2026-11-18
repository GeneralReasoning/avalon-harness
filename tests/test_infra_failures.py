"""Infra failures: a fourth `ok=False` category (SPEC §12.1, `error.kind = "infra"`).

The harness or the environment failed to carry out the call, so the call
itself was not judged: a tool's `execute` raised, the env session died, the
env transport or the env's own tool code failed, the grader failed. These
land on `Session.infra_failure_events` / `RolloutResult.infra_failure_events`,
never on the tool-error stream a trainer penalises. The one env-side failure
that IS the model's fault — the env rejecting the call as malformed, which the
OpenReward SDK raises as `ToolCallError` — stays a plain tool error.
"""

from __future__ import annotations

import json

import pytest
from openreward.api.errors import ToolCallError, ToolFailed

from agentorg.env_adapter import ENV_SESSION_DEAD_CODE
from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    TokenUsage,
    ToolCallChunk,
)
from agentorg.loop import _dispatch_tools, _serialize_tool_result
from agentorg.openreward_env import _OpenRewardTool
from agentorg.session import SessionConfig
from agentorg.tools.context import ToolContext, infra_result
from agentorg.types import Message, ToolCall, ToolDefinition


class _RaisingTool:
    definition = ToolDefinition(
        name="flaky",
        description="raises",
        input_schema={"type": "object", "properties": {}},
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx):
        raise RuntimeError("disk on fire")


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


class _OneCallProvider(InferenceProvider):
    """First turn calls `flaky` once; second turn stops."""

    def __init__(self):
        self._calls = 0

    async def stream(self, request):
        self._calls += 1
        if self._calls == 1:
            yield ToolCallChunk(id="f1", name="flaky", args={})
            yield CompletedChunk(
                usage=TokenUsage(), stop_reason="tool_calls", response_id="chatcmpl-f1"
            )
        else:
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


def _config(tmp_path) -> SessionConfig:
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="do the thing",
    )


async def _session(tmp_path, inference=None):
    session = await Harness(inference=inference or _Stub()).start_session(
        config=_config(tmp_path)
    )
    session.tools.register(_RaisingTool())
    return session


async def _call(session, agent, name: str, call_id: str):
    call = ToolCall(id=call_id, name=name, args={})
    asst = Message(role="assistant", tool_calls=[call])
    await session.append_message(agent.name, asst)
    await _dispatch_tools(session, agent, asst, [call])
    return [m for m in session.transcript(agent.name) if m.role == "tool"][-1].tool_result


def _events(session, event_type):
    path = session.paths.events_jsonl
    rows = (json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    return [r for r in rows if r.get("type") == event_type]


class _Adapter:
    def __init__(self, exc):
        class _Sess:
            async def call_tool(self, name, args):
                raise exc

        self.session = _Sess()


_ENV_DEFN = ToolDefinition(
    name="place_bet",
    description="bet",
    input_schema={"type": "object", "properties": {}},
    scope="env",
)


@pytest.mark.asyncio
async def test_openreward_malformed_call_is_a_tool_error(tmp_path):
    """Only `input_validation` is the model's mistake: the env's own input
    schema rejected the arguments it sent."""
    session = await _session(tmp_path)
    ctx = ToolContext(agent=session.agent(session.ceo_name), session=session, args={}, call_id="c1")

    exc = ToolCallError(reason="input_validation", detail="amount must be > 0")
    r = await _OpenRewardTool(_ENV_DEFN, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]
    assert r.ok is False
    assert r.error.code == "env-invalid-call"
    assert r.error.kind == "error"
    assert "input_validation" in r.error.message


@pytest.mark.asyncio
async def test_env_registration_faults_are_not_the_models_mistake(tmp_path):
    """The harness advertises env tools from the env's own `list_tools()`. If
    the env then says the tool is unknown or collides, its registration moved
    under us — an infra fault. Penalising the emitting turn trains on noise.

    The env server sets `reason` at all three of its RunToolError sites, so
    the discriminator is always present; a pre-`reason` server has its three
    message prefixes matched by the SDK's own fallback.
    """
    session = await _session(tmp_path)
    ctx = ToolContext(agent=session.agent(session.ceo_name), session=session, args={}, call_id="c3")

    for reason in ("not_found", "name_collision"):
        exc = ToolCallError(reason=reason, detail="'place_bet' is not a valid tool")
        r = await _OpenRewardTool(_ENV_DEFN, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]
        assert r.ok is False, reason
        assert r.error.code == "env-tool-unavailable", reason
        assert r.error.kind == "infra", reason

    # `bad_input_shape` is raised client-side, before anything reaches the env:
    # agentorg's own marshalling. It is left to propagate so §4.2 records it as
    # `exception` against the harness, instead of `env-failure` — which would
    # blame the env in the rollout record and feed the env circuit breaker.
    exc = ToolCallError(reason="bad_input_shape", detail="input must be a dictionary")
    with pytest.raises(ToolCallError):
        await _OpenRewardTool(_ENV_DEFN, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_env_refusal_after_episode_finished_is_a_rejection(tmp_path):
    """The env server refuses every call once a tool on the session returned
    finished=True (another agent's concurrent submit won). Nothing ran and the
    locked reward stands, so it is neither the model's mistake nor an infra
    fault: a rejection, the same code the harness-side guard uses."""
    session = await _session(tmp_path)
    ctx = ToolContext(agent=session.agent(session.ceo_name), session=session, args={}, call_id="c4")

    exc = ToolCallError(reason="episode_finished", detail="the episode has finished")
    r = await _OpenRewardTool(_ENV_DEFN, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]
    assert r.ok is False
    assert r.error.code == "episode-finished"
    assert r.error.kind == "rejection"


@pytest.mark.asyncio
async def test_openreward_other_exceptions_are_infra_failures(tmp_path):
    """Everything else out of `call_tool` is the env or its transport
    failing: a transport error, an unknown exception. kind is "infra".

    `ToolFailed` is infra too, but NOT retryable: the SDK calls `_mark_dead`
    and POSTs `/delete` before raising it, so the session is already gone and
    every later call raises SessionTerminatedError. It gets the dead-session
    code, like the other unrecoverable deaths."""
    session = await _session(tmp_path)
    ctx = ToolContext(agent=session.agent(session.ceo_name), session=session, args={}, call_id="c2")

    class MaxRetriesError(Exception):
        """Mimics the SDK class name the adapter classifies on."""

    cases = [
        (ToolFailed("ZeroDivisionError in env code"), ENV_SESSION_DEAD_CODE),
        (ValueError("transient hiccup"), "env-failure"),
        (MaxRetriesError("(environment session sid=abc) Exceeded 5 retries"), ENV_SESSION_DEAD_CODE),
    ]
    for exc, code in cases:
        r = await _OpenRewardTool(_ENV_DEFN, _Adapter(exc)).execute(ctx)  # type: ignore[arg-type]
        assert r.ok is False
        assert r.error.code == code, exc
        assert r.error.kind == "infra", exc


@pytest.mark.asyncio
async def test_raising_tool_is_an_infra_failure(tmp_path):
    """§4.2: a tool whose `execute` raises surfaces as `exception`, kind
    "infra", and lands on the infra stream only."""
    session = await _session(tmp_path)
    ceo = session.agent(session.ceo_name)

    result = await _call(session, ceo, "flaky", "k1")
    assert result.ok is False
    assert result.error.code == "exception"
    assert result.error.kind == "infra"
    assert result.error.message == "RuntimeError: disk on fire"
    [ev] = session.infra_failure_events
    assert (ev.code, ev.tool_call_id, ev.agent_name) == ("exception", "k1", ceo.name)
    assert session.tool_error_events == []
    assert session.harness_rejection_events == []
    assert session.parse_error_events == []


@pytest.mark.asyncio
async def test_infra_kind_is_rendered_to_the_model(tmp_path):
    session = await _session(tmp_path)
    ceo = session.agent(session.ceo_name)

    rendered = json.loads(_serialize_tool_result(await _call(session, ceo, "flaky", "r1")))
    assert rendered == {
        "ok": False,
        "error": {"code": "exception", "message": "RuntimeError: disk on fire", "kind": "infra"},
    }
    assert json.loads(_serialize_tool_result(infra_result("env-failure", "x")))["error"]["kind"] == "infra"


@pytest.mark.asyncio
async def test_infra_kind_is_persisted_in_the_event_copy(tmp_path):
    """§12.4: both persisted copies agree."""
    session = await _session(tmp_path)
    ceo = session.agent(session.ceo_name)
    await _call(session, ceo, "flaky", "p1")

    [completed] = [
        e for e in _events(session, "tool.completed") if e["payload"].get("call_id") == "p1"
    ]
    assert completed["payload"]["error"]["kind"] == "infra"


@pytest.mark.asyncio
async def test_rollout_result_exports_the_infra_stream(tmp_path):
    harness = Harness(inference=_OneCallProvider())
    session = await harness.start_session(config=_config(tmp_path))
    session.tools.register(_RaisingTool())
    result = await harness.run_to_completion(session, max_wall_s=10)

    [ev] = result.infra_failure_events
    assert ev.code == "exception"
    assert ev.provider_response_id == "chatcmpl-f1"
    assert result.tool_error_events == []


def test_tool_failed_is_a_session_death_sign():
    """`is_session_death_message` is the single source of truth shared by the
    adapter and training-data builders. `ToolFailed` belongs in it:
    the SDK calls `_mark_dead` and POSTs `/delete` before raising, so the
    session is gone and nothing on it can be retried."""
    from agentorg.openreward_errors import is_session_death_message

    assert is_session_death_message("ToolFailed: ZeroDivisionError in env code")
    assert not is_session_death_message("ValueError: transient hiccup")
    assert not is_session_death_message("env tool reported ToolFailedValidation")
