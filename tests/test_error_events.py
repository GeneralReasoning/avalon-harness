"""Per-turn parse-error / tool-error events on Session + RolloutResult.

Exercises the two error surfaces a trainer applying a per-turn tool-error
penalty would read:

* Provider-level malformed-JSON args → ``ToolCallChunk(parse_error=True)`` →
  Session.parse_error_events (code=``"malformed-json"``).
* Loop-level ``input_schema`` validation → ``ToolResult(ok=False,
  error.code="invalid-args")`` → Session.parse_error_events (code=
  ``"invalid-args"``).
* Tool execution failure → ``ToolResult(ok=False, error.code="env-tool-error"
  | ...)`` → Session.tool_error_events.
* The malformed-JSON + downstream invalid-args pair must NOT double-record.
* RolloutResult surfaces both event streams verbatim.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
    ToolCallChunk,
)
from agentorg.session import Session, SessionConfig
from agentorg.tools.context import (
    ToolContext,
    err_result,
    infra_result,
    ok_result,
    rejected_result,
)
from agentorg.tools.registry import ToolRegistry
from agentorg.types import ToolDefinition


class _MalformedArgsProvider(InferenceProvider):
    """Emits one tool call whose args JSON was unparseable at the provider,
    then a tool-less turn on subsequent calls (session ends)."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            yield ToolCallChunk(
                id="call_bad",
                name=self._tool_name,
                args={"__raw_arguments__": '{"x": ', "__parse_error__": True},
                parse_error=True,
            )
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="tool_calls",
                response_id="chatcmpl-bad",
            )
        else:
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="stop",
                response_id="chatcmpl-done",
            )

    def model_ids(self):
        return ["mock"]


class _WellFormedFailingProvider(InferenceProvider):
    """Emits one well-formed tool call that the tool will reject at
    execution (ok=False, env-tool-error), then a tool-less turn."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            yield ToolCallChunk(
                id="call_ok",
                name=self._tool_name,
                args={"x": 1},
            )
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="tool_calls",
                response_id="chatcmpl-ok",
            )
        else:
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="stop",
                response_id="chatcmpl-done",
            )

    def model_ids(self):
        return ["mock"]


class _WellFormedSessionDeadProvider(InferenceProvider):
    """Well-formed tool call that the tool rejects with `env-session-dead`.
    The loop's env-tool circuit breaker aborts on the FIRST such code, so we
    only need one turn."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        yield ToolCallChunk(
            id="call_dead",
            name=self._tool_name,
            args={"x": 1},
        )
        yield CompletedChunk(
            usage=TokenUsage(),
            stop_reason="tool_calls",
            response_id="chatcmpl-dead",
        )

    def model_ids(self):
        return ["mock"]


class _EchoTool:
    """Local tool that returns whatever `code` it's constructed with as an
    err_result (or ok if code is None). Used to synthesise specific
    ToolResult shapes without standing up a real env."""

    def __init__(self, name: str, err_code: str | None = None, err_message: str = "boom"):
        self.definition = ToolDefinition(
            name=name,
            description=f"echo tool ({err_code or 'ok'})",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
        self._err_code = err_code
        self._err_message = err_message

    async def execute(self, ctx: ToolContext):
        if self._err_code is None:
            return ok_result({"x": ctx.args.get("x")})
        return err_result(self._err_code, self._err_message)


class _RejectingTool(_EchoTool):
    """Declines every call on policy grounds (kind="rejection"),
    the way the hire tools do when the budget is spent."""

    async def execute(self, ctx: ToolContext):
        return rejected_result(self._err_code, self._err_message)


class _InfraTool(_EchoTool):
    """Fails every call on the platform's side (kind="infra"), the way the
    OpenReward adapter does for a dead session or a transport error."""

    async def execute(self, ctx: ToolContext):
        return infra_result(self._err_code, self._err_message)


def _config(tmp_path):
    return SessionConfig(
        workspace_path=str(tmp_path),
        model="mock",
        budget=0,
        mode="default",
        headless=True,
        permission_policy="bypass",
        initial_user_message="do the thing",
    )


# ---------------------------------------------------------------------------
# Direct Session unit tests
# ---------------------------------------------------------------------------


def test_record_parse_error_appends_and_returns_copies(tmp_path):
    """Direct API: record_parse_error accumulates in emission order and the
    read-only property returns independent copies."""
    s = Session(config=_config(tmp_path), inference=_MalformedArgsProvider("x"),
                tools=ToolRegistry())
    s.record_parse_error(
        agent_name="CEO", provider_response_id="chatcmpl-1", tool_call_id="c1",
        tool_name="x", code="malformed-json", message="bad json",
    )
    s.record_parse_error(
        agent_name="CEO", provider_response_id="chatcmpl-2", tool_call_id="c2",
        tool_name="x", code="invalid-args", message="schema fail",
    )
    events = s.parse_error_events
    assert [e.tool_call_id for e in events] == ["c1", "c2"]
    assert [e.code for e in events] == ["malformed-json", "invalid-args"]
    # Mutating a returned event does not corrupt the stored one.
    events[0].tool_call_id = "hacked"
    assert s.parse_error_events[0].tool_call_id == "c1"


def test_record_tool_error_appends(tmp_path):
    s = Session(config=_config(tmp_path), inference=_MalformedArgsProvider("x"),
                tools=ToolRegistry())
    s.record_tool_error(
        agent_name="CEO", provider_response_id="chatcmpl-1", tool_call_id="c1",
        tool_name="x", code="env-tool-error", message="boom",
    )
    events = s.tool_error_events
    assert len(events) == 1 and events[0].code == "env-tool-error"


# ---------------------------------------------------------------------------
# End-to-end: chunk → loop → session → RolloutResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_json_records_parse_error_only(tmp_path):
    """A ToolCallChunk(parse_error=True) drives a parse-error event with the
    turn's provider_response_id, and the downstream `invalid-args` result
    (schema validation catches `__parse_error__` sentinel as not-in-schema)
    does NOT double-record for the same tool_call_id."""
    tool = _EchoTool("x")  # never actually runs — invalid-args short-circuits
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_MalformedArgsProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))
    session.tools = registry  # inject our tool over the harness's registered set
    result = await harness.run_to_completion(session, max_wall_s=10)

    parse_events = [e for e in result.parse_error_events if e.tool_call_id == "call_bad"]
    assert len(parse_events) == 1, parse_events
    assert parse_events[0].code == "malformed-json"
    assert parse_events[0].provider_response_id == "chatcmpl-bad"
    assert parse_events[0].tool_name == "x"
    assert result.tool_error_events == []


@pytest.mark.asyncio
async def test_execution_failure_records_tool_error(tmp_path):
    tool = _EchoTool("x", err_code="env-tool-error", err_message="upstream 500")
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_WellFormedFailingProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))
    session.tools = registry
    result = await harness.run_to_completion(session, max_wall_s=10)

    assert result.parse_error_events == []
    assert len(result.tool_error_events) == 1
    ev = result.tool_error_events[0]
    assert ev.code == "env-tool-error"
    assert ev.tool_call_id == "call_ok"
    assert ev.provider_response_id == "chatcmpl-ok"
    assert "upstream 500" in ev.message


@pytest.mark.asyncio
async def test_harness_rejection_records_in_its_own_stream(tmp_path):
    """A call the harness declined is neither a parse error nor a
    tool error: it lands on `harness_rejection_events`, which the trainer
    leaves unpenalised unless configured otherwise (§12.1)."""
    tool = _RejectingTool("x", err_code="budget-exhausted", err_message="no slots")
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_WellFormedFailingProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))
    session.tools = registry
    result = await harness.run_to_completion(session, max_wall_s=10)

    assert result.parse_error_events == []
    assert result.tool_error_events == []
    assert len(result.harness_rejection_events) == 1
    ev = result.harness_rejection_events[0]
    assert ev.code == "budget-exhausted"
    assert ev.tool_call_id == "call_ok"
    assert ev.provider_response_id == "chatcmpl-ok"
    assert "no slots" in ev.message


@pytest.mark.asyncio
async def test_env_session_dead_records_as_infra_failure(tmp_path):
    """env-session-dead is an infra failure; a training driver reads it off
    the infra stream to drop shaping on the whole rollout. It must land in
    infra_failure_events with the code preserved, and nowhere else."""
    tool = _InfraTool("x", err_code="env-session-dead", err_message="pod evicted")
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_WellFormedSessionDeadProvider("x"))
    config = _config(tmp_path)
    # The provider repeats one identical call and the stub is not env-scoped,
    # so neither the breaker nor (with the §A.8.6 rule off) a refusal ends
    # the run early: every event is the same infra failure.
    config.repeat_refuse_after = 0
    session = await harness.start_session(config=config)
    session.tools = registry
    result = await harness.run_to_completion(session, max_wall_s=10)

    assert {e.code for e in result.infra_failure_events} == {"env-session-dead"}
    assert result.tool_error_events == []
    assert result.harness_rejection_events == []


class _ForgedSentinelProvider(InferenceProvider):
    """Model emits VALID JSON args that happen to contain the literal
    `__parse_error__` key. The provider parsed them fine (parse_error stays
    False on the chunk), so the downstream schema failure is a genuine
    model error and MUST still be recorded — suppression keys on the
    provider-set ToolCall.parse_error flag, never on args content."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            yield ToolCallChunk(
                id="call_forge",
                name=self._tool_name,
                args={"__parse_error__": True, "x": 1},
            )
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="tool_calls",
                response_id="chatcmpl-forge",
            )
        else:
            yield CompletedChunk(
                usage=TokenUsage(),
                stop_reason="stop",
                response_id="chatcmpl-done",
            )

    def model_ids(self):
        return ["mock"]


@pytest.mark.asyncio
async def test_forged_parse_error_sentinel_does_not_suppress_recording(tmp_path):
    """Regression: suppression used to key on args["__parse_error__"], which
    a model can forge through ordinary emitted JSON to mute its own error
    events. The forged key fails the tool's schema (additionalProperties:
    false) -> invalid-args, and that MUST be recorded."""
    tool = _EchoTool("x")
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_ForgedSentinelProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))
    session.tools = registry
    result = await harness.run_to_completion(session, max_wall_s=10)

    forged = [e for e in result.parse_error_events if e.tool_call_id == "call_forge"]
    assert len(forged) == 1, forged
    assert forged[0].code == "invalid-args"
    assert forged[0].provider_response_id == "chatcmpl-forge"


class _NoResponseIdProvider(InferenceProvider):
    """A provider that never sets CompletedChunk.response_id (the exact
    shape of a training-provider bug this suite originally missed). Events still record, degraded to provider_response_id=None —
    the trainer's join drops them, but the streams stay truthful."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._calls = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            yield ToolCallChunk(id="call_noid", name=self._tool_name, args={"x": 1})
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
        else:
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


@pytest.mark.asyncio
async def test_provider_without_response_id_degrades_to_none_key(tmp_path):
    tool = _EchoTool("x", err_code="env-tool-error", err_message="boom")
    registry = ToolRegistry()
    registry.register(tool)
    harness = Harness(inference=_NoResponseIdProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))
    session.tools = registry
    result = await harness.run_to_completion(session, max_wall_s=10)

    evs = [e for e in result.tool_error_events if e.tool_call_id == "call_noid"]
    assert len(evs) == 1
    assert evs[0].provider_response_id is None


@pytest.mark.asyncio
async def test_terminal_tool_grading_failure_records_infra_failure(tmp_path):
    """_dispatch_terminal_tool's except branch must emit an InfraFailureEvent
    (code "terminal-tool-error") keyed on the graded turn's response id. The
    grader broke, the answer was not judged: measurable, never a tool error."""
    from agentorg.loop import _dispatch_terminal_tool
    from agentorg.types import ToolCall

    harness = Harness(inference=_WellFormedFailingProvider("x"))
    session = await harness.start_session(config=_config(tmp_path))

    class _DeadGrader:
        async def call_terminal_tool(self, text):
            raise RuntimeError("grader down")

    session.set_env_adapter(_DeadGrader())
    agent = session._agents[session.ceo_name]
    call = ToolCall(id="term1", name="terminal", args={"answer": "42"}, synthetic=True)

    await _dispatch_terminal_tool(session, agent, call, response_id="chatcmpl-term")

    evs = session.infra_failure_events
    assert len(evs) == 1
    assert evs[0].code == "terminal-tool-error"
    assert evs[0].provider_response_id == "chatcmpl-term"
    assert evs[0].tool_call_id == "term1"
    assert session.tool_error_events == []
