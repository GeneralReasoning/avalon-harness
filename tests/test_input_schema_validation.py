"""Input-schema validation (SPEC §5.1).

The harness MUST validate `call.args` against `tool.definition.input_schema`
before invoking `execute`; failure surfaces as
`ToolResult(ok=False, error.code="invalid-args")`.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.loop import _validate_tool_args
from agentorg.session import SessionConfig
from agentorg.tools.context import ToolContext, ok_result
from agentorg.types import ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Unit tests for the validator helper itself.
# ---------------------------------------------------------------------------


def test_validate_empty_schema_accepts_everything():
    assert _validate_tool_args({}, {"arbitrary": "args"}) is None
    assert _validate_tool_args({"type": "object", "properties": {}}, {}) is None


def test_validate_missing_required_field():
    schema = {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    }
    err = _validate_tool_args(schema, {})
    assert err is not None
    assert "file_path" in err


def test_validate_wrong_type():
    schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    }
    err = _validate_tool_args(schema, {"limit": "five"})
    assert err is not None
    # Path / message should be informative enough for the model to self-correct.
    assert "/limit" in err
    assert "integer" in err or "string" in err  # mentions the type mismatch


def test_validate_broken_schema_lets_call_through():
    """If the SCHEMA itself is invalid (operator error), we don't block the
    model — that would silently break every tool call. Log and pass."""
    busted = {"type": "object", "properties": "not-a-dict"}
    # Should not raise; should return None (allow through).
    assert _validate_tool_args(busted, {}) is None


def test_validate_nested_path():
    schema = {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"role": {"type": "string"}},
                    "required": ["role"],
                },
            }
        },
    }
    err = _validate_tool_args(schema, {"agents": [{"role": "ok"}, {}]})
    assert err is not None
    assert "/agents/1" in err


# ---------------------------------------------------------------------------
# Integration: invalid args returned via the dispatch boundary.
# ---------------------------------------------------------------------------


class _StrictTool:
    """Test tool with a strict schema; would crash on bad args if not validated."""

    definition = ToolDefinition(
        name="StrictEcho",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["message"],
        },
        read_only=True,
        scope="local",
    )

    def __init__(self):
        self.execute_calls = 0

    async def execute(self, ctx: ToolContext) -> ToolResult:
        self.execute_calls += 1
        # Would KeyError if validation didn't fire and message was missing.
        return ok_result({"echoed": ctx.args["message"]})


class _ProgrammableProvider(InferenceProvider):
    """Emits scripted tool calls (one per stream)."""

    def __init__(self, calls):
        self._calls = list(calls)
        self._idx = 0

    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        if self._idx >= len(self._calls):
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
            return
        name, args = self._calls[self._idx]
        self._idx += 1
        yield ToolCallChunk(id=f"c{self._idx}", name=name, args=args)
        yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_dispatch_returns_invalid_args_for_missing_required(tmp_path):
    """Missing required field → invalid-args error code, execute NOT called."""
    tool = _StrictTool()
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        headless=True,
        permission_policy="bypass",
        initial_user_message="go",
    )
    harness = Harness(inference=_ProgrammableProvider([
        ("StrictEcho", {}),  # missing required "message"
    ]))
    session = await harness.start_session(config=config)
    session.tools.register(tool)
    await harness.run_to_completion(session, max_wall_s=5)

    # execute() must NOT have been called.
    assert tool.execute_calls == 0

    # The transcript records a tool message whose result is the
    # invalid-args error.
    transcript = session.transcript(session.ceo_name)
    tool_msgs = [m for m in transcript if m.role == "tool"]
    assert len(tool_msgs) == 1
    res = tool_msgs[0].tool_result
    assert res is not None
    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "invalid-args"
    assert "message" in res.error.message


@pytest.mark.asyncio
async def test_dispatch_returns_invalid_args_for_wrong_type(tmp_path):
    """Wrong type for a property → invalid-args."""
    tool = _StrictTool()
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        headless=True,
        permission_policy="bypass",
        initial_user_message="go",
    )
    harness = Harness(inference=_ProgrammableProvider([
        ("StrictEcho", {"message": "hi", "count": "not-an-int"}),
    ]))
    session = await harness.start_session(config=config)
    session.tools.register(tool)
    await harness.run_to_completion(session, max_wall_s=5)

    assert tool.execute_calls == 0
    transcript = session.transcript(session.ceo_name)
    tool_msgs = [m for m in transcript if m.role == "tool"]
    assert len(tool_msgs) == 1
    res = tool_msgs[0].tool_result
    assert res.ok is False
    assert res.error.code == "invalid-args"


@pytest.mark.asyncio
async def test_dispatch_valid_args_pass_through(tmp_path):
    """Valid args → execute IS called, normal result."""
    tool = _StrictTool()
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        headless=True,
        permission_policy="bypass",
        initial_user_message="go",
    )
    harness = Harness(inference=_ProgrammableProvider([
        ("StrictEcho", {"message": "hello", "count": 2}),
    ]))
    session = await harness.start_session(config=config)
    session.tools.register(tool)
    await harness.run_to_completion(session, max_wall_s=5)

    assert tool.execute_calls == 1
    transcript = session.transcript(session.ceo_name)
    tool_msgs = [m for m in transcript if m.role == "tool"]
    assert len(tool_msgs) == 1
    res = tool_msgs[0].tool_result
    assert res.ok is True
    assert res.output == {"echoed": "hello"}
