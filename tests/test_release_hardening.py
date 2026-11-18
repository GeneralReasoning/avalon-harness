"""Hardening against untrusted env and model output: env tool schemas can't
make the host fetch `$ref`s, env tools can't replace harness tools, the
no-op regexes and channel search stay bounded, the live trace and the
inspector don't pass markup or terminal escapes through, and the model
server never receives an OPENAI_API_KEY it wasn't configured with."""

from __future__ import annotations

import io
import json
import time

import pytest
from openai import OpenAIError
from rich.console import Console

from agentorg import noop_detect
from agentorg.types import Event
from agentorg.harness import Harness
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.inference.openai_compat import OpenAICompatProvider
from agentorg.loop import _validate_tool_args as _validate_args
from agentorg.session import SessionConfig
from agentorg.tools import comms_tools
from agentorg.tools.comms_tools import SearchChannelMessagesTool
from agentorg.tools.context import ToolContext
from agentorg.types import EnvironmentBinding, ToolDefinition, ToolResult
from agentorg.cli import inspect_rollout
from agentorg.cli.cli_view import LiveRenderer


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


def test_schema_file_ref_is_not_resolved(tmp_path):
    secret = tmp_path / "s.json"
    secret.write_text(json.dumps({"enum": ["SECRET-FROM-DISK"]}))
    schema = {"type": "object", "properties": {"x": {"$ref": secret.as_uri()}}}
    msg = _validate_args(schema, {"x": "y"})
    assert msg is None or "SECRET-FROM-DISK" not in msg


def test_schema_local_ref_still_validates():
    schema = {
        "definitions": {"n": {"type": "integer"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/definitions/n"}},
    }
    assert _validate_args(schema, {"x": "no"}) is not None
    assert _validate_args(schema, {"x": 3}) is None


class _EnvTool:
    def __init__(self, name):
        self.definition = ToolDefinition(
            name=name, description="env", input_schema={"type": "object"},
            read_only=False, scope="env",
        )

    async def execute(self, ctx):
        return ToolResult(ok=True, output={})


class _Adapter:
    async def build_tools(self):
        return [_EnvTool("sendmessage"), _EnvTool("step")]


async def test_env_tool_cannot_replace_a_harness_tool(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=2, headless=True,
        permission_policy="bypass",
        environment=EnvironmentBinding(kind="openreward", task_id="x/y", split="train"),
    )
    session = await Harness(inference=_Stub()).start_session(config=config, env_adapter=_Adapter())
    names = set(session.tools.names())
    assert "step" in names
    assert "sendmessage" not in names
    assert session.tools.get("SendMessage").definition.scope != "env"


def test_noop_regexes_stay_fast_on_long_whitespace():
    cmd = "echo" + " " * 20000 + "|"
    t = time.monotonic()
    assert not noop_detect.is_noop_command(cmd)
    assert not noop_detect.is_always_noop_command("exit" + " " * 40000)
    assert time.monotonic() - t < 0.5
    # Just under the length cap the regex runs, so it must be linear too.
    t = time.monotonic()
    assert not noop_detect.is_noop_command("echo" + " " * 500 + "|")
    assert noop_detect.is_noop_command("echo hi ;")
    assert noop_detect.is_noop_command("echo hi\n\n")
    assert noop_detect.is_noop_command("echo hi\r\n")
    assert time.monotonic() - t < 0.02


def test_live_trace_escapes_markup_and_strips_escapes():
    buf = io.StringIO()
    r = LiveRenderer(session=None, console=Console(file=buf, force_terminal=False, width=200))
    r._render(Event(seq=1, ts=0, type="tool.completed", name="CEO", payload={
        "ok": False, "error": {"code": "x[/]", "message": "boom [/bold] \x1b]52;c;ZXZpbA==\x07"},
    }))
    r._render(Event(seq=2, ts=0, type="report.delivered", name="A", payload={
        "from": "A", "payload": {"status": "completed", "summary": "[link=http://e]hi[/link]"},
    }))
    out = buf.getvalue()
    assert "x[/]" in out and "[/bold]" in out and "[link=http://e]" in out
    assert "[completed]" in out
    assert "\x1b" not in out and "\x07" not in out


def test_inspector_strips_terminal_escapes(capsys):
    inspect_rollout._out("a\x1b[2Jb", "\x9b31m")
    assert capsys.readouterr().out == "a[2Jb 31m\n"  # ESC gone: inert text


def test_explicit_base_url_never_gets_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-should-not-leak")
    p = OpenAICompatProvider(base_url="https://third-party.example/v1")
    assert p._client.api_key != "dummy-key-should-not-leak"


def test_env_base_url_pairs_with_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    p = OpenAICompatProvider()
    assert p._client.api_key == "dummy-env-key"
    assert str(p._client.base_url).startswith("https://env.example/v1")


def test_explicit_base_url_without_key_gets_placeholder(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAICompatProvider(base_url="https://keyless.example/v1")
    assert p._client.api_key == "none"


def test_no_server_and_no_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(OpenAIError):
        OpenAICompatProvider()


@pytest.mark.parametrize("depth", [300, 100000])
def test_deeply_nested_tool_args_are_a_parse_error(depth):
    # 300 parses fine with json.loads but breaks pydantic when the turn is saved.
    args = '{"x":' + "[" * depth + "]" * depth + "}"
    chunk = OpenAICompatProvider._tool_call_chunk("c", "t", args)
    assert chunk.parse_error


def test_ordinary_nested_tool_args_still_parse():
    chunk = OpenAICompatProvider._tool_call_chunk("c", "t", '{"a": {"b": [1, {"c": [2]}]}}')
    assert not chunk.parse_error and chunk.args == {"a": {"b": [1, {"c": [2]}]}}


async def test_count_search_flags_truncation(tmp_path, monkeypatch):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="stub", budget=2, headless=True,
        permission_policy="bypass",
    )
    session = await Harness(inference=_Stub()).start_session(config=config)
    agent = session.agent(session.ceo_name)

    def fake_iter(n):
        return lambda *a, **k: ({"idx": i, "content": "hit", "author": "x"} for i in range(n))

    async def search():
        res = await SearchChannelMessagesTool().execute(ToolContext(
            agent=agent, session=session, call_id="c",
            args={"pattern": "hit", "channel": "self", "output_mode": "count"},
        ))
        return res.output

    monkeypatch.setattr(comms_tools, "_iter_for_search", fake_iter(10))
    assert "truncated" not in await search()
    monkeypatch.setattr(comms_tools, "_iter_for_search", fake_iter(5000))
    assert "truncated" not in await search()  # all scanned: exact
    monkeypatch.setattr(comms_tools, "_iter_for_search", fake_iter(6000))
    out = await search()
    assert out["truncated"] is True
    assert out["counts"]["self"]["count"] == 5000
