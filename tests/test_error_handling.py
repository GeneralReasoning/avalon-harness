"""Error-handling tests — inference retry/classification, mid-stream
guardrail, loop meta-message on failure, and the env-tool circuit breaker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceChunk,
    InferenceProvider,
    InferenceRequest,
    ToolCallChunk,
    TokenUsage,
)
from agentorg.inference.openai_compat import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_RETRY_AFTER_S,
    OpenAICompatProvider,
    backoff_seconds,
    classify_inference_error,
    parse_retry_after,
)
from agentorg.loop import ENV_TOOL_CIRCUIT_BREAKER_LIMIT
from agentorg.session import SessionConfig
from agentorg.tools.context import ToolContext, err_result, ok_result
from agentorg.types import EnvironmentBinding, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Helpers to fabricate openai SDK errors with realistic shapes.
# ---------------------------------------------------------------------------


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/v1/chat/completions")


def _fake_response(status: int, headers: Optional[dict] = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, request=_fake_request())


def _status_error(
    cls: type[APIStatusError],
    status: int,
    *,
    body: object = None,
    headers: Optional[dict] = None,
) -> APIStatusError:
    """Build an APIStatusError of the given subclass with a fabricated response."""
    return cls("err", response=_fake_response(status, headers), body=body)


def _connection_error(msg: str = "connection reset") -> APIConnectionError:
    return APIConnectionError(message=msg, request=_fake_request())


# ---------------------------------------------------------------------------
# classify_inference_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "err_factory,expected",
    [
        # Retryable: network / timeout
        (lambda: _connection_error(), "retryable"),
        (lambda: APITimeoutError(request=_fake_request()), "retryable"),
        # Retryable: 408 / 409 / 429
        (lambda: _status_error(APIStatusError, 408), "retryable"),
        (lambda: _status_error(ConflictError, 409), "retryable"),
        (lambda: _status_error(RateLimitError, 429), "retryable"),
        # Retryable: 5xx
        (lambda: _status_error(InternalServerError, 500), "retryable"),
        (lambda: _status_error(InternalServerError, 502), "retryable"),
        (lambda: _status_error(InternalServerError, 503), "retryable"),
        # Retryable: 400 with no body (edge-layer reject — proxy/CDN/LB, not a real schema error)
        (lambda: _status_error(BadRequestError, 400, body=None), "retryable"),
        (lambda: _status_error(BadRequestError, 400, body={}), "retryable"),
        # Terminal: 400 with body (real schema/validation error from the model server)
        (lambda: _status_error(BadRequestError, 400, body={"error": "bad schema"}), "terminal"),
        # Terminal: 401/403/404/422
        (lambda: _status_error(AuthenticationError, 401), "terminal"),
        (lambda: _status_error(PermissionDeniedError, 403), "terminal"),
        (lambda: _status_error(NotFoundError, 404), "terminal"),
        (lambda: _status_error(UnprocessableEntityError, 422), "terminal"),
    ],
)
def test_classify_inference_error_buckets(err_factory, expected):
    err = err_factory()
    assert classify_inference_error(err) == expected


def test_classify_unknown_error_terminal():
    # Bare exception with no status -> terminal (we don't retry blindly).
    assert classify_inference_error(ValueError("boom")) == "terminal"


# ---------------------------------------------------------------------------
# parse_retry_after
# ---------------------------------------------------------------------------


def test_parse_retry_after_ms_extension():
    err = _status_error(RateLimitError, 429, headers={"retry-after-ms": "2500"})
    assert parse_retry_after(err) == pytest.approx(2.5)


def test_parse_retry_after_seconds():
    err = _status_error(RateLimitError, 429, headers={"retry-after": "3"})
    assert parse_retry_after(err) == pytest.approx(3.0)


def test_parse_retry_after_capped():
    # Server tells us to wait forever; we cap.
    err = _status_error(RateLimitError, 429, headers={"retry-after": "999"})
    assert parse_retry_after(err) == pytest.approx(MAX_RETRY_AFTER_S)


def test_parse_retry_after_missing():
    err = _status_error(RateLimitError, 429, headers={})
    assert parse_retry_after(err) is None


def test_parse_retry_after_connection_error_no_response():
    err = _connection_error()
    assert parse_retry_after(err) is None


# ---------------------------------------------------------------------------
# backoff_seconds
# ---------------------------------------------------------------------------


def test_backoff_grows_exponentially():
    # Without jitter, attempt n delay should be base * 2^(n-1).
    base = 0.5
    deterministic = type("R", (), {"uniform": staticmethod(lambda a, b: 0.0)})()
    assert backoff_seconds(1, base=base, jitter=0.0, rng=deterministic) == pytest.approx(0.5)
    assert backoff_seconds(2, base=base, jitter=0.0, rng=deterministic) == pytest.approx(1.0)
    assert backoff_seconds(3, base=base, jitter=0.0, rng=deterministic) == pytest.approx(2.0)
    assert backoff_seconds(4, base=base, jitter=0.0, rng=deterministic) == pytest.approx(4.0)


def test_backoff_honors_retry_after():
    err = _status_error(RateLimitError, 429, headers={"retry-after-ms": "1234"})
    # Even attempt=10, the retry-after wins.
    assert backoff_seconds(10, err) == pytest.approx(1.234)


def test_backoff_caps_at_max():
    # Huge attempt -> capped at MAX_RETRY_AFTER_S.
    deterministic = type("R", (), {"uniform": staticmethod(lambda a, b: 0.0)})()
    assert backoff_seconds(20, base=1.0, jitter=0.0, rng=deterministic) == pytest.approx(
        MAX_RETRY_AFTER_S
    )


# ---------------------------------------------------------------------------
# OpenAICompatProvider.stream() — pre-stream retry behavior
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, delta_content=None, finish_reason=None):
        self.delta = type("D", (), {
            "content": delta_content,
            "tool_calls": None,
            "reasoning_content": None,
        })()
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCompletions:
    def __init__(self, client: "_FakeChatClient"):
        self._client = client

    async def create(self, **_kwargs):
        self._client.call_count += 1
        if not self._client._behaviors:
            raise RuntimeError("FakeChatClient: no behaviors left")
        behavior = self._client._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior


class _FakeChat:
    def __init__(self, client: "_FakeChatClient"):
        self.completions = _FakeCompletions(client)


class _FakeChatClient:
    """Stand-in for `provider._client`. The constructor takes a list of
    `behaviors` — each either an exception to raise (pre-stream) or an
    async-iterable stream to return.
    """

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.call_count = 0
        self.chat = _FakeChat(self)


def _ok_stream(content="hello"):
    """A stream that yields one content delta and a completion."""
    return _FakeStream([
        _FakeChunk(choices=[_FakeChoice(delta_content=content)]),
        _FakeChunk(choices=[_FakeChoice(finish_reason="stop")]),
        _FakeChunk(usage=type("U", (), {
            "prompt_tokens": 1, "completion_tokens": 1,
            "completion_tokens_details": None, "prompt_tokens_details": None,
        })()),
    ])


def _make_provider(behaviors, max_retries=DEFAULT_MAX_ATTEMPTS):
    provider = OpenAICompatProvider(api_key="x", base_url="http://x", max_retries=max_retries)
    provider._client = _FakeChatClient(behaviors)
    return provider


@pytest.mark.asyncio
async def test_stream_retries_pre_stream_then_succeeds(monkeypatch):
    """Pre-stream errors retry up to max; on success, emit chunks normally."""
    # Make sleep instant so the test runs in <100ms.
    async def _instant_sleep(*_args, **_kwargs):
        return None
    monkeypatch.setattr("agentorg.inference.openai_compat.asyncio.sleep", _instant_sleep)

    behaviors = [
        _status_error(InternalServerError, 500),
        _connection_error(),
        _ok_stream("hi"),
    ]
    provider = _make_provider(behaviors)
    sink_payloads = []
    request = InferenceRequest(
        model="m",
        messages=[],
        on_attempt_failure=sink_payloads.append,
    )
    chunks: list[InferenceChunk] = []
    async for ch in provider.stream(request):
        chunks.append(ch)

    # Two failures recorded, both retryable, both will_retry=True.
    assert len(sink_payloads) == 2
    assert all(p["classification"] == "retryable" for p in sink_payloads)
    assert all(p["will_retry"] is True for p in sink_payloads)
    # Then a successful stream: content + completion.
    assert any(isinstance(c, ContentDeltaChunk) and c.text == "hi" for c in chunks)
    assert any(isinstance(c, CompletedChunk) for c in chunks)


@pytest.mark.asyncio
async def test_stream_terminal_error_no_retry():
    """A 400 should not be retried — fail immediately."""
    behaviors = [_status_error(BadRequestError, 400, body={"error": "bad"})]
    provider = _make_provider(behaviors)
    sink_payloads = []
    request = InferenceRequest(
        model="m", messages=[], on_attempt_failure=sink_payloads.append
    )
    with pytest.raises(BadRequestError):
        async for _ in provider.stream(request):
            pass
    # One attempt logged, classification=terminal, will_retry=False.
    assert len(sink_payloads) == 1
    assert sink_payloads[0]["classification"] == "terminal"
    assert sink_payloads[0]["will_retry"] is False


@pytest.mark.asyncio
async def test_stream_exhausts_retries(monkeypatch):
    """max_retries+1 retryable failures -> raise the last one."""
    async def _instant_sleep(*_args, **_kwargs):
        return None
    monkeypatch.setattr("agentorg.inference.openai_compat.asyncio.sleep", _instant_sleep)

    behaviors = [_connection_error() for _ in range(10)]
    provider = _make_provider(behaviors, max_retries=3)
    sink_payloads = []
    request = InferenceRequest(
        model="m", messages=[], on_attempt_failure=sink_payloads.append
    )
    with pytest.raises(APIConnectionError):
        async for _ in provider.stream(request):
            pass
    # 3 retries + final failure = 4 logged attempts; the 4th has will_retry=False.
    assert len(sink_payloads) == 4
    assert sink_payloads[-1]["will_retry"] is False


@pytest.mark.asyncio
async def test_stream_mid_stream_failure_does_not_retry(monkeypatch):
    """A failure AFTER a chunk has been yielded must propagate, not retry.

    This is the duplicate-yield bug the old code had — replaying the whole
    stream after partial yield would emit duplicate content / tool calls.
    """
    async def _instant_sleep(*_args, **_kwargs):
        return None
    monkeypatch.setattr("agentorg.inference.openai_compat.asyncio.sleep", _instant_sleep)

    class _FailingMidStream:
        """Yields one chunk then raises a retryable error."""
        def __init__(self):
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise _connection_error("dropped after one chunk")
            self._done = True
            return _FakeChunk(choices=[_FakeChoice(delta_content="partial")])

    behaviors = [_FailingMidStream(), _ok_stream("never-yielded")]
    provider = _make_provider(behaviors)
    sink_payloads = []
    request = InferenceRequest(
        model="m", messages=[], on_attempt_failure=sink_payloads.append
    )
    yielded = []
    with pytest.raises(APIConnectionError):
        async for ch in provider.stream(request):
            yielded.append(ch)
    # We yielded the partial chunk, then raised. No duplicates.
    assert len(yielded) == 1
    assert isinstance(yielded[0], ContentDeltaChunk)
    assert yielded[0].text == "partial"
    # The mid-stream failure is logged with phase=mid-stream and will_retry=False.
    assert len(sink_payloads) == 1
    assert sink_payloads[0]["phase"] == "mid-stream"
    assert sink_payloads[0]["will_retry"] is False
    # Crucial: only one attempt — we did NOT consume the second behavior.
    assert provider._client.call_count == 1


@pytest.mark.asyncio
async def test_stream_retry_after_honored(monkeypatch):
    """If the server sends Retry-After, we wait that many seconds."""
    sleeps: list[float] = []

    async def _capture_sleep(delay):
        sleeps.append(delay)
        return None

    monkeypatch.setattr("agentorg.inference.openai_compat.asyncio.sleep", _capture_sleep)

    err = _status_error(RateLimitError, 429, headers={"retry-after-ms": "1500"})
    behaviors = [err, _ok_stream("ok")]
    provider = _make_provider(behaviors)
    request = InferenceRequest(model="m", messages=[])
    async for _ in provider.stream(request):
        pass
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Loop integration: meta message on inference failure
# ---------------------------------------------------------------------------


class _AlwaysFailsProvider(InferenceProvider):
    """Inference provider that raises immediately on every stream attempt."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        if False:  # pragma: no cover — signal async-generator return type
            yield  # type: ignore[unreachable]
        raise self._exc

    def model_ids(self):
        return ["fail"]


@pytest.mark.asyncio
async def test_inference_failure_leaves_transcript_clean(tmp_path):
    """Per SPEC §8.2: failure is visible via inference-errors.jsonl and the
    event stream, never via a synthetic transcript entry.
    """
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="fail",
        budget=0,
        headless=True,
        permission_policy="bypass",
        initial_user_message="hi",
    )
    err = _status_error(BadRequestError, 400, body={"error": "schema"})
    harness = Harness(inference=_AlwaysFailsProvider(err))
    session = await harness.start_session(config=config)
    await harness.run_to_completion(session, max_wall_s=5)

    # Transcript must contain only real exchanges — no synthetic meta/assistant.
    transcript = session.transcript(session.ceo_name)
    assert all(m.source != "meta" for m in transcript), (
        "no meta message should be injected on inference failure"
    )
    # No synthetic assistant turn either — the failed inference yielded none.
    assert not any(
        m.role == "assistant" and (m.content is None or m.content == "")
        and not m.tool_calls
        for m in transcript
    )

    # The terminal record is in inference-errors.jsonl — the canonical place.
    err_path = Path(session.paths.inference_errors)
    assert err_path.exists()
    lines = [json.loads(line) for line in err_path.read_text().splitlines() if line]
    assert any(line.get("phase") == "terminal" for line in lines)


# ---------------------------------------------------------------------------
# Env-tool circuit breaker
# ---------------------------------------------------------------------------


class _BrokenEnvTool:
    """An env-scoped tool that always errs with the same signature."""

    def __init__(self, name: str = "BrokenEnvCall"):
        self.definition = ToolDefinition(
            name=name,
            description="Always fails for testing.",
            input_schema={"type": "object", "properties": {}},
            read_only=False,
            scope="env",
        )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        return err_result("env-tool-error", "deterministic failure")


class _ScriptedProvider(InferenceProvider):
    """Always emits a single tool call to the same env tool.

    The harness will fire the loop until the session aborts (which the
    circuit breaker triggers after `ENV_TOOL_CIRCUIT_BREAKER_LIMIT`
    consecutive identical failures).
    """

    def __init__(self, tool_name: str):
        self._tool_name = tool_name

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        yield ToolCallChunk(id="c0", name=self._tool_name, args={})
        yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")

    def model_ids(self):
        return ["scripted"]


@pytest.mark.asyncio
async def test_env_tool_circuit_breaker_aborts_session(tmp_path):
    """Five consecutive identical env-tool failures abort the session."""
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="scripted",
        budget=0,
        headless=True,
        permission_policy="bypass",
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
        initial_user_message="break it",
    )

    harness = Harness(inference=_ScriptedProvider("BrokenEnvCall"))
    session = await harness.start_session(config=config)
    # Manually register the broken env tool (the harness registers env tools
    # only when an env_adapter is supplied — we skip that to keep the test
    # self-contained).
    session.tools.register(_BrokenEnvTool("BrokenEnvCall"))

    await harness.run_to_completion(session, max_wall_s=10)

    # Streak should be at or above the limit, and the session should have
    # been told to stop via abort.
    assert session._env_tool_failure_streak >= ENV_TOOL_CIRCUIT_BREAKER_LIMIT
    assert session.abort_requested


@pytest.mark.asyncio
async def test_env_tool_breaker_resets_on_success(tmp_path):
    """A successful env-tool call clears the failure streak."""
    from agentorg.session import Session
    from agentorg.tools.registry import ToolRegistry
    from agentorg.types import ToolCall

    class _Stub(InferenceProvider):
        async def stream(self, request):  # noqa: D401
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
    )
    session = Session(config=config, inference=_Stub(), tools=ToolRegistry())
    session.tools.register(_BrokenEnvTool("ETool"))

    from agentorg.loop import _update_env_tool_breaker

    call = ToolCall(id="c", name="ETool", args={})
    fail = err_result("env-tool-error", "boom")
    succeed = ok_result({"ok": True})

    # Two failures bump the streak.
    _update_env_tool_breaker(session, call, fail)
    _update_env_tool_breaker(session, call, fail)
    assert session._env_tool_failure_streak == 2

    # A success resets the streak and clears the signature.
    _update_env_tool_breaker(session, call, succeed)
    assert session._env_tool_failure_streak == 0
    assert session._env_tool_failure_signature is None


def test_env_tool_breaker_aborts_immediately_on_session_death(tmp_path):
    """An unrecoverable env-session death aborts on the FIRST failure, without
    waiting for the consecutive-failure streak limit."""
    from agentorg.session import Session
    from agentorg.tools.registry import ToolRegistry
    from agentorg.types import ToolCall

    class _Stub(InferenceProvider):
        async def stream(self, request):  # noqa: D401
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
    )
    session = Session(config=config, inference=_Stub(), tools=ToolRegistry())
    session.tools.register(_BrokenEnvTool("place_bet"))

    from agentorg.loop import _update_env_tool_breaker

    from agentorg.env_adapter import ENV_SESSION_DEAD_CODE

    call = ToolCall(id="c", name="place_bet", args={})
    # The adapter classifies an unrecoverable env death to this generic code;
    # the loop reacts to the CODE, not the (provider-specific) message text.
    dead = err_result(ENV_SESSION_DEAD_CODE, "MaxRetriesError: (environment session sid=abc) ...")

    # One death-coded failure is enough — no need to reach the streak limit.
    _update_env_tool_breaker(session, call, dead)
    assert session.abort_requested
    # And the streak counter was not used for this path.
    assert session._env_tool_failure_streak == 0


def test_env_tool_breaker_ignores_local_tools(tmp_path):
    """Local-scoped tool failures don't move the env-tool streak counter."""
    from agentorg.session import Session
    from agentorg.tools.registry import ToolRegistry
    from agentorg.types import ToolCall

    class _Stub(InferenceProvider):
        async def stream(self, request):  # noqa: D401
            yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

        def model_ids(self):
            return ["stub"]

    class _LocalTool:
        definition = ToolDefinition(
            name="LocalTool",
            description="local",
            input_schema={"type": "object", "properties": {}},
            scope="local",
        )

        async def execute(self, ctx):
            return err_result("local-err", "x")

    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        environment=EnvironmentBinding(kind="openreward", task_id="x"),
    )
    session = Session(config=config, inference=_Stub(), tools=ToolRegistry())
    session.tools.register(_LocalTool())

    from agentorg.loop import _update_env_tool_breaker

    for _ in range(10):
        _update_env_tool_breaker(
            session,
            ToolCall(id="c", name="LocalTool", args={}),
            err_result("local-err", "x"),
        )
    # Local-tool failures never touched the env-tool counter.
    assert session._env_tool_failure_streak == 0
    assert not session.abort_requested
