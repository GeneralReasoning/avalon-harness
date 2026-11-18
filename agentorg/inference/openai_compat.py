"""OpenAI-compatible streaming provider for self-hosted endpoints.

Works with SGLang, vLLM and other servers that speak the OpenAI Chat
Completions interface. Handles streaming content + tool calls + (when
the model emits it) reasoning deltas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, AsyncIterator, Literal, Optional

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceChunk,
    InferenceProvider,
    InferenceRequest,
    ReasoningDeltaChunk,
    TokenUsage,
    ToolCallChunk,
)

log = logging.getLogger(__name__)


DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_S = 0.5
DEFAULT_BACKOFF_JITTER = 0.2

MAX_RETRY_AFTER_S = 60.0

# Deeper tool-call args are treated as unparseable. Real tool args are a few
# levels deep; pydantic refuses to serialize past ~255 when the turn is saved.
_MAX_ARGS_DEPTH = 64


def _nesting_exceeds(obj: Any, limit: int) -> bool:
    """True iff `obj` nests lists/dicts more than `limit` levels. Iterative."""
    stack = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            node = list(node.values())
        elif not isinstance(node, list):
            continue
        if depth > limit:
            return True
        stack.extend((child, depth + 1) for child in node)
    return False


Classification = Literal["retryable", "terminal"]


def classify_inference_error(err: BaseException) -> Classification:
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return "retryable"
    status = getattr(err, "status_code", None)
    if status is None:
        # Bare APIError without status (rare) — treat as terminal.
        return "terminal"
    if status >= 500:
        return "retryable"
    if status in (408, 409, 429):
        return "retryable"
    if status == 400:
        # `body` is None when no response body was attached, {} when the
        # body was present but empty, or non-empty when the server sent a
        # real error. Treat the first two as transient.
        body = getattr(err, "body", None)
        if not body:
            return "retryable"
        return "terminal"
    return "terminal"


def parse_retry_after(err: BaseException) -> Optional[float]:
    response = getattr(err, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    ms = headers.get("retry-after-ms") if hasattr(headers, "get") else None
    if ms:
        try:
            seconds = float(ms) / 1000.0
            return min(seconds, MAX_RETRY_AFTER_S)
        except (TypeError, ValueError):
            pass
    sec = headers.get("retry-after") if hasattr(headers, "get") else None
    if sec:
        try:
            return min(float(sec), MAX_RETRY_AFTER_S)
        except (TypeError, ValueError):
            return None
    return None


def backoff_seconds(
    attempt: int,
    err: Optional[BaseException] = None,
    *,
    base: float = DEFAULT_BACKOFF_BASE_S,
    jitter: float = DEFAULT_BACKOFF_JITTER,
    rng: Optional[random.Random] = None,
) -> float:
    """`base * 2^(attempt-1)` with ±jitter, but honor Retry-After if present.

    `attempt` is 1-indexed. Caps at `MAX_RETRY_AFTER_S` regardless of input.
    """
    if err is not None:
        ra = parse_retry_after(err)
        if ra is not None:
            return ra
    delay = base * (2 ** max(0, attempt - 1))
    r = rng if rng is not None else random
    delay += r.uniform(-jitter, jitter) * base
    return max(0.0, min(delay, MAX_RETRY_AFTER_S))


def _error_payload(
    err: BaseException, *, attempt: int, classification: Classification, will_retry: bool
) -> dict[str, Any]:
    status = getattr(err, "status_code", None)
    code = getattr(err, "code", None)
    request_id = getattr(err, "request_id", None)
    body = getattr(err, "body", None)
    return {
        "ts": int(time.time() * 1000),
        "attempt": attempt,
        "type": err.__class__.__name__,
        "message": str(err)[:500],
        "status": status,
        "code": code,
        "request_id": request_id,
        "body_excerpt": _truncate(json.dumps(body, default=str)) if body else None,
        "classification": classification,
        "will_retry": will_retry,
    }


def _truncate(s: str, limit: int = 500) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _notify_failure(request: InferenceRequest, payload: dict[str, Any]) -> None:
    sink = request.on_attempt_failure
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:  # noqa: BLE001 — logging hook must never break the loop
        log.exception("on_attempt_failure sink raised; ignoring")


class OpenAICompatProvider(InferenceProvider):
    """Drives any OpenAI-Chat-Completions-compatible endpoint."""

    # Retryable error union — `APIConnectionError` covers timeouts via
    # subclass; `APIStatusError` catches 4xx/5xx; bare `APIError` is the
    # last-resort base.
    _RETRYABLE_EXCS = (APIConnectionError, APIStatusError, APIError)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_ATTEMPTS,
        request_timeout: Optional[float] = 600,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_jitter: float = DEFAULT_BACKOFF_JITTER,
        non_streaming: Optional[bool] = None,
    ):
        # The OPENAI_* env vars are a pair: OPENAI_API_KEY is only used when
        # the server also comes from OPENAI_BASE_URL (or defaults), never sent
        # to a base_url the caller chose. Keyless custom servers get a
        # placeholder, since the SDK would otherwise read OPENAI_API_KEY
        # itself. With no server and no key, the SDK fails fast as before.
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
        if base_url and not api_key:
            api_key = "none"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=0,  # we do our own retries below
        )
        # Non-streaming mode: one create(stream=False) call whose response is
        # replayed through the same chunk interface. Needed wherever streamed
        # chat completions are unavailable — notably inspect_ai's agent_bridge,
        # which intercepts in-process model calls but rejects stream=True.
        # Constructor arg wins; else the AGENTORG_NON_STREAMING env var (the
        # same route the other AGENTORG_* knobs take).
        if non_streaming is None:
            non_streaming = os.getenv("AGENTORG_NON_STREAMING", "").lower() in ("1", "true", "yes")
        self._non_streaming = non_streaming
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_jitter = backoff_jitter

    def model_ids(self) -> list[str]:
        return []  # caller passes a model in InferenceRequest

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        attempt = 0
        emitted_any = False
        while True:
            attempt += 1
            try:
                once = self._complete_once if self._non_streaming else self._stream_once
                async for chunk in once(request):
                    emitted_any = True
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except self._RETRYABLE_EXCS as e:
                classification = classify_inference_error(e)
                # Mid-stream guardrail — once we've yielded a chunk, retrying
                # would replay content/tool-call deltas to the caller and
                # corrupt the assistant message. Fail-fast.
                if emitted_any:
                    _notify_failure(
                        request,
                        _error_payload(
                            e,
                            attempt=attempt,
                            classification=classification,
                            will_retry=False,
                        )
                        | {"phase": "mid-stream"},
                    )
                    log.warning(
                        "mid-stream inference failure (no retry): %s",
                        e.__class__.__name__,
                    )
                    raise
                if classification == "terminal" or attempt > self._max_retries:
                    _notify_failure(
                        request,
                        _error_payload(
                            e,
                            attempt=attempt,
                            classification=classification,
                            will_retry=False,
                        ),
                    )
                    raise
                delay = backoff_seconds(
                    attempt,
                    e,
                    base=self._backoff_base_s,
                    jitter=self._backoff_jitter,
                )
                _notify_failure(
                    request,
                    _error_payload(
                        e,
                        attempt=attempt,
                        classification=classification,
                        will_retry=True,
                    )
                    | {"backoff_s": round(delay, 3)},
                )
                log.warning(
                    "Inference transient error (attempt %d/%d): %s — retrying in %.2fs",
                    attempt,
                    self._max_retries,
                    e.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)

    def _request_kwargs(self, request: InferenceRequest, *, stream: bool) -> dict[str, Any]:
        messages = _to_openai_messages(request.messages)
        tools = _to_openai_tools(request.tools)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        # OpenAI's `max_tokens` already means "completion tokens". An
        # explicit output cap is folded in with min() so neither field can
        # ever LOOSEN the other — a caller may pass the context window as
        # max_tokens, which is harmlessly loose as a completion cap here.
        # Window-aware output capping, where a backend needs it, is the
        # backend's job, not this class's.
        if request.max_completion_tokens is not None:
            kwargs["max_tokens"] = (
                min(request.max_completion_tokens, request.max_tokens)
                if request.max_tokens is not None
                else request.max_completion_tokens
            )
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.extra:
            kwargs.update(request.extra)
        return kwargs

    @staticmethod
    def _usage_from(u: Any) -> TokenUsage:
        usage = TokenUsage()
        if u is None:
            return usage
        usage.input_tokens = getattr(u, "prompt_tokens", 0) or 0
        usage.output_tokens = getattr(u, "completion_tokens", 0) or 0
        cdetails = getattr(u, "completion_tokens_details", None)
        if cdetails is not None:
            usage.reasoning_tokens = getattr(cdetails, "reasoning_tokens", 0) or 0
        pdetails = getattr(u, "prompt_tokens_details", None)
        if pdetails is not None:
            usage.cached_input_tokens = getattr(pdetails, "cached_tokens", 0) or 0
        return usage

    @staticmethod
    def _tool_call_chunk(call_id: str, name: str, args_str: str) -> ToolCallChunk:
        parse_error = False
        try:
            args = json.loads(args_str or "{}")
            if _nesting_exceeds(args, _MAX_ARGS_DEPTH):
                # json.loads accepts ~10k levels, but persisting the turn
                # (pydantic, ~255) and other recursive walkers do not.
                raise RecursionError
        except (json.JSONDecodeError, RecursionError):
            # Malformed JSON args — keep the raw text under the sentinel key so
            # the loop can echo it back to the model (see Session.record_parse_error).
            args = {"__raw_arguments__": args_str, "__parse_error__": True}
            parse_error = True
        return ToolCallChunk(id=call_id, name=name, args=args, parse_error=parse_error)

    async def _complete_once(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        """One non-streaming completion, replayed through the chunk interface.

        Emits the same sequence _stream_once produces (reasoning, content, tool
        calls, CompletedChunk), so callers cannot tell the modes apart. The
        whole response arrives before anything is yielded, which also means a
        retry can never replay partial deltas.
        """
        kwargs = self._request_kwargs(request, stream=False)
        resp = await self._client.chat.completions.create(
            extra_headers=request.extra_headers or None, **kwargs
        )
        choice = resp.choices[0] if resp.choices else None
        message = choice.message if choice is not None else None
        if message is not None:
            reasoning = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )
            if reasoning:
                yield ReasoningDeltaChunk(text=reasoning)
            if message.content:
                yield ContentDeltaChunk(text=message.content)
            for idx, tc in enumerate(message.tool_calls or []):
                fn = getattr(tc, "function", None)
                yield self._tool_call_chunk(
                    tc.id or f"call_{idx}",
                    (fn.name if fn is not None else "") or "",
                    (fn.arguments if fn is not None else "") or "{}",
                )
        yield CompletedChunk(
            usage=self._usage_from(resp.usage),
            stop_reason=(choice.finish_reason if choice is not None else None) or "stop",
            response_id=getattr(resp, "id", None) or None,
        )

    async def _stream_once(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        kwargs = self._request_kwargs(request, stream=True)

        # Per-tool-call accumulators (Chat Completions streams arguments as
        # incremental delta strings keyed by tool_call.index).
        pending_calls: dict[int, dict[str, Any]] = {}
        last_finish_reason: Optional[str] = None
        # OpenAI Chat Completions caches prompt prefixes automatically (no
        # cache_control directive). `cached_tokens` below reflects whatever
        # the endpoint reused; cache-friendliness is determined by how stable
        # the message prefix is across turns (see messages_for_inference).
        usage = TokenUsage()

        extra_headers = request.extra_headers or None
        stream = await self._client.chat.completions.create(
            extra_headers=extra_headers, **kwargs
        )
        response_id: Optional[str] = None
        async for event in stream:
            if response_id is None:
                response_id = getattr(event, "id", None) or None
            if not event.choices:
                # Final usage chunk (stream_options={"include_usage": True}).
                if event.usage is not None:
                    usage = self._usage_from(event.usage)
                continue

            choice = event.choices[0]
            delta = choice.delta

            # Reasoning deltas — supported by Kimi-K2 and a few others via the
            # `reasoning_content` extension field.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning:
                yield ReasoningDeltaChunk(text=reasoning)

            if delta.content:
                yield ContentDeltaChunk(text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    rec = pending_calls.setdefault(
                        idx, {"id": None, "name": "", "args_buf": ""}
                    )
                    if tc.id:
                        rec["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            rec["name"] = fn.name
                        if fn.arguments:
                            rec["args_buf"] += fn.arguments

            if choice.finish_reason:
                last_finish_reason = choice.finish_reason

        # Materialize tool calls now that streaming completed.
        for idx in sorted(pending_calls.keys()):
            rec = pending_calls[idx]
            yield self._tool_call_chunk(
                rec["id"] or f"call_{idx}", rec["name"] or "", rec["args_buf"] or "{}"
            )

        yield CompletedChunk(
            usage=usage,
            stop_reason=last_finish_reason or "stop",
            response_id=response_id,
        )


def _to_openai_messages(messages) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": _stringify(m.content),
                }
            )
            continue
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if m.content is not None:
                entry["content"] = m.content if isinstance(m.content, str) else _stringify(
                    m.content
                )
            if m.reasoning:
                entry["reasoning_content"] = m.reasoning
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            out.append(entry)
            continue
        # system / user
        out.append({"role": m.role, "content": m.content if m.content is not None else ""})
    return out


def _to_openai_tools(tools) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except TypeError:
        return str(v)
