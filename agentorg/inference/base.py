"""Inference provider interface (§8.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Optional, Protocol, Union

from agentorg.types import MessageUsage as TokenUsage


@dataclass
class ReasoningDeltaChunk:
    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass
class ContentDeltaChunk:
    text: str
    kind: Literal["content_delta"] = "content_delta"


@dataclass
class ToolCallChunk:
    id: str
    name: str
    args: dict[str, Any]
    # True when the provider could not parse the tool call's argument JSON.
    # `args` still carries the model's raw text (typically as
    # `{"__raw_arguments__": ...}`) so the loop can feed the failure back to
    # the model, but the harness also records a `parse_error_event` on the
    # session (Session.parse_error_events) so training can penalize the
    # exact turn that emitted the malformed call.
    parse_error: bool = False
    kind: Literal["tool_call"] = "tool_call"


@dataclass
class CompletedChunk:
    usage: TokenUsage
    stop_reason: str
    # Provider-assigned completion id (OpenAI `id`, "chatcmpl-..."). Our
    # training server keys its generated-token spans on it, so env
    # reward events can be pinned to exact token positions at training time
    # (Session.record_reward / reward_events). None when the provider
    # doesn't report one.
    response_id: Optional[str] = None
    kind: Literal["completed"] = "completed"


InferenceChunk = Union[
    ReasoningDeltaChunk, ContentDeltaChunk, ToolCallChunk, CompletedChunk
]


@dataclass
class ProviderMessage:
    """Provider-agnostic message shape. Closer to OpenAI's chat shape."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None  # str or list[{"type": ..., ...}]
    reasoning: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ProviderTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class InferenceRequest:
    model: str
    messages: list[ProviderMessage]
    tools: list[ProviderTool] = field(default_factory=list)
    parallel_tool_calls: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Cap on GENERATED tokens only, independent of prompt size. Distinct from
    # `max_tokens` (`SessionConfig.max_tokens`), which some window-aware
    # backends map to a TOTAL input+output budget — passing an output cap
    # through `max_tokens` there silently starves the completion when the
    # prompt is large (a zero-output `length` stop). Callers that mean
    # "generate at most N tokens" must use this field.
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    # OpenAI-compatible `tool_choice`. `None` (default) → server default
    # ("auto" when tools are present). Set to `"required"` to force the
    # model to emit at least one tool call this turn — used by the
    # post-nudge recovery turn in `loop.py`. The provider ignores this
    # field when `tools` is empty (the API rejects `tool_choice` without
    # tools).
    tool_choice: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Additional HTTP headers forwarded to the inference endpoint. Used to
    # propagate metadata like X-Rollout-Id and X-Agent-Id so the server can
    # attribute requests (e.g. for trainers that group requests by rollout).
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Per-attempt error logging hook. The provider calls this once per
    # failed attempt (retried or terminal) with a structured payload. The
    # loop wires this to <session>.inference-errors.jsonl so retries are
    # visible in trajectories without coupling the provider to session paths.
    on_attempt_failure: Optional[Callable[[dict], None]] = None


class InferenceProvider(Protocol):
    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        ...

    def model_ids(self) -> list[str]:
        ...
