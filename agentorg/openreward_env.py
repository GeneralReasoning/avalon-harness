"""OpenReward environment adapter (§12.3).

Lives outside `agentorg.cli` so it is importable without the CLI
(`from agentorg.openreward_env import OpenRewardEnvAdapter`) by out-of-tree
consumers such as a trainer's rollout workers.

Implements the `agentorg.env_adapter.EnvAdapter` protocol consumed by
`Harness.start_session`. Wraps an `openreward.AsyncSession` and exposes
its tools through the harness registry. Tool results that include
`finished=True` (with optional `reward`) are surfaced in the result
payload; the dispatch loop applies `Session.mark_terminal` AFTER
persisting the tool-result message so the trajectory is durable before
the supervisor cancels the CEO (see `loop._apply_env_terminal_if_needed`).

Error handling:

- `call_tool` failures surface to the model as a tool result so it can
  react (apologise, retry, give up) rather than crashing the loop:
  `env-invalid-call` when the env's input schema rejected the arguments
  (`ToolCallError(reason="input_validation")`; kind "error" — the model's
  mistake), `env-tool-unavailable` when the env cannot offer the tool at all
  (`not_found` / `name_collision`; kind "infra" — an env registration
  fault), `env-failure` when the env or its transport failed (kind "infra",
  §12.1). `bad_input_shape` is agentorg's own marshalling fault and is left
  to raise, so §4.2 records it as `exception` against the harness.
  The loop tracks consecutive identical env-tool
  failures and aborts the session if the streak hits a limit (§4.4).
  The exception is an UNRECOVERABLE env-session death (pod crashed/evicted
  or the SDK exhausted its retries — see `openreward_errors`): those map to
  the generic `env-session-dead` code, which the loop aborts on at the FIRST
  occurrence (no env left to retry).
- `list_tools` is wrapped so a misconfigured env produces a clear
  diagnostic at harness start rather than an opaque traceback from inside
  the OpenReward SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from openreward import AsyncOpenReward, AsyncSession
from openreward.api.errors import ToolCallError

from agentorg.env_adapter import ENV_SESSION_DEAD_CODE
from agentorg.openreward_errors import is_session_death_message
from agentorg.tools.context import (
    MAX_TOOL_OUTPUT_BYTES,
    ToolContext,
    ToolImpl,
    err_result,
    infra_result,
    ok_result,
    rejected_result,
)
from agentorg.types import ToolDefinition, ToolResult

log = logging.getLogger(__name__)


# Env tools that LOOK read-only by name. The OpenReward ToolSpec doesn't
# carry a read-only flag, so without this heuristic every env tool is
# treated as state-mutating — which means the CEO wait-gate (SPEC §4.5.2)
# blocks even inspectors like `view_economy` / `ViewPolicy`, leaving the
# CEO with no way to read state while specialists are running.
_READ_ONLY_VERBS = frozenset({
    "view", "list", "get", "read", "show", "inspect",
    "describe", "query", "find", "search",
    "observe", "grep", "glob", "diff", "stat", "ls",
})


def _looks_read_only(name: str) -> bool:
    # First token of either snake_case or CamelCase. Handles `view_economy`,
    # `ViewPolicy`, `viewPolicy`, and bare `view`.
    parts = re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", name)
    return bool(parts) and parts[0].lower() in _READ_ONLY_VERBS


class OpenRewardEnvUnavailable(RuntimeError):
    """Raised when the env can't be reached at session-build time.

    Wraps the underlying SDK exception with the env name so the diagnostic
    points at *which* env failed rather than at the SDK internals.
    """


@dataclass
class OpenRewardEnvAdapter:
    """Holds the AsyncSession used by env-side tools.

    Implements `agentorg.env_adapter.EnvAdapter` via `build_tools()`.
    """

    client: AsyncOpenReward
    session: AsyncSession
    deployment_name: str
    split: Optional[str] = None
    task_index: Optional[int] = None
    toolset_name: Optional[str] = None

    async def build_tools(self) -> list[ToolImpl]:
        """Fetch the env's tools and wrap each as a harness Tool.

        Failures here are unrecoverable for the rollout (we can't run an
        env if we don't know its tools), so we raise
        `OpenRewardEnvUnavailable` with the env name attached. The caller
        (`sampling.run_openreward_rollouts`) catches per-rollout exceptions
        via `asyncio.gather(..., return_exceptions=True)`, so this fails
        the affected rollout cleanly without taking down the batch.
        """
        try:
            raw = await self.session.list_tools()  # list[ToolSpec]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap with env context
            raise OpenRewardEnvUnavailable(
                f"list_tools failed for env={self.deployment_name!r} "
                f"(split={self.split!r}, task_index={self.task_index!r}): "
                f"{e.__class__.__name__}: {e}"
            ) from e
        out: list[ToolImpl] = []
        for ts in raw:
            schema = ts.input_schema or {"type": "object", "properties": {}}
            definition = ToolDefinition(
                name=ts.name,
                description=(ts.description or "").strip() or f"Environment tool {ts.name}",
                input_schema=schema,
                read_only=_looks_read_only(ts.name),
                scope="env",
            )
            out.append(_OpenRewardTool(definition, self))
        return out

    async def terminal_tool(self) -> Optional[Any]:
        """The env's @terminal tool spec (name + arg name), or None if the env
        has none or the openreward SDK is too old to expose it. Feature-detected
        with getattr so pre-0.1.126 SDKs (which lack the method entirely) still
        return None instead of raising."""
        fn = getattr(self.session, "terminal_tool", None)
        if fn is None:
            return None
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 — probe should not kill the rollout
            log.warning("terminal_tool probe failed: %s", e)
            return None

    async def call_terminal_tool(self, message: str = "") -> Any:
        """Grade the assistant's plain final message via the env's @terminal
        tool. Returns the openreward `ToolOutput` verbatim (reward on
        `.reward`, `finished=True`). Raises if the env has no terminal tool
        or the SDK is too old — callers must check `terminal_tool()` first.
        """
        return await self.session.call_terminal_tool(message)

    async def aclose(self) -> None:
        """Close THIS adapter's env session only.

        The `client` and `client.environments` API objects are owned by
        the caller (the batch driver in `agentorg/cli/lib/sampling.py`) — they
        may be SHARED across many concurrent rollouts/adapters, and
        closing them here would kill sibling rollouts. The batch driver
        closes the client exactly once at end-of-batch.
        """
        try:
            await self.session.__aexit__(None, None, None)  # type: ignore[arg-type]
        except Exception as e:
            log.warning("OpenReward session close error: %s", e)


class _OpenRewardTool:
    """Wraps a single tool exposed by the OpenReward env-side toolset."""

    def __init__(self, definition: ToolDefinition, adapter: OpenRewardEnvAdapter):
        self.definition = definition
        self._adapter = adapter

    async def execute(self, ctx: ToolContext) -> ToolResult:
        adapter = self._adapter
        try:
            output = await adapter.session.call_tool(self.definition.name, ctx.args)
        except ToolCallError as e:
            # `reason` discriminates whose fault the rejection is; the class
            # alone does not. The env server sets it at all three of its
            # RunToolError sites, and for a server too old to send it the SDK
            # infers it from the message prefix.
            rendered = f"{e.__class__.__name__}: {e}"
            if e.reason == "input_validation":
                # The env's own input schema rejected the arguments: the
                # model's mistake, so a plain tool error (§12.1).
                return err_result("env-invalid-call", rendered)
            if e.reason in ("not_found", "name_collision"):
                # The env cannot offer a tool the harness advertised to the
                # model from the env's own `list_tools()` — its tool set moved
                # under us, or two tools share a name. An env registration
                # fault; penalising the model for it trains on noise.
                return infra_result("env-tool-unavailable", rendered)
            if e.reason == "episode_finished":
                # The env server refused the call because an earlier call on
                # this session already returned finished=True (another agent's
                # concurrent submit won the race). Nothing ran, and the locked
                # reward stands: a rejection, like the harness-side guard in
                # `loop._dispatch_tool`.
                return rejected_result("episode-finished", rendered)
            # `bad_input_shape` — raised client-side before the call leaves
            # the process (input not a mapping, or non-string keys), so it is
            # agentorg's own marshalling, not the env. Unreachable while args
            # come from parsed JSON, which always has string keys. Let it
            # raise: the dispatch wrapper records it as `exception` (§4.2),
            # which attributes the fault to the harness instead of the env and
            # keeps an agentorg bug out of the env circuit breaker's signature.
            raise
        except Exception as e:
            rendered = f"{e.__class__.__name__}: {e}"
            # Everything else is the env or its transport failing, not the
            # call: `ToolFailed` (the env's tool raised), `SessionTerminatedError`,
            # the `TransportError` family, an HTTP error from the proxy. An
            # infra failure (§12.1), never penalised.
            #
            # A dead env session is unrecoverable — surface a distinct, generic
            # code so the harness aborts the session instead of letting the
            # agent retry a corpse (see `loop._update_env_tool_breaker`).
            # `ToolFailed` is one of those: the SDK marks the session dead and
            # deletes it server-side before raising, so `is_session_death_message`
            # matches it too. Every other failure is `env-failure` so the model
            # can still react and retry it.
            code = (
                ENV_SESSION_DEAD_CODE
                if is_session_death_message(rendered)
                else "env-failure"
            )
            return infra_result(code, rendered)

        # ToolOutput has blocks, metadata, reward, finished
        blocks = []
        for block in output.blocks:
            btype = getattr(block, "type", None)
            if btype == "text":
                blocks.append({"type": "text", "text": block.text})
            elif btype == "image":
                blocks.append(
                    {
                        "type": "image",
                        "mime_type": block.mimeType,
                        # Truncate image data in the tool result to avoid context blow-up.
                        "data_bytes": len(block.data) if block.data else 0,
                    }
                )
            else:
                blocks.append({"type": btype})

        # Reward accounting (§12.1).
        # - EVERY graded step lands in the per-step stream, terminal included.
        #   Downstream consumers treat `step_rewards` / `cumulative_reward` as
        #   the full episode return: RL reward reducers (episode return,
        #   average with zero-reward masking) expect the terminal grade as a stream member (a binary-terminal
        #   env whose grade is missing reads as all-zero and gets masked out),
        #   and agentorg's own `build_trajectory` re-derives the cumulative
        #   from the persisted payloads — which always carried the terminal
        #   step's reward. This also matches the @terminal-tool grading path
        #   (`loop._dispatch_terminal_tool`). No double-count in
        #   `RolloutResult.reward`: `effective_final_reward` prefers the
        #   explicit `_final_reward` set at `mark_terminal` over the sum.
        # - On a terminal step, the env's `reward` is also the authoritative
        #   final value. We deliberately do NOT call `mark_terminal` (or emit
        #   `environment.terminal`) here: doing so triggers `session.stop()`,
        #   which races with `_persist_tool_result` in the dispatch loop —
        #   the supervisor can cancel the CEO before the tool-result message
        #   is flushed to messages.jsonl and message observers (e.g. a
        #   telemetry sink) fire. The harness dispatch loop applies terminal
        #   semantics after persisting; see `loop._apply_env_terminal_if_needed`.
        if output.reward is not None:
            ctx.session.record_reward(
                float(output.reward),
                agent=ctx.agent.name,
                response_id=ctx.turn_response_id,
                terminal=bool(output.finished),
            )

        payload = {
            "blocks": blocks,
            "metadata": output.metadata,
            "reward": output.reward,
            "finished": output.finished,
        }
        return ok_result(_enforce_output_cap(payload))


# Reserve enough bytes for the wrapper's own JSON scaffolding (the keys,
# braces, and quotes around `blocks`/`metadata`/`reward`/`finished`/`note`
# plus the truncation marker text itself). Sized empirically to leave
# room for typical metadata dicts; pathological metadata will still
# overshoot and is handled by the recheck at the bottom of the helper.
_CAP_WRAPPER_OVERHEAD = 512

# Over the cap, metadata gets the bytes the text leaves, but never less than this.
_META_MIN_BUDGET = 4 * 1024
# Metadata strings at least this long that occur verbatim in a text block are a
# mirror of the output the model already reads.
_META_MIRROR_MIN_CHARS = 256
# Metadata strings are never cut shorter than this.
_META_MIN_STRING_CHARS = 64


def _json_len(obj: Any) -> int:
    """Serialized UTF-8 size, as the model receives it; past the cap if it won't serialize."""
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return MAX_TOOL_OUTPUT_BYTES + 1


def _map_strings(obj: Any, fn) -> Any:
    """Copy of JSON-like `obj` with `fn` applied to every string value (keys untouched).
    A value `json.dumps(default=str)` would render as `str(value)` is treated as that string."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_map_strings(v, fn) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return fn(str(obj))


def _shrink_metadata(metadata: Any, texts: list[str], budget: int) -> Any:
    """A copy of `metadata` that serializes within `budget` bytes, for the model-facing
    payload only. Scalars and short strings are kept as they are; long strings go first.

    1. A long string that occurs verbatim in a text block is replaced by a marker: the
       model reads it in the text already.
    2. Remaining strings are cut to a per-string limit that halves until the metadata fits
       (head kept, original length marked). Each pass walks the metadata once and only
       serializes the cut copy, so a multi-MB value costs a few linear passes.
    3. If many small values still overshoot, top-level containers are replaced by a
       marker and top-level scalars are kept; failing that, the whole value is a marker.
    """

    def drop_mirror(s: str) -> str:
        if len(s) >= _META_MIRROR_MIN_CHARS and any(s in t for t in texts):
            return f"[omitted: same as the tool output text above ({len(s)} chars)]"
        return s

    shrunk = _map_strings(metadata, drop_mirror)
    if _json_len(shrunk) <= budget:
        return shrunk
    limit, candidate = budget, shrunk
    while limit >= _META_MIN_STRING_CHARS:

        def cut(s: str, limit: int = limit) -> str:
            if len(s) <= limit:
                return s
            return s[:limit] + f"...[truncated: first {limit} of {len(s)} chars kept]"

        candidate = _map_strings(shrunk, cut)
        if _json_len(candidate) <= budget:
            return candidate
        limit //= 2
    if isinstance(candidate, dict):
        top = {
            k: (
                f"[omitted: {type(v).__name__} too large for the output cap]"
                if isinstance(v, (dict, list))
                else v
            )
            for k, v in candidate.items()
        }
        if _json_len(top) <= budget:
            return top
    return "[omitted: metadata too large for the output cap]"


def _head_end(raw: bytes, limit: int) -> int:
    """End of the kept head: after the last newline in `raw[:limit]`, unless that would
    discard more than half of the window; then `limit`, backed off to a UTF-8 boundary."""
    nl = raw.rfind(b"\n", 0, limit)
    if nl >= 0 and limit - (nl + 1) <= limit // 2:
        return nl + 1
    end = limit
    while end > 0 and raw[end] & 0xC0 == 0x80:
        end -= 1
    return end


def _tail_start(raw: bytes, start: int) -> int:
    """Start of the kept tail: the first line start at or after `start`, unless that
    would discard more than half of the window; then `start`, moved forward to a UTF-8
    boundary. The tail is never left empty by a trailing newline."""
    window = len(raw) - start
    nl = raw.find(b"\n", start - 1, len(raw) - 1)
    if nl >= 0 and (nl + 1) - start <= window // 2:
        return nl + 1
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    return start


def _text_capper(payload: dict) -> Callable[[int], dict]:
    """A function of `text_budget` returning `payload` with its text cut to that many raw
    UTF-8 bytes: the head and the tail are kept (half the budget each, at line boundaries
    where possible) and the middle is replaced by a marker giving the omitted line range
    (1-based, of the original text) and sizes, so the agent can fetch exactly that range.

    The text of all text blocks is treated as one string in block order; the marker sits
    at the cut point, and a block left with no text is dropped. Non-text blocks are kept
    whole. The text is encoded and its newlines counted once, so each call costs only the
    kept bytes."""
    blocks = payload["blocks"]
    pieces = [
        (b.get("text", "") or "").encode("utf-8") if b.get("type") == "text" else None
        for b in blocks
    ]
    raw = b"".join(p for p in pieces if p is not None)
    total = len(raw)
    total_nl = raw.count(b"\n")
    total_lines = total_nl + (1 if total and not raw.endswith(b"\n") else 0)

    def capped(text_budget: int) -> dict:
        new_blocks: list[dict] = []
        dropped_bytes = 0
        dropped_blocks = 0
        if total <= text_budget:
            new_blocks = list(blocks)
        else:
            head_end = _head_end(raw, text_budget // 2)
            tail_start = _tail_start(raw, total - (text_budget - text_budget // 2))
            dropped_bytes = tail_start - head_end
            first = raw.count(b"\n", 0, head_end) + 1
            last = total_nl - raw.count(b"\n", tail_start - 1, total) + 1
            lines = f"line {first}" if first == last else f"lines {first}\u2013{last}"
            marker = (
                ("" if head_end == 0 or raw[head_end - 1] == 0x0A else "\n")
                + f"[\u2026 {lines} omitted ({dropped_bytes} bytes); output was "
                f"{total_lines} lines / {total} bytes total \u2026]"
                + ("\n" if tail_start < total else "")
            )
            pos = 0
            for block, piece in zip(blocks, pieces):
                if piece is None:
                    new_blocks.append(block)
                    continue
                s, e = pos, pos + len(piece)
                pos = e
                if s == e:
                    new_blocks.append(block)
                    continue
                text = ""
                if s < head_end:
                    text += raw[s:min(e, head_end)].decode("utf-8")
                if s <= head_end < e:
                    text += marker
                if e > tail_start:
                    text += raw[max(s, tail_start):e].decode("utf-8")
                if text:
                    new_blocks.append({"type": "text", "text": text})
                else:
                    dropped_blocks += 1

        if dropped_bytes:
            note = (
                f"Env tool output capped at {MAX_TOOL_OUTPUT_BYTES} bytes: the start and "
                f"end of the text are kept and {dropped_bytes} bytes in the middle are "
                "omitted; the marker at the cut gives the omitted line range. Re-invoke "
                "with narrower arguments (e.g. a line range or a search pattern) to read it."
            )
        else:
            note = (
                f"Env tool output capped at {MAX_TOOL_OUTPUT_BYTES} bytes: the text is "
                "complete; long metadata values were shortened (marked in place)."
            )
        out: dict = {
            "blocks": new_blocks,
            "metadata": payload.get("metadata"),
            "reward": payload.get("reward"),
            "finished": payload.get("finished"),
            "truncated": True,
            "dropped_bytes": dropped_bytes,
            "note": note,
        }
        if dropped_blocks:
            out["dropped_blocks"] = dropped_blocks
        return out

    return capped


def _enforce_output_cap(payload: dict) -> dict:
    """Truncate the env tool's text blocks so the serialized payload fits
    `MAX_TOOL_OUTPUT_BYTES`.

    Env tools (`scope="env"`) bypass the per-tool cap that local agentorg
    tools apply inside their own `execute` methods — the env may hand back
    arbitrarily large bodies (a full file via `view`, the full stdout of a
    long build). Without a cap here a single env call can pin tens or
    hundreds of KB into the transcript and dominate context on every
    subsequent turn.

    Keep the head and the tail of the text, half the budget each, with a
    marker at the cut naming the omitted line range and sizes (`_text_capper`):
    after a long run the end holds the result and errors, and the marker lets
    the agent re-fetch exactly the missing lines. `truncated=True` and
    `dropped_bytes` are surfaced too. Non-text blocks (e.g. images) are kept
    whole — they're already summarised to a byte count at this point. Oversized
    metadata is shrunk first (`_shrink_metadata`), so a long value mirrored
    there does not cost the model the text itself.
    """
    try:
        serialized_len = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        # Pathological metadata that won't even serialize: skip cap, leave
        # the caller to surface the error. Better than silently dropping
        # data here.
        return payload
    if serialized_len <= MAX_TOOL_OUTPUT_BYTES:
        return payload

    # Text is what the model reads; metadata is often a mirror of it (`output`,
    # `content`) next to small structured fields (`exit_code`, `score`). So metadata
    # gets the bytes the rest of the payload leaves, with a floor, and is shrunk to
    # that before any text is cut. Doing it first also keeps the binary search below
    # from re-serializing a multi-MB metadata value on every probe.
    metadata = payload.get("metadata")
    meta_len = _json_len(metadata)
    meta_budget = max(
        _META_MIN_BUDGET,
        MAX_TOOL_OUTPUT_BYTES - _CAP_WRAPPER_OVERHEAD - (serialized_len - meta_len),
    )
    if meta_len > meta_budget:
        texts = [b.get("text") or "" for b in payload["blocks"] if b.get("type") == "text"]
        payload = {**payload, "metadata": _shrink_metadata(metadata, texts, meta_budget)}

    # The budget is in RAW text bytes, but the cap is on the SERIALIZED payload, and
    # serializing escapes every newline, quote and backslash (and control characters to
    # six-byte \uXXXX), and the marker and note add their own bytes. Text trimmed to the
    # raw budget can therefore still overshoot, so shrink the text budget until the
    # serialized payload fits. Binary search: ~15
    # serializations of a <=32 KB payload.
    hi = MAX_TOOL_OUTPUT_BYTES - _CAP_WRAPPER_OVERHEAD
    capped_payload = _text_capper(payload)
    capped = capped_payload(hi)
    final_len = _json_len(capped)
    if final_len > MAX_TOOL_OUTPUT_BYTES:
        lo = 0
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = capped_payload(mid)
            if _json_len(candidate) <= MAX_TOOL_OUTPUT_BYTES:
                best, lo = candidate, mid + 1
            else:
                hi = mid - 1
        if best is not None:
            capped, final_len = best, _json_len(best)
    # If even an EMPTY text budget overshoots (huge block structure), collapse to a
    # stub. Keeps the cap a hard guarantee rather than a best-effort.
    if final_len > MAX_TOOL_OUTPUT_BYTES:
        return {
            "blocks": [
                {"type": "text", "text": "[env tool output exceeded cap before content rendering]"}
            ],
            "metadata": None,
            "reward": payload.get("reward"),
            "finished": payload.get("finished"),
            "truncated": True,
            "dropped_bytes": serialized_len,
            "note": (
                f"Env tool output exceeded {MAX_TOOL_OUTPUT_BYTES} bytes even "
                "after truncation (metadata or block structure too large). "
                "Re-invoke with narrower arguments."
            ),
        }
    return capped


