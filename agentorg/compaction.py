"""Compaction (§11.4).

Two tiers:

- Microcompaction: replace the body of older filesystem/shell tool results
  with a sentinel string. Preserves recent results.
- Compaction (summarization): collapse the older prefix of an agent's
  transcript into a single `compaction` message and persist a boundary
  marker.

Both fire when the transcript crosses a model-aware threshold. Tokens are
estimated on the RENDERED prompt — the ProviderMessages the loop will
actually send, with the session's reasoning-history gate applied — anchored
on the provider's reported `usage.input_tokens` from the most recent
assistant turn, with per-message estimates only for what was appended since
(pure per-message estimates only when no assistant has reported usage yet).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Optional

from agentorg.inference.base import (
    InferenceRequest,
    ProviderMessage,
    CompletedChunk,
    ContentDeltaChunk,
    ReasoningDeltaChunk,
)
from agentorg.storage import append_jsonl_async
from agentorg.types import AgentName, Message, ToolResult

if TYPE_CHECKING:
    # Only needed for type hints. Importing Session at runtime would create a
    # cycle: session.py imports CompactionConfig (below) from this module.
    from agentorg.session import Session

log = logging.getLogger(__name__)

OLD_RESULT_SENTINEL = "[Old tool result content cleared]"

# Per SPEC §11.4.1, only bulky filesystem / shell tool outputs are eligible
# for microcompaction. Env-side tools are added at runtime by inspecting the
# session's tool registry (scope == "env"). Coordination, channel, archive,
# and report tool results are intentionally excluded because they carry
# persistent state the agent needs to keep referring to (e.g. channel
# scrollback, archive paths, report bodies).
BULKY_TOOLS: frozenset[str] = frozenset(
    {"Bash", "Edit", "Read", "Glob", "Grep", "Write", "WebFetch", "WebSearch"}
)
DEFAULT_PRESERVE_RECENT = 5

# ---------------------------------------------------------------------------
# Model-aware compaction thresholds
# ---------------------------------------------------------------------------
#
# Thresholds are derived from the model's context window so they scale with
# the model, instead of being hardcoded to one specific window size. The math
# is: reserve a slab for the summary call, back off by a buffer,
# trigger microcompaction below that.
#
# Every number below is a *default*. A library user can override any of them
# per session through `CompactionConfig` (carried on `SessionConfig.compaction`)
# — point an unknown/custom model at the right window, retune the buffers, or
# resize the summary output cap — without forking the library. The module
# constants stay the single source of truth for those defaults.

DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000

# Slack we always keep clear so the summary call fits when it fires. Covers
# the call's own system prompt, reasoning tokens, the SUMMARY_MAX_TOKENS
# output budget, and bytes/4 estimation imprecision.
RESERVED_SUMMARY_OUTPUT_TOKENS = 20_000

# Distance from the effective limit at which full compaction fires — gives
# the summary call time to run before the transcript would actually overflow.
AUTO_COMPACT_BUFFER_TOKENS = 13_000

# Distance below the auto-compact threshold at which microcompaction fires.
# Microcompaction is cheaper, so it gets first crack at relieving pressure.
MICROCOMPACT_BUFFER_TOKENS = 20_000

# Floor under both derived thresholds. Guards against a tiny context window or
# aggressive buffers driving the trigger point to (or below) zero, which would
# fire compaction on a near-empty transcript.
MIN_THRESHOLD_TOKENS = 20_000

# Output cap for the summary call. RESERVED_SUMMARY_OUTPUT_TOKENS (above) is
# the context-window reservation; this is the per-call output cap. The 12k
# cap is sized to cover a reasoning model's `reasoning_content` plus a
# multi-section prose summary.
SUMMARY_MAX_TOKENS = 12_000

# Default number of recent agent turns full compaction keeps verbatim; the
# older prefix is summarized away. One "turn" is one API round: an assistant
# message plus its trailing tool/user follow-ups (see
# `_group_messages_by_api_round`).
DEFAULT_AUTO_COMPACT_PRESERVE_RECENT_TURNS = 8

# Default for `CompactionConfig.min_prefix_rounds`: minimum number of API
# rounds the summarized prefix must span. One is the structural floor: an
# empty prefix pays a summary call, recovers no budget (the transcript
# re-renders at the same cost), and re-fires immediately. Whether a
# compaction is worth its summary call at all is the TRIGGER THRESHOLD's
# job, not this floor's — selection only runs on transcripts rendering at or
# above it, and rounds are no proxy for mass now that preserved reasoning
# can put a whole threshold's worth of tokens into a single round. (A floor
# of 2 would force every mandate + 2-giant-round transcript into
# max_keep = 0 — a guaranteed zero-round wipe that still pays the summary
# call it claims to be saving.)
MIN_PREFIX_ROUNDS = 1

# When `auto_compact_preserve_suffix_max_tokens` is unset, the preserved
# suffix budget derives from the model's context window divided by this,
# additionally capped threshold-aware for small windows (see
# `get_suffix_token_budget`). Sized so the post-compaction prompt
# (system + summary + suffix) restarts well below the microcompaction
# threshold on every supported window.
SUFFIX_BUDGET_WINDOW_DIVISOR = 4

# Flat per-message overhead added to rendered cost estimates: role framing,
# message separators, tool-call envelope tokens.
_MESSAGE_OVERHEAD_TOKENS = 4

# `estimate_tokens` anchors on the provider's reported `usage.input_tokens`.
# Some servers report that figure NET of prefix-cache hits with
# `cached_tokens` left at 0, which can make the anchor ~1% of the real prompt
# so compaction never fires proactively. A report is trusted
# only when the density it implies — reported tokens, less the prompt's
# already-exact parts, over its chars/4 rendering — is at least this. 
ANCHOR_MIN_REPORTED_RATIO = 0.5

# A trusted anchor also measures how densely this model tokenises this
# content, in tokens per four characters. That density is applied to the
# chars/4 part of everything rendered after the anchor. Only plausible
# densities are used; anything outside the band (a tiny transcript behind a
# large tool schema, a synthetic usage) falls back to 1.0. The band is a
# guard, not a correction: the measurement also carries whatever fixed
# per-request overhead the rendering omits (tool schemas above all), which
# inflates it most when the prompt is small.
CALIBRATION_MIN = 1.0
CALIBRATION_MAX = 2.0


@dataclass
class CompactionConfig:
    """User-tunable knobs for the two-tier compaction system (§11.4).

    Threshold resolution, per agent, before each turn:

      1. `auto_compact_threshold` / `micro_compact_threshold`, if set, win
         outright — an absolute trigger point that bypasses the derivation
         below entirely. Use these when you know exactly where to fire.
      2. Otherwise the threshold is derived::

             window = effective context window (see below)
             auto   = max(min_threshold_tokens,
                          window - reserved_summary_output_tokens
                                 - auto_compact_buffer_tokens)
             micro  = max(min_threshold_tokens,
                          auto - microcompact_buffer_tokens)

    The effective context window is, in priority order:

      - `context_window_tokens` if set. The session's model is fixed, so this
        single override is all that's needed to correct an unknown or custom
        model's window while keeping the threshold derivation, else
      - the built-in table (gemini→1M, claude→200k, avalon→131,072,
        gpt/kimi/glm→128k), else
      - `DEFAULT_CONTEXT_WINDOW_TOKENS` (128k).
    """

    # Absolute trigger overrides. Set either to a token count to fire that tier
    # at exactly that point, skipping the model-aware derivation. None = derive
    # (the common case).
    auto_compact_threshold: Optional[int] = None
    micro_compact_threshold: Optional[int] = None

    # This session's model context window. None = look it up from the model
    # name (built-in table, else DEFAULT_CONTEXT_WINDOW_TOKENS). Set it to
    # correct a model the table doesn't know (e.g. nemotron) while still getting
    # the model-aware threshold derivation.
    context_window_tokens: Optional[int] = None

    # Derivation knobs (see the threshold math in the class docstring).
    reserved_summary_output_tokens: int = RESERVED_SUMMARY_OUTPUT_TOKENS
    auto_compact_buffer_tokens: int = AUTO_COMPACT_BUFFER_TOKENS
    microcompact_buffer_tokens: int = MICROCOMPACT_BUFFER_TOKENS
    min_threshold_tokens: int = MIN_THRESHOLD_TOKENS

    # Per-call output budget for the summarization call.
    summary_max_tokens: int = SUMMARY_MAX_TOKENS

    # How many recent items each tier keeps verbatim.
    microcompact_preserve_recent: int = DEFAULT_PRESERVE_RECENT
    auto_compact_preserve_recent_turns: int = DEFAULT_AUTO_COMPACT_PRESERVE_RECENT_TURNS

    # Minimum API rounds the summarized prefix must span before full
    # compaction fires; on a shorter transcript compaction WAITS (no summary
    # call, no wipe) for the next attempt. 1 (the default) is the structural
    # floor — an empty prefix recovers no budget — and is right for small
    # windows, where a single near-cap round can cross the trigger threshold
    # and deferring costs output headroom. Raise it on large windows to
    # summarize less often and in bigger chunks; values above 1 recreate
    # deferral pressure on small windows (the transcript keeps growing until
    # the prefix qualifies, backstopped by PTL truncation). Clamped to >= 1.
    min_prefix_rounds: int = MIN_PREFIX_ROUNDS

    # Token budget for the verbatim-preserved suffix at full compaction. None
    # (default) = derive from the model's context window, capped so the
    # post-compaction prompt restarts well below the trigger threshold on
    # small windows too (see `get_suffix_token_budget`).
    # Full compaction keeps up to `auto_compact_preserve_recent_turns` recent
    # rounds but drops rounds — all the way down to an EMPTY suffix (the
    # transcript becomes pinned mandate + summary) — until the kept tail's
    # rendered estimate fits this budget. The summary call always sees
    # everything it drops, trailing unseen tool results included, so nothing
    # vanishes unsummarized. There is deliberately no unlimited setting: an
    # uncapped fixed turn count is exactly what pinned reasoning-heavy
    # transcripts (reasoning is always re-sent) above the compaction
    # threshold forever. Set a huge explicit value if you truly want
    # fixed-count behavior.
    auto_compact_preserve_suffix_max_tokens: Optional[int] = None

    # Reasoning-strip ladder at full compaction. When True, boundary
    # selection may keep MORE rounds inside the suffix budget by stripping
    # reasoning from the OLDEST kept rounds (content, tool calls and tool
    # results survive verbatim; only the thinking channel is dropped), one
    # round at a time, most-recent rounds keeping reasoning the longest.
    # Motivation: on reasoning-heavy transcripts a single round's re-sent
    # thinking can exceed the whole suffix budget, so the keep-whole-rounds
    # walk degrades to an EMPTY suffix — the agent resumes from mandate +
    # summary with no tool chain and often dies tool-less, especially on
    # small (e.g. 64k) windows. A stripped round costs a small fraction
    # of a full one, so the entire tool chain fits where zero rounds fit
    # before. Stripping happens only at a compaction splice — a point that
    # already rewrites the rendered context — never between turns.
    strip_reasoning_to_fit: bool = False


def avalon_compaction_config() -> CompactionConfig:
    """The compaction settings Avalon-M1-Mini was trained with.

    Full compaction fires at 114,688 tokens of a 131,072-token window, keeps up
    to 32,768 tokens of recent turns verbatim, and caps the summary at 12,000
    tokens. Microcompaction was off in training (its threshold is never reached).
    """
    return CompactionConfig(
        context_window_tokens=131_072,
        auto_compact_threshold=114_688,
        auto_compact_preserve_suffix_max_tokens=32_768,
        summary_max_tokens=12_000,
        micro_compact_threshold=1_000_000_000,
    )


def get_model_context_window(
    model: str, config: Optional[CompactionConfig] = None
) -> int:
    """Token budget for the named model.

    Matching is substring-based so variants like `kimi-k2.6`, `gpt-5-mini`,
    or `claude-opus-4-7` all resolve correctly. A `config.context_window_tokens`
    override wins when set; otherwise the built-in table and the
    `DEFAULT_CONTEXT_WINDOW_TOKENS` fallback are used.
    """
    cfg = config if config is not None else CompactionConfig()
    if cfg.context_window_tokens is not None:
        return cfg.context_window_tokens
    n = model.lower()
    if "gemini" in n:
        return 1_000_000
    if "claude" in n:
        return 200_000
    if "avalon" in n:
        return 131_072  # Avalon-M1-Mini training window
    if any(s in n for s in ("gpt-4", "gpt-5", "kimi", "glm")):
        return 128_000
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def get_auto_compact_threshold(
    model: str, config: Optional[CompactionConfig] = None
) -> int:
    """Token threshold at which full compaction (summarization) fires."""
    cfg = config if config is not None else CompactionConfig()
    return max(
        cfg.min_threshold_tokens,
        get_model_context_window(model, cfg)
        - cfg.reserved_summary_output_tokens
        - cfg.auto_compact_buffer_tokens,
    )


def get_microcompact_threshold(
    model: str, config: Optional[CompactionConfig] = None
) -> int:
    """Token threshold at which microcompaction fires.

    Sits below the auto-compact threshold so microcompaction has a chance to
    relieve pressure cheaply before the heavier summarize-the-prefix path
    takes over.
    """
    cfg = config if config is not None else CompactionConfig()
    return max(
        cfg.min_threshold_tokens,
        get_auto_compact_threshold(model, cfg) - cfg.microcompact_buffer_tokens,
    )


def get_suffix_token_budget(
    model: str, config: Optional[CompactionConfig] = None
) -> int:
    """Token budget for the verbatim-preserved suffix at full compaction.

    An explicit `auto_compact_preserve_suffix_max_tokens` wins; otherwise the
    budget derives from the context window so it scales with the model:

        min(window // SUFFIX_BUDGET_WINDOW_DIVISOR,
            derived_auto_threshold // 2 - summary_max_tokens)

    The second term is the invariant that makes compaction terminate in a
    healthy state: the post-compaction prompt (system + pinned mandate +
    summary + suffix) must restart WELL BELOW the trigger threshold, or
    compaction re-fires almost immediately and thrashes. On windows >= 128k
    the window share already satisfies it; on small windows (e.g. a 64k
    session) the fixed reservations — summary slab, trigger buffer,
    summary output cap — dominate the threshold math and the cap binds.
    Clamped at 0: on windows so small nothing fits (~40k and below with the
    default 12k summary cap), every compaction keeps an empty suffix and the
    loud zero-rounds warning tells you to tune the config explicitly.

    Both terms use the DERIVED threshold, never an absolute
    `auto_compact_threshold` override: a test or caller pinning the trigger
    to 1 must not collapse the budget and wipe every suffix.
    """
    cfg = config if config is not None else CompactionConfig()
    if cfg.auto_compact_preserve_suffix_max_tokens is not None:
        return cfg.auto_compact_preserve_suffix_max_tokens
    window_share = get_model_context_window(model, cfg) // SUFFIX_BUDGET_WINDOW_DIVISOR
    headroom_cap = (
        get_auto_compact_threshold(model, cfg) // 2 - cfg.summary_max_tokens
    )
    return max(0, min(window_share, headroom_cap))


# Retry once on a bad summary. Kimi's tool-call priming failure (see
# NO_TOOLS_PREAMBLE comment) usually recovers when the retry prompt explicitly
# names the previous failure mode. Two retries don't help — if the model is
# broken on this turn, more attempts just burn tokens.
MAX_BAD_SUMMARY_RETRIES = 1

# Retry up to N times on `prompt_too_long` (413) errors during the summary
# call. Each retry drops ~20% of the oldest API rounds from the rendered
# history and re-issues. Bounded so a pathologically large transcript can't
# stall the rollout forever.
MAX_PROMPT_TOO_LONG_RETRIES = 3

# Cap on the exception text carried by an unavailable-summary placeholder. The
# placeholder becomes the agent's context, and a provider can serialize the
# whole request into its error text; the tail is kept because that is where
# such wrappers (and tracebacks) put the actual error.
SUMMARY_ERROR_MAX_CHARS = 500

# Marker prepended to the rendered history when a PTL retry truncates the
# head. Shown to the model so it knows the older context was dropped (rather
# than silently never existing). 
_PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"

# Heuristic detection for prompt-too-long errors across providers. We also
# check `status_code` / `status` attributes when the exception carries one.
_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt too long|context length|maximum context length|input too large|too many tokens",
    re.IGNORECASE,
)

# Chat-template delimiters reserved by kimi / Qwen / a few others. If they
# show up in the summary it means the model tried to emit a tool call in
# response to the no-tools system prompt and the template renderer leaked
# the raw delimiters as content. 
_CHAT_TEMPLATE_TOKEN_RE = re.compile(r"<\|[a-z][a-z_]*\|>", re.IGNORECASE)

_NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do not call tools.

- Do NOT use any tools.
- Tool calls will be rejected and will waste your only turn.
- Your entire response must be plain text containing an <analysis> block followed by a <summary> block."""


def _rendered_view(transcript: list[Message]) -> list[Any]:
    """The transcript as it will actually be sent: one ProviderMessage per
    transcript message, index-aligned. Estimating on this — instead of the
    raw stored Messages — keeps the estimate in lockstep with the rendering
    (reasoning is preserved thinking, re-sent in full, and priced as such).

    Lazy import: loop.py imports this module (via session.py) at load time,
    so the reverse import must be deferred to call time.
    """
    from agentorg.loop import _to_provider_message

    return [_to_provider_message(m) for m in transcript]


def _rendered_message_cost(
    msg: Message, pm: Any, include_reasoning: bool = True
) -> int:
    """Estimated prompt cost of one rendered message.

    chars/4 over the fields that actually go on the wire. The reasoning part
    uses the provider's own `usage.reasoning_tokens` count when available
    (chars/4 only as fallback). ``include_reasoning=False`` prices the message
    as it would render with its reasoning stripped — the ladder's unit in
    `_select_compact_boundary` (content + tool calls survive, thinking does
    not).
    """
    approx, exact = _rendered_message_cost_parts(msg, pm, include_reasoning)
    return approx + exact


def _rendered_message_cost_parts(
    msg: Message, pm: Any, include_reasoning: bool = True
) -> tuple[int, int]:
    """`_rendered_message_cost` split into (chars/4 estimate, exact tokens).

    The first part is what the density calibration in `estimate_tokens`
    scales; the second is the provider's own `usage.reasoning_tokens` count
    for the message's reasoning and is already exact.
    """
    chars = 0
    if pm.content:
        chars += (
            len(pm.content)
            if isinstance(pm.content, str)
            else len(_json_dumps_safe(pm.content))
        )
    if pm.tool_calls:
        chars += len(_json_dumps_safe(pm.tool_calls))
    if pm.tool_call_id:
        chars += len(pm.tool_call_id)
    approx = chars // 4 + _MESSAGE_OVERHEAD_TOKENS
    exact = 0
    if pm.reasoning and include_reasoning:
        if msg.usage is not None and msg.usage.reasoning_tokens > 0:
            exact = msg.usage.reasoning_tokens
        else:
            approx += len(pm.reasoning) // 4
    return approx, exact


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


@dataclass(frozen=True)
class TokenEstimate:
    """`estimate_tokens` with its provenance, for the `compaction.fired`
    payload: which assistant anchored it (None = pure rendered fallback) and
    the tokens-per-four-chars density applied to the rendered part."""

    tokens: int
    anchor_index: int | None
    calibration: float


def estimate_tokens(
    transcript: list[Message],
    *,
    system_prompt: str | None = None,
) -> int:
    """Token-count estimate for the NEXT prompt this transcript will produce.

    See `estimate_tokens_detail`; this returns only the number.
    """
    return estimate_tokens_detail(transcript, system_prompt=system_prompt).tokens


def estimate_tokens_detail(
    transcript: list[Message],
    *,
    system_prompt: str | None = None,
) -> TokenEstimate:
    """Token-count estimate for the NEXT prompt this transcript will produce.

    Costs are measured on the rendered form (`_rendered_view`). Anchored on
    the provider's reported `usage.input_tokens` from the most recent
    assistant turn whose report is TRUSTED — ground truth for everything up
    to that call — plus that turn's own re-sent output (reasoning included),
    plus per-message estimates for everything appended after, scaled by the
    density the anchor measured. Falls back to a pure rendered estimate plus
    `system_prompt` when no assistant has reported usage yet (first turn) or
    no report passes the gate.

    A report is trusted when the density it implies is at least
    `ANCHOR_MIN_REPORTED_RATIO`. Lanes that report prompt tokens net of
    prefix-cache hits fail the gate on every warm turn and the walk continues
    to an older assistant — after a compaction, the first cache-cold turn —
    instead of anchoring ~99% below the real prompt and leaving compaction to
    the prompt-too-long forced path.

    Density is measured against the chars/4 part of the prompt only: the
    exact parts of that prompt (provider-counted reasoning tokens) are
    subtracted from the report first, since they are already tokens and
    scaling them would drag the measurement toward 1.0. Within
    [CALIBRATION_MIN, CALIBRATION_MAX] the density scales the chars/4 part of
    everything after the anchor (code tokenises at ~1.4 on the audited runs,
    so unscaled trailing cost hides a third of a long post-compaction
    segment). Exact parts — the anchor's usage and provider-counted reasoning
    tokens — are never scaled.
    """
    rendered = _rendered_view(transcript)
    parts = [
        _rendered_message_cost_parts(m, pm) for m, pm in zip(transcript, rendered)
    ]
    system_tokens = len(system_prompt) // 4 if system_prompt else 0
    # Prefix/suffix sums so the walk below is O(n) even when every report
    # fails the gate (the net-of-cache lane, where it visits every assistant).
    pre_approx = list(accumulate((a for a, _ in parts), initial=system_tokens))
    pre_exact = list(accumulate((e for _, e in parts), initial=0))
    total_approx, total_exact = pre_approx[-1], pre_exact[-1]
    for i in range(len(transcript) - 1, -1, -1):
        msg = transcript[i]
        if not (
            msg.role == "assistant"
            and msg.usage is not None
            and msg.usage.input_tokens > 0
        ):
            continue
        prompt_approx = pre_approx[i]
        prompt_exact = pre_exact[i]
        density = (
            (msg.usage.input_tokens - prompt_exact) / prompt_approx
            if prompt_approx > 0
            else 0.0
        )
        if density < ANCHOR_MIN_REPORTED_RATIO:
            continue
        calibration = density if CALIBRATION_MIN <= density <= CALIBRATION_MAX else 1.0
        # The size this estimate predicts is the *next* prompt: the prompt
        # that produced this assistant (`input_tokens`) + the assistant's
        # own output that gets re-sent + everything appended after it.
        # Omitting the resend term hides a single oversized turn until the
        # *next* inference reports it via `input_tokens` — one turn too late.
        resent_output = msg.usage.output_tokens
        if rendered[i].reasoning is None:
            # Reasoning the provider counted but never returned as text
            # (so it can't be re-sent) doesn't reach the next prompt.
            resent_output = max(0, resent_output - msg.usage.reasoning_tokens)
        trailing_approx = total_approx - pre_approx[i + 1]
        trailing_exact = total_exact - pre_exact[i + 1]
        tokens = (
            msg.usage.input_tokens
            + resent_output
            + int(trailing_approx * calibration)
            + trailing_exact
        )
        return TokenEstimate(tokens, i, calibration)
    return TokenEstimate(total_approx + total_exact, None, 1.0)


async def microcompact_if_needed(
    session: Session,
    agent_name: AgentName,
) -> bool:
    """Apply the cheap pass once per agent (§11.4.1).

    The trigger threshold is `session.config.compaction.micro_compact_threshold`
    if set, otherwise it is derived from the session's model and
    `session.config.compaction` via `get_microcompact_threshold`.

    Microcompaction is capped at one fire per agent because rewriting
    older tool-result contents in place invalidates that agent's prefix
    cache; the cost is only amortised over many subsequent turns if we
    don't pay it again. The cap is per-agent (not session-global) since
    each agent has its own transcript and its own prefix cache.
    """
    agent = session.agent(agent_name)
    if agent is None or agent.microcompacted:
        return False
    transcript = session.transcript(agent_name)
    cfg = session.config.compaction
    # Absolute override from config, else model-aware derivation.
    threshold = cfg.micro_compact_threshold
    if threshold is None:
        threshold = get_microcompact_threshold(session.state.model, cfg)
    estimate = estimate_tokens(
        transcript,
        system_prompt=agent.system_prompt,
    )
    if estimate < threshold:
        return False
    bulky_names = BULKY_TOOLS | {
        d.name for d in session.tools.definitions() if d.scope == "env"
    }
    changed = _microcompact(
        transcript,
        preserve_recent=cfg.microcompact_preserve_recent,
        bulky_tool_names=bulky_names,
    )
    if changed:
        agent.microcompacted = True
        await session.update_agent(agent)
    return changed


def _microcompact(
    transcript: list[Message],
    *,
    preserve_recent: int,
    bulky_tool_names: frozenset[str] | set[str],
) -> bool:
    """Mutate transcript in-place; replace eligible tool result bodies with a sentinel.

    Per SPEC §11.4.1, only "bulky" tools are eligible: the harness's
    filesystem/shell tools listed in `BULKY_TOOLS`, plus any env-scoped tools.
    Coordination, channel, archive, and report tool results are skipped
    because they carry persistent state the agent needs to keep referring to.
    """
    # Build a tool_call_id → tool_name map by walking the assistant messages
    # that preceded each tool-result record.
    call_id_to_name: dict[str, str] = {}
    for msg in transcript:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    call_id_to_name[tc.id] = tc.name

    # Find eligible (bulky + successful) tool indices; spare the most recent N.
    eligible_indices: list[int] = []
    for i, msg in enumerate(transcript):
        if msg.role != "tool" or msg.tool_result is None or not msg.tool_result.ok:
            continue
        tool_name = call_id_to_name.get(msg.tool_call_id or "", "")
        if tool_name not in bulky_tool_names:
            continue
        eligible_indices.append(i)
    if len(eligible_indices) <= preserve_recent:
        return False
    changed = False
    for i in eligible_indices[:-preserve_recent]:
        msg = transcript[i]
        result = msg.tool_result
        if result is None or not isinstance(result.output, (dict, list, str)):
            continue
        if isinstance(result.output, dict) and result.output.get("_compacted"):
            continue
        msg.tool_result = ToolResult(
            ok=result.ok,
            output={"_compacted": True, "summary": OLD_RESULT_SENTINEL},
            error=result.error,
        )
        changed = True
    return changed


def _select_compact_boundary(
    transcript: list[Message],
    preserve_recent_turns: int,
    suffix_budget_tokens: int,
    strip_reasoning: bool = False,
    min_prefix_rounds: int = MIN_PREFIX_ROUNDS,
) -> Optional[tuple[int, int]]:
    """(boundary, strip_before): where the summarized prefix ends, and how
    much of the preserved suffix loses its reasoning.

    Works in whole API rounds (`_group_messages_by_api_round`), so a boundary
    can never split an assistant turn from its trailing tool results — the
    suffix always opens on an assistant round, never a dangling `role="tool"`
    message that providers would reject. Tries `preserve_recent_turns` rounds
    first, then fewer, down to an empty suffix (boundary == len(transcript))
    as the last resort, returning the boundary for the LARGEST keep whose
    rendered suffix fits `suffix_budget_tokens`.

    With ``strip_reasoning``, each keep-count gets a second chance before
    shrinking: strip reasoning from the oldest kept rounds — one round at a
    time, least stripping that fits wins — down to a fully bare suffix.
    Suffix messages with index in [boundary, strip_before) are to be
    re-spliced with ``reasoning=None``; strip_before == boundary means no
    stripping (always the case when the flag is off). Round count dominates
    reasoning retention: a longer bare tool chain beats a shorter full one
    (the suffix's job is continuity; the summarized prose carries the rest).

    The summarized prefix always spans at least ``min_prefix_rounds``
    assistant rounds (a leading user-only group — the mandate — doesn't
    count), so a fired compaction always recovers budget. Returns None on
    transcripts too short for that: there the only possible plans are a
    no-op and a zero-round wipe, and waiting strictly improves the menu —
    the next attempt can keep these rounds and summarize them instead.
    """
    min_prefix_rounds = max(1, min_prefix_rounds)
    groups = _group_messages_by_api_round(transcript)
    assistant_rounds = sum(1 for g in groups if g[0].role == "assistant")
    if assistant_rounds <= min_prefix_rounds:
        return None
    max_keep = min(preserve_recent_turns, assistant_rounds - min_prefix_rounds)
    starts: list[int] = []
    pos = 0
    for g in groups:
        starts.append(pos)
        pos += len(g)
    n = len(transcript)

    # Rendered cost of transcript[j:] for every candidate boundary j — full
    # and reasoning-stripped — built once in a single backward pass from the
    # earliest candidate. Any (boundary b, strip_before s) plan then prices
    # in O(1): stripped cost of [b, s) plus full cost of [s, n).
    rendered = _rendered_view(transcript)
    earliest = starts[len(groups) - max_keep] if max_keep > 0 else n
    full_from: dict[int, int] = {n: 0}
    bare_from: dict[int, int] = {n: 0}
    run_full = run_bare = 0
    for j in range(n - 1, earliest - 1, -1):
        run_full += _rendered_message_cost(transcript[j], rendered[j])
        run_bare += _rendered_message_cost(
            transcript[j], rendered[j], include_reasoning=False
        )
        full_from[j] = run_full
        bare_from[j] = run_bare

    for keep in range(max_keep, -1, -1):
        boundary = starts[len(groups) - keep] if keep > 0 else n
        if full_from[boundary] <= suffix_budget_tokens:
            return boundary, boundary  # fits with all reasoning intact
        if not strip_reasoning or keep == 0:
            continue  # over budget -> keep fewer rounds
        # The ladder: strip_before at each kept round's start strips every
        # older kept round; n strips them all (bare tool chain).
        for s in [starts[len(groups) - keep + r] for r in range(1, keep)] + [n]:
            cost = (bare_from[boundary] - bare_from[s]) + full_from[s]
            if cost <= suffix_budget_tokens:
                return boundary, s
    return None


async def auto_compact_if_needed(
    session: Session,
    agent_name: AgentName,
    *,
    preserve_recent_turns: Optional[int] = None,
    force: bool = False,
) -> bool:
    """Full compaction (§11.4.2). Summarizes older prefix into a single message.

    The trigger threshold is `session.config.compaction.auto_compact_threshold`
    if set, otherwise it is derived from the session's model and
    `session.config.compaction` via `get_auto_compact_threshold`.
    `preserve_recent_turns=None` falls back to
    `session.config.compaction.auto_compact_preserve_recent_turns`.

    `force=True` skips the threshold trigger check — for callers holding
    ground truth that the prompt is already too long (the run loop's
    prompt-too-long recovery). The estimate that gated proactive compaction
    was wrong by construction there; boundary selection and everything else
    run unchanged.

    The preserved suffix is chosen in whole API rounds: up to
    `preserve_recent_turns` recent rounds, shrunk — possibly to an empty
    suffix — until the rendered tail fits the suffix token budget
    (`get_suffix_token_budget`). The transcript's opening user message (the
    agent's mandate) is pinned: it survives every compaction verbatim, ahead
    of the summary, so the task definition never round-trips through a lossy
    summary-of-a-summary chain.
    """
    cfg = session.config.compaction
    transcript = session.transcript(agent_name)
    # Absolute override from config, else model-aware derivation.
    threshold = cfg.auto_compact_threshold
    if threshold is None:
        threshold = get_auto_compact_threshold(session.state.model, cfg)
    if preserve_recent_turns is None:
        preserve_recent_turns = cfg.auto_compact_preserve_recent_turns
    agent = session.agent(agent_name)
    estimate_detail = estimate_tokens_detail(
        transcript,
        system_prompt=agent.system_prompt if agent is not None else None,
    )
    estimate = estimate_detail.tokens
    if not force and estimate < threshold:
        return False

    suffix_budget = get_suffix_token_budget(session.state.model, cfg)
    selected = _select_compact_boundary(
        transcript,
        preserve_recent_turns,
        suffix_budget,
        strip_reasoning=cfg.strip_reasoning_to_fit,
        min_prefix_rounds=cfg.min_prefix_rounds,
    )
    if selected is None:
        # Transcript shorter than min_prefix_rounds — nothing could be kept
        # yet. The next attempt, one round richer, keeps these rounds and
        # summarizes them; microcompaction and PTL truncation cover the
        # interim.
        return False
    boundary, strip_before = selected
    prefix = transcript[:boundary]
    # Pin the mandate: the opening user message is the agent's task
    # definition. The summarizer still sees it (it's in `prefix`, useful
    # context for a good summary), but it also survives verbatim ahead of the
    # summary instead of living only inside lossy summarized prose.
    pinned = [prefix[0]] if prefix and prefix[0].role == "user" else []

    result = await _summarize_prefix(session, agent_name, prefix)
    summary = result.summary
    new_msg = Message(role="user", source="meta", content=f"[compaction]\n{summary}", compaction=True)

    # Splice under the transcript lock, RE-SLICING the suffix from the live
    # list: the summary call above is a long await, and any message appended
    # during it (externally injected user input, inbox delivery) landed on the list
    # object being replaced — a snapshot taken before the await would drop it.
    # Indices are stable because the in-memory transcript is append-only
    # (microcompaction mutates bodies in place, never order). The boundary
    # record is written under the same lock so no message record can land
    # between the splice and the boundary line on disk — replay counts
    # `suffix_count` records back from the boundary line.
    async with session.transcript_lock(agent_name):
        live = session.transcript(agent_name)
        # Suffix messages cross the splice as copies with (a) reasoning
        # stripped for indices < strip_before — the ladder's plan; messages
        # appended during the summary await sit past the pre-await snapshot
        # so they always keep theirs — and (b) `usage` cleared on every
        # assistant: usage.input_tokens was measured against the
        # PRE-compaction rendering, and estimate_tokens anchors on the latest
        # usage-bearing assistant, so a stale anchor holds the estimate above
        # the trigger threshold and re-fires compaction on the very next turn,
        # immediately re-summarizing the just-spliced suffix (under the old
        # MIN_PREFIX_ROUNDS = 2 floor, a zero-round wipe). Cleared, the next estimate
        # falls back to the pure rendered cost of the spliced transcript,
        # which is the quantity the trigger is meant to compare. Disk records
        # are untouched (append-only); replay re-applies both via the
        # boundary record's strip_prefix_count (see Session._load_transcript).
        suffix: list[Message] = []
        stripped_rounds = 0
        for k, m in enumerate(live[boundary:]):
            update: dict[str, Any] = {}
            if m.role == "assistant":
                if m.usage is not None:
                    update["usage"] = None
                if boundary + k < strip_before and m.reasoning:
                    update["reasoning"] = None
                    stripped_rounds += 1
            suffix.append(m.model_copy(update=update) if update else m)
        session._transcripts[agent_name] = pinned + [new_msg] + suffix

        # Append-only update (§11.1, §12.4). Pre-compaction prefix stays on
        # disk for SFT/RL. Boundary + summary (+ pinned mandate) live in a
        # SINGLE JSONL record so a crash mid-write can't leave a boundary on
        # disk that drops the prefix without a replacement summary.
        boundary_record: dict[str, Any] = {
            "type": "compaction-boundary",
            "seq": session.events.last_seq,
            "suffix_count": len(suffix),
            # How many leading suffix messages replay must re-strip reasoning
            # from (and every suffix assistant's stale usage is cleared on
            # replay regardless) — the in-memory splice above and a reloaded
            # transcript must render identically.
            "strip_prefix_count": max(0, strip_before - boundary),
            "summary": new_msg.model_dump(by_alias=True),
        }
        if pinned:
            # Replay prepends these ahead of the summary after dropping the
            # prefix — see Session._load_transcript. Stored in the record so
            # the pin is atomic with the boundary, like the summary.
            boundary_record["pinned"] = [
                m.model_dump(by_alias=True) for m in pinned
            ]
        # Persist the FULL summarization call — the request exactly as sent
        # (system prompt + the rendered prefix the model saw, microcompaction
        # sentinels and any PTL truncation included) plus the raw completion
        # and provenance — so a training-data builder can emit a
        # `compaction-summary` example training the model to PRODUCE the
        # summary, not just consume it. Stored verbatim (not reconstructed) so
        # it matches the distribution the policy faces at RL time; logged for
        # every outcome so a later dataset rebuild loses nothing.
        training: dict[str, Any] = {
            "outcome": result.outcome,
            "attempts": result.attempts,
            "ptl_truncations": result.ptl_truncations,
        }
        if result.completion is not None:
            completion_msg = Message(
                role="assistant",
                content=result.completion,
                reasoning=result.reasoning,
            )
            training["completion"] = completion_msg.model_dump(by_alias=True)
        if result.system_prompt is not None:
            training["system"] = result.system_prompt
        if result.prompt_body is not None:
            training["prompt"] = result.prompt_body
        if result.issue_kind is not None:
            training["issue"] = result.issue_kind
        if result.stop_reason is not None:
            training["stop_reason"] = result.stop_reason
        if result.usage is not None:
            training["usage"] = result.usage
        boundary_record["summary_training"] = training

        await append_jsonl_async(
            session.transcript_path(agent_name), boundary_record
        )

    # Telemetry: one record per compaction so the trainer can surface the
    # empty-summary rate — and the zero-round / stripped-round shape — without
    # re-parsing the transcript (Session.record_compaction). After the splice
    # so the recorded suffix_count is the spliced truth, not the pre-await
    # selection.
    session.record_compaction(
        result.outcome,
        result.issue_kind,
        suffix_count=len(suffix),
        stripped_rounds=stripped_rounds,
    )

    if not suffix:
        log.warning(
            "compaction for %s kept ZERO rounds (suffix budget %d tokens): the "
            "most recent round alone exceeds the budget — the agent resumes "
            "from mandate + summary only. Recurring empty suffixes mean the "
            "budget is too small for this model/reasoning configuration"
            "%s.",
            agent_name, suffix_budget,
            "" if session.config.compaction.strip_reasoning_to_fit
            else " (strip_reasoning_to_fit is off)",
        )

    # §11.4.2 step 5: full compaction resets this agent's microcompaction
    # slot.
    if agent is not None and agent.microcompacted:
        agent.microcompacted = False
        await session.update_agent(agent)

    # Emit after the boundary is durable on disk so any subscriber that reacts
    # to the event sees the persisted view. Payload carries the shape of the
    # compaction (sizes), the outcome, and provenance for the summary call —
    # enough for dashboards to surface the moment without re-parsing the file.
    await session.events.emit(
        "compaction.fired",
        {
            "name": agent_name,
            "prefix_count": len(prefix),
            "suffix_count": len(suffix),
            "stripped_rounds": stripped_rounds,
            "pinned_count": len(pinned),
            "estimate_tokens": estimate,
            # `forced`: fired from the loop's prompt-too-long recovery, not the
            # estimate gate. `anchor_valid` False means no usage report passed
            # ANCHOR_MIN_REPORTED_RATIO and the estimate was pure chars/4.
            "forced": force,
            "anchor_valid": estimate_detail.anchor_index is not None,
            "calibration": estimate_detail.calibration,
            "suffix_budget": suffix_budget,
            "outcome": result.outcome,
            "attempts": result.attempts,
            "ptl_truncations": result.ptl_truncations,
            "issue": result.issue_kind,
            "stop_reason": result.stop_reason,
            "usage": result.usage,
        },
        name=agent_name,
    )

    log.info("compacted transcript for %s: %d -> %d messages", agent_name, len(transcript), len(session._transcripts[agent_name]))
    return True


@dataclass
class _BadSummaryReason:
    kind: str  # "empty" | "tool-template-leak" | "format-violation"
    detail: str


@dataclass
class _SummaryResult:
    """Outcome of one summarization call.

    `summary` is the extracted `<summary>` body that gets injected into the
    next segment as the agent's new context (the consume side — always set).

    The rest captures the *producer* side for SFT/RL: the request EXACTLY as
    sent and the raw completion the model returned, so the builder can train
    the model to *produce* compaction summaries on the same distribution it
    will hit at RL time. Populated for EVERY accepted summary — including ones
    that needed a bad-summary retry or a prompt-too-long head-truncation — and
    None only when the summary was ultimately unavailable. We do NOT filter at
    capture time: a recovered summary is still a good summary, and retries /
    truncations are real situations the policy faces in RL. `system_prompt` and
    `prompt_body` are stored verbatim rather than reconstructed because the
    rendered prefix reflects in-place microcompaction sentinels and any PTL
    head-truncation that the on-disk message records do NOT — reconstructing
    from disk would yield a prompt the policy never actually sees. `attempts` /
    `ptl_truncations` are provenance so the builder can filter downstream
    (e.g. clean-first-attempt only) without us discarding data here.
    """

    summary: str
    completion: Optional[str] = None
    reasoning: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt_body: Optional[str] = None
    attempts: int = 0
    ptl_truncations: int = 0
    # "accepted" | "bad-summary" | "ptl-failed" | "inference-error". We record
    # EVERY outcome (not just accepted) so a future SFT rebuild can study or
    # train on the failures too — the builder filters to "accepted" by default.
    outcome: str = "accepted"
    issue_kind: Optional[str] = None
    # Completion metadata of the summary call, for the build-time quality gate:
    # `stop_reason == "length"` => truncated at SUMMARY_MAX_TOKENS; `usage`
    # carries the input/output token counts (compaction ratio).
    stop_reason: Optional[str] = None
    usage: Optional[dict] = None


async def _summarize_prefix(
    session: Session, agent_name: AgentName, prefix: list[Message]
) -> _SummaryResult:
    """Produce a detailed prose summary of the older transcript prefix.

    1. **History is rendered as a single prose block in one user message**,
       not as native assistant/tool/tool_result messages. Replaying a long
       history of assistant turns with `tool_calls` primes chat-template
       models (kimi-k2.6 in particular) to emit a tool call next; with no
       `tools` array in the summary request, the renderer leaks raw template
       delimiters as content.
    2. **Output is structured** as <analysis>...</analysis> followed by
       <summary>...</summary> tags; we extract only the <summary> body.
    3. **Two retry tiers**, orthogonal:
       - PTL retries (up to MAX_PROMPT_TOO_LONG_RETRIES): if the call fails
         with a prompt-too-long error, drop ~20% of the oldest API rounds
         and re-issue. Doesn't consume the bad-summary budget.
       - Bad-summary retry (up to MAX_BAD_SUMMARY_RETRIES): if the call
         succeeds but the output is empty, leaks chat-template tokens, or
         omits the <summary> tag, retry once with a stronger prompt that
         names the failure. Two retries don't help once the model is broken
         on this turn.
    """
    last_bad: Optional[_BadSummaryReason] = None
    messages_to_summarize: list[Message] = list(prefix)
    ptl_attempts = 0
    bad_summary_attempts = 0
    last_reasoning_chars = 0

    while True:
        user_body = _render_conversation_for_compaction(messages_to_summarize)
        system_prompt = _build_compact_prompt(last_bad)
        # Proactively shrink the prefix until prompt + the FULL output budget
        # fits the model window. Without this, a fire that overshot the
        # threshold (a single huge turn can add up to the whole per-turn cap
        # between compaction checks) produces a prompt so large the server
        # clamps the summary's output budget — a truncated or empty summary
        # that no retry can fix, since the retry re-sends the same prompt
        # (16.7% of fires across the 260723 runs). chars//3 overestimates
        # tokens for math-heavy text, so we truncate slightly early — the
        # safe direction; the reactive PTL tier below stays as backstop.
        _window = get_model_context_window(
            session.state.model, session.config.compaction
        )
        _prompt_budget = (
            _window - session.config.compaction.summary_max_tokens - 1024
        )
        while (len(user_body) + len(system_prompt)) // 3 > _prompt_budget:
            _shrunk = _truncate_head_for_ptl_retry(messages_to_summarize)
            if _shrunk is None:
                break  # one giant round left — let the reactive tiers handle it
            messages_to_summarize = _shrunk
            user_body = _render_conversation_for_compaction(messages_to_summarize)
        # Carry the same routing headers as the main turn loop (loop.py):
        # X-Rollout-Id for trainers that group requests by rollout, and
        # X-Agent-Id to keep the summary call attributed to this agent's trace.
        extra_headers: dict[str, str] = {"X-Agent-Id": agent_name}
        if session.config.rollout_id:
            extra_headers["X-Rollout-Id"] = session.config.rollout_id
        request = InferenceRequest(
            model=session.state.model,
            messages=[
                ProviderMessage(role="system", content=system_prompt),
                ProviderMessage(role="user", content=user_body),
            ],
            tools=[],
            parallel_tool_calls=False,
            # Output cap ONLY — never `max_tokens`, which some servers map to
            # a total input+output budget: with a large rendered prefix that
            # can silently leave zero generation budget, and the summary dies
            # as a zero-output `length` stop.
            max_completion_tokens=session.config.compaction.summary_max_tokens,
            temperature=session.config.temperature,
            reasoning_effort=session.config.reasoning_effort,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )

        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        completed: Optional[CompletedChunk] = None
        try:
            async for ch in session.inference.stream(request):
                if isinstance(ch, ContentDeltaChunk):
                    content_chunks.append(ch.text)
                elif isinstance(ch, ReasoningDeltaChunk):
                    reasoning_chunks.append(ch.text)
                elif isinstance(ch, CompletedChunk):
                    completed = ch
        except Exception as e:
            if (
                _is_prompt_too_long_error(e)
                and ptl_attempts < MAX_PROMPT_TOO_LONG_RETRIES
            ):
                truncated = _truncate_head_for_ptl_retry(messages_to_summarize)
                if truncated is None:
                    log.warning(
                        "compaction summary PTL but no more head to drop for %s: %s",
                        agent_name, e,
                    )
                    return _SummaryResult(
                        f"[summary unavailable: {_summary_error_text(e)}]",
                        ptl_truncations=ptl_attempts,
                        outcome="ptl-failed",
                    )
                ptl_attempts += 1
                log.warning(
                    "compaction summary PTL retry %d for %s: %d → %d messages",
                    ptl_attempts, agent_name,
                    len(messages_to_summarize), len(truncated),
                )
                messages_to_summarize = truncated
                continue
            log.warning("compaction summary inference failed for %s: %s", agent_name, e)
            return _SummaryResult(
                f"[summary unavailable: {_summary_error_text(e)}]",
                ptl_truncations=ptl_attempts,
                outcome="inference-error",
            )

        last_reasoning_chars = sum(len(s) for s in reasoning_chunks)
        raw_content = "".join(content_chunks)
        summary, had_summary_tag = _extract_response_summary(raw_content)
        issue = _validate_summary(summary, had_summary_tag=had_summary_tag)
        reasoning = "".join(reasoning_chunks).strip() or None
        # Completion metadata from this attempt — `stop_reason == "length"`
        # flags a summary truncated at SUMMARY_MAX_TOKENS, and `usage` gives the
        # input->output compaction ratio. Both feed the build-time quality gate
        # and are unrecoverable from disk if not captured here.
        stop_reason = completed.stop_reason if completed else None
        usage = completed.usage.model_dump() if completed and completed.usage else None
        if issue is None:
            if bad_summary_attempts > 0 or ptl_attempts > 0:
                log.info(
                    "compaction summary recovered for %s (bad_summary_retries=%d, ptl_retries=%d)",
                    agent_name, bad_summary_attempts, ptl_attempts,
                )
            # Capture the request EXACTLY as sent (microcompaction sentinels and
            # any PTL truncation included) plus the raw completion, for every
            # accepted summary — retried/truncated ones too. This is the
            # (prompt, summary) distribution the policy faces at RL time.
            return _SummaryResult(
                summary=summary,
                completion=raw_content,
                reasoning=reasoning,
                system_prompt=system_prompt,
                prompt_body=user_body,
                attempts=bad_summary_attempts + 1,
                ptl_truncations=ptl_attempts,
                outcome="accepted",
                stop_reason=stop_reason,
                usage=usage,
            )

        if bad_summary_attempts >= MAX_BAD_SUMMARY_RETRIES:
            log.warning(
                "compaction summary unusable after %d attempts for %s (%s, reasoning_chars=%d): %s",
                bad_summary_attempts + 1, agent_name, issue.kind, last_reasoning_chars, issue.detail,
            )
            # Still log the failed request + bad completion so a later dataset
            # rebuild can use it (e.g. as a negative). The builder skips it as a
            # training target via outcome != "accepted".
            return _SummaryResult(
                summary=f"[summary unavailable: {issue.kind} — {issue.detail}]",
                completion=raw_content,
                reasoning=reasoning,
                system_prompt=system_prompt,
                prompt_body=user_body,
                attempts=bad_summary_attempts + 1,
                ptl_truncations=ptl_attempts,
                outcome="bad-summary",
                issue_kind=issue.kind,
                stop_reason=stop_reason,
                usage=usage,
            )

        bad_summary_attempts += 1
        last_bad = issue
        log.warning(
            "compaction summary bad on attempt %d for %s (%s): %s — retrying",
            bad_summary_attempts, agent_name, issue.kind, issue.detail,
        )


def _summary_error_text(error: BaseException) -> str:
    text = str(error).strip()
    if len(text) > SUMMARY_ERROR_MAX_CHARS:
        text = "..." + text[-SUMMARY_ERROR_MAX_CHARS:]
    return f"{type(error).__name__}: {text}"


def _is_prompt_too_long_error(error: BaseException) -> bool:
    """Detect 413 / context-overflow errors across providers.

    OpenAI-style exceptions carry a `status` or `status_code` attribute; some
    custom wrappers stringify the upstream message. Cover both.
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status == 413:
        return True
    return bool(_PROMPT_TOO_LONG_RE.search(str(error)))


def _group_messages_by_api_round(messages: list[Message]) -> list[list[Message]]:
    """Split into "API rounds" (one assistant turn + its trailing tool/user
    follow-ups belong to the same round). Used by PTL truncation to drop
    whole rounds at a time so we don't break tool-call/tool-result pairs.
    """
    groups: list[list[Message]] = []
    current: list[Message] = []
    for m in messages:
        if m.role == "assistant" and current:
            groups.append(current)
            current = [m]
            continue
        current.append(m)
    if current:
        groups.append(current)
    return groups


def _is_ptl_marker(m: Message) -> bool:
    return m.role == "user" and m.source == "meta" and m.content == _PTL_RETRY_MARKER


def _truncate_head_for_ptl_retry(messages: list[Message]) -> Optional[list[Message]]:
    """Drop the oldest ~20% of API rounds for a PTL retry.

    The opening round is pinned when it starts with a user message: the
    agent's mandate, plus — after the first compaction — the summary that
    follows it, so a summary is never written from a segment with no task in
    it. (`auto_compact_if_needed` pins the mandate alone; this pins the whole
    opening round, which post-compaction is mandate + previous summary.)
    Rounds are dropped from just after the pinned round instead.

    Returns None when truncation can't help (nothing droppable would remain).
    On success, a USER-meta marker sits where the drop happened so the model
    sees that history was truncated rather than missing.
    """
    # Strip the marker from a prior PTL retry so we don't pile them up.
    messages = [m for m in messages if not _is_ptl_marker(m)]

    groups = _group_messages_by_api_round(messages)
    pinned: list[Message] = []
    if groups and groups[0] and groups[0][0].role == "user":
        pinned = groups[0]
        groups = groups[1:]
    if len(groups) < 2:
        return None

    drop_count = min(len(groups) - 1, max(1, len(groups) // 5))
    sliced: list[Message] = []
    for g in groups[drop_count:]:
        sliced.extend(g)
    if not sliced:
        return None

    # If the first surviving message is an assistant, insert the marker so
    # the rendered prose doesn't run from the pinned round (or from nothing)
    # straight into a bare assistant turn. If the first survivor is already a
    # user message, the model has user-context to ground the assistant turn
    # that follows.
    if sliced[0].role == "assistant":
        marker = Message(role="user", source="meta", content=_PTL_RETRY_MARKER)
        return [*pinned, marker, *sliced]
    return [*pinned, *sliced]


def _render_conversation_for_compaction(messages: list[Message]) -> str:
    """Render the transcript prefix as a single labeled prose block.

    See `_summarize_prefix` docstring for why this is one user message in
    prose form rather than the native role-tagged history.
    """
    blocks: list[str] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            if not m.content:
                continue
            tag = "USER (meta)" if m.source == "meta" else "USER"
            blocks.append(f"{tag}:\n{_stringify_content(m.content)}")
            continue
        if m.role == "assistant":
            parts: list[str] = []
            text = _stringify_content(m.content)
            if text.strip():
                parts.append(text)
            for tc in (m.tool_calls or []):
                try:
                    args_str = json.dumps(tc.args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_str = str(tc.args)
                parts.append(f"(tool call) {tc.name}({args_str})")
            body = "\n".join(parts) if parts else "[no content]"
            blocks.append(f"ASSISTANT:\n{body}")
            continue
        if m.role == "tool":
            if m.tool_result is None:
                continue
            if m.tool_result.ok:
                tag = "TOOL RESULT"
                body = (
                    json.dumps(m.tool_result.output, default=str, ensure_ascii=False)
                    if m.tool_result.output is not None
                    else ""
                )
            else:
                tag = "TOOL ERROR"
                err = m.tool_result.error
                body = f"{err.code}: {err.message}" if err else ""
            blocks.append(f"{tag} ({m.tool_call_id or '?'}):\n{body}")
    return "Here is the conversation to summarize:\n\n" + "\n\n---\n\n".join(blocks)


def _stringify_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for seg in content:
            text = getattr(seg, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _extract_response_summary(content: str) -> tuple[str, bool]:
    """Return (summary, had_summary_tag).

    If <summary>...</summary> is present, that's the summary. Otherwise the
    fallback is the content minus any <analysis> block — the validator will
    then reject it as a format violation, since on a non-compliant response
    that fallback is whatever junk the model emitted instead of a summary.
    """
    summary_match = re.search(
        r"<summary>([\s\S]*?)</summary>", content, re.IGNORECASE
    )
    if summary_match:
        return summary_match.group(1).strip(), True
    fallback = re.sub(
        r"<analysis>[\s\S]*?</analysis>", "", content, flags=re.IGNORECASE
    ).strip()
    return fallback, False


def _validate_summary(
    summary: str, *, had_summary_tag: bool
) -> Optional[_BadSummaryReason]:
    """Three failure modes, in priority order:

    - **empty**: extraction returned nothing.
    - **tool-template-leak**: raw chat-template delimiters appear in the
      summary (kimi tool-call priming).
    - **format-violation**: no <summary> tag — the fallback content is
      whatever the model emitted instead of a summary and shouldn't be
      injected into the next compaction segment.
    """
    trimmed = summary.strip()
    if not trimmed:
        return _BadSummaryReason("empty", "summary was empty after extraction")
    leak = _CHAT_TEMPLATE_TOKEN_RE.search(trimmed)
    if leak:
        # Named without its <| |> delimiters: this detail is echoed into the
        # retry prompt and the unavailable-summary placeholder, and a server
        # that tokenizes text with special tokens live (some training stacks do) would turn
        # the literal delimiter back into the real token.
        name = leak.group(0)[2:-2]
        return _BadSummaryReason(
            "tool-template-leak",
            f"summary contains chat-template tokens ({name}) — the model "
            "emitted a tool-call template instead of prose. This means the summary "
            "request was answered with a (broken) tool call rather than text.",
        )
    if not had_summary_tag:
        return _BadSummaryReason(
            "format-violation",
            "response did not contain a <summary>...</summary> block; the model "
            "ignored the requested output format. Treating the raw content as a "
            "summary risks dropping stray text into the next segment.",
        )
    return None


def _build_compact_prompt(retry_hint: Optional[_BadSummaryReason]) -> str:
    retry_section = ""
    if retry_hint is not None:
        retry_section = (
            "\n\n## Previous Attempt Was Rejected\n"
            f"Reason: {retry_hint.detail}\n\n"
            # No literal token names here: a server that tokenizes prompt text
            # with special tokens live (some training stacks do) turns e.g. `<|im_start|>`
            # into the real turn marker, splitting this system prompt. The
            # generic `<|...|>` pattern is plain text (checked on Nemotron 3).
            "This is a retry. Output PROSE ONLY. Do not emit chat-template tokens of "
            "any kind — nothing matching the pattern `<|...|>`. "
            "Those tokens are reserved chat-template delimiters and their presence "
            "means the summary is unreadable. The summary must be a multi-paragraph "
            "prose description of what happened in the conversation — files touched, "
            "errors hit, decisions made, current task state — wrapped in "
            "`<analysis>...</analysis>` followed by `<summary>...</summary>` tags."
        )
    return (
        f"{_NO_TOOLS_PREAMBLE}\n\n"
        "Your task is to create a detailed summary of the conversation so far, paying "
        "close attention to the user's explicit requests and previous actions.\n"
        "This summary should preserve enough technical detail to continue the work "
        "without losing context.\n\n"
        "Before providing your final summary, wrap your reasoning in <analysis> tags. "
        "Then provide the final summary in <summary> tags.\n\n"
        "Your summary should include these sections:\n"
        "1. Primary Request and Intent\n"
        "2. Key Technical Concepts\n"
        "3. Files and Code Sections\n"
        "4. Errors and Fixes\n"
        "5. Problem Solving\n"
        "6. All User Messages\n"
        "7. Pending Tasks\n"
        "8. Current Work\n"
        "9. Optional Next Step\n\n"
        "Be specific about:\n"
        "- the user's explicit requests, constraints, and corrections\n"
        "- exact files, commands, errors, and code changes\n"
        "- the most recent unfinished work and where it stopped\n\n"
        "Do not omit recent user feedback that changes how the work should be done."
        f"{retry_section}"
    )
