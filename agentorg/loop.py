"""Per-agent run loop (§4).

One coroutine per agent. Builds context, streams inference, dispatches tool
calls, persists results, and handles the wait gate. Multiple loops run
concurrently in the same Session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from typing import Any, Optional

import jsonschema
import referencing

from agentorg.env_adapter import ENV_SESSION_DEAD_CODE
from agentorg.inbox import render_attachment_marker, render_inbox
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceRequest,
    ProviderMessage,
    ProviderTool,
    ReasoningDeltaChunk,
    ToolCallChunk,
)
from agentorg.noop_detect import (
    always_noop_calls,
    extract_command,
    is_always_noop_call,
    is_noop_call,
    turn_is_noop,
)
from agentorg.org import fire_org_agent
from agentorg.session import Session
from agentorg.storage import append_jsonl, append_jsonl_async
from agentorg.tools.context import ToolContext, ToolImpl, rejected_result
from agentorg.tools.permissions import get_policy
from agentorg.types import (
    Agent,
    InboxItem,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultError,
    now_ms,
)

# Env-tool circuit breaker. If the same env-side tool fails with the same
# error signature this many times in a row across the session, we abort:
# the model is stuck in a loop the harness can't break out of, and burning
# more rollout budget on it serves no one. Reset on any env-tool success.
ENV_TOOL_CIRCUIT_BREAKER_LIMIT = 5

# Pre-execution failures that keep `kind="error"` because they are the model's
# own mistake: it named a tool that does not exist, or sent args the schema
# rejects. The env never saw either, so — exactly like the `kind="rejection"`
# results — they say nothing about env health and must not feed the circuit
# breaker. Every OTHER refusal `_dispatch_tool` produces before execution is
# `kind="rejection"` and is recognised by that field, so it does not need
# listing here.
PRE_EXECUTION_ERROR_CODES = frozenset({"unknown-tool", "invalid-args"})

# §A.8.6 — the one tool exempt from the repeated-call rule: its args are
# always empty, its output is the wake payload, and re-calling it on the
# quiescent marker is handled by its own short-circuit.
REPEAT_EXEMPT_TOOLS = frozenset({"WaitForMessages"})

# §A.8.6 — output fields the content digest ignores: per-call bookkeeping that
# says nothing about whether the call did anything new. `reward` is here
# because it is an env-controlled float that some envs drift on every call; 
# digesting it would hand every repeat a fresh key and disable the
# rule for exactly the polling loops it exists to stop. `finished` is kept:
# it is bounded, and flipping it really does end the episode.
_DIGEST_IGNORED_KEYS = frozenset(
    {"metadata", "reward", "shaping_reward", "shaping_reason"}
)


log = logging.getLogger(__name__)


# Qwen chain-of-thought delimiter. Stripped from the CEO's plain final message
# before it's sent to an openreward @terminal grader — the trace splits the
# reasoning into a separate ReasoningItem, so leaving it in would duplicate it,
# and simple regex/exact-match graders would misgrade.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# §A.8.2 — when the CEO is in env-bound mode with budget > 0 and has
# never hired an org agent, fire a "you can hire" reminder every N
# countable tool calls. Suppressed once any agent has been hired (even
# if since fired): the CEO has already exercised the option, and the
# nudge text presumes they haven't.
# "Countable" excludes plan-mode plumbing and trivial reads; we want the
# nudge to track substantive solo work in the env. The interval is in
# *assistant turns that did substantive work*, not raw tool-call counts —
# one nudge per N qualifying turns.
SOLO_NUDGE_INTERVAL = 2
SOLO_NUDGE_EXCLUDE = frozenset(
    {
        "EnterOrgPlanMode",
        "ExitOrgPlanMode",
        "WaitForMessages",
        "ListChannels",
        "ListArchive",
        "TodoWrite",
    }
)


WAIT_GATE_TOOLS = frozenset(
    {
        # Coordination
        "SendMessage",
        "WaitForMessages",
        "PostToChannel",
        "ListChannels",
        "ViewChannelMessages",
        "SearchChannelMessages",
        "SetChannelSubscription",
        "ReactToMessage",
        "CreateChannels",
        # Archive
        "ListArchive",
        "ViewArchiveFile",
        "WriteToArchive",
        "EditArchiveFile",
        "MoveArchiveFile",
        # Org-management
        "HireOrgAgent",
        "FireOrgAgent",
        "UpdateOrgAgentTools",
        "EnterOrgPlanMode",
        "ExitOrgPlanMode",
        # Productivity
        "TodoWrite",
    }
)


def visible_tool_defs(
    session: Session,
    agent: Agent,
    *,
    waiting_for_org_agents: bool = False,
) -> list[ToolDefinition]:
    allowed = agent.tool_allowlist
    disallowed = agent.tool_disallowlist or []
    # Under wait-gate the visible set depends on which peers hold which
    # tools, so cache invalidation needs the peer-tool signature.
    peers_signature: tuple = ()
    if waiting_for_org_agents and agent.kind == "ceo":
        peers_signature = tuple(sorted(
            (
                p.name,
                tuple(p.tool_allowlist) if p.tool_allowlist is not None else None,
                tuple(p.tool_disallowlist or ()),
            )
            for p in _ceo_unreported_live_org_agents(session)
        ))
    episode_finished = session.terminal_event is not None
    cache_key = (
        tuple(allowed) if allowed is not None else None,
        tuple(disallowed),
        bool(waiting_for_org_agents),
        peers_signature,
        episode_finished,
    )
    cached = session._vtd_cache.get(agent.name)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    if agent.kind == "ceo" and "ReportToCEO" not in disallowed:
        disallowed = list(disallowed) + ["ReportToCEO"]
    restrict_to: Optional[set[str]] = None
    if session.state.mode == "org-plan" and agent.kind == "ceo":
        # Plan mode permission policy denies state-mutating tools globally;
        # we expose only the read-only + plan-management + filesystem
        # write-to-plan path. The model can still call others — they'll be
        # denied at dispatch time with a clear error.
        pass
    if waiting_for_org_agents and agent.kind == "ceo":
        # CEO tool restriction while waiting for unreported org agents (§4.5.2).
        # Always include WAIT_GATE_TOOLS and any read_only tool. Beyond that,
        # surface tools the CEO holds exclusively — no unreported, non-parked
        # peer can call them, so there's no race to gate.
        readonly = {t.name for t in session.tools.definitions() if t.read_only}
        unreported = _ceo_unreported_live_org_agents(session)
        exclusive = {
            t.name for t in session.tools.definitions()
            if not _peers_holding_tool(unreported, t.name)
        }
        restrict_to = set(WAIT_GATE_TOOLS) | readonly | exclusive
    defs = session.tools.visible(
        allowed=allowed,
        disallowed=disallowed,
        restrict_to=restrict_to,
    )
    # §6.5: once the env signalled finished=True the episode is over — every
    # env tool leaves every agent's tool list for any turn that starts before
    # the session stops. `_dispatch_tool` refuses any that is called anyway.
    if episode_finished:
        defs = [d for d in defs if d.scope != "env"]
    session._vtd_cache[agent.name] = (cache_key, defs)
    return defs


def messages_for_inference(
    session: Session,
    agent: Agent,
    *,
    extra_user: Optional[str] = None,
) -> list[ProviderMessage]:
    """Assemble the message array sent to the LLM this turn (§4.3).

    Every assistant turn's ``reasoning_content`` is passed through — preserved
    thinking.
    """
    out: list[ProviderMessage] = [
        ProviderMessage(role="system", content=agent.system_prompt or "")
    ]
    for msg in session.transcript(agent.name):
        out.append(_to_provider_message(msg))
    if extra_user is not None and extra_user.strip():
        out.append(ProviderMessage(role="user", content=extra_user))
    return out


def _to_provider_message(msg: Message) -> ProviderMessage:
    if msg.role == "tool":
        return ProviderMessage(
            role="tool",
            content=_serialize_tool_result(msg.tool_result),
            tool_call_id=msg.tool_call_id,
        )
    if msg.role == "assistant":
        content: Any = msg.content if isinstance(msg.content, str) else None
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        return ProviderMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning=msg.reasoning,
        )
    # system / user
    if isinstance(msg.content, list):
        # Flatten to text — most providers don't accept rich user segments here.
        texts = [seg.text for seg in msg.content if seg.type == "text" and seg.text]
        paths = [seg.archive_path for seg in msg.content
                 if not (seg.type == "text" and seg.text) and seg.archive_path]
        text = "\n".join(t for t in [*texts, render_attachment_marker(paths)] if t)
        return ProviderMessage(role=msg.role, content=text)
    return ProviderMessage(role=msg.role, content=msg.content or "")


def _serialize_tool_result(result: Optional[ToolResult]) -> str:
    if result is None:
        return ""
    if result.ok:
        return json.dumps(result.output, default=str, ensure_ascii=False)
    error: dict[str, Any] | None = None
    if result.error:
        error = {"code": result.error.code, "message": result.error.message}
        # A rejection or an infra failure is not a failure of the call: tell
        # the model which it was. Plain errors keep their historical rendering.
        if result.error.kind != "error":
            error["kind"] = result.error.kind
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


def tool_defs_to_provider_tools(defs: list[ToolDefinition]) -> list[ProviderTool]:
    return [
        ProviderTool(name=d.name, description=d.description, input_schema=d.input_schema)
        for d in defs
    ]


def _validate_tool_args(schema: dict[str, Any], args: dict[str, Any]) -> Optional[str]:
    """Return None if `args` validates against `schema`, otherwise an error string.

    Per SPEC §5.1, the harness MUST validate args against input_schema
    before calling execute; the failure surfaces as a ToolResult with
    `error.code = "invalid-args"`. We use Draft7 because that's what the
    tool-call schemas across providers (OpenAI, Anthropic, Claude Code)
    converge on for `function.parameters` / `input_schema`.

    Empty / trivially-permissive schemas (`{}` or `{"type":"object","properties":{}}`)
    accept any object — matches model behaviour expectations.
    """
    if not schema:
        return None
    try:
        # An empty registry: env-supplied schemas can't make the host fetch a
        # remote or file:// `$ref` (jsonschema's default resolves them with
        # urlopen). In-schema `#/...` refs still resolve.
        jsonschema.Draft7Validator(schema, registry=referencing.Registry()).validate(args)
    except jsonschema.ValidationError as e:
        # Provide enough context for the model to self-correct without
        # leaking the full schema (the model already has it).
        path = "/" + "/".join(str(p) for p in e.absolute_path) if e.absolute_path else ""
        return f"input_schema validation failed at {path or '/'}: {e.message}"
    except Exception as e:  # noqa: BLE001
        # The schema itself is broken (SchemaError, or a structural issue
        # that surfaces as AttributeError / TypeError mid-validation).
        # Operator error, not the model's fault — log loudly and let the
        # call through; better than blocking every call on a misconfigured
        # tool. The tool's own `execute` may still reject the args.
        log.error(
            "tool input_schema is invalid (allowing call through): %s: %s",
            e.__class__.__name__,
            e,
        )
        return None
    return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def run_agent(session: Session, agent: Agent) -> None:
    """Coroutine that drives a single agent until fired or session ends.

    For the CEO this is the main session loop; for org agents these run in
    parallel as background tasks (`session._agent_tasks`).
    """
    name = agent.name
    log.info("agent loop started: %s (%s)", name, agent.kind)
    turn_index = -1
    try:
        while True:
            # After env-terminal nothing is left to read: the episode is over.
            if session.stop_requested() and (
                session.terminal_event is not None
                or not await session.inbox.has_pending(name)
            ):
                break
            if agent.state == "fired":
                break
            # Decide whether to run inference this iteration. A turn fires iff
            # the agent has something unanswered — either the transcript ends
            # in a non-assistant message (a tool result, user input, or the
            # initial mandate) or there are inbox items waiting. We only *peek*
            # here; the inbox is drained later (after compaction, just before
            # inference) so items delivered during a slow compaction or a
            # queued inference are still folded into this turn rather than
            # deferred a whole turn (freshness invariant, §4.3).
            if not _has_pending_input(session, name) and not await session.inbox.has_pending(name):
                if not _should_park(session, agent):
                    log.info("agent %s exiting loop (nothing to do)", name)
                    return
                # No timeout. Liveness comes from the quiescence invariant
                # (SPEC §4.5.1): the CEO can't be parked while every org
                # agent is also parked. The helper handles both the CEO
                # short-circuit (inject quiescent marker + don't park) and
                # the org-agent trigger (wake CEO if last to park).
                await _park_with_quiescence_guard(session, agent)
                if session.stop_requested():
                    continue
                continue

            turn_index += 1

            # Compaction before inference if we're approaching the limit.
            try:
                from agentorg.compaction import (
                    auto_compact_if_needed,
                    microcompact_if_needed,
                )

                await microcompact_if_needed(session, name)
                await auto_compact_if_needed(session, name)
            except Exception as e:
                log.warning("compaction failed for %s: %s", name, e)

            # Drain the inbox into this turn's user message — AFTER compaction
            # and immediately before inference (freshness invariant, §4.3).
            # Anything delivered during a slow/queued compaction, or during the
            # previous turn's long inference, is folded into *this* request
            # instead of being deferred to the next turn's drain. Draining
            # post-compaction also keeps brand-new messages verbatim rather
            # than summarising them away.
            items = await session.inbox.drain(name)
            if items:
                await _persist_turn_user(session, agent, render_inbox(items, recipient=name))

            await session.events.emit(
                "turn.started", {"name": name, "turn_index": turn_index}, name=name
            )

            try:
                tool_calls, assistant_msg = await _stream_turn(session, agent)
            except Exception as e:
                # A provider-reported prompt-too-long is ground truth the
                # token estimator missed: had the estimate crossed the
                # threshold, compaction would already have fired above. PTL
                # is also deterministic — the identical prompt fails forever
                # — so without intervention the agent dies below. Force a
                # compaction (skipping the estimate gate) and retry the turn
                # ONCE; if compaction can't fire or the retry fails too,
                # fall through to the terminal handling as before.
                from agentorg.compaction import (
                    _is_prompt_too_long_error,
                    auto_compact_if_needed,
                )

                recovered = False
                if _is_prompt_too_long_error(e):
                    log.warning(
                        "prompt too long on main inference for %s (%s) — "
                        "forcing compaction and retrying the turn",
                        name, e,
                    )
                    try:
                        compacted = await auto_compact_if_needed(
                            session, name, force=True
                        )
                    except Exception:
                        log.exception("forced PTL compaction failed for %s", name)
                        compacted = False
                    if compacted:
                        try:
                            tool_calls, assistant_msg = await _stream_turn(
                                session, agent
                            )
                            recovered = True
                        except Exception as retry_err:
                            log.warning(
                                "turn retry after forced PTL compaction still "
                                "failed for %s: %s",
                                name, retry_err,
                            )
                            e = retry_err
                if not recovered:
                    log.exception("inference failed for %s: %s", name, e)
                    # Per-attempt sink already wrote one line per failed attempt
                    # during retries; this one captures the terminal exception
                    # with full context. The transcript is deliberately left
                    # untouched on failure — the failure is visible via the
                    # event stream and inference-errors.jsonl. See SPEC §8.2.
                    await append_jsonl_async(
                        session.paths.inference_errors,
                        {
                            "name": name,
                            "ts": now_ms(),
                            "turn_index": turn_index,
                            "phase": "terminal",
                            "type": e.__class__.__name__,
                            "message": str(e)[:500],
                            "status": getattr(e, "status_code", None),
                            "code": getattr(e, "code", None),
                            "request_id": getattr(e, "request_id", None),
                        },
                    )
                    if agent.kind == "ceo":
                        # The supervisor restarts the CEO when new input
                        # arrives; if the session ends first, it ends as
                        # aborted with this reason, not "complete" (§4.4).
                        session.ceo_inference_error = f"inference-error:{e.__class__.__name__}"
                    await session.events.emit(
                        "turn.ended", {"name": name, "turn_index": turn_index}, name=name
                    )
                    return
            if agent.kind == "ceo":
                session.ceo_inference_error = None
                session.ceo_retry_offered = False

            # Terminal-tool routing (openreward @terminal envs). If the env
            # grades the assistant's plain final message, mutate the assistant
            # message in place to carry a synthetic ToolCall pointing at the
            # env's terminal tool, then let the normal dispatch path grade it
            # via _dispatch_terminal_tool below (post-persist). Doing this
            # BEFORE persist is what keeps the trace shape
            # AssistantMessage(answer) -> ToolCall(terminal) -> ToolResult:
            # openreward's UI attaches reward badges to ToolResult events and
            # ignores them on AssistantMessage, so a trace ending on the plain
            # message would store the reward but never render it inline.
            terminal_call: Optional[ToolCall] = None
            if not tool_calls and _is_terminal_grading_candidate(session, agent):
                terminal_call = await _prepare_terminal_call(session, assistant_msg)
                if terminal_call is not None:
                    tool_calls = [terminal_call]
                    assistant_msg.tool_calls = list(assistant_msg.tool_calls or []) + [terminal_call]

            # Persist assistant message.
            await session.append_message(name, assistant_msg)

            if terminal_call is not None:
                # Grade the CEO's answer via the env's @terminal tool, persist
                # the synthetic ToolResult message, and mark the session
                # terminal. Returns without falling into the tool-less nudge
                # path or the normal _dispatch_tools call (both would be wrong
                # here — the tool isn't in the harness registry and the
                # rollout is already over).
                await _dispatch_terminal_tool(
                    session,
                    agent,
                    terminal_call,
                    response_id=assistant_msg.provider_response_id,
                )
                await session.events.emit(
                    "turn.ended", {"name": name, "turn_index": turn_index}, name=name
                )
                return

            if not tool_calls:
                # Tool-less turn — structural fallback per §4.5.1.
                await session.events.emit(
                    "turn.ended", {"name": name, "turn_index": turn_index}, name=name
                )
                if agent.kind == "ceo":
                    # §6.5: the env already finished (a peer's call, while
                    # this turn was generating). The episode is over: no
                    # park, no nudge.
                    if session.terminal_event is not None:
                        return
                    if _ceo_has_live_org_agents(session):
                        await _park_with_quiescence_guard(session, agent)
                        continue
                    # Interactive (end_on_idle=False): a tool-less turn means the
                    # CEO has answered and is now idle. Park instead of ending so
                    # later external input (send_user_input) re-engages it; the
                    # session ends only on explicit close/abort.
                    if not session.config.end_on_idle:
                        await _park_with_quiescence_guard(session, agent)
                        continue
                    # Env-bound CEO with no live agents and tool-less turn:
                    # §4.5.1 prescribes a last-chance nudge before terminating.
                    # First consecutive miss → nudge and loop; second → end.
                    if session.state.environment is not None:
                        session._env_tool_less_streak += 1
                        if session._env_tool_less_streak == 1:
                            await _inject_env_tool_less_nudge(
                                session,
                                truncated=assistant_msg.stop_reason == "length",
                            )
                            continue
                        log.info(
                            "env-bound CEO produced %d consecutive tool-less turns — ending run",
                            session._env_tool_less_streak,
                        )
                    return
                # Org agent: park (always reachable via DM / channel). Same
                # quiescence trigger as the WaitForMessages tool path —
                # parking org agent wakes CEO if it's the last to park.
                await _park_with_quiescence_guard(session, agent)
                continue

            # A successful tool call clears the tool-less streak; the CEO is
            # back to making real progress.
            if agent.kind == "ceo" and session._env_tool_less_streak:
                session._env_tool_less_streak = 0

            # Dispatch tool calls. §A.8.5: decide up front whether this whole
            # turn was a specialist doing nothing, so the penalty stamped onto
            # each tool result and the nudge fired below agree by construction.
            noop_turn = turn_is_noop(tool_calls)
            await _dispatch_tools(
                session, agent, assistant_msg, tool_calls, noop_turn=noop_turn
            )

            # §A.8.5 — warn, and eventually park, a specialist signalling that
            # it is finished via the shell. A turn that did real work clears
            # the streak.
            if noop_turn:
                if await _handle_noop_turn(session, agent, tool_calls):
                    await session.events.emit(
                        "turn.ended", {"name": name, "turn_index": turn_index}, name=name
                    )
                    await _park_with_quiescence_guard(session, agent)
                    continue
            else:
                session.record_productive_turn(name)
                # A turn that did real work but still spent a call on
                # something that could never do anything. Not a stall, so no
                # nudge and no park — but the call is still wasted, still
                # penalised, and still worth counting.
                strays = always_noop_calls(tool_calls)
                if strays:
                    await _record_stray_noops(session, agent, strays)

            # §A.8.2 — bump the solo-thrash counter and fire the reminder
            # when threshold is reached. Relevant for any ATTENDED CEO — env-
            # bound or interactive (end_on_idle=False, no env bound) —
            # with a non-zero budget that has never hired (fired agents count
            # as having exercised the option, so the nudge stays off). We
            # count *qualifying turns* (turns with ≥1 non-excluded tool call),
            # not raw tool calls — that way the nudge cadence matches the
            # CEO's mental model of "how many turns have I worked solo".
            if (
                agent.kind == "ceo"
                and (
                    session.state.environment is not None
                    or not session.config.end_on_idle
                )
                and session.state.budget.max_org_agents > 0
                and not any(a.kind == "org" for a in session.all_agents())
            ):
                has_substantive = any(
                    tc.name not in SOLO_NUDGE_EXCLUDE for tc in tool_calls
                )
                if has_substantive:
                    session._solo_turn_count += 1
                    if session._solo_turn_count >= SOLO_NUDGE_INTERVAL:
                        session._solo_turn_count = 0
                        await _inject_solo_thrash_nudge(session)

            await session.events.emit(
                "turn.ended", {"name": name, "turn_index": turn_index}, name=name
            )

            if session.stop_requested():
                break

            # §6.5: once the env has finished, the CEO starts no further turn,
            # even before the harness has fired the org and stopped the
            # session.
            if agent.kind == "ceo" and session.terminal_event is not None:
                return
    finally:
        log.info("agent loop ended: %s", name)


def _ceo_has_live_org_agents(session: Session) -> bool:
    return any(a.kind == "org" and a.state != "fired" for a in session.all_agents())


# First token of an env tool's name that marks it as the way to hand in an
# answer. Mirrors `openreward_env._looks_read_only`, which classifies env tools
# by the same trick.
_SUBMIT_VERBS = frozenset(
    {"submit", "answer", "finish", "final", "complete", "respond", "solve"}
)


def _env_submit_tools(session: Session) -> list[str]:
    """Env tools that actually end a rollout, by name. Possibly empty.

    "The env tools" is not a usable stand-in: under a claude-code toolset
    `bash`, `ls` and `grep` are all `scope = "env"`, and a nudge that offers
    them as ways to "submit your answer and end the rollout" is telling the
    model to end the episode with a shell call.

    An empty result means "none of these looks like a submission tool", NOT
    "this env has none" — the check is a name heuristic, and an env is free to
    call its submission tool something the verb list has never heard of. A
    caller MUST phrase the empty case as uncertainty rather than as a fact
    about the environment. It is also the common outcome: nothing matches in
    many environments.

    A matching tool that has already returned `finished=False` is excluded:
    the nudge tells the model the tool "ends the rollout", and this rollout has
    shown it doesn't (e.g. a `submit` that grades and continues).
    """
    return sorted(
        d.name
        for d in session.tools.definitions()
        if d.scope == "env"
        and re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", d.name)[0].lower() in _SUBMIT_VERBS
        and d.name not in session._env_tools_seen_unfinished
    )


async def _inject_env_tool_less_nudge(
    session: Session, truncated: bool = False
) -> None:
    """§4.5.1 — last-chance nudge before terminating on a tool-less turn.

    Env-bound solo CEO produced a turn without `tool_calls`. Two distinct
    causes get distinct wording:

    * ``truncated=False`` — a completed turn that only wrote content. Naming
      the env's submission tool explicitly tends to recover this: the model
      often wrote the answer into content and forgot the structured
      submission.
    * ``truncated=True`` — the engine cut the turn at a token limit
      (``stop_reason == "length"``) before any tool call. The model may not
      realize its response never finished — telling it, and telling it to be
      briefer, is what makes the recovery turn behave differently from the
      turn that just failed. (After a mid-reasoning cut the stored message may
      have no content at all, so the completed-turn wording would describe a
      message the model can't even see in its history.)
    """
    submit_tools = _env_submit_tools(session)
    # Submission is gated on the agent actually wanting to stop, in every
    # branch. An unconditional "to submit, call X" reads as an instruction to
    # a model looking for a way out, and premature submission caps the episode
    # return as surely as a done-loop wastes it (§A.8.5).
    if len(submit_tools) == 1:
        route = (
            "If you want to finish now and stop working, call "
            f"`{submit_tools[0]}` with your final answer — that ends the "
            "rollout. Otherwise keep working: call any other available tool "
            "and carry on improving the result."
        )
    elif submit_tools:
        names = ", ".join(f"`{n}`" for n in submit_tools)
        route = (
            f"If you want to finish now and stop working, call one of {names} "
            "with your final answer — that ends the rollout. Otherwise keep "
            "working: call any other available tool and carry on improving "
            "the result."
        )
    else:
        # Recognition is a name heuristic, so "no match" is not the same as
        # "this env has none" and must not be phrased as though it were. It is
        # also the common case, not a fallback: nothing matches in many envs,
        # including every claude-code-toolset env.
        route = (
            "If one of your tools submits an answer and you want to finish "
            "now, call it with your final answer. If none obviously does, "
            "your default assumption should be that there is more to do: keep "
            "working, call any available tool and carry on improving the "
            "result."
        )
    # §A.8.5 lives one nudge away from this one, and without this line the two
    # contradict: an agent told to "call an environment tool" reaches for
    # `bash`, reaches for the cheapest thing to put in it, and gets penalised
    # for a call this message asked for.
    route += (
        " Do not reach for a no-op like `exit 0` or `echo done` to satisfy "
        "this — those change nothing and carry a negative reward. Make a real "
        "call."
    )

    warning = (
        " If you don't make a tool call on the next turn either, the "
        "rollout will terminate."
    )
    if truncated:
        body = (
            "Your previous response was cut off at the per-turn output limit "
            "before it made a tool call, and truncated text is not graded. "
            "Do not restart the same long response — be much more concise "
            f"and end with a tool call this turn. {route}{warning}"
        )
    else:
        body = (
            "Your previous response had content but no tool call. Bare text is "
            "not graded — only tool calls advance the environment. "
            f"{route}{warning}"
        )
    await session.inbox.append(
        session.ceo_name, InboxItem(kind="meta", content=body)
    )
    session.wakes.wake(session.ceo_name)
    log.info(
        "fired env-bound tool-less-turn nudge for %s (truncated=%s)",
        session.ceo_name,
        truncated,
    )


async def _inject_solo_thrash_nudge(session: Session) -> None:
    """§A.8.2 — recurring 'remember you can hire' nudge."""
    budget = session.state.budget.max_org_agents
    # Closing line depends on how this CEO finishes: an env-bound CEO submits
    # via the env tool and terminates; an interactive (envless) CEO has no
    # submission tool — it just replies and parks for the next human turn.
    close_out = (
        "submit via the env's submission tool and terminate."
        if session.state.environment is not None
        else "just reply with your answer."
    )
    body = (
        f"Reminder: you have {budget} specialist agent(s) you can hire. "
        "You've been working solo for several turns. If the answer isn't already in reach, "
        "strongly prefer to design an org (call WritePlan with an agents-org block, then "
        "ExitOrgPlanMode) over persisting solo — parallel specialists shorten long-horizon "
        f"tasks. If you're already close to a confident answer, {close_out}"
    )
    session._solo_thrash_nudges_fired += 1
    await session.inbox.append(
        session.ceo_name,
        InboxItem(kind="meta", content=body),
    )
    session.wakes.wake(session.ceo_name)
    log.info(
        "fired solo-thrash nudge #%d for %s",
        session._solo_thrash_nudges_fired,
        session.ceo_name,
    )


def _noop_nudge_body(
    session: Session, agent: Agent, command: str, count: int, threshold: int
) -> str:
    """§A.8.5 warning text for a specialist that emitted a no-op turn.

    `count` is how many of the last `noop_window` turns did nothing, not a
    consecutive run. Escalates from the second strike on: a model that ignored
    the first warning is unlikely to read the same paragraph differently, so
    the repeat leads with the count and what happens when the strikes run out.
    """
    steps = [
        "1. ReportToCEO with your status and an explicit ask for the next piece "
        "of work, then call WaitForMessages to park. Parking costs nothing and "
        "you wake the moment anyone messages you.",
    ]
    if _agent_holds_tool(agent, "PostToChannel"):
        steps.append(
            "2. If the CEO does not come back, PostToChannel tagging "
            f"@{session.ceo_name} with a specific question, then WaitForMessages "
            "again."
        )
    steps.append(
        f"{len(steps) + 1}. If still nothing, pick your assignment back up and "
        "find a way to raise the quality of what you already delivered. There is "
        "always something to improve."
    )
    escalation = ""
    if count > 1:
        remaining = max(0, threshold - count)
        tail = (
            "the harness will park you on your next one"
            if remaining <= 1
            else f"the harness will park you after {remaining} more"
        )
        escalation = (
            f"{count} of your recent turns have done nothing and you were "
            f"already warned. Keep going and {tail}. "
        )
    return (
        escalation
        + f"That turn did nothing. `{command}` is a no-op: it changes nothing in "
        "the environment and it does not end your assignment. There is no "
        '"mission complete", the rollout ends when the CEO submits, not when '
        "you stand down. No shell command can end it, and each no-op call is "
        "recorded and scored against you (see the negative `shaping_reward` "
        "on that tool result).\n\n"
        "If you have no work in hand, in this order:\n"
        + "\n".join(steps)
        + "\n\nNever emit `exit`, `echo done`, or similar to stand down."
    )


def _noop_ceo_nudge_body(session: Session, command: str, count: int) -> str:
    """§A.8.5 warning text for a CEO that emitted a no-op turn.

    The specialist wording does not transfer: the CEO has nobody to report to
    and nothing to park for. Nor does the obvious inverse, "submit if you are
    done" — plenty of environments expose no submission tool at all, and where
    one exists, offering it to a CEO that has already decided it is finished
    just hands it a legitimate way to stop early. Premature submission caps the
    reward as surely as a done-loop wastes it. So the message is that the
    environment has NOT ended and there is more to do.
    """
    escalation = ""
    if count > 1:
        escalation = (
            f"{count} of your recent turns have done nothing and you were "
            "already warned. "
        )
    lines = [
        "- Re-read the task and check what you have actually delivered "
        "against what it asks for. Partial coverage is the usual gap.",
    ]
    if session.live_org_agents():
        lines.append(
            "- Review your specialists' work for errors, thin spots, and "
            "claims nobody verified. Direct them with SendMessage or "
            "PostToChannel, or FireOrgAgent anyone genuinely finished (each "
            "needs an exit review)."
        )
    lines.append(
        "- If the work really is covered, raise its quality rather than "
        "standing down. There is almost always something left to improve."
    )
    return (
        escalation
        + f"That turn did nothing. `{command}` is a no-op: it changes nothing "
        "in the environment and it ends nothing. The environment has not "
        'ended and this rollout is still running. There is no "mission '
        'complete" to signal. Each no-op '
        "call is recorded and scored against you (see the negative "
        "`shaping_reward` on that tool result).\n\nKeep working:\n"
        + "\n".join(lines)
        + "\n\nNever emit `exit`, `echo done`, or similar to wind down."
    )


def _noop_ceo_repeat_body(command: str, count: int) -> str:
    """Terse re-injection for a CEO that has already had the full warning.

    Rationed by `Session.claim_noop_nudge`, and short by the same reasoning
    `_noop_nudge_body` gives for escalating: a model that ignored the full
    paragraph will not read it differently the fourth time, so repeating it
    buys nothing and costs a kilobyte of context each go.
    """
    return (
        f"Still no-ops: `{command}`. {count} of your recent turns have done "
        "nothing and each one is scored against you. The environment has not "
        "ended. Keep working and find something to improve."
    )


def _noop_park_body(count: int) -> str:
    """§A.8.5 final message, left pending so the agent reads it on wake."""
    return (
        f"{count} of your recent turns have done nothing, after being warned. "
        "The harness has parked you. You will wake as soon as the CEO or a "
        "teammate messages you.\n\n"
        "When you wake, do not emit another no-op. ReportToCEO with your status "
        "and ask for work, or pick your assignment back up and improve on what "
        "you delivered. There is no way to end your assignment from the shell."
    )


async def _record_stray_noops(
    session: Session, agent: Agent, strays: list[ToolCall]
) -> None:
    """Count and surface no-op calls that rode along with real work.

    Deliberately quieter than `_handle_noop_turn`: no nudge, no strike, no
    effect on the window. The turn was productive, and warning about a stray
    line in a turn that did real work would be noise. The event carries
    `turn_noop: false` so analysis can keep the two apart.
    """
    session.record_noop_calls(agent.name, len(strays), kind=agent.kind)
    first = strays[0]
    command = " ".join((extract_command(first.args) or first.name).split())[:120]
    await session.events.emit(
        "agent.noop_detected",
        {
            "name": agent.name,
            "kind": agent.kind,
            "tool": first.name,
            "command": command,
            "calls": len(strays),
            "turn_noop": False,
            "penalty": float(session.config.noop_penalty),
            "parked": False,
        },
        name=agent.name,
    )


async def _handle_noop_turn(
    session: Session, agent: Agent, tool_calls: list[ToolCall]
) -> bool:
    """§A.8.5 — warn an agent that just spent a turn doing nothing.

    Returns True when the caller should force-park the agent, which only ever
    happens for a specialist. The CEO gets the gate and the penalty but not the
    remedy: it has nobody to report to and nothing to park for, so parking it
    would strand the rollout. Its nudge tells it the environment has not ended
    and sends it back to the work — see `_noop_ceo_nudge_body` for why it does
    not offer submission as the alternative.

    The CEO's nudge is rationed to one per `noop_window` turns, with a terse
    repeat after the first: it has no park to end the ladder, so re-injecting
    the full paragraph on every no-op turn would pile harness meta into the
    transcript and drive compaction churn in exactly the rollouts already
    going wrong.

    Strikes are counted over a sliding window of the last `noop_window` turns,
    not as a strict run. A CEO alternating `echo done` with the occasional real
    `python - <<PY` never accumulates a consecutive run but is every bit as
    stuck; requiring consecutiveness let exactly that shape through.

    The ladder is `noop_park_strikes` no-op turns within the window, then one
    strike for a specialist already parked once: a loop that resumes after a
    wake does not get the grace period again.

    Neither branch calls `wakes.wake`. The agent is demonstrably running (we
    are mid-turn, and the transcript ends in a tool result, so the loop will
    iterate again and drain the inbox on its own), and a stale wake event is
    actively harmful: `WakeBroker.park_and_wait` returns immediately when the
    event is already set, so a wake left over from an earlier nudge would make
    the park strike a no-op. A genuine wake from a DM or channel notification
    during the turn still short-circuits the park, which is correct — an agent
    with real input to read should not be parked.
    """
    first = tool_calls[0]
    command = " ".join((extract_command(first.args) or first.name).split())[:120]

    window_size = max(1, int(session.config.noop_window))
    run, in_window = session.record_noop_turn(
        agent.name, len(tool_calls), kind=agent.kind
    )
    threshold = (
        1 if session.noop_park_count(agent.name) else session.config.noop_park_strikes
    )
    park = agent.kind == "org" and in_window >= max(1, threshold)

    await session.events.emit(
        "agent.noop_detected",
        {
            "name": agent.name,
            "kind": agent.kind,
            "tool": tool_calls[0].name if tool_calls else None,
            "command": command,
            "calls": len(tool_calls),
            "turn_noop": True,
            "streak": run,
            "in_window": in_window,
            "window": window_size,
            "penalty": float(session.config.noop_penalty),
            "parked": park,
        },
        name=agent.name,
    )

    if agent.kind == "ceo":
        # The CEO is never parked, so nothing else bounds this. Ration the
        # text to at most one per `noop_window` turns and shorten the repeat;
        # the penalty above and the event just emitted still fire every time.
        sent = session.claim_noop_nudge(agent.name, every=window_size)
        if sent is not None:
            body = (
                _noop_ceo_nudge_body(session, command, in_window)
                if sent == 1
                else _noop_ceo_repeat_body(command, in_window)
            )
            await session.inbox.append(
                agent.name, InboxItem(kind="meta", content=body)
            )
            log.info(
                "fired no-op nudge #%d for CEO %s (%d in last %d, command=%r)",
                sent, agent.name, in_window, window_size, command,
            )
        return False

    if park:
        body = _noop_park_body(in_window)
        session.record_noop_park(agent.name)
        # Drop a stale wake before parking. `spawn_org_agent` sets one to
        # deliver the mandate and nothing consumes it (the agent's turns are
        # driven by the transcript, not by a park), so without this the park
        # would short-circuit and the agent would keep looping. Only safe when
        # nothing else is pending: a wake that came with a real DM or channel
        # notification must survive, so the agent unparks and reads it.
        if not await session.inbox.has_pending(agent.name):
            session.wakes.clear(agent.name)
        await session.inbox.append(agent.name, InboxItem(kind="meta", content=body))
        log.info(
            "%d no-op turns in the last %d by %s — force-parking (command=%r)",
            in_window, window_size, agent.name, command,
        )
        return True

    await session.inbox.append(
        agent.name,
        InboxItem(
            kind="meta",
            content=_noop_nudge_body(session, agent, command, in_window, threshold),
        ),
    )
    log.info(
        "fired no-op nudge #%d for %s (command=%r)", in_window, agent.name, command
    )
    return False


def _has_pending_input(session: Session, name: str) -> bool:
    """An agent has pending input iff its transcript ends in a non-assistant message."""
    transcript = session.transcript(name)
    if not transcript:
        return False
    return transcript[-1].role != "assistant"


def _ceo_unreported_live_org_agents(session: Session) -> list[Agent]:
    # An agent gates the CEO when it's live, hasn't reported, AND isn't
    # currently parked in WaitForMessages. A parked agent can't race the CEO
    # on shared state (it's blocked waiting for input), so the CEO can act
    # on what the agent already published via channels/archive/tagged DMs
    # without waiting for an explicit ReportToCEO call.
    reported = set(session.state.reported_agent_names)
    return [
        a for a in session.all_agents()
        if a.kind == "org"
        and a.state != "fired"
        and a.name not in reported
        and not session.wakes.is_waiting(a.name)
    ]


def _agent_holds_tool(agent: Agent, tool_name: str) -> bool:
    """True if `tool_name` survives this agent's allow / disallow lists."""
    if agent.tool_allowlist is not None and tool_name not in agent.tool_allowlist:
        return False
    if agent.tool_disallowlist and tool_name in agent.tool_disallowlist:
        return False
    return True


def _peers_holding_tool(peers: list[Agent], tool_name: str) -> list[Agent]:
    """Subset of `peers` whose effective tool set includes `tool_name`."""
    return [a for a in peers if _agent_holds_tool(a, tool_name)]


def _should_park(session: Session, agent: Agent) -> bool:
    """True → park indefinitely; False → exit the loop (§4.5.1).

    - CEO with live org agents: park (waiting for reports/notifications).
    - Solo CEO env-bound: exit (the env-bound CEO must submit to advance).
    - Solo CEO no env: exit on quiescence (end_on_idle=True, the RL default),
      or — for interactive sessions (end_on_idle=False) — park so that later
      external input can re-engage it; ends only on explicit close/abort.
    - Org agent: always park (peers/CEO can DM/notify at any time).
    """
    if agent.state == "fired":
        return False
    if agent.kind == "ceo":
        # §6.5: never park the CEO after env-terminal — the episode is over
        # and the harness is firing the org; nothing should wake it.
        if session.terminal_event is not None:
            return False
        # Interactive sessions keep the CEO parked on idle so external input
        # (send_user_input → wakes.wake) can re-engage it. RL/rollout sessions
        # leave end_on_idle=True and fall through to the quiescence logic.
        if not session.config.end_on_idle:
            return True
        return _ceo_has_live_org_agents(session)
    return True


async def _park_with_quiescence_guard(session: Session, agent: Agent) -> None:
    """Park `agent` under SPEC §4.5.1 quiescence invariant.

    - CEO short-circuit (checkpoint 1): if every live org agent is already
      parked, signal quiescence to the CEO (meta inbox item + wake) and
      return without parking. The next loop iteration drains the marker
      as a turn_user_message and re-engages the model.
    - Org-agent trigger (checkpoint 2): on parking, if every live agent
      (incl. the CEO) is now in the waiting set, signal quiescence to
      the CEO. The triggering org agent itself stays parked.

    Both branches route through `session.signal_quiescence_to_ceo()` —
    the centralised helper keeps the wake / meta-item / event emission
    in one place across this path and the `WaitForMessagesTool` path.
    """
    name = agent.name
    if agent.kind == "ceo":
        if session.is_org_quiescent() and session.live_org_agents():
            await session.signal_quiescence_to_ceo(source="ceo-entry")
            return
        await session.wakes.park_and_wait(name)
        return

    # Org agent: park with the checkpoint-2 trigger. `on_parked` runs
    # synchronously under WakeBroker._lock with `name` already added to
    # the waiting set, so the issubset check is race-free.
    ceo_name = session.ceo_name

    def _on_parked() -> None:
        live = {a.name for a in session.live_agents()}
        if ceo_name not in live:
            return
        if live.issubset(session.wakes.waiting_agents()):
            asyncio.create_task(
                session.signal_quiescence_to_ceo(source="org-trigger")
            )

    await session.wakes.park_and_wait(name, on_parked=_on_parked)


async def _persist_turn_user(session: Session, agent: Agent, body: str) -> None:
    await session.append_message(
        agent.name, Message(role="user", source="env", content=body)
    )


async def _stream_turn(
    session: Session, agent: Agent
) -> tuple[list[ToolCall], Message]:
    waiting_for_org = (
        agent.kind == "ceo"
        and session.config.ceo_tool_restriction
        and len(_ceo_unreported_live_org_agents(session)) > 0
    )
    tool_defs = visible_tool_defs(session, agent, waiting_for_org_agents=waiting_for_org)

    # Per-attempt failure logger — synchronous, writes one JSONL line per
    # failed inference attempt (retried or terminal). The provider invokes
    # this from inside its retry loop; the loop's outer try/except still
    # writes the terminal record so a failed turn has both granular retry
    # history *and* a clean terminal entry.
    err_path = session.paths.inference_errors

    def _record_attempt(payload: dict) -> None:
        try:
            append_jsonl(
                err_path,
                {"name": agent.name, "session_id": session.session_id, **payload},
            )
        except Exception:  # noqa: BLE001 — logging must never break inference
            log.exception("failed to append inference-error attempt")

    # §4.5.1 + tool_choice combo: if the env-bound solo CEO just received a
    # last-chance nudge (streak > 0), force a tool call on this recovery
    # turn so the model can't whiff a second time. The streak is reset
    # immediately after a successful tool call, so subsequent normal turns
    # revert to the model's free choice.
    tool_choice = None
    if (
        agent.kind == "ceo"
        and session.state.environment is not None
        and session._env_tool_less_streak > 0
    ):
        tool_choice = "required"

    extra_headers: dict[str, str] = {"X-Agent-Id": agent.name}
    if session.config.rollout_id:
        extra_headers["X-Rollout-Id"] = session.config.rollout_id

    request = InferenceRequest(
        model=session.state.model,
        messages=messages_for_inference(session, agent),
        tools=tool_defs_to_provider_tools(tool_defs),
        parallel_tool_calls=session.config.parallel_tool_calls,
        temperature=session.config.temperature,
        top_p=session.config.top_p,
        max_tokens=session.config.max_tokens,
        max_completion_tokens=session.config.max_completion_tokens,
        reasoning_effort=session.config.reasoning_effort,  # type: ignore[arg-type]
        tool_choice=tool_choice,
        on_attempt_failure=_record_attempt,
        extra_headers=extra_headers,
    )
    # Persist a snapshot of the LLM request to messages.jsonl so the rollout is
    # replayable from disk. system_prompt, tool defs, and mode are all mutable
    # in-memory state that gets overwritten in agent.json — without this record,
    # the on-disk transcript can't tell you what the model actually saw on any
    # given turn. `_load_transcript` skips these by `type`, so they don't get
    # re-fed as a system message on the next call.
    await append_jsonl_async(
        session.transcript_path(agent.name),
        {
            "type": "inference-request",
            "ts": now_ms(),
            "agent": agent.name,
            "system_prompt": agent.system_prompt or "",
            "tools": [
                {
                    "name": d.name,
                    "description": d.description,
                    "input_schema": d.input_schema,
                }
                for d in tool_defs
            ],
            "model": session.state.model,
            "params": {
                "temperature": session.config.temperature,
                "top_p": session.config.top_p,
                "max_tokens": session.config.max_tokens,
                "max_completion_tokens": session.config.max_completion_tokens,
                "reasoning_effort": session.config.reasoning_effort,
                "parallel_tool_calls": session.config.parallel_tool_calls,
                "tool_choice": tool_choice,
            },
            "mode": session.state.mode,
            "waiting_for_org_agents": waiting_for_org,
        },
    )
    await session.events.emit(
        "inference.requested", {"name": agent.name, "model": session.state.model}, name=agent.name
    )

    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = None
    stop_reason = None
    response_id = None

    # Stream the inference. Per SPEC §4.5.1 the peer activity card carries
    # no inflight/streaming time fields any more — wedged-stream detection
    # is delegated to the provider's HTTP timeout (§8.2). The loop here
    # just collects chunks.
    # Track tool-call chunks whose provider-level JSON parse failed; we
    # can't record the ParseErrorEvent until we've seen the CompletedChunk
    # (which carries the provider_response_id — chunks arrive in order:
    # deltas, tool_calls, completed).
    parse_error_calls: list[tuple[str, str, str]] = []  # (id, name, raw_args)
    async for chunk in session.inference.stream(request):
        if isinstance(chunk, ContentDeltaChunk):
            content_chunks.append(chunk.text)
        elif isinstance(chunk, ReasoningDeltaChunk):
            reasoning_chunks.append(chunk.text)
        elif isinstance(chunk, ToolCallChunk):
            tool_calls.append(ToolCall(
                id=chunk.id, name=chunk.name, args=chunk.args,
                parse_error=chunk.parse_error,
            ))
            if chunk.parse_error:
                raw = ""
                if isinstance(chunk.args, dict):
                    raw = str(chunk.args.get("__raw_arguments__") or "")
                parse_error_calls.append((chunk.id, chunk.name, raw))
        elif isinstance(chunk, CompletedChunk):
            usage = chunk.usage
            stop_reason = chunk.stop_reason
            response_id = chunk.response_id
        await session.events.emit(
            "inference.chunk", _chunk_to_payload(chunk), name=agent.name
        )

    await session.events.emit(
        "inference.completed",
        {
            "name": agent.name,
            "usage": (usage.__dict__ if usage else {}),
            "stop_reason": stop_reason or "stop",
        },
        name=agent.name,
    )

    for _id, _name, _raw in parse_error_calls:
        session.record_parse_error(
            agent_name=agent.name,
            provider_response_id=response_id,
            tool_call_id=_id,
            tool_name=_name,
            code="malformed-json",
            message=f"provider could not parse tool-call arguments as JSON: {_raw[:200]}",
        )

    assistant = Message(
        role="assistant",
        content="".join(content_chunks) if content_chunks else None,
        reasoning="".join(reasoning_chunks) if reasoning_chunks else None,
        tool_calls=tool_calls or None,
        usage=usage,
        stop_reason=stop_reason,
        provider_response_id=response_id,
    )
    return tool_calls, assistant


def _chunk_to_payload(chunk) -> dict:
    return {
        "kind": chunk.kind,
        **{k: v for k, v in chunk.__dict__.items() if k != "kind"},
    }


# ---------------------------------------------------------------------------
# Tool dispatch (§4.2)
# ---------------------------------------------------------------------------


def _record_tool_outcome_error(
    session: Session,
    agent: Agent,
    call: ToolCall,
    result: ToolResult,
    turn_id: Optional[str],
) -> None:
    """Record a ParseErrorEvent (invalid-args), a HarnessRejectionEvent
    (``error.kind == "rejection"``: a call the harness declined to run, so
    nothing executed), an InfraFailureEvent (``error.kind == "infra"``: the
    harness or env failed to carry the call out) or a ToolErrorEvent
    (everything else with ok=False) so training can attribute each to the
    emitting turn's tokens and penalise the categories it chooses to (§12.1).

    Skipped when the provider flagged the call as unparseable
    (``ToolCall.parse_error``, carried from ``ToolCallChunk.parse_error``) —
    that path already recorded a `malformed-json` parse error in
    `_stream_turn`, and its downstream dispatch failure would double-record
    for the same tool_call_id. Keyed on the provider-set flag, NOT the
    ``__parse_error__`` args sentinel: the sentinel is model-forgeable (a
    model emitting literal ``"__parse_error__": true`` inside valid-JSON args
    could suppress recording of its genuine failures).
    """
    if result.ok:
        return
    if call.parse_error:
        return
    code = result.error.code if result.error else "unknown-error"
    message = result.error.message if result.error else ""
    if code == "invalid-args":
        session.record_parse_error(
            agent_name=agent.name,
            provider_response_id=turn_id,
            tool_call_id=call.id,
            tool_name=call.name,
            code="invalid-args",
            message=message,
        )
    elif result.error is not None and result.error.kind == "rejection":
        session.record_harness_rejection(
            agent_name=agent.name,
            provider_response_id=turn_id,
            tool_call_id=call.id,
            tool_name=call.name,
            code=code,
            message=message,
        )
    elif result.error is not None and result.error.kind == "infra":
        session.record_infra_failure(
            agent_name=agent.name,
            provider_response_id=turn_id,
            tool_call_id=call.id,
            tool_name=call.name,
            code=code,
            message=message,
        )
    else:
        session.record_tool_error(
            agent_name=agent.name,
            provider_response_id=turn_id,
            tool_call_id=call.id,
            tool_name=call.name,
            code=code,
            message=message,
        )


def _canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args or {}, sort_keys=True, default=str, ensure_ascii=False)


def _result_content_digest(result: ToolResult) -> str:
    """§A.8.6 content digest: what the call *returned*, minus per-call
    bookkeeping (`_DIGEST_IGNORED_KEYS`). A dict output as JSON without those
    keys; a non-dict output as JSON; a failure as its error code and message.

    An env payload's `blocks` are digested whole rather than as their joined
    text: an image block carries no text, so text-only digesting made every
    screenshot identical and locked a screenshotting tool out for good.
    """
    if result.ok:
        out = result.output
        if isinstance(out, dict):
            text = json.dumps(
                {k: v for k, v in out.items() if k not in _DIGEST_IGNORED_KEYS},
                sort_keys=True, default=str, ensure_ascii=False,
            )
        else:
            text = json.dumps(out, sort_keys=True, default=str, ensure_ascii=False)
    else:
        err = result.error
        text = f"{err.code}\n{err.message}" if err else ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _record_call_for_repeat_rule(session: Session, agent: Agent, call: ToolCall, result: ToolResult) -> None:
    """Fold a completed call into the agent's §A.8.6 run.

    A refusal extends the run it belongs to instead of re-keying it: its
    digest is of the error, not of the output the run was keyed on, so keying
    it would reset the run and the next identical call would execute.

    Every OTHER rejection neither grows a run nor becomes its key. Nothing
    executed, so the call is no evidence that repeating it is futile. Keying
    it would let a transient gate take a tool away for good: fifteen identical
    `wait-gate-restricted` results would build a run, and then the first call
    after the gate lifts — the one that would finally have run — is refused
    instead, and stays refused, because the refusal extends that run.

    It does still CLEAR a run counting a different call. An agent that tried
    something else, even unsuccessfully, is no longer repeating what the run
    was on; leaving the run standing would keep the earlier tool refused
    across arbitrarily many intervening rejections, and each refusal would
    extend it further. Clearing is not keying: the rejected call does not take
    the slot, so no run is ever built out of refusals.
    """
    if result.error is not None and result.error.kind == "rejection":
        if result.error.code == "repeated-call":
            session.extend_repeat_run(agent.name)
        else:
            session.clear_repeat_run_unless(
                agent.name, call.name, _canonical_args(call.args)
            )
        return
    session.record_call_key(
        agent.name,
        (call.name, _canonical_args(call.args), _result_content_digest(result)),
    )


async def _refuse_repeated_call(
    session: Session, agent: Agent, call: ToolCall, run: int, threshold: int
) -> ToolResult:
    """§A.8.6 refusal: a rejection (nothing runs) carrying the harness
    shaping penalty on its payload, stamped here so both persisted copies
    (§12.4) agree. The message offers no exit: not firing, not submission,
    not `finish` (§4.5.2, §A.8.5 rationale)."""
    result = rejected_result(
        "repeated-call",
        f"Not run. This is identical to your last {run} calls, and every one of "
        "them returned the same output. Repeating it cannot produce anything "
        "new. Do something different: change the call, act on the output you "
        "already have, or — if you are waiting on something outside your "
        "control — park with WaitForMessages instead of polling.",
    )
    penalty = session.repeat_penalty
    if penalty != 0.0:
        result.output = {"shaping_reward": penalty, "shaping_reason": "repeated-call"}
    await session.events.emit(
        "agent.repeat_refused",
        {
            "name": agent.name,
            "kind": agent.kind,
            "tool": call.name,
            "run": run,
            "threshold": threshold,
            "penalty": penalty,
        },
        name=agent.name,
    )
    return result


def _stamp_noop_penalty(session: Session, result: ToolResult) -> None:
    """Write the §A.8.5 penalty onto a no-op shell call's result payload.

    Under `shaping_reward`, NOT `reward`. `reward` means "an env graded this
    step" to everything downstream — `trajectory.build_trajectory` and
    downstream consumers that sum `reward` off `tool.completed`. A `-0.05`
    there would be indistinguishable from a real env grade. A separate key
    keeps the harness's opinion out of the environment's, and
    `session.record_reward` stays uninvolved either way (see its docstring).
    The stamp is shown to the model in the tool result (as in training) but
    is never added to the reported reward; it was not part of
    Avalon-M1-Mini's training reward.

    Called from `_execute_tool` rather than from the dispatch loop, so the
    stamp lands before `tool.completed` is emitted. §12.4 requires the two
    persisted copies of a ToolResult (events.jsonl and messages.jsonl) to
    match; stamping after the emit left them disagreeing.

    Only a successful dict payload is touched — the env tool's
    `{blocks, metadata, reward, finished}` and host `Bash`'s
    `{stdout, stderr, exit_code}` both qualify. No guard against an existing
    `reward` or a `finished` payload is needed any more: the keys no longer
    collide.
    """
    if not result.ok or not isinstance(result.output, dict):
        return
    result.output["shaping_reward"] = float(session.config.noop_penalty)
    result.output["shaping_reason"] = "noop-termination-signal"


async def _dispatch_tools(
    session: Session,
    agent: Agent,
    assistant_msg: Message,
    tool_calls: list[ToolCall],
    *,
    noop_turn: bool = False,
) -> None:
    """Partition into read-only (parallel) and state-mutating (sequential) (§4.2).

    `noop_turn` is the caller's §A.8.5 verdict on the whole turn (every call a
    no-op shell call). It is passed in rather than recomputed so the penalty
    and the nudge can never disagree about what happened.

    A call is penalised when the turn was a no-op turn, or — whatever else the
    turn did — when the call could never have accomplished anything at all
    (`exit`, `true`, `:`). Pairing one of those with a real call is the obvious
    way to slip a termination signal past a whole-turn gate, and there is no
    legitimate use to protect.
    """
    read_only: list[ToolCall] = []
    state_mutating: list[ToolCall] = []
    for call in tool_calls:
        # Use case-insensitive resolve so the partition decision is correct
        # even when the model emits the wrong case (e.g. `Bash` for env `bash`).
        # `_execute_tool` will canonicalise the name when it runs the call.
        resolved = session.tools.resolve(call.name)
        tool = resolved[1] if resolved is not None else None
        if tool is not None and tool.definition.read_only:
            read_only.append(call)
        else:
            state_mutating.append(call)

    turn_id = assistant_msg.provider_response_id

    # Parallel read-only execution.
    if read_only:
        async def _run_one(c):
            return c, await _execute_tool(
                session, agent, c, turn_id=turn_id,
                noop=(noop_turn and is_noop_call(c)) or is_always_noop_call(c),
            )

        results = await asyncio.gather(*[_run_one(c) for c in read_only])
        for call, result in results:
            await _persist_tool_result(session, agent, call, result)
            _record_tool_outcome_error(session, agent, call, result, turn_id)
            _update_env_tool_breaker(session, call, result)
            await _apply_env_terminal_if_needed(session, agent, call, result)

    for call in state_mutating:
        result = await _execute_tool(
            session, agent, call, turn_id=turn_id,
            noop=(noop_turn and is_noop_call(call)) or is_always_noop_call(call),
        )
        await _persist_tool_result(session, agent, call, result)
        _record_tool_outcome_error(session, agent, call, result, turn_id)
        _update_env_tool_breaker(session, call, result)
        await _apply_env_terminal_if_needed(session, agent, call, result)


async def _apply_env_terminal_if_needed(
    session: Session,
    agent: Agent,
    call: ToolCall,
    result: ToolResult,
) -> None:
    """Apply env-tool terminal semantics AFTER the tool-result message is
    persisted (§12.3).

    The env tool's payload (`{blocks, metadata, reward, finished}`) carries
    `finished=True` when the env signals the rollout is done. We deferred
    `mark_terminal` out of the env tool's `execute` because
    `session.stop()` would race with `_persist_tool_result`: the supervisor
    could cancel the CEO before the tool-result `Message` was flushed to
    messages.jsonl and before message observers fired.
    Running here guarantees the trajectory is durable first.
    """
    tool = session.tools.get(call.name)
    if tool is None or tool.definition.scope != "env":
        return
    if not result.ok or not isinstance(result.output, dict):
        return
    if not result.output.get("finished"):
        session._env_tools_seen_unfinished.add(call.name)
        return
    reward = result.output.get("reward")
    terminal_payload = {
        "reward": reward,
        "tool": call.name,
        "by": agent.name,
        "metadata": result.output.get("metadata"),
    }
    await session.events.emit(
        "environment.terminal", terminal_payload, name=agent.name
    )
    await _settle_env_terminal(session, agent, terminal_payload, reward)


async def _settle_env_terminal(
    session: Session,
    agent: Agent,
    terminal_payload: dict[str, Any],
    reward: Optional[float],
) -> None:
    """Shared §6.5 tail for every env-terminal path (`_apply_env_terminal_if_
    needed` and `_dispatch_terminal_tool`): the episode is over, so the org
    dies with it.

    Lock in the terminal event and reward first, so any env call a peer makes
    while the org is being torn down is refused (`_dispatch_tool`). Then fire
    every org agent still on the roster (`source="harness"`,
    `reason="env-terminal"`, no exit review) and stop the session. The stop
    comes last: it makes the supervisor cancel every agent task, including
    the one running this, so the fires must be persisted before it.
    """
    session.mark_terminal(terminal_payload, reward=reward, stop=False)
    try:
        for a in session.live_org_agents():
            await fire_org_agent(session, a.name, source="harness", reason="env-terminal")
    finally:
        # A failed fire must not leave the session running past its episode.
        session.stop()


def _update_env_tool_breaker(
    session: Session, call: ToolCall, result: ToolResult
) -> None:
    """Track consecutive identical env-tool failures; abort if limit hit.

    The signature is `<tool_name>:<error_code or 'exception'>:<message[:200]>`
    — coarse enough that genuinely the same failure mode clusters, fine
    enough that legitimately different errors reset the streak.

    An adapter signals an unrecoverable env-session death (the backing
    process/pod died, or its client gave up reaching it) with the generic
    `ENV_SESSION_DEAD_CODE`. That short-circuits the streak and aborts on the
    FIRST occurrence: there is no env left to call, so the
    `ENV_TOOL_CIRCUIT_BREAKER_LIMIT` retries the streak would otherwise allow
    are guaranteed to fail the same way — just more flailing in the transcript.
    """
    tool = session.tools.get(call.name)
    if tool is None or tool.definition.scope != "env":
        return
    if result.ok:
        # Any env-tool success clears the streak — model has made progress.
        session._env_tool_failure_streak = 0
        session._env_tool_failure_signature = None
        return
    err = result.error
    code = err.code if err else "unknown"
    message = err.message if err else ""
    if code == ENV_SESSION_DEAD_CODE:
        log.warning(
            "env session dead on %s — aborting session: %s",
            call.name,
            message[:200],
        )
        session.abort(f"env-session-dead:{call.name}")
        return
    if (err is not None and err.kind == "rejection") or code in PRE_EXECUTION_ERROR_CODES:
        # Harness policy or a malformed call, not env health. Leave the streak
        # untouched: this call neither reached the env (so it is not a failure
        # to count) nor proved the env works (so it is not a success to reset
        # on). Driven by `kind` so a rejection code added later is covered
        # without being listed anywhere; the code set carries only the two
        # pre-execution codes `kind` cannot identify.
        return
    signature = f"{call.name}:{code}:{message[:200]}"
    if signature == session._env_tool_failure_signature:
        session._env_tool_failure_streak += 1
    else:
        session._env_tool_failure_signature = signature
        session._env_tool_failure_streak = 1
    if session._env_tool_failure_streak >= ENV_TOOL_CIRCUIT_BREAKER_LIMIT:
        log.warning(
            "env-tool circuit breaker tripped (%d× '%s') — aborting session",
            session._env_tool_failure_streak,
            signature,
        )
        # Include the error code so abort_reason consumers can tell a
        # policy-flavored trip (e.g. plan-mode-restricted spam) from an
        # env-flavored one (persistent env-failure) post hoc.
        session.abort(f"env-tool-circuit-breaker:{call.name}:{code}")


async def _execute_tool(
    session: Session,
    agent: Agent,
    call: ToolCall,
    turn_id: Optional[str] = None,
    *,
    noop: bool = False,
) -> ToolResult:
    """Emit `tool.requested`, dispatch the call, emit `tool.completed`.

    Exactly one of each per call, whatever the outcome. The rejections
    `_dispatch_tool` returns before the tool ever runs — scoping, schema,
    plan mode, wait gate, permission — used to emit neither event, so they
    existed only in messages.jsonl and any error count taken from
    events.jsonl read zero for every gated or denied call.
    """
    # Case-insensitive resolution — rescues `Bash` → `bash` (env-side) and
    # symmetrically the other direction. Rewrite `call.name` to the canonical
    # registered spelling so downstream allow/disallow checks, event logs,
    # and the assistant transcript all reference the same name. Without this
    # the model wastes ~2 turns rediscovering the right case every time.
    resolved = session.tools.resolve(call.name)
    if resolved is not None:
        canonical_name = resolved[0]
        if canonical_name != call.name:
            log.info(
                "tool name case-rescued: %r -> %r (agent=%s)",
                call.name, canonical_name, agent.name,
            )
            call.name = canonical_name
    await session.events.emit(
        "tool.requested",
        {"name": agent.name, "tool": call.name, "args": call.args, "call_id": call.id},
        name=agent.name,
    )
    result = await _dispatch_tool(
        session, agent, call, resolved, turn_id=turn_id, noop=noop
    )
    # §A.8.6 — every completed call, refusals included, feeds the run the
    # next dispatch reads. Before `tool.completed` so nothing observes a
    # result the run has not seen.
    _record_call_for_repeat_rule(session, agent, call, result)
    await session.events.emit(
        "tool.completed",
        {"call_id": call.id, "ok": result.ok, "output": result.output, "error": result.error.model_dump() if result.error else None},
        name=agent.name,
    )
    return result


async def _dispatch_tool(
    session: Session,
    agent: Agent,
    call: ToolCall,
    resolved: tuple[str, ToolImpl] | None,
    turn_id: str | None = None,
    *,
    noop: bool = False,
) -> ToolResult:
    """Resolve permission and run the call. Every early return here refuses
    the call before the env sees it: a `kind="rejection"` result, or one of
    `PRE_EXECUTION_ERROR_CODES`.
    """
    if resolved is None:
        return ToolResult(
            ok=False,
            error=ToolResultError(code="unknown-tool", message=f"No such tool: {call.name}"),
        )
    tool = resolved[1]

    # §6.5 — the env already returned finished=True: the episode, and its
    # reward, are settled. No env tool runs again, whoever calls it — a second
    # `submit` would otherwise re-grade and overwrite the locked-in reward.
    if tool.definition.scope == "env" and session.terminal_event is not None:
        return ToolResult(
            ok=False,
            error=ToolResultError(
                code="episode-finished",
                kind="rejection",
                message=(
                    f"{call.name} is an environment tool and the episode has "
                    "already finished; the reward is locked in. Environment "
                    "tools are no longer available."
                ),
            ),
        )

    # Allow/disallow scoping (§5.2 step 3-4). Case-insensitive so a CEO-typed
    # mis-cased name in the agents-org JSON (e.g. `"disallowed_tools": ["Bash"]`
    # when only env-side `bash` exists) still gates the right tool.
    disallow_lower = {d.lower() for d in (agent.tool_disallowlist or [])}
    allow_lower = (
        {a.lower() for a in agent.tool_allowlist}
        if agent.tool_allowlist is not None
        else None
    )
    if call.name.lower() in disallow_lower:
        return ToolResult(
            ok=False,
            error=ToolResultError(
                code="not-permitted",
                kind="rejection",
                message=f"{call.name} is on the agent's disallow list.",
            ),
        )
    if allow_lower is not None and call.name.lower() not in allow_lower:
        return ToolResult(
            ok=False,
            error=ToolResultError(
                code="not-permitted",
                kind="rejection",
                message=f"{call.name} is not in the agent's allow list.",
            ),
        )

    # §5.1 — validate args against the tool's input_schema before any
    # permission decision or execute call. Sit between scoping (above) and
    # plan-mode / wait-gate / permission policy (below) so disallowed
    # tools don't masquerade as schema errors and vice versa.
    schema_err = _validate_tool_args(tool.definition.input_schema, call.args)
    if schema_err is not None:
        return ToolResult(
            ok=False,
            error=ToolResultError(code="invalid-args", message=schema_err),
        )

    # Plan-mode restriction (§5.3 "plan" policy): deny state-mutating tools
    # except those needed to write the plan and exit the mode.
    #
    # Permissive org-plan (§A.3.1 org-plan mode): in ATTENDED sessions with
    # budget > 0 the plan-mode tool restriction is lifted, so the CEO may act
    # directly (submit / work solo) OR design an org instead of being force-
    # locked into WritePlan. Attended = env-bound OR interactive
    # (end_on_idle=False, no env bound). The prompt + recurring solo-thrash
    # nudge keep the org-design path salient. Lifting the restriction is also
    # what lets an envless CEO's host tools reach the host-ask permission flow —
    # they'd otherwise be denied here, before the policy runs.
    plan_mode_permissive = (
        (
            session.state.environment is not None
            or not session.config.end_on_idle
        )
        and session.state.budget.max_org_agents > 0
    )
    if (
        session.state.mode == "org-plan"
        and not tool.definition.read_only
        and agent.kind == "ceo"
        and not plan_mode_permissive
        and call.name not in {"ExitOrgPlanMode", "WritePlan", "EnterOrgPlanMode"}
    ):
        return ToolResult(
            ok=False,
            error=ToolResultError(
                code="plan-mode-restricted",
                kind="rejection",
                message=(
                    f"{call.name} is denied in org-plan mode. "
                    "Use WritePlan to author the plan file, then call ExitOrgPlanMode."
                ),
            ),
        )

    # CEO tool restriction while waiting for unreported org agents (§4.5.2).
    # Per-tool race rule: a tool gates the CEO only if some unreported,
    # non-parked peer also holds it in their effective tool set. If the CEO
    # is the sole holder (the common case for env tools the CEO reserved at
    # hire time via `disallowed_tools`), there's nothing to race on.
    # Off entirely with `config.ceo_tool_restriction = False`.
    if agent.kind == "ceo" and session.config.ceo_tool_restriction:
        unreported = _ceo_unreported_live_org_agents(session)
        if unreported:
            readonly_names = {d.name for d in session.tools.definitions() if d.read_only}
            always_allowed = WAIT_GATE_TOOLS | readonly_names
            if call.name not in always_allowed:
                contenders = _peers_holding_tool(unreported, call.name)
                if contenders:
                    held_by = ", ".join(a.name for a in contenders)
                    # Only the CEO is gated. Several hires may hold the tool
                    # at once and are expected to coordinate its use among
                    # themselves; the gate keeps the CEO coordinating them
                    # rather than acting on the same state alongside them.
                    # UpdateOrgAgentTools takes ONE agent per call, so the
                    # override has to be spelled out per holder — a CEO told
                    # to act "on them" sends one call, reads a success, and
                    # hits the gate again on the next try.
                    # The cost comes before the recipe: the message is read
                    # as instructions, and revoking is not a loan to the CEO —
                    # it takes the tool off the holders and stalls their work.
                    if len(contenders) == 1:
                        holders = (
                            f"{call.name} is held by {held_by}, who is still working. "
                            "You coordinate them rather than act on the same state "
                            "alongside them. It returns to you when they report or "
                            "park; until then, call WaitForMessages and read what "
                            "they post."
                        )
                        override = (
                            f"Revoking it is not a loan: it takes {call.name} away "
                            f"from {held_by} until you restore it, and any of their "
                            f"work that needs {call.name} stops. Revoke only if you "
                            f"no longer want them using it, not to make a call "
                            f"yourself. To revoke, call UpdateOrgAgentTools on "
                            f"{held_by} with their full disallowed_tools list "
                            f"plus {call.name} — it replaces theirs wholesale."
                        )
                    else:
                        holders = (
                            f"{call.name} is held by {held_by}, who are still working. "
                            "They coordinate its use among themselves; you coordinate "
                            "them rather than act on the same state alongside them. It "
                            "returns to you as each of them reports or parks; until "
                            "then, call WaitForMessages and read what they post."
                        )
                        override = (
                            f"Revoking it is not a loan: it takes {call.name} away "
                            "from each agent you revoke it from until you restore it, "
                            f"and any of their work that needs {call.name} stops. "
                            "Revoke only from agents you no longer want using it, "
                            "not to make a call yourself. To revoke, call "
                            "UpdateOrgAgentTools once per holder — one call per "
                            "agent, carrying that agent's full disallowed_tools list "
                            f"plus {call.name}. Each call replaces that agent's list "
                            "wholesale."
                        )
                    return ToolResult(
                        ok=False,
                        error=ToolResultError(
                            code="wait-gate-restricted",
                            kind="rejection",
                            message=f"{holders} {override}",
                        ),
                    )

    # §A.8.6 — refuse a call identical to the agent's current run. After the
    # policy gates above (a gated call is refused on its own grounds first)
    # and before the permission flow (never ask a human about a call that
    # will not run).
    threshold = int(session.config.repeat_refuse_after)
    if threshold > 0 and tool.definition.name not in REPEAT_EXEMPT_TOOLS:
        run = session.repeat_run(agent.name, call.name, _canonical_args(call.args))
        if run >= threshold:
            return await _refuse_repeated_call(session, agent, call, run, threshold)

    # Permission policy.
    policy = get_policy(session.state.permission_policy)
    ctx = ToolContext(
        agent=agent,
        session=session,
        args=call.args,
        call_id=call.id,
        turn_response_id=turn_id,
    )
    decision = policy.decide(call.name, ctx)
    if decision == "deny":
        return ToolResult(
            ok=False,
            error=ToolResultError(
                code="permission-denied",
                kind="rejection",
                message=f"Policy denied {call.name}",
            ),
        )
    if decision == "ask":
        await session.events.emit(
            "tool.permission.asked",
            {"call_id": call.id, "tool": call.name, "args": call.args},
        )
        if not session.config.end_on_idle:
            # Interactive session: a human resolver is attached (the invariant is
            # interactive ⟺ end_on_idle False ⟺ envless live session). Suspend
            # this call until decide_permission resolves it — an UNBOUNDED wait:
            # the prompt waits for the human, like Claude Code. Liveness on
            # disconnect is the embedding host's job: cancelling this task
            # aborts the wait, and the try/finally then guarantees the pending
            # future is never leaked.
            fut = session.permissions.register(call.id)
            try:
                decision = await fut
            finally:
                session.permissions.discard(call.id)
            await session.events.emit(
                "tool.permission.decided", {"call_id": call.id, "decision": decision}
            )
            if decision == "deny":
                return ToolResult(
                    ok=False,
                    error=ToolResultError(
                        code="permission-denied",
                        kind="rejection",
                        message=f"Permission denied for {call.name}",
                    ),
                )
        elif session.headless:
            # Headless rollout, no human resolver (the RL/env path): auto-allow.
            decision = "allow"
            await session.events.emit("tool.permission.decided", {"call_id": call.id, "decision": "allow"})
        else:
            return ToolResult(
                ok=False,
                error=ToolResultError(
                    code="permission-required",
                    kind="rejection",
                    message="Permission required; no interactive resolver attached.",
                ),
            )

    await session.events.emit("tool.started", {"call_id": call.id}, name=agent.name)
    try:
        result = await tool.execute(ctx)
    except Exception as e:
        log.exception("tool %s failed", call.name)
        # The tool implementation broke, not the call: an infra failure
        # (§12.1), never a tool error the model is penalised for.
        result = ToolResult(
            ok=False,
            error=ToolResultError(
                code="exception",
                kind="infra",
                message=f"{e.__class__.__name__}: {e}",
            ),
        )
    # §A.8.5 — stamp the no-op penalty BEFORE `tool.completed` is emitted, so
    # the event log and messages.jsonl carry the same ToolResult verbatim
    # (§12.4 replay). Stamping after the emit would leave the two persisted
    # records of one result disagreeing about its reward. `_execute_tool`
    # emits it once this returns.
    if noop:
        _stamp_noop_penalty(session, result)
    return result


async def _persist_tool_result(
    session: Session, agent: Agent, call: ToolCall, result: ToolResult
) -> None:
    msg = Message(
        role="tool",
        tool_call_id=call.id,
        tool_result=result,
    )
    await session.append_message(agent.name, msg)


# ---- terminal-tool grading (openreward @terminal envs) --------------------
#
# Envs that declare a @terminal tool hide it from list_tools() so the model
# never chooses it. The CEO's plain final message IS the answer; the harness
# routes it via session.env_adapter.call_terminal_tool(text) to get graded,
# and records the returned reward (session.record_reward) + mark_terminal(...).
#
# Trace shape: mutate the pre-persist assistant message to carry a synthetic
# ToolCall pointing at the terminal tool, then persist a ToolResult with the
# grader's output. This is what puts the reward badge on a ToolResult event in
# the openreward UI (which never displays it on AssistantMessage events).


def _is_terminal_grading_candidate(session: Session, agent: Agent) -> bool:
    """Cheap sync gate: is this the right moment to consider terminal-tool
    grading? The env-adapter probe (_prepare_terminal_call) is async, so we
    skip it entirely when the surrounding conditions rule it out."""
    if agent.kind != "ceo":
        return False
    if session.terminal_event is not None:
        return False
    if session.state.environment is None:
        return False
    return session.env_adapter is not None


async def _prepare_terminal_call(session: Session, assistant_msg: Message) -> Optional[ToolCall]:
    """Probe the env for a @terminal tool spec (cached per session on
    session._terminal_tool_probed) and build a synthetic ToolCall for it.
    Returns None when the env has no terminal tool or the SDK is too old to
    expose one — in which case the caller falls back to the normal
    tool-less turn path (nudge, then terminate)."""
    spec = session._terminal_tool_probed
    if spec is ...:
        adapter = session.env_adapter
        probe = getattr(adapter, "terminal_tool", None) if adapter is not None else None
        if probe is None:
            spec = None
        else:
            try:
                spec = await probe()
            except Exception as e:  # noqa: BLE001 — probe failure shouldn't kill rollout
                log.warning("terminal-tool probe failed: %s", e)
                spec = None
        session._terminal_tool_probed = spec
    if spec is None:
        return None

    # Strip Qwen chain-of-thought — the trace already emits reasoning as a
    # separate ReasoningItem via convert_trace_to_openreward, and simple
    # graders would misgrade on the reasoning content. Fall back to the raw
    # content if stripping empties the string (Qwen's known failure mode of
    # placing the answer inside <think> without closing the tag).
    raw = assistant_msg.content or ""
    stripped = _THINK_RE.sub("", raw).strip() if isinstance(raw, str) else ""
    answer_text = stripped or (raw if isinstance(raw, str) else "")
    args = {} if spec.arg is None else {spec.arg: answer_text}
    return ToolCall(
        id=f"call_terminal_{uuid.uuid4().hex[:16]}",
        name=spec.name,
        args=args,
        # Harness-injected, not model output — training-data builders strip
        # this call (and its paired ToolResult) so the training target stays
        # the genuine tool-less answer.
        synthetic=True,
    )


async def _dispatch_terminal_tool(
    session: Session,
    agent: Agent,
    call: ToolCall,
    *,
    response_id: Optional[str] = None,
) -> None:
    """Grade the CEO's answer via session.env_adapter.call_terminal_tool,
    persist a matching ToolResult, and mark the session terminal.

    ``response_id`` is the provider completion id of the assistant turn whose
    final message is being graded — it keys the terminal grade in
    `session.reward_events` to that turn's generated-token span.

    Kept out of `_dispatch_tools` because the terminal tool is HIDDEN from
    session.tools by design (hidden from the model), so `_execute_tool`
    can't resolve it. `_apply_env_terminal_if_needed` is likewise bypassed —
    it also gates on `session.tools.get(call.name)` being an env-scope tool,
    which by construction returns None here — but both paths share the same
    §6.5 settlement via `_settle_env_terminal`.
    """
    adapter = session.env_adapter
    # Args carry the stripped answer under the terminal tool's own arg name;
    # for a no-arg terminal tool the args dict is empty and call_terminal_tool
    # ignores the message.
    text = next(iter(call.args.values())) if call.args else ""
    try:
        output = await adapter.call_terminal_tool(text)
    except Exception as e:  # noqa: BLE001 — grading failure shouldn't crash the loop
        log.warning("terminal-tool grading failed: %s", e)
        result = ToolResult(
            ok=False,
            error=ToolResultError(
                code="terminal-tool-error",
                kind="infra",
                message=f"{e.__class__.__name__}: {e}",
            ),
        )
        await _persist_tool_result(session, agent, call, result)
        # Record the grading failure as an infra-failure event (code
        # "terminal-tool-error", §12.1): the grader broke, the answer was not
        # judged, so it is measurable per rollout but never penalised.
        _record_tool_outcome_error(session, agent, call, result, response_id)
        # Same §6.5 settlement as the success path, minus the reward: a failed
        # grade still ends the episode.
        await _settle_env_terminal(
            session,
            agent,
            {"kind": "terminal_tool", "tool": call.name,
             "by": agent.name, "error": str(e)},
            None,
        )
        return

    reward = output.reward
    if reward is not None:
        session.record_reward(
            float(reward),
            agent=agent.name,
            response_id=response_id,
            terminal=True,
        )

    blocks_out: list[dict[str, Any]] = []
    for b in getattr(output, "blocks", []) or []:
        btype = getattr(b, "type", None)
        if btype == "text":
            blocks_out.append({"type": "text", "text": b.text})
        else:
            blocks_out.append({"type": btype})
    payload = {
        "blocks": blocks_out,
        "metadata": getattr(output, "metadata", None),
        "reward": reward,
        "finished": True,
    }
    result = ToolResult(ok=True, output=payload)
    await _persist_tool_result(session, agent, call, result)

    terminal_payload = {
        "kind": "terminal_tool",
        "tool": call.name,
        "by": agent.name,
        "reward": reward,
        "metadata": getattr(output, "metadata", None),
    }
    await session.events.emit(
        "environment.terminal", terminal_payload, name=agent.name
    )
    await _settle_env_terminal(session, agent, terminal_payload, reward)
