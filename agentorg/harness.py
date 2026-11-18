"""Harness — the top-level entry point for spawning and running sessions.

Owns:
- registering built-in tools (org/local/env)
- starting the CEO loop + any org-agent loops
- driving rollouts to a terminal state and returning a `RolloutResult`
  (the flattened trajectory view is `agentorg.trajectory.build_trajectory`)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anyio

from agentorg.env_adapter import EnvAdapter
from agentorg.inference.base import InferenceProvider
from agentorg.inference.openai_compat import OpenAICompatProvider
from agentorg.loop import run_agent
from agentorg.metrics import compute_rollout_metrics
from agentorg.org import refresh_ceo_system_prompt, retire_crashed_org_agent
from agentorg.session import Session, SessionConfig
from agentorg.tools.fs_tools import all_local_tools
from agentorg.tools.org_tools import (
    CreateChannelsTool,
    EditArchiveFileTool,
    EnterOrgPlanModeTool,
    ExitOrgPlanModeTool,
    FireOrgAgentTool,
    HireOrgAgentTool,
    ListArchiveTool,
    ListChannelsTool,
    MoveArchiveFileTool,
    PostToChannelTool,
    ReactToMessageTool,
    ReportToCEOTool,
    SearchChannelMessagesTool,
    SearchMyHistoryTool,
    SendMessageTool,
    SetChannelSubscriptionTool,
    TodoWriteTool,
    UpdateOrgAgentToolsTool,
    ViewArchiveFileTool,
    ViewChannelMessagesTool,
    ViewMyHistoryTool,
    WaitForMessagesTool,
    WritePlanTool,
    WriteToArchiveTool,
)
from agentorg.tools.registry import ToolRegistry
from agentorg.types import (
    Agent,
    CEO_NAME,
    HarnessRejectionEvent,
    InboxItem,
    InfraFailureEvent,
    Message,
    now_ms,
    ParseErrorEvent,
    ToolErrorEvent,
)

log = logging.getLogger(__name__)

# Upper bound on waiting for cancelled agent tasks to finish unwinding before
# closing the event bus. They normally finish within one event-loop pass; the
# bound only matters if something swallows the cancel.
AGENT_UNWIND_TIMEOUT_S = 5.0


# Tools that only make sense with multiple agents — channels, peer messaging,
# and org management. Disabled entirely in the solo case (budget <= 0): a lone
# CEO has no peers to message, no channels to post to, and no budget to hire, so
# surfacing them is pure noise. The solo system prompt already strips every
# reference to these (see build_ceo_system_prompt's `solo` branch).
MULTI_AGENT_ONLY_TOOLS = frozenset({
    "SendMessage",
    "PostToChannel",
    "CreateChannels",
    "ListChannels",
    "ViewChannelMessages",
    "SearchChannelMessages",
    "SetChannelSubscription",
    "ReactToMessage",
    "WaitForMessages",
    "HireOrgAgent",
    "FireOrgAgent",
    "UpdateOrgAgentTools",
})


def org_level_tools(*, solo: bool = False) -> list:
    tools = [
        SendMessageTool(),
        PostToChannelTool(),
        CreateChannelsTool(),
        ListChannelsTool(),
        ViewChannelMessagesTool(),
        SearchChannelMessagesTool(),
        SetChannelSubscriptionTool(),
        ReactToMessageTool(),
        WaitForMessagesTool(),
        ReportToCEOTool(),
        ListArchiveTool(),
        ViewArchiveFileTool(),
        WriteToArchiveTool(),
        EditArchiveFileTool(),
        MoveArchiveFileTool(),
        EnterOrgPlanModeTool(),
        ExitOrgPlanModeTool(),
        WritePlanTool(),
        HireOrgAgentTool(),
        FireOrgAgentTool(),
        UpdateOrgAgentToolsTool(),
        TodoWriteTool(),
        # No AskUserQuestion: there is no one to ask. Headless mode answered it with the
        # agent's own first option, which agents then cited as operator approval, and
        # interactive mode has no resolver. AskUserQuestionTool stays for when one exists.
    ]
    if solo:
        tools = [t for t in tools if t.definition.name not in MULTI_AGENT_ONLY_TOOLS]
        # Solo replacements for #self recovery: read-only views over the agent's
        # own trace (full history, across compaction boundaries), named without
        # channel vocabulary so there's no channel confusion.
        tools += [ViewMyHistoryTool(), SearchMyHistoryTool()]
    return tools


@dataclass
class RolloutResult:
    session_id: str
    workspace_path: str
    terminal_event: Optional[dict[str, Any]]
    reward: Optional[float]
    ceo_name: str
    org_agents: list[Agent]
    duration_s: float
    used_turns: int
    cumulative_reward: float = 0.0
    reward_steps: int = 0
    # Full per-step env reward stream, in the order the env graded them.
    # cumulative_reward/reward_steps above are the sum/count; downstream
    # consumers may need the list to reduce it under an env-specific policy (e.g.
    # mean over non-zero steps).
    step_rewards: list[float] = field(default_factory=list)
    terminated_by_env: bool = False
    # Why the session ended: "env-terminal", the abort reason (e.g.
    # "max_wall_s", "max_total_tokens", "env-tool-circuit-breaker:…"), or
    # "complete" when the CEO ended it.
    end_reason: str = "complete"
    # Org-wide activity metrics (channel/archive token spend, reaction/hire/fire
    # /channel-creation counts, @mention tagging, role distribution). See
    # agentorg.metrics.compute_rollout_metrics.
    metrics: dict[str, Any] = field(default_factory=dict)
    # Turn-level error streams, for trainers that want per-turn signals (none
    # were part of Avalon-M1-Mini's training reward). Each event carries the
    # `provider_response_id` of the assistant turn that emitted the failing
    # call, so a trainer can attribute per-token penalties to the exact
    # generated-token span (given an inference server that maps response_id
    # to generated-token ranges). Split by cause so shapers can
    # apply distinct penalties: parse errors are turn-formatting issues,
    # tool errors are execution failures.
    parse_error_events: list[ParseErrorEvent] = field(default_factory=list)
    tool_error_events: list[ToolErrorEvent] = field(default_factory=list)
    # Calls the harness declined to run (error.kind ==
    # "rejection"). A third stream so trainers can leave these unpenalised
    # by default and price them separately when they choose to.
    harness_rejection_events: list[HarnessRejectionEvent] = field(default_factory=list)
    # Calls the harness or the env failed to carry out (error.kind ==
    # "infra"): a raising tool, a dead env session, a transport or grader
    # failure. Never penalised; exported so infra failure rates are
    # measurable per rollout.
    infra_failure_events: list[InfraFailureEvent] = field(default_factory=list)


class Harness:
    """High-level entry point: `start_session(...)`, then `run_to_completion(...)`."""

    def __init__(
        self,
        *,
        inference: Optional[InferenceProvider] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.inference: InferenceProvider = inference or OpenAICompatProvider(
            api_key=api_key, base_url=base_url
        )

    async def start_session(
        self,
        *,
        config: SessionConfig,
        env_adapter: Optional[EnvAdapter] = None,
        initial_inbox: Optional[list[InboxItem]] = None,
    ) -> Session:
        registry = ToolRegistry()
        registry.register_many(org_level_tools(solo=config.budget <= 0))
        # Local workspace tools — only when not env-bound. In env-bound
        # sessions the local fs tools are NOT registered, whatever the env's
        # toolset: they would run on the host, outside the env's sandbox, and
        # an env toolset that brings its own shell/files (e.g. OpenReward's
        # `claude-code`, all lowercase) would sit beside them under
        # case-distinct names. The env supplies whatever filesystem
        # the agent needs, and the plan-file workflow goes through
        # `WritePlan` (a dedicated org-level tool). Registering host-side
        # `Read`/`Write`/`Edit` in that mode used to coexist with the env's
        # own lowercase `read`/`write`/`edit` (no case-sensitive name
        # collision), confusing the model and silently writing the plan
        # into the env sandbox where ExitOrgPlanMode couldn't see it.
        if config.environment is None:
            for tool in all_local_tools():
                registry.register(tool)
        # An env tool may not replace a harness tool: one named e.g.
        # `SendMessage` or `WritePlan` would receive the org's messages or plan.
        if env_adapter is not None:
            env_tools = await env_adapter.build_tools()
            reserved = {n.lower() for n in registry.names()}
            for tool in env_tools:
                if tool.definition.name.lower() in reserved:
                    log.warning(
                        "env tool %r collides with a harness tool; not registered",
                        tool.definition.name,
                    )
                    continue
                registry.register(tool)

        session = Session(config=config, inference=self.inference, tools=registry)
        # Give the session a back-reference so the loop can reach adapter-level
        # capabilities not exposed as tools (openreward @terminal-tool grading
        # in particular — see loop._prepare_terminal_call / _dispatch_terminal_tool).
        if env_adapter is not None:
            session.set_env_adapter(env_adapter)
        await session.persist_state()
        await session.events.emit(
            "session.started",
            {
                "session_id": session.session_id,
                "model": session.state.model,
                "mode": session.state.mode,
                "budget": session.state.budget.model_dump(),
                "env": config.environment.model_dump() if config.environment else None,
            },
        )
        if config.environment is not None:
            await session.events.emit(
                "environment.bound", config.environment.model_dump()
            )

        # Register CEO agent. Its `name` is the reserved literal "CEO" (§6.6).
        ceo = Agent(
            name=CEO_NAME,
            session_id=session.session_id,
            kind="ceo",
            role="CEO",
            description="Root agent of the session.",
            work_dir=session.state.workspace_path,
            system_prompt="",  # filled below
        )
        await session.register_agent(ceo)
        await refresh_ceo_system_prompt(session)
        await session.events.emit(
            "agent.spawned", {"name": ceo.name, "kind": "ceo", "role": "CEO"}, name=ceo.name
        )

        # Seed CEO's initial user message — task prompt + env preamble.
        if config.initial_user_message:
            await session.append_message(
                ceo.name,
                Message(role="user", source="human", content=config.initial_user_message),
            )

        # If starting in org-plan mode, ensure the system prompt reflects it.
        if config.mode == "org-plan":
            await session.events.emit("plan.mode.entered", {"mode": "org-plan"})

        # Pre-seed inbox items (e.g. environment preamble).
        if initial_inbox:
            for item in initial_inbox:
                await session.inbox.append(ceo.name, item)

        return session

    async def run_to_completion(
        self,
        session: Session,
        *,
        max_wall_s: Optional[float] = None,
        max_total_tokens: Optional[int] = None,
        wall_clock: Optional[Callable[[], float]] = None,
    ) -> RolloutResult:
        """Drive the session until terminal, or until a host guard fires.

        `max_wall_s` caps wall-clock time; `max_total_tokens` caps cumulative
        total tokens (input + output across every agent's turns, each distinct
        token counted once). Either may be None to disable that guard.

        `wall_clock` replaces real time as the §4.6 session clock, which the
        `max_wall_s` deadline reads: seconds, never decreasing. A host that freezes
        the session's inference for stretches it should not be charged for
        (an RL trainer taking an optimizer step on the GPUs that serve it)
        passes a clock that stops during them. `RolloutResult.duration_s`
        stays real time.
        """
        if wall_clock is not None:
            session.clock = wall_clock
        start = now_ms() / 1000.0
        clock_start = session.clock()
        deadline = (clock_start + max_wall_s) if max_wall_s else None
        # §4.6 — the session clock's start and horizon (the host wall).
        session.clock_start_s = clock_start
        session.clock_total_s = max_wall_s or None

        # Launch one task per agent. New org agents register additional tasks
        # via the supervisor loop below.
        async def supervisor() -> None:
            running: dict[str, asyncio.Task] = {}
            exited: set[str] = set()  # agents whose loop returned at least once
            while True:
                # Check terminal / abort.
                if session.stop_requested():
                    break
                if deadline and session.clock() > deadline:
                    log.info("max wall time reached; aborting session")
                    session.abort("max_wall_s")
                    break
                if max_total_tokens and session.total_tokens > max_total_tokens:
                    log.info(
                        "max total tokens reached (%d > %d); aborting session",
                        session.total_tokens,
                        max_total_tokens,
                    )
                    session.abort("max_total_tokens")
                    break
                # Reap done tasks FIRST so we don't immediately restart them.
                done_names = [n for n, t in running.items() if t.done()]
                for name in done_names:
                    t = running.pop(name)
                    exited.add(name)
                    # A live org agent always parks, so its loop only exits
                    # on fire, stop, or env-terminal (which also cancels org
                    # tasks). Any other exit is a crash: an uncaught
                    # exception, or run_agent's return on a failed inference.
                    cause = "its model call failed"
                    try:
                        t.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        log.exception("agent task %s crashed", name)
                        cause = f"{e.__class__.__name__}: {str(e)[:200]}"
                    agent = session.agent(name)
                    if (
                        agent is not None
                        and agent.kind == "org"
                        and agent.state != "fired"
                        and not session.stop_requested()
                        and session.terminal_event is None
                    ):
                        log.warning("org agent %s crashed (%s); retiring", name, cause)
                        await retire_crashed_org_agent(session, name, cause)

                # Start any new agents (or restart exited ones with pending input).
                env_terminal_seen = session.terminal_event is not None
                for agent in session.all_agents():
                    if agent.state == "fired":
                        continue
                    # §6.5: after env-terminal, freeze the org roster. Between
                    # the terminal result and the session stop the harness is
                    # firing every org agent; don't start or resurrect one.
                    if env_terminal_seen and agent.kind == "org":
                        continue
                    if agent.name in running:
                        continue  # already in flight
                    if agent.name in exited:
                        if not await session.inbox.has_pending(agent.name):
                            continue
                        exited.discard(agent.name)
                    task = asyncio.create_task(
                        run_agent(session, agent), name=f"loop:{agent.name}"
                    )
                    running[agent.name] = task
                    session._agent_tasks[agent.name] = task

                # §4.5.1: a CEO whose model call failed restarts only on new
                # input. If every live org agent is parked, none will come:
                # restart it once with the quiescent marker, and if it fails
                # again with the org still quiescent, end the session.
                if (
                    session.ceo_inference_error
                    and session.ceo_name not in running
                    and session.terminal_event is None
                    and not await session.inbox.has_pending(session.ceo_name)
                ):
                    org = {a.name for a in session.live_org_agents()}
                    if org and org <= session.wakes.waiting_agents():
                        if not session.ceo_retry_offered:
                            session.ceo_retry_offered = True
                            await session.signal_quiescence_to_ceo(source="ceo-retry")
                            continue
                        session.abort(session.ceo_inference_error)
                        break

                # End condition: nothing currently in flight, and nothing
                # delivered to the CEO since the restart pass above.
                if not running:
                    if await session.inbox.has_pending(session.ceo_name):
                        continue
                    if session.ceo_inference_error and session.terminal_event is None:
                        # Nothing left to restart the CEO after its model
                        # call failed: say so rather than end as "complete".
                        # A graded (env-terminal) episode keeps its ending.
                        session.abort(session.ceo_inference_error)
                    break
                await asyncio.sleep(0.1)

            # Cancel remaining tasks.
            for name, task in list(running.items()):
                if not task.done():
                    task.cancel()
            for name, task in list(running.items()):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            await supervisor()
        except asyncio.CancelledError:
            # The host cancelled us (e.g. inspect's time limit). That lands on
            # an await inside the supervisor loop and skips its task cleanup,
            # leaving every agent running after the sample ended. Abort and
            # cancel them synchronously, before any await.
            session.abort("cancelled")
            for task in session._agent_tasks.values():
                task.cancel()
            raise
        finally:
            # Shielded so a level-triggered host cancel (anyio, as inspect
            # uses) can't cut teardown short at its first await. Every wait
            # below is bounded, so the shield can't hang the host.
            with anyio.CancelScope(shield=True):
                # Let cancelled agent tasks unwind while the bus is still open.
                # No-op on a normal exit: the supervisor already awaited them.
                pending = [t for t in session._agent_tasks.values() if not t.done()]
                if pending:
                    _, still_running = await asyncio.wait(
                        pending, timeout=AGENT_UNWIND_TIMEOUT_S
                    )
                    if still_running:
                        # Something swallowed the cancel. asyncio can't force
                        # a task to end; the stop flag ends its loop at the
                        # next turn boundary.
                        log.warning(
                            "agent tasks still running %gs after cancel: %s",
                            AGENT_UNWIND_TIMEOUT_S,
                            sorted(t.get_name() for t in still_running),
                        )
                # §5.6 — every running background task is terminated on session
                # shutdown. Run this BEFORE emitting `session.ended` so the
                # final `task_update` notifications make it into the
                # trajectory while the event bus is still open.
                try:
                    await session.background_tasks.shutdown()
                except Exception:
                    log.exception("background-task shutdown failed")
                # Emit session.aborted before session.ended for paths that signalled
                # abort from inside the loop (§4.4 — e.g. max-wall-s, env-tool
                # circuit breaker, host cancel). External abort_session emits
                # eagerly itself; the idempotency in emit_aborted_if_needed
                # dedupes either way.
                await session.emit_aborted_if_needed()
                ended_reason = "aborted" if session.abort_requested else "complete"
                await session.events.emit("session.ended", {"reason": ended_reason})
                await session.events.close()

        end = now_ms() / 1000.0
        turn_count = sum(
            1
            for m in session.transcript(session.ceo_name)
            if m.role == "assistant"
        )
        if session.terminal_event is not None:
            end_reason = "env-terminal"
        elif session.abort_requested:
            end_reason = session.abort_reason or "aborted"
        else:
            end_reason = "complete"
        return RolloutResult(
            end_reason=end_reason,
            session_id=session.session_id,
            workspace_path=session.state.workspace_path,
            terminal_event=session.terminal_event,
            reward=session.effective_final_reward(),
            ceo_name=session.ceo_name,
            org_agents=[a for a in session.all_agents() if a.kind == "org"],
            duration_s=(end - start),
            used_turns=turn_count,
            cumulative_reward=session.cumulative_reward,
            reward_steps=session.reward_step_count,
            step_rewards=session.step_rewards,
            terminated_by_env=session.terminal_event is not None,
            metrics=compute_rollout_metrics(session),
            parse_error_events=session.parse_error_events,
            tool_error_events=session.tool_error_events,
            harness_rejection_events=session.harness_rejection_events,
            infra_failure_events=session.infra_failure_events,
        )
