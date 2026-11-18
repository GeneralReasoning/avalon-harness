"""Session — the central runtime state of one AgentOrg (§3.1)."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agentorg.archive import ArchiveStore
from agentorg.background_tasks import BackgroundTaskRegistry
from agentorg.channels import ChannelStore
from agentorg.compaction import CompactionConfig
from agentorg.events import EventBus, PermissionBroker, WakeBroker
from agentorg.inbox import InboxStore
from agentorg.inference.base import InferenceProvider
from agentorg.prompts import QUIESCENT_CEO_MARKER_BODY
from agentorg.storage import (
    SessionPaths,
    iter_jsonl,
    read_json,
    to_jsonable,
    write_json_async,
)
from agentorg.tools.registry import ToolRegistry
from agentorg.types import (
    Agent,
    AgentName,
    Budget,
    CEO_NAME,
    EnvironmentBinding,
    HarnessRejectionEvent,
    InboxItem,
    InfraFailureEvent,
    Message,
    new_id,
    now_ms,
    ParseErrorEvent,
    SessionState,
    ToolErrorEvent,
)

log = logging.getLogger(__name__)


def wall_seconds() -> float:
    """Real time in seconds: the default session clock."""
    return now_ms() / 1000.0


@dataclass
class SessionConfig:
    workspace_path: str
    model: str
    budget: int = 2  # max live org agents (SPEC §6.2); 0 = CEO works solo
    mode: str = "default"  # "default" | "org-plan"
    environment: Optional[EnvironmentBinding] = None
    headless: bool = True
    permission_policy: str = "bypass"  # default for headless rollouts
    # Lifecycle. True (default) ends the session on quiescence — the RL/rollout
    # path. False keeps a solo CEO parked on idle (instead of exiting) so later
    # external input can re-engage it; such a session ends only on close/abort.
    # Opt-in for live interactive sessions; never set for RL.
    end_on_idle: bool = True
    # True (default) applies the §4.5.2 CEO tool restriction: while org agents
    # are unreported and not parked, the CEO can't call (or see) a non-read-only
    # tool any of them also holds. False lets the CEO use every tool it holds
    # at all times, alongside its agents.
    ceo_tool_restriction: bool = True
    external_labels: list[str] = field(default_factory=list)
    initial_user_message: Optional[str] = None
    workspace_agents_md: Optional[str] = None  # contents of AGENTS.md
    plan_filename: str = "session-plan.md"
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    toolset_name: Optional[str] = None
    date_str: Optional[str] = None
    # Inference knobs
    temperature: Optional[float] = 1.0  # Avalon-M1-Mini was trained at 1.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Cap on GENERATED tokens per call, independent of prompt size. `max_tokens`
    # above cannot express this — see InferenceRequest.max_completion_tokens.
    # Unset, nothing bounds a single response: one turn can fill the context and
    # leave the next prompt over the window, ending the session on a 400.
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    # Backend overrides
    parallel_tool_calls: bool = True
    # Optional rollout id forwarded as X-Rollout-Id header on inference calls,
    # for trainers that group requests by rollout.
    rollout_id: Optional[str] = None
    # §A.8.5 no-op termination signal (kept as trained). `noop_penalty` is
    # stamped as `shaping_reward` onto the ToolResult payload of a no-op shell
    # call (see `loop._stamp_noop_penalty`); it is shown to the model but never
    # added to the reported reward. `noop_park_strikes` is how many no-op turns
    # within the window it takes before the harness force-parks a specialist.
    noop_penalty: float = -0.05
    noop_park_strikes: int = 3
    # How many recent turns the strike count is taken over. Strict
    # consecutiveness is trivially defeated: an agent alternating `echo done`
    # with a real call never accumulates a run, but is just as stuck.
    noop_window: int = 8
    # §A.8.6 repeated identical call. Refuse a call once the agent's run of
    # consecutive identical (tool, args, output) calls reaches this length;
    # 0 disables. `repeat_penalty` is stamped on the refused result's payload
    # as `shaping_reward`; None means "the value of `noop_penalty`".
    repeat_refuse_after: int = 15
    repeat_penalty: float | None = None


class Session:
    """One AgentOrg session: a CEO, a roster of org agents, persistent state.

    The Session owns:
      - the on-disk paths,
      - the EventBus (§9),
      - the channel store (§7),
      - the archive store (§7.1),
      - the inbox store (§3.7),
      - the tool registry (§5.4),
      - the agent records and per-agent transcripts.

    The agent loop lives in `loop.py` and drives this Session.
    """

    def __init__(
        self,
        *,
        config: SessionConfig,
        inference: InferenceProvider,
        tools: ToolRegistry,
        session_id: Optional[str] = None,
    ):
        self.config = config
        sid = session_id or new_id("ses_")
        self._paths = SessionPaths(config.workspace_path, sid)
        self._paths.ensure()
        self.inference = inference
        self.tools = tools

        self.state = SessionState(
            session_id=sid,
            workspace_path=str(Path(config.workspace_path).resolve()),
            model=config.model,
            budget=Budget(max_org_agents=config.budget),
            mode=config.mode,  # type: ignore[arg-type]
            environment=config.environment,
            headless=config.headless,
            permission_policy=config.permission_policy,  # type: ignore[arg-type]
            context_window_tokens=config.compaction.context_window_tokens,
        )

        self.events = EventBus(self._paths, starting_seq=0)
        self.wakes = WakeBroker()
        self.permissions = PermissionBroker()
        self.channels = ChannelStore(self._paths)
        self.archive = ArchiveStore(self._paths)
        self.background_tasks = BackgroundTaskRegistry(self)

        def _pending_for(name: str) -> Path:
            if name == CEO_NAME:
                return self._paths.pending_jsonl
            return self._paths.agent_pending(name)

        self.inbox = InboxStore(CEO_NAME, _pending_for)

        # Per-agent in-memory records (kept in sync with agent.json + state).
        self._agents: dict[str, Agent] = {}
        # Per-agent transcripts as in-memory lists for hot reads — backed by
        # messages.jsonl on disk for persistence.
        self._transcripts: dict[str, list[Message]] = {}
        self._transcript_locks: dict[str, asyncio.Lock] = {}
        # Lifetime assistant-message counts per agent, maintained incrementally
        # by `append_message` so the peer activity card (§4.5.1) doesn't re-scan
        # the full transcript on every render.
        self._assistant_turn_counts: dict[str, int] = {}
        # Per-agent visible-tool-defs cache (consumed by loop.visible_tool_defs).
        # Stores one (key, defs) entry per agent; the key encodes everything the
        # function reads (allowlist, disallowlist, waiting_for_org_agents,
        # peer tools, episode finished) so a state change forces a recompute
        # on the next call. List is shared across calls — callers MUST NOT mutate it.
        self._vtd_cache: dict[str, tuple] = {}
        # Per-(viewer, channel) cursor — the latest `idx` the viewer has been
        # shown by ViewChannelMessages. Used to default no-cursor calls to
        # "since last seen" so the CEO doesn't repeatedly re-fetch trace
        # tails. In-memory; resets on session restart.
        self._vcm_cursors: dict[tuple[str, str], int] = {}
        # Per-(viewer, peer) — peer's lifetime `turn_count` the last time the
        # viewer saw the activity card via WaitForMessages. Used to compute
        # `turn_delta` so the CEO can spot silent-but-working peers at a glance.
        self._peer_turn_count_seen: dict[tuple[str, str], int] = {}

        # Multi-coroutine coordination
        self._agent_tasks: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._abort = asyncio.Event()
        self._terminal_event: dict[str, Any] | None = None
        self._final_reward: Optional[float] = None
        # SINGLE source of truth for env rewards (§12.1): one entry per graded
        # value — step deltas AND the terminal grade — in the order
        # record_reward was called. Each entry: {"kind": "step"|"terminal",
        # "reward": float, "agent": str|None, "response_id": str|None} —
        # `agent` is who earned it, `response_id` the provider completion id
        # of the turn whose tool call earned it, for trainers that want to
        # place rewards on exact generated-token spans (Avalon-M1-Mini used
        # the collective env reward only). `step_rewards` / `cumulative_reward` /
        # `reward_step_count` are DERIVED views of this list; nothing else
        # may record rewards, so the views can never drift.
        self._reward_events: list[dict[str, Any]] = []
        # Per-turn error events, for trainers that want per-turn signals.
        # Populated at the existing error sites in the loop (`_stream_turn` for parse errors,
        # `_dispatch_tools` for tool-result-ok=False). Each event carries the
        # ``provider_response_id`` of the assistant turn that emitted the
        # failing tool call, so a trainer can attribute a penalty to the
        # exact generated-token span. Read-only views via the
        # `parse_error_events` / `tool_error_events` properties.
        self._parse_error_events: list[ParseErrorEvent] = []
        self._tool_error_events: list[ToolErrorEvent] = []
        self._harness_rejection_events: list[HarnessRejectionEvent] = []
        self._infra_failure_events: list[InfraFailureEvent] = []
        # Compaction summarization health (telemetry only, NOT a reward path).
        # One entry per full-compaction summary call — {"outcome": str,
        # "issue": str|None} — so trainers can surface the empty-summary rate
        # (a reasoning model can spend its whole output budget in the thinking
        # channel and emit empty content, so the summary is replaced by a
        # "[summary unavailable]" placeholder and the next epoch resumes with no
        # context) without re-parsing the transcript. Mirrors `reward_events`.
        self._compaction_outcomes: list[dict[str, Any]] = []
        # Cached openreward @terminal tool spec: unset (Ellipsis) until first
        # probed by the CEO loop; then either a TerminalToolSpec (env grades the
        # CEO's plain final message via session.env.call_terminal_tool) or
        # None (env has no terminal tool, or the SDK is too old to expose one).
        # Cached because the probe hits /tools and is stable for the session.
        self._terminal_tool_probed: Any = ...
        # Cumulative total tokens (input + output) across every agent's
        # assistant turns, counting each distinct token exactly once. Each
        # turn re-sends the whole transcript as input, so we add only the
        # *new* input that turn (what grew since the prior turn, minus our own
        # prior output, which we already counted) plus this turn's output.
        # In the no-compaction case this equals "final context size + every
        # generated token"; after a compaction the shrunken context clamps the
        # delta to 0 rather than recounting the summary. Used by the harness
        # host-level `max_total_tokens` guard (§4.4).
        self._total_tokens: int = 0
        # Per-agent (input_tokens, output_tokens) from that agent's previous
        # assistant turn, for the new-input delta above.
        self._prev_usage: dict[str, tuple[int, int]] = {}
        # §A.8.2 solo-thrash nudge bookkeeping. Counter is in
        # *qualifying assistant turns* (turns with ≥1 substantive tool call),
        # not raw tool calls.
        self._solo_turn_count: int = 0
        self._solo_thrash_nudges_fired: int = 0

        # Consecutive tool-less turns by the env-bound solo CEO.
        # Reset on any tool call. After one streak step we fire a last-chance
        # nudge; a second consecutive tool-less turn terminates the run.
        self._env_tool_less_streak: int = 0
        # Env tools that have returned `finished=False` at least once. The
        # tool-less nudge must not tell the model such a tool ends the rollout.
        self._env_tools_seen_unfinished: set[str] = set()
        # §4.6 session clock, started by `Harness.run_to_completion`. Both are
        # None until then; `clock_total_s` stays None when no horizon is known.
        self.clock_start_s: Optional[float] = None
        self.clock_total_s: Optional[float] = None
        # What the session clock and the `max_wall_s` deadline read, in
        # seconds. Real time unless the host passes
        # `run_to_completion(wall_clock=...)`.
        self.clock: Callable[[], float] = wall_seconds

        self._env_tool_failure_streak: int = 0
        self._env_tool_failure_signature: Optional[str] = None

        # §A.8.5 — per-agent no-op history. `_noop_window` is the last
        # `noop_window` turn verdicts (True = the turn did nothing), and its
        # True-count is what trips the strike threshold: a run-based counter
        # resets on any real call, so an agent interleaving `echo done` with
        # occasional genuine work never trips one. `_noop_run` tracks the
        # strictly consecutive run alongside, purely for the `noop_max_run`
        # metric. `parks` counts
        # force-parks; an agent parked once gets no further grace.
        self._noop_window: dict[str, deque[bool]] = {}
        self._noop_run: dict[str, int] = {}
        self._noop_stats: dict[str, dict[str, Any]] = {}
        # Rate-limiting for the CEO nudge, which has no terminal state to
        # bound it the way a specialist's park does. `_turns_since_noop_nudge`
        # gates re-injection; `_noop_nudges_sent` counts them within the
        # current spell so the repeat can be shorter than the first.
        self._turns_since_noop_nudge: dict[str, int] = {}
        self._noop_nudges_sent: dict[str, int] = {}
        # §A.8.6 — per-agent current run: (key, length), where key is
        # (tool, canonical args, content digest) of the last completed call.
        self._repeat_run: dict[str, tuple[tuple[str, str, str], int]] = {}

        # §4.4 — first abort reason wins; session.aborted is emitted exactly
        # once via `emit_aborted_if_needed()` regardless of how many callers
        # raced into abort(). Lets the sync circuit-breaker path set the
        # reason without needing to own the event emit.
        self._abort_reason: Optional[str] = None
        # Set while the CEO's last model call failed for good (§4.4
        # `inference-error`); cleared by its next successful call.
        self.ceo_inference_error: Optional[str] = None
        # Whether the supervisor has already restarted the failed CEO with
        # the quiescent marker (§4.5.1); cleared with ceo_inference_error.
        self.ceo_retry_offered: bool = False
        self._aborted_emitted: bool = False

        # External-sender labels
        self._external_labels = set(config.external_labels)

        # Optional back-reference to the EnvAdapter that supplied this session's
        # env tools. Set by Harness.start_session for env-bound sessions; None
        # otherwise. The loop uses it to reach adapter-level capabilities that
        # aren't exposed as tools (openreward @terminal-tool grading in
        # particular — see loop._prepare_terminal_call / _dispatch_terminal_tool).
        self._env_adapter: Any = None

        # Message observers — fired after every successful `append_message`
        # (post-persist). Used by external telemetry sinks to stream the
        # trajectory as it happens.
        # Observers are sync callbacks; raising is caught + logged but
        # doesn't break the transcript path.
        self._message_observers: list[Callable[[AgentName, Message], None]] = []

    # ---- properties --------------------------------------------------------

    @property
    def paths(self) -> SessionPaths:
        return self._paths

    @property
    def ceo_name(self) -> AgentName:
        return CEO_NAME

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def external_labels(self) -> set[str]:
        return self._external_labels

    @property
    def headless(self) -> bool:
        return self.state.headless

    @property
    def plan_path(self) -> str:
        # Under the session dir, not the workspace root: concurrent sessions share a
        # workspace (e.g. an evaluation run that uses one workspace for every trial),
        # and a root-level plan file would let one CEO's WritePlan overwrite another's
        # before its ExitOrgPlanMode re-read it — the second CEO would then spawn the
        # first's roster.
        return str(self.paths.root / self.config.plan_filename)

    @property
    def terminal_event(self) -> Optional[dict[str, Any]]:
        return self._terminal_event

    @property
    def final_reward(self) -> Optional[float]:
        return self._final_reward

    @property
    def cumulative_reward(self) -> float:
        """Sum of every recorded env reward, terminal grade included — the
        episode return (§12.1). Derived from `reward_events`."""
        return sum(e["reward"] for e in self._reward_events)

    @property
    def step_rewards(self) -> list[float]:
        """Full env reward stream as bare floats, in record order (terminal
        grade included). Downstream consumers reduce this to a scalar under
        an env-specific policy (e.g. sum, or mean over non-zero steps).
        Derived read-only view of `reward_events`, so callers can't corrupt
        the stream."""
        return [e["reward"] for e in self._reward_events]

    @property
    def reward_step_count(self) -> int:
        """Number of recorded env rewards (terminal grade included)."""
        return len(self._reward_events)

    @property
    def env_adapter(self) -> Any:
        """The EnvAdapter that supplied this session's env tools, or None."""
        return self._env_adapter

    def set_env_adapter(self, adapter: Any) -> None:
        """Called once by Harness.start_session after Session construction; the
        adapter can't be a __init__ param without breaking every existing test
        that instantiates Session directly."""
        self._env_adapter = adapter

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens (input + output) across all assistant turns,
        counting each distinct token once (re-sent context is not recounted)."""
        return self._total_tokens

    def record_reward(
        self,
        reward: float,
        *,
        agent: Optional[str] = None,
        response_id: Optional[str] = None,
        terminal: bool = False,
    ) -> None:
        """Record a graded env reward (§12.1) — the ONLY reward write path.

        Appends one entry to `reward_events`; `step_rewards`,
        `cumulative_reward`, and `reward_step_count` are all derived from
        that list, so the views cannot drift.

        ``agent`` is the name of the agent whose tool call earned the reward;
        ``response_id`` is the provider completion id of that turn
        (``ToolContext.turn_response_id``), letting training attribute the
        reward to its exact generated-token span. Terminal grades flow
        through here too (``terminal=True`` — see `_OpenRewardTool.execute`
        and `loop._dispatch_terminal_tool`), so the stream carries the full
        episode return with the terminal grade exactly once; `mark_terminal`
        only locks in `_final_reward` and never appends.

        The §A.8.5 no-op penalty deliberately does NOT come through here, and
        does not ride the payload's `reward` field either. It is a harness
        opinion about a call, not an env grade, so it is stamped under
        `shaping_reward` (`loop._stamp_noop_penalty`), shown to the model, and
        never added to the reported reward — every reader of `output["reward"]`
        (`trajectory.build_trajectory`, downstream consumers that sum
        `reward`) sees env grades alone. Keeping both this stream and that
        field pure is what makes reward comparable across runs.
        """
        self._reward_events.append(
            {
                "kind": "terminal" if terminal else "step",
                "reward": float(reward),
                "agent": agent,
                "response_id": response_id,
            }
        )

    @property
    def reward_events(self) -> list[dict[str, Any]]:
        """Time-ordered env reward stream (step deltas + terminal grade), each
        entry ``{"kind": "step"|"terminal", "reward": float, "agent":
        str|None, "response_id": str|None}``. The sum of rewards is the
        episode return."""
        return [dict(e) for e in self._reward_events]

    def record_parse_error(
        self,
        *,
        agent_name: str,
        provider_response_id: Optional[str],
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        """Record a tool call the harness could not accept as valid (§12.1).

        Called at the two existing detection sites in the loop:
        ``_stream_turn`` when a ToolCallChunk arrives with ``parse_error``
        set (malformed JSON args), and ``_dispatch_tools`` when the
        tool's ``input_schema`` validator returns an ``invalid-args``
        ToolResult. Both keyed on ``provider_response_id`` so training can
        pin the shaping penalty to the exact generated-token span, exactly
        like ``record_reward`` does for step rewards.
        """
        self._parse_error_events.append(
            ParseErrorEvent(
                agent_name=agent_name,
                provider_response_id=provider_response_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                code=code,  # type: ignore[arg-type]
                message=message,
            )
        )

    def record_tool_error(
        self,
        *,
        agent_name: str,
        provider_response_id: Optional[str],
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        """Record a tool that ran and returned ``ok=False`` (§12.1).

        Called from ``_dispatch_tools`` after the ToolResult comes back
        with ``ok=False``, ``error.kind == "error"`` and a code other than
        ``"invalid-args"`` (that case belongs to ``record_parse_error`` —
        it's a bad tool call, not a failed execution). Rejections and infra
        failures have their own recorders below.
        """
        self._tool_error_events.append(
            ToolErrorEvent(
                agent_name=agent_name,
                provider_response_id=provider_response_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                code=code,
                message=message,
            )
        )

    def record_harness_rejection(
        self,
        *,
        agent_name: str,
        provider_response_id: str | None,
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        """Record a call the harness declined to run (§12.1); nothing executed.

        Called from ``_dispatch_tools`` when the ToolResult is ``ok=False``
        with ``error.kind == "rejection"``. Kept apart from
        ``record_tool_error`` so a trainer can leave these unpenalised (the
        default) or price them separately from execution failures.
        """
        self._harness_rejection_events.append(
            HarnessRejectionEvent(
                agent_name=agent_name,
                provider_response_id=provider_response_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                code=code,
                message=message,
            )
        )

    def record_infra_failure(
        self,
        *,
        agent_name: str,
        provider_response_id: str | None,
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        """Record a call the harness or the environment failed to carry out
        (§12.1); the call itself was not judged.

        Called from ``_dispatch_tools`` / ``_dispatch_terminal_tool`` when the
        ToolResult is ``ok=False`` with ``error.kind == "infra"``. Kept apart
        from ``record_tool_error`` so a trainer never prices these; a
        training driver can also read ``env-session-dead`` here to drop shaping
        for the whole rollout.
        """
        self._infra_failure_events.append(
            InfraFailureEvent(
                agent_name=agent_name,
                provider_response_id=provider_response_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                code=code,
                message=message,
            )
        )

    @property
    def repeat_penalty(self) -> float:
        """§A.8.6 stamp value: `repeat_penalty` if configured, else `noop_penalty`."""
        p = self.config.repeat_penalty
        return float(self.config.noop_penalty if p is None else p)

    def repeat_run(self, name: str, tool: str, canonical_args: str) -> int:
        """Length of `name`'s current run if it is on (`tool`, `canonical_args`),
        else 0 (§A.8.6). Read before dispatch to decide a refusal."""
        entry = self._repeat_run.get(name)
        if entry is None:
            return 0
        (t, a, _digest), n = entry
        return n if (t, a) == (tool, canonical_args) else 0

    def clear_repeat_run_unless(self, name: str, tool: str, canonical_args: str) -> None:
        """Drop `name`'s run unless it is already on (`tool`, `canonical_args`).

        For a rejected call (§A.8.6): nothing executed, so the call must not
        become the run's key — keying it would build a run out of refusals,
        and once the §A.8.6 refusal started extending that run the tool would
        be lost for good. But an agent that tried something else, even
        unsuccessfully, is no longer repeating what the run was counting, so
        the run does not survive it either. Clear without re-keying.
        """
        entry = self._repeat_run.get(name)
        if entry is None:
            return
        (t, a, _digest), _n = entry
        if (t, a) != (tool, canonical_args):
            del self._repeat_run[name]

    def record_call_key(self, name: str, key: tuple[str, str, str]) -> int:
        """Fold a completed call into `name`'s run (§A.8.6): extend it when
        the key matches the current run, else start a run of one. Returns the
        new length."""
        entry = self._repeat_run.get(name)
        n = entry[1] + 1 if entry is not None and entry[0] == key else 1
        self._repeat_run[name] = (key, n)
        return n

    def extend_repeat_run(self, name: str) -> int:
        """Extend `name`'s current run without re-keying it (§A.8.6). A refusal
        is an ``ok=False`` result, so its digest is of an error, never of the
        output the run was keyed on; keying it would start a fresh run of one
        and the next identical call would execute."""
        entry = self._repeat_run.get(name)
        if entry is None:
            return 0
        key, n = entry
        self._repeat_run[name] = (key, n + 1)
        return n + 1

    def _noop_entry(self, name: str, kind: str = "org") -> dict[str, Any]:
        return self._noop_stats.setdefault(
            name,
            {"kind": kind, "calls": 0, "turns": 0, "max_run": 0, "parks": 0},
        )

    def _noop_deque(self, name: str) -> deque:
        window = self._noop_window.get(name)
        size = max(1, int(self.config.noop_window))
        if window is None or window.maxlen != size:
            window = deque(window or (), maxlen=size)
            self._noop_window[name] = window
        return window

    def record_noop_turn(self, name: str, n_calls: int, *, kind: str = "org") -> tuple[int, int]:
        """Record one no-op turn by `name`.

        Returns ``(consecutive_run, noops_in_window)``. The window count is
        what the strike threshold reads; the run is kept for the `noop_max_run`
        metric.

        `n_calls` is how many no-op tool calls the turn made — the turn is the
        unit for striking, but calls are what those scans count.
        """
        window = self._noop_deque(name)
        window.append(True)
        self._turns_since_noop_nudge[name] = (
            self._turns_since_noop_nudge.get(name, 0) + 1
        )
        run = self._noop_run.get(name, 0) + 1
        self._noop_run[name] = run
        stats = self._noop_entry(name, kind)
        stats["kind"] = kind
        stats["calls"] += int(n_calls)
        stats["turns"] += 1
        stats["max_run"] = max(stats["max_run"], run)
        return run, sum(window)

    def record_noop_calls(self, name: str, n_calls: int, *, kind: str = "org") -> None:
        """Count no-op calls that rode along with real work in the same turn.

        Bumps `calls` only. The turn was productive, so it must not touch
        `turns`, the run, or the window — those measure stalls, and a stray
        line in a working turn is not one.
        """
        stats = self._noop_entry(name, kind)
        stats["kind"] = kind
        stats["calls"] += int(n_calls)

    def record_productive_turn(self, name: str) -> None:
        """Record a turn that did real work. Breaks the consecutive run, and
        ages one slot off the window — but does NOT clear it: interleaving real
        calls between no-ops is the evasion the window exists to catch."""
        self._noop_run.pop(name, None)
        if name in self._noop_window:
            self._noop_deque(name).append(False)
            self._turns_since_noop_nudge[name] = (
                self._turns_since_noop_nudge.get(name, 0) + 1
            )

    def claim_noop_nudge(self, name: str, *, every: int) -> Optional[int]:
        """Ask to send `name` a no-op nudge, at most once per `every` turns.

        Returns how many have now been sent (1 for the first), or None if one
        went out too recently. The first always goes through; every repeat is
        rationed, and a productive turn in between does NOT lift that.

        Resetting on a productive turn would disable the rationing for the
        exact shape this feature exists to catch: an agent alternating
        `echo done` with a real call is never more than one turn away from a
        reset. The counter is per agent for the life of the session; only a
        park resets it, because that is a real change of state. A specialist's
        ladder is bounded by the park; the CEO has no terminal state, so
        without this the same ~1 KB paragraph would be re-injected on every
        no-op turn, piling harness meta into the transcripts already going
        wrong and driving compaction churn. The penalty and the event still
        fire every time; only the text is rationed.
        """
        sent_before = self._noop_nudges_sent.get(name, 0)
        if sent_before and self._turns_since_noop_nudge.get(name, 0) < max(1, every):
            return None
        self._turns_since_noop_nudge[name] = 0
        sent = sent_before + 1
        self._noop_nudges_sent[name] = sent
        return sent

    def record_noop_park(self, name: str) -> int:
        """Record a force-park of `name`; returns how many times it has parked.

        Clears the run and the window: the ladder restarts, but `parks > 0` is
        what removes the grace period on the next no-op.
        """
        stats = self._noop_entry(name)
        stats["parks"] += 1
        self._noop_run.pop(name, None)
        self._noop_window.pop(name, None)
        self._turns_since_noop_nudge.pop(name, None)
        self._noop_nudges_sent.pop(name, None)
        return stats["parks"]

    def noop_park_count(self, name: str) -> int:
        return self._noop_stats.get(name, {}).get("parks", 0)

    @property
    def noop_stats(self) -> dict[str, dict[str, Any]]:
        """Per-agent no-op counters:
        ``{agent: {kind, calls, turns, max_run, parks}}``.

        Read-only view — callers get copies, so they cannot corrupt the
        counters behind the `noop_*` rollout metrics.
        """
        return {k: dict(v) for k, v in self._noop_stats.items()}

    @property
    def parse_error_events(self) -> list[ParseErrorEvent]:
        """Read-only view of every recorded parse-error event, in emission
        order. Returned copies are safe to mutate."""
        return [e.model_copy() for e in self._parse_error_events]

    @property
    def tool_error_events(self) -> list[ToolErrorEvent]:
        """Read-only view of every recorded tool-execution-error event, in
        emission order. Returned copies are safe to mutate."""
        return [e.model_copy() for e in self._tool_error_events]

    @property
    def harness_rejection_events(self) -> list[HarnessRejectionEvent]:
        """Read-only view of every recorded harness-rejection event, in
        emission order. Returned copies are safe to mutate."""
        return [e.model_copy() for e in self._harness_rejection_events]

    @property
    def infra_failure_events(self) -> list[InfraFailureEvent]:
        """Read-only view of every recorded infra-failure event, in
        emission order. Returned copies are safe to mutate."""
        return [e.model_copy() for e in self._infra_failure_events]

    def record_compaction(
        self,
        outcome: str,
        issue: Optional[str] = None,
        *,
        suffix_count: Optional[int] = None,
        stripped_rounds: Optional[int] = None,
    ) -> None:
        """Record one full-compaction summary outcome (telemetry). ``outcome``
        is the summary call's result ("accepted" | "bad-summary" |
        "inference-error" | "ptl-failed"); ``issue`` is the bad-summary kind
        ("empty" | "format-violation" | "tool-template-leak") or None.
        ``suffix_count`` is how many messages survived the splice verbatim —
        0 marks a zero-round compaction, the resume-from-mandate+summary-only
        state that strands a tool-using agent. ``stripped_rounds`` is how many
        kept assistant rounds lost their reasoning to the strip ladder
        (CompactionConfig.strip_reasoning_to_fit). Called once per compaction;
        drives the trainer's empty/bad-summary/zero-round rates."""
        self._compaction_outcomes.append({
            "outcome": outcome,
            "issue": issue,
            "suffix_count": suffix_count,
            "stripped_rounds": stripped_rounds,
        })

    @property
    def compaction_outcomes(self) -> list[dict[str, Any]]:
        """Per-compaction summary outcomes in call order, each
        ``{"outcome": str, "issue": str|None, "suffix_count": int|None,
        "stripped_rounds": int|None}``. Empty until the first full
        compaction fires."""
        return [dict(e) for e in self._compaction_outcomes]

    def effective_final_reward(self) -> Optional[float]:
        """Final reward to report: explicit terminal value if any, else the cumulative."""
        if self._final_reward is not None:
            return self._final_reward
        if self._reward_events:
            return self.cumulative_reward
        return None

    @property
    def abort_requested(self) -> bool:
        return self._abort.is_set()

    @property
    def abort_reason(self) -> Optional[str]:
        """Reason of the first abort() call (e.g. "max_wall_s",
        "env-session-dead:<tool>", "external"), or None if never aborted.
        Lets harness consumers distinguish infra-caused session deaths from
        policy-attributable budget exhaustion after run_to_completion."""
        return self._abort_reason

    # ---- persistence -------------------------------------------------------

    async def persist_state(self) -> None:
        await write_json_async(self._paths.session_json, to_jsonable(self.state))

    def transcript_path(self, name: AgentName) -> Path:
        if name == CEO_NAME:
            return self._paths.messages_jsonl
        return self._paths.agent_messages(name)

    def transcript_lock(self, name: AgentName) -> asyncio.Lock:
        if name not in self._transcript_locks:
            self._transcript_locks[name] = asyncio.Lock()
        return self._transcript_locks[name]

    def transcript(self, name: AgentName) -> list[Message]:
        if name not in self._transcripts:
            loaded = self._load_transcript(name)
            self._transcripts[name] = loaded
            self._assistant_turn_counts[name] = sum(1 for m in loaded if m.role == "assistant")
        return self._transcripts[name]

    def _load_transcript(self, name: AgentName) -> list[Message]:
        """Reconstruct the live (post-compaction) transcript from messages.jsonl.

        The file is append-only and contains the full history. A
        `compaction-boundary` record means: drop everything we've
        accumulated except the last `suffix_count` messages, then
        prepend the embedded `summary` (see §11.4.2 and
        `agentorg/compaction.py::auto_compact_if_needed`). Boundary +
        summary share one JSONL record so the prefix-drop is atomic.
        """
        path = self.transcript_path(name)
        live: list[Message] = []
        for raw in iter_jsonl(path):
            if raw.get("type") == "compaction-boundary":
                suffix_count = int(raw.get("suffix_count") or 0)
                strip_prefix = int(raw.get("strip_prefix_count") or 0)
                live = live[-suffix_count:] if suffix_count > 0 else []
                # Mirror the in-memory splice exactly (see
                # auto_compact_if_needed): the first strip_prefix suffix
                # messages lose their reasoning (the strip ladder's plan),
                # and every suffix assistant loses its stale `usage` — its
                # input_tokens measured the pre-compaction rendering and
                # would poison the estimate_tokens anchor after a reload.
                for k, m in enumerate(live):
                    if m.role != "assistant":
                        continue
                    update: dict[str, Any] = {}
                    if m.usage is not None:
                        update["usage"] = None
                    if k < strip_prefix and m.reasoning:
                        update["reasoning"] = None
                    if update:
                        live[k] = m.model_copy(update=update)
                # Pinned messages (the agent's mandate) survive every
                # compaction verbatim, ahead of the summary — see
                # agentorg/compaction.py::auto_compact_if_needed.
                head: list[Message] = []
                for pinned_raw in raw.get("pinned") or []:
                    try:
                        head.append(Message.model_validate(pinned_raw))
                    except Exception:
                        pass
                summary_raw = raw.get("summary")
                if summary_raw is not None:
                    try:
                        head.append(Message.model_validate(summary_raw))
                    except Exception:
                        pass
                live = head + live
                continue
            if raw.get("type") == "inference-request":
                # Metadata-only record (system prompt + tool defs + params
                # snapshot for replay). Not a transcript message.
                continue
            try:
                msg = Message.model_validate(raw)
            except Exception:
                continue
            live.append(msg)
        return live

    def full_transcript(self, name: AgentName) -> list[Message]:
        """The COMPLETE append-only history for `name`, including turns that
        `transcript()` hides behind a compaction boundary.

        `transcript()` returns the live (post-compaction) view: at each
        `compaction-boundary` it drops the summarised prefix down to the kept
        suffix and prepends the summary. This returns every message ever
        appended, in file order, skipping only the boundary / `inference-request`
        metadata records (and the embedded summary) — so trace views can surface
        the original reasoning and tool I/O that compaction summarised away.

        Read from disk each call (not cached): the raw history is deliberately
        not held in memory after compaction, and the virtual-channel read path
        that uses this is low-frequency (diagnosis / recovery), not the
        inference hot path.
        """
        path = self.transcript_path(name)
        msgs: list[Message] = []
        for raw in iter_jsonl(path):
            if raw.get("type") in ("compaction-boundary", "inference-request"):
                continue
            try:
                msgs.append(Message.model_validate(raw))
            except Exception:
                continue
        return msgs

    def epoch_history(self, name: AgentName) -> list[tuple[int, Message]]:
        """`full_transcript()` with each message's compaction epoch: epoch k
        messages were generated under the context produced by the k-th
        compaction splice (k=0 before any compaction). File order is
        generation order and the boundary record lands after the preserved
        suffix, so suffix messages keep the epoch they were GENERATED in.

        Provenance view for trainers: credit assignment must not chain
        across a compaction (the context was rewritten), and `transcript()`
        has already dropped the summarized-away turns it would need.
        """
        path = self.transcript_path(name)
        out: list[tuple[int, Message]] = []
        epoch = 0
        for raw in iter_jsonl(path):
            if raw.get("type") == "compaction-boundary":
                epoch += 1
                continue
            if raw.get("type") == "inference-request":
                continue
            try:
                out.append((epoch, Message.model_validate(raw)))
            except Exception:
                continue
        return out

    async def append_message(self, name: AgentName, msg: Message) -> None:
        from agentorg.storage import append_jsonl_async

        path = self.transcript_path(name)
        async with self.transcript_lock(name):
            self.transcript(name).append(msg)
            if msg.role == "assistant":
                self._assistant_turn_counts[name] = self._assistant_turn_counts.get(name, 0) + 1
                if msg.usage is not None:
                    prev_in, prev_out = self._prev_usage.get(name, (0, 0))
                    # New input this turn = context growth not attributable to
                    # our own prior output (already counted). Clamps to 0 on a
                    # compaction, where the context shrinks.
                    new_input = max(0, msg.usage.input_tokens - prev_in - prev_out)
                    self._total_tokens += new_input + msg.usage.output_tokens
                    self._prev_usage[name] = (msg.usage.input_tokens, msg.usage.output_tokens)
            await append_jsonl_async(path, msg)
        for cb in self._message_observers:
            try:
                cb(name, msg)
            except Exception:
                log.exception("message observer failed for agent=%s", name)

    def add_message_observer(
        self, cb: Callable[[AgentName, Message], None]
    ) -> None:
        """Register a sync callback fired after every persisted message.

        Observers run *after* the transcript lock is released, so they can
        safely re-enter Session APIs. They are intended for telemetry sinks
        — keep them fast and non-blocking, or
        offload work to a background queue.
        """
        self._message_observers.append(cb)

    # ---- agents -----------------------------------------------------------

    def agent(self, name: AgentName) -> Optional[Agent]:
        return self._agents.get(name)

    def live_agents(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.state != "fired"]

    def all_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def live_org_agents(self) -> list[Agent]:
        return [
            a for a in self._agents.values()
            if a.kind == "org" and a.state != "fired"
        ]

    def live_agent_names(self) -> set[str]:
        names = {a.name for a in self.live_agents()}
        return names

    def all_agent_names(self) -> set[str]:
        return {a.name for a in self._agents.values()}

    def is_org_quiescent(self) -> bool:
        """Snapshot: every live org agent is currently parked in the wait.

        Used by `WaitForMessagesTool` (SPEC §4.5.1 checkpoint 1) to decide
        whether the CEO should short-circuit instead of parking. Best-effort
        — read outside the WakeBroker lock; tolerates benign races (rare
        false positive: CEO sees quiescent marker right as an org agent
        unparks; the CEO acts on the marker and the world stays consistent).

        If there are no live org agents, treats the org as trivially
        quiescent — the CEO has no peers that can wake it. (The caller
        normally distinguishes this case via `live_org_agents()` and emits
        a `no_live_peers` marker instead.)
        """
        org_names = {a.name for a in self.live_org_agents()}
        if not org_names:
            return True
        return org_names.issubset(self.wakes.waiting_agents())

    async def signal_quiescence_to_ceo(self, *, source: str) -> None:
        """SPEC §4.5.1 — deliver the quiescent meta-marker to the CEO.

        Used by both quiescence-trigger paths:
          - `tools.org_tools.WaitForMessagesTool` `on_parked` (org-agent
            arrives last to park while CEO is also parked),
          - `loop._park_with_quiescence_guard` org-agent branch
            (same condition on the structural-fallback path).

        Appends a `[Harness meta]` InboxItem carrying
        `QUIESCENT_CEO_MARKER_BODY`, wakes the CEO, and emits the
        `agent.quiesced` event with the given `source`. Centralizing this
        on Session avoids a `loop.py` ↔ `org_tools.py` cross-import.
        """
        ceo = self.ceo_name
        await self.inbox.append(
            ceo, InboxItem(kind="meta", content=QUIESCENT_CEO_MARKER_BODY)
        )
        self.wakes.wake(ceo)
        await self.events.emit(
            "agent.quiesced", {"ceo": ceo, "source": source}, name=ceo
        )

    def assistant_turn_count(self, name: AgentName) -> int:
        """Lifetime count of assistant messages for `name`'s transcript.

        Used by the per-peer activity card (SPEC §4.5.1) as a time-free
        relative progress signal: the CEO can diff `turn_count` across
        successive snapshots to tell whether a peer has produced work
        since the last check.
        """
        # Touch `transcript()` so the count is populated on first read.
        if name not in self._assistant_turn_counts:
            self.transcript(name)
        return self._assistant_turn_counts.get(name, 0)

    def fired_agents(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.state == "fired"]

    def predecessor_for_role(self, role: str) -> Optional[Agent]:
        """Most-recently-hired fired agent with the same role (case-insensitive) (§6.4 step 4).

        Ordering is recovered from `state.org_agent_names` (hire order) walked in reverse.
        """
        role_lower = role.lower()
        for name in reversed(self.state.org_agent_names):
            agent = self._agents.get(name)
            if (
                agent is not None
                and agent.state == "fired"
                and agent.role
                and agent.role.lower() == role_lower
            ):
                return agent
        return None

    async def register_agent(self, agent: Agent) -> None:
        self._agents[agent.name] = agent
        if agent.kind == "ceo":
            pass
        else:
            self._paths.ensure_agent(agent.name)
            if agent.name not in self.state.org_agent_names:
                self.state.org_agent_names.append(agent.name)
        await write_json_async(self._agents_json_path(agent), to_jsonable(agent))
        await self.persist_state()

    def _agents_json_path(self, agent: Agent) -> Path:
        if agent.kind == "ceo":
            return self._paths.root / "ceo.json"
        return self._paths.agent_json(agent.name)

    async def update_agent(self, agent: Agent) -> None:
        self._agents[agent.name] = agent
        await write_json_async(self._agents_json_path(agent), to_jsonable(agent))

    def remaining_budget(self) -> int:
        slots_used = sum(1 for a in self.live_org_agents())
        return max(0, self.state.budget.max_org_agents - slots_used)

    # ---- control --------------------------------------------------------------

    def stop(self) -> None:
        """Clean stop — does NOT set the abort flag. `abort_requested` feeds
        the `session.ended` reason; a clean stop (env-terminal) must end with
        reason "complete", not "aborted"."""
        self._stop.set()
        for name in list(self._agents.keys()):
            self.wakes.wake(name)

    def abort(self, reason: str = "external") -> None:
        # First abort wins the reason — subsequent calls (e.g. supervisor
        # racing with external) don't overwrite it.
        if self._abort_reason is None:
            self._abort_reason = reason
        self._stop.set()
        self._abort.set()
        log.info("session abort requested: %s", reason)
        for name in list(self._agents.keys()):
            self.wakes.wake(name)

    async def emit_aborted_if_needed(self) -> None:
        """Emit `session.aborted` exactly once with the stored reason (§4.4).

        Multiple paths may want to surface the abort (the external interface
        wants the event before returning to its caller; the harness supervisor
        wants to emit on the way out for paths that abort from inside the
        loop). Idempotency lets both call this without double-emitting.
        """
        if self._aborted_emitted or self._abort_reason is None:
            return
        self._aborted_emitted = True
        await self.events.emit("session.aborted", {"reason": self._abort_reason})

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def mark_terminal(
        self,
        event: dict[str, Any],
        *,
        reward: Optional[float] = None,
        stop: bool = True,
    ) -> None:
        # Locks in the authoritative final value ONLY. The terminal grade's
        # stream entry (`reward_events`) was already recorded by
        # `record_reward(..., terminal=True)` at the site that earned it —
        # recording here too would double-count the episode return.
        self._terminal_event = event
        if reward is not None:
            self._final_reward = reward
        if stop:
            self.stop()

    # ---- workspace agents md ------------------------------------------------

    def workspace_agents_md(self) -> Optional[str]:
        return self.config.workspace_agents_md

    # ---- resume ------------------------------------------------------------

    @classmethod
    def resume(cls, *, workspace_path: str, session_id: str, inference, tools) -> "Session":
        paths = SessionPaths(workspace_path, session_id)
        state_raw = read_json(paths.session_json)
        if not state_raw:
            raise FileNotFoundError(f"No session at {paths.session_json}")
        state = SessionState.model_validate(state_raw)
        cfg = SessionConfig(
            workspace_path=state.workspace_path,
            model=state.model,
            budget=state.budget.max_org_agents,
            mode=state.mode,
            environment=state.environment,
            headless=state.headless,
            permission_policy=state.permission_policy,
            compaction=CompactionConfig(
                context_window_tokens=state.context_window_tokens
            ),
        )
        sess = cls(
            config=cfg,
            inference=inference,
            tools=tools,
            session_id=state.session_id,
        )
        sess.state = state
        sess.events = EventBus(paths, starting_seq=state.last_seq)
        # Load agent records.
        ceo_raw = read_json(paths.root / "ceo.json")
        if ceo_raw:
            sess._agents[CEO_NAME] = Agent.model_validate(ceo_raw)
        for name in state.org_agent_names:
            raw = read_json(paths.agent_json(name))
            if raw:
                sess._agents[name] = Agent.model_validate(raw)
        # Initialize the inbox sequence counter past the highest seq on
        # disk so freshly-appended post-resume items don't get a lower
        # `seq` than items written before the restart (which would put
        # them OUT OF ORDER in the rendered inbox).
        sess.inbox.init_counter_from_disk([CEO_NAME] + state.org_agent_names)
        return sess
