"""Core data types for the AgentOrg harness, per SPEC.md §3."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

AgentName = str
ExternalLabel = str
Json = Union[dict[str, Any], list[Any], str, int, float, bool, None]

# The CEO's name is the literal string "CEO" (§6.6). It is reserved:
# - the org-agent name generator must not mint it,
# - channel names matching it resolve as the CEO's virtual-trace channel.
CEO_NAME: AgentName = "CEO"


def now_ms() -> int:
    """Wall-clock timestamp in milliseconds since epoch.

    Host-side utility ONLY — used by `harness.py` for `RolloutResult.duration_s`
    bookkeeping. Per SPEC §4.5 + §12.4, no data-model record (`Message`,
    `InboxItem`, `ChannelMessage`, `ReactionEvent`, `Event`, `Agent`,
    `Plan`, `Channel`) exposes a `ts` field; the spec exposes no notion of
    time to agents. Do NOT default-factory a new field with this helper.
    """
    return int(time.time() * 1000)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Session / Agent / Budget / Mode  (§3.1, §3.2)
# ---------------------------------------------------------------------------


SessionMode = Literal["default", "org-plan"]
AgentKind = Literal["ceo", "org"]
AgentState = Literal["idle", "running", "fired"]


class Budget(_Base):
    max_org_agents: int = 0


class EnvironmentBinding(_Base):
    """§12.3 environment binding."""

    kind: str
    task_id: str
    split: Optional[str] = None
    client_config: dict[str, Any] = Field(default_factory=dict)


class SessionState(_Base):
    """Persisted session.json — the durable shape of a Session.

    The CEO's name is the literal "CEO" (§6.6) and is not recorded separately.
    """

    session_id: str
    workspace_path: str
    model: str
    budget: Budget
    mode: SessionMode = "default"
    org_agent_names: list[AgentName] = Field(default_factory=list)
    environment: Optional[EnvironmentBinding] = None
    headless: bool = False
    permission_policy: Literal["default", "accept-edits", "host-ask", "bypass", "plan"] = "default"
    last_seq: int = 0  # monotonic event seq across the session
    last_turn_index: int = -1  # for turn.started/ended events
    reported_agent_names: list[AgentName] = Field(default_factory=list)
    # Model's true context window (tokens), as configured at launch. Persisted
    # so `Session.resume` can rebuild the CompactionConfig override instead of
    # silently falling back to the model-name table (None → use the table).
    context_window_tokens: Optional[int] = None


ExitCategory = Literal["mission_complete", "underperforming", "redundant", "capacity"]


class ExitReview(_Base):
    """§3.2 / §6.5 exit review — the CEO's read on an agent's tenure."""

    category: ExitCategory
    delivered: str
    shortfalls: str
    score: int = Field(ge=0, le=10)


class Agent(_Base):
    """§3.2 — durable Agent record on disk."""

    name: AgentName
    session_id: str
    kind: AgentKind
    role: Optional[str] = None
    description: str = ""
    parent_name: Optional[AgentName] = None
    work_dir: str
    tool_allowlist: Optional[list[str]] = None
    tool_disallowlist: Optional[list[str]] = None
    system_prompt: str = ""
    state: AgentState = "idle"
    exit_review: Optional[ExitReview] = None
    fired_source: Optional[Literal["ceo", "harness"]] = None
    fired_reason: Optional[str] = None
    # Per-agent microcompaction slot (§11.4.1). Excluded from persistence
    # because microcompaction is a cache-busting in-memory mutation; the
    # corresponding tool-result records on disk are still verbatim. On resume
    # the slot must be available again so the freshly re-hydrated transcript
    # can be microcompacted from scratch.
    microcompacted: bool = Field(default=False, exclude=True)


# ---------------------------------------------------------------------------
# Messages  (§3.3)
# ---------------------------------------------------------------------------


class UserSegment(_Base):
    type: Literal["text", "attachment"]
    text: Optional[str] = None
    archive_path: Optional[str] = None
    snapshot: Optional[str] = None


class ToolCall(_Base):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    # True for harness-injected calls the model never emitted (currently only
    # the @terminal grading call, see loop._prepare_terminal_call). Training
    # -data builders strip these so the model's genuine tool-less answer stays
    # the target; the flag never reaches the provider payload (the request
    # builder reads only id/name/args).
    synthetic: bool = False
    # True when the provider flagged this call's arguments as unparseable
    # (ToolCallChunk.parse_error). _stream_turn already recorded the
    # malformed-json ParseErrorEvent from the chunk, so
    # _record_tool_outcome_error suppresses on this flag to avoid
    # double-recording the downstream dispatch failure. Provider-controlled —
    # unlike the `__parse_error__` args sentinel, the model cannot forge it
    # through emitted JSON.
    parse_error: bool = False


class ToolResultError(_Base):
    code: str
    message: str
    # Why there is no usable result (§12.1). ``"error"``: the call was wrong
    # or its execution failed. ``"rejection"``: the harness declined to run
    # the call on policy grounds (tool scoping, plan mode, the §4.5.2 wait
    # gate, permission policy, hire budget); nothing executed, so whether the
    # call would have succeeded is unknown. ``"infra"``: the harness or the
    # environment failed to carry out the call (a tool's ``execute`` raised,
    # the env session died, the env transport or grader failed); the call
    # itself was not judged. Set where the result is produced
    # (``tools.context.err_result`` / ``rejected_result`` / ``infra_result``).
    kind: Literal["error", "rejection", "infra"] = "error"


class ToolResult(_Base):
    ok: bool
    output: Any = None
    error: Optional[ToolResultError] = None


class ParseErrorEvent(_Base):
    """A tool call the harness could not accept as valid before execution.

    Emitted at the two places a bad tool call surfaces:
    * ``"malformed-json"`` — the provider could not parse the arguments as
      JSON (openai_compat sees a raw string it can't decode). The turn is
      persisted with the model's raw text so the loop can feed the failure
      back, but the harness records this event so training can penalize
      the exact turn that emitted the malformed call.
    * ``"invalid-args"`` — the args parsed but failed the tool's
      ``input_schema`` (see ``loop._validate_tool_args``); dispatch
      short-circuits with an ``invalid-args`` ToolResult.
    ``provider_response_id`` is the completion id of the assistant turn
    that emitted the call (``ToolContext.turn_response_id`` / the same field
    on the assistant ``Message``), so training can attribute the penalty
    to the exact generated-token span.
    """
    agent_name: str
    provider_response_id: Optional[str] = None
    tool_call_id: str
    tool_name: str
    code: Literal["malformed-json", "invalid-args"]
    message: str


class ToolErrorEvent(_Base):
    """A tool that ran and returned ``ok=False`` with ``error.kind == "error"``:
    the call was wrong or its execution failed on its own terms (a local
    tool code, or ``env-invalid-call`` for a call the env rejected as
    malformed).

    Distinct from ``ParseErrorEvent`` (fires before execution),
    ``HarnessRejectionEvent`` (the harness declined to run the call) and
    ``InfraFailureEvent`` (the harness or env failed to carry it out).
    Recorded for trainers that want per-turn penalties on the tokens of the
    failing assistant turn — attributed via ``provider_response_id``, in the
    same way env reward events are (§12.1). Avalon-M1-Mini was trained on the
    collective env reward only; these events were not part of its reward.
    """
    agent_name: str
    provider_response_id: Optional[str] = None
    tool_call_id: str
    tool_name: str
    code: str  # "env-invalid-call" | local tool codes
    message: str


class HarnessRejectionEvent(_Base):
    """A tool call the harness declined to run (§12.1).

    Recorded when a ToolResult comes back ``ok=False`` with
    ``error.kind == "rejection"``: tool scoping (``not-permitted``), org-plan
    mode, the §4.5.2 wait gate, permission policy, the hire budget, the
    §A.8.6 repeated-call refusal. Nothing executed, so whether the call would
    have succeeded is unknown; it is neither a ``ParseErrorEvent`` nor a
    ``ToolErrorEvent``, and a trainer that penalises tool errors would
    normally leave these unpenalised by default. Same attribution key as
    the other streams.
    """
    agent_name: str
    provider_response_id: str | None = None
    tool_call_id: str
    tool_name: str
    code: str
    message: str


class InfraFailureEvent(_Base):
    """A tool call the harness or the environment failed to carry out (§12.1).

    Recorded when a ToolResult comes back ``ok=False`` with
    ``error.kind == "infra"``: a tool's ``execute`` raised (``exception``),
    the env session died (``env-session-dead``), the env transport or the
    env's own tool code failed (``env-failure``), the env could not offer a
    tool it advertised (``env-tool-unavailable``), or the grading call failed
    (``terminal-tool-error``). The call
    itself was not judged, so a trainer should not penalise these; the stream
    exists so infra failure rates are measurable per rollout. Same
    attribution key as the other streams.
    """
    agent_name: str
    provider_response_id: str | None = None
    tool_call_id: str
    tool_name: str
    code: str
    message: str


class MessageUsage(_Base):
    """Provider-reported token usage for an assistant turn.

    Populated by the inference loop from `CompletedChunk.usage`. Used by
    compaction's token estimator to anchor on the real prompt-token count
    instead of the bytes/4 heuristic, so thresholds fire at the right place
    even when the heuristic is off.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0


class Message(_Base):
    """A single transcript record (system / user / assistant / tool).

    Per SPEC §3.3, the record carries no wall-clock timestamp.
    Transcript order is the append order of `messages.jsonl`.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list[UserSegment]]] = None
    reasoning: Optional[str] = None
    source: Optional[
        Literal["human", "report", "channel-notification", "dm", "env", "meta"]
    ] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    # Compaction marker
    compaction: bool = False
    # Provider-reported usage for assistant turns; None on user/tool/system.
    usage: Optional[MessageUsage] = None
    # Provider finish reason for assistant turns ("stop" | "length" |
    # "tool_calls" | provider-specific); None on user/tool/system rows and on
    # transcripts written before the field existed. "length" marks a turn the
    # engine cut at a token limit — the loop uses it to tell the model its
    # response was truncated instead of scolding it for bare text.
    stop_reason: Optional[str] = None
    # Provider-assigned completion id of the inference call that produced this
    # assistant turn (None on non-assistant rows and providers that don't
    # report one). Env reward events are keyed on it (Session.reward_events)
    # so training can attribute each step reward to the exact generated-token
    # span of the turn that earned it.
    provider_response_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Channels  (§3.4)
# ---------------------------------------------------------------------------


SubscriptionLevel = Literal["full", "light", "muted"]


class Mention(_Base):
    kind: Literal["agent", "external", "stale"]
    name: Optional[AgentName] = None
    label: Optional[ExternalLabel] = None


class Channel(_Base):
    name: str
    session_id: str
    default_subscription: SubscriptionLevel = "light"
    subscriptions: dict[AgentName, SubscriptionLevel] = Field(default_factory=dict)


class Attachment(_Base):
    archive_path: str


class ChannelMessage(_Base):
    """Per SPEC §3.4: addressed by per-channel monotonic `idx`.

    `msg_key` is `"<channel>:<idx>"` — the immutable identity used
    by reactions. The record carries no wall-clock timestamp.
    """

    channel: str
    idx: int                       # per-channel monotonic post index, assigned by ChannelStore.post
    msg_key: str                   # "<channel>:<idx>"
    author: str  # AgentName or ExternalLabel — disambiguated by `is_external`
    is_external: bool = False
    content: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)


class NotificationCounts(_Base):
    """§3.7 — delivery summary returned to the poster."""

    full: int = 0
    light: int = 0
    muted: int = 0
    tagged: list[str] = Field(default_factory=list)
    # §A.4 / §A.5 — names that were tagged with @<name> but didn't resolve
    # to a live teammate (fired agent or typo). The notification is a silent
    # no-op for these, but the poster needs to know so they can retarget.
    tagged_but_not_running: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reactions  (§3.4 + §7.5)
# ---------------------------------------------------------------------------


ReactionOp = Literal["add", "remove"]


class ReactionEvent(_Base):
    """Append-only entry in the per-channel reaction log.

    Per SPEC §3.4, ordering is by append position in the
    `<channel>.reactions.jsonl` log; the record carries no
    wall-clock timestamp.
    """

    channel: str
    msg_key: str
    emoji: str
    reactor: str  # AgentName or ExternalLabel
    op: ReactionOp


class ReactionGroup(_Base):
    """Per-emoji folded view rendered to callers of channel reads.

    `first_reaction_idx` is the append-order index in the reaction log
    of the earliest still-in-effect add for this `(msg_key, emoji)`
    group. Used to keep chip ordering stable across toggles.
    """

    emoji: str
    count: int
    reactors: list[str] = Field(default_factory=list)
    first_reaction_idx: int


# ---------------------------------------------------------------------------
# Plans  (§3.5)
# ---------------------------------------------------------------------------


class OrgAgentSpec(_Base):
    role: str
    description: str
    allowed_tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    work_dir: Optional[str] = None
    subscriptions: dict[str, SubscriptionLevel] = Field(default_factory=dict)
    archive_attachments: Optional[list[str]] = None


class OrgChannelSpec(_Base):
    """Channel declared in the CEO's org plan; pre-created on ExitOrgPlanMode."""
    name: str
    default_subscription: SubscriptionLevel = "light"
    subscriptions: dict[AgentName, SubscriptionLevel] = Field(default_factory=dict)


class OrgBlock(_Base):
    agents: list[OrgAgentSpec] = Field(default_factory=list)
    channels: list[OrgChannelSpec] = Field(default_factory=list)


class Plan(_Base):
    plan_id: str
    session_id: str
    markdown: str = ""
    org_block: Optional[OrgBlock] = None
    # Presence of `approved_by` indicates the plan was approved.
    # Approval ordering relative to other plan state is recoverable
    # from `plan.mode.exited` event seqs in `events.jsonl`.
    approved_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Tools  (§3.6)
# ---------------------------------------------------------------------------


ToolScope = Literal["local", "env"]


class ToolDefinition(_Base):
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False
    scope: ToolScope = "local"


# ---------------------------------------------------------------------------
# Inbox  (§3.7)
# ---------------------------------------------------------------------------


InboxKind = Literal[
    "user",
    "dm",
    "channel_post",
    "channel_preview",
    "reaction",
    "report",
    "meta",
    "tool_update",
    "task_update",
    "predecessor_handoff",
]


class InboxItem(_Base):
    """Per SPEC §3.7: `seq` is the sole ordering primitive on inbox content."""

    kind: InboxKind
    seq: int = 0  # arrival ordering within session; assigned by InboxStore on append
    # Common payload fields — populated based on `kind`.
    content: Optional[str] = None
    preview: Optional[str] = None
    from_: Optional[str] = Field(default=None, alias="from")
    author: Optional[str] = None
    channel: Optional[str] = None
    attachments: list[Attachment] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    payload: Optional[dict[str, Any]] = None
    # Reaction-specific fields (kind="reaction"; §7.5).
    msg_key: Optional[str] = None
    emoji: Optional[str] = None
    reactor: Optional[str] = None
    target_content_preview: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


ReportStatus = Literal["completed", "blocked", "failed"]


class ReportPayload(_Base):
    status: ReportStatus
    summary: str
    artifacts: list[Attachment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Events  (§9)
# ---------------------------------------------------------------------------


EventType = Literal[
    "session.started",
    "session.ended",
    "session.aborted",
    "turn.started",
    "turn.ended",
    "inference.requested",
    "inference.chunk",
    "inference.completed",
    "tool.requested",
    "tool.permission.asked",
    "tool.permission.decided",
    "tool.started",
    "tool.completed",
    "agent.spawned",
    "agent.state.changed",
    "agent.fired",
    "agent.quiesced",
    "agent.noop_detected",
    "agent.repeat_refused",
    "channel.created",
    "channel.posted",
    "channel.reaction.added",
    "channel.reaction.removed",
    "dm.delivered",
    "report.delivered",
    "plan.mode.entered",
    "plan.mode.exited",
    "external.input.received",
    "environment.bound",
    "environment.terminal",
    "compaction.fired",
]


class Event(_Base):
    """Per SPEC §9: `seq` is the sole ordering primitive (§12.4); renderings
    into any model-facing surface MUST NOT include wall-clock timestamps.

    The `ts` field (epoch-ms) is written to events.jsonl for human-facing
    postmortem tooling only, per §9.2 / §12.4.
    """

    seq: int
    ts: int = Field(default_factory=now_ms)
    name: Optional[AgentName] = None
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
