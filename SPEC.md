# AgentOrg Protocol Specification

AgentOrg is the organisation protocol Avalon-M1-Mini was trained in;
Avalon Harness (this repository, Python package `agentorg`) implements it.

**Version:** 1.1

## 0. Why AgentOrg

Multi-agent LLM systems today are mostly a workaround: they parallelize work and split up context to dodge the limits of a single long-running agent. Under this framing, the whole is at most the sum of its parts: give one agent infinite runtime, context and no degradation, and it should match or beat any decomposition of itself.

Human societies suggest this framing is incomplete. Output per person scales superlinearly with city size, and the mechanism is not just "more people" - it is denser interaction among heterogeneous specialists, which compounds into work that no individual could produce alone. The interesting question for AI is whether we can induce the same property: enough functional diversity and structured interaction between agents that the organization becomes capable of things no single agent, however well-resourced, could do.

We propose AgentOrg, a multi-agent system led by a CEO agent that holds the full task in view and decides how to attack it - decomposing the problem, hiring and reassigning specialists, supervising and redirecting them, and firing them when needed. Communication runs through topic-scoped channels and private DMs. The CEO is the linchpin: by staying one level above the work, it avoids the context rot that drags a single end-to-end agent into the weeds, and is positioned to form strong views on how to proceed and revise them quickly when an employee surfaces evidence against them.

## 1. Overview & design principles

A session in this harness is one such organization: a CEO agent and a roster of specialist agents working in a shared workspace, with persistent memory and communication across the run. This section describes the high-level choices that shape it.

### Principles

1. **Heterogeneous specialists.** A specialist differs from the CEO not by ability but by *scope*: a narrower system prompt, a restricted tool set, a private work directory, a single mandate. The CEO produces functional diversity by varying these scopes. Diversity is the precondition for the superlinear effect §0 bets on.

2. **One CEO, one level above the work.** Exactly one agent — the CEO — holds the global goal and the full task in view. Its work is decomposition, delegation, supervision, and direction; not execution. Staying one level above the work lets the CEO avoid the context rot that drags a single end-to-end agent into the weeds, hold strong views about how to proceed, and revise them quickly when a specialist surfaces evidence against them. A single accountable role also keeps the org decisive where flat multi-agent systems waste tokens on negotiation. Specialists cannot hire further specialists in v1: nesting adds coordination overhead we want to understand at a single level first.

3. **Two communication primitives - channels and DMs.** **Channels** are topic-scoped broadcast streams. Each subscriber picks one of three modes per channel: **full** for tight coordination (every post lands in the agent's next turn), **light** for situational awareness (the agent gets a notification count and can fetch full context on demand), and **muted** for archival access (the agent is not notified but may peruse contents at will). Across modes, an `@`-mention escalates the addressed agent to full notification for that message. **DMs** sit alongside as private 1:1 messages for targeted exchange that should not broadcast. Together these let the org calibrate — per topic, per agent, per message — how aware each member is in real time. This is the "denser interaction" of §0 made operational.

4. **Composable roster.** The CEO declares an initial roster upfront in **org-plan mode**: each specialist named, scoped, and given a mandate. This deliberate upfront decomposition matters — it forces the CEO to think through the structure of the work before any of it begins. But the design is not the last word: as work unfolds, the CEO can hire, fire, retool, reassign, or fundamentally reshape the org in response to what it learns.

5. **Bounded budget.** Concurrent specialists are capped, but within the cap the CEO has full latitude over how many to actually use — sometimes none, sometimes the full allowance. The cap bounds session cost and forces explicit prioritization when the work pushes against it.

6. **Plan-then-act.** Before doing state-mutating work, the CEO enters **org-plan mode**, which pauses execution until a written plan with an org roster is drafted. This separates strategy from execution and surfaces the plan for review. Planning is optional due to a subtle failure mode: for some types of problems such as math the CEO reasons so deeply during planning that the work is essentially done — if it then spawns specialists they only redo the same reasoning, and the org becomes overhead rather than leverage. The harness watches for this and pushes the CEO toward one of two productive exits: create an org and delegate, or finish directly via the environment's answer tool. The org is a tool, not a hard requirement.

7. **Shared workspace as external memory.** Agents read, write, and reference a shared **archive** of markdown documents. The archive is the org's working memory: persistent across the session, citable in messages, and a common substrate that no single context window has to hold.

8. **Identity over IDs.** Every agent has a stable display name (e.g. `RedAristotle`). Names make `@`-mentions meaningful, give specialists felt accountability, and let humans inspect the org without memorizing UUIDs.

9. **The org is a continuing entity.** A session's transcripts, channels, archive, and roster persist on disk; specialists revive on resume with their accumulated context. An AgentOrg is not a single conversation that ends with a final message — it is an organization that can be paused, joined, and continued.

### Same org, many frontends

The same organization can be driven by a CLI, a programmatic host, an RL training loop, or an evaluation run. The harness exposes a uniform interface for state inspection and external input; the org itself knows nothing about how a human or a trainer chooses to interact with it. This is what lets one code path serve interactive debugging, evaluation, and large-scale sampling alike.

---

## 2. Concepts & glossary

These are the primitives an AgentOrg is built from — the nouns the rest of the spec uses.

| Term | Definition |
|---|---|
| **Harness** | The agent runtime defined by this spec. |
| **Session** | One execution of the harness, scoped to a workspace and a session id. Owns one CEO and zero or more org agents. |
| **Workspace** | A filesystem directory the CEO operates in. Holds persistence under `.<harness>/`. |
| **Agent** | An LLM context with its own message history, tools, and (optionally) work directory. Two kinds: **CEO** and **org agent**. |
| **CEO** | The root agent of a session. One per session. Drives org planning, hire/fire, etc. |
| **Org agent** | A specialist agent hired by the CEO under a budget. Persists for the lifetime of the session (survives reports, pauses, and resumes; ends only when fired). Distinct from short-lived "subagent" / "task" delegates in single-agent harnesses, which run once and return. See §6.1. |
| **External sender** | A non-agent participant (a human, a frontend, an external service) that can post messages and DMs through the **external interface**. |
| **Channel** | A named, persistent, append-only message stream. Agents subscribe per channel. |
| **Virtual channel** | A read-only view exposing an existing on-disk transcript (e.g. an agent's own message log) through the channel read tools. Has no underlying `<name>.jsonl` file; writes are rejected. See §7.8. |
| **Direct message (DM)** | A message sent privately to one agent's inbox. |
| **Report** | A message sent by an org agent to the CEO on task completion. |
| **Tool** | A callable capability available to an agent (read a file, post to a channel, hire an agent, etc.). |
| **Org-plan mode** | A controlled mode in which the CEO drafts a plan to disk that includes an `org-block` specifying specialists to spawn. |
| **Org block** | A fenced code block of type `agents-org` inside the plan markdown that defines the specialist roster. |
| **Archive** | A per-session shared markdown directory that agents and external senders can read/write/list as attachments. |
| **Budget** | The maximum number of concurrent org agents permitted in a session. |
| **Inbox** | Per-agent queue of pending DMs, channel notifications, and reports awaiting delivery on the agent's next turn. |
| **Mention** | An `@<name>` token in a channel message that names a recipient (agent or external sender) and elevates the notification to "tagged". |
| **Reaction** | An emoji toggle attached by an agent or external sender to a specific `ChannelMessage`. Append-only; the latest add/remove per `(message, emoji, reactor)` decides current state. Reactions do not modify the target message. See §7.5. |
| **Wait gate** | Umbrella term for the two coordination mechanisms in §4.5: the **wait** (the CEO calling `WaitForMessages`, or — as a backstop — the harness structurally parking an agent whose previous turn made no tool calls until new activity arrives) and the **CEO tool restriction** (the CEO is limited to coordination, archive, read-only, and org-management tools whenever it has live unreported org agents). The wait is purely event-driven — there is no timeout, and the spec exposes no notion of wall-clock time to agents. Liveness is preserved structurally via the **quiescence invariant** (§4.5.1): the CEO MUST NOT be parked while every live org agent is also parked. |
| **Quiescence** | The state in which every live agent in the session is parked in the wait simultaneously. The harness enforces the **quiescence invariant** (§4.5.1) so the org cannot enter this state and deadlock. |

---

## 3. Data model

An AgentOrg is, structurally, a small graph: a CEO, a roster of specialists, channels they exchange messages on and documents they have drafted. This section is the JSON shape of that graph — what the harness writes down so an org can be inspected, persisted, and resumed. All durable types are JSON-serializable. Field names are snake_case in the on-disk form; implementations MAY use idiomatic casing in memory.

### 3.1 Session

```ts
type Session = {
  session_id: string;
  workspace_path: string;
  model: string;
  budget: { max_org_agents: number };
  mode: "default" | "org-plan";
  org_agent_names: AgentName[];    // all org agents ever hired this session, in hire order; current state lives on each Agent record. The CEO's name is the literal "CEO" (§6.6) and is not recorded separately.
  environment?: EnvironmentBinding;
};
```

### 3.2 Agent

```ts
type Agent = {
  name: AgentName;               // unique within session; identity. e.g. "RedAristotle"
  session_id: string;
  kind: "ceo" | "org";
  role?: string;                 // "Feature Auditor", "Policy Strategist"...
  description: string;           // human-readable purpose
  parent_name?: AgentName;       // org: the CEO that hired it
  work_dir: string;              // private cwd; defaults to workspace_path
  tool_allowlist?: string[];     // null = inherit from kind defaults
  tool_disallowlist?: string[];
  system_prompt: string;         // current materialized system prompt; rebuilt by the harness on hire/fire and org-plan-mode entry/exit (§4.3)
  state: "idle" | "running" | "fired";
  exit_review?: ExitReview;      // CEO's structured review of the agent's tenure; recorded at fire time (§6.5). Absent on running agents and on harness-initiated terminations.
};

// The spec exposes no creation or termination wall-clock time on the Agent
// record. Lifecycle ordering is recovered from `agent.spawned` / `agent.fired`
// event seqs (§9). Implementations MAY record `created_at` / `ended_at`
// timestamps internally for human-facing trajectory inspection only — they
// MUST NOT be surfaced to agents in any tool result or rendered context.

// AgentName: a string matching [A-Za-z0-9_-]+, length 1–50, unique within
// a session (live + fired). Filename-safe so it can be used as a directory
// name. Always generated by the harness (e.g. <Color><Philosopher>) — the
// CEO does not pick names. The CEO's own name is the literal "CEO" (§6.6).

type ExitReview = {
  category: "mission_complete" | "underperforming" | "redundant" | "capacity";
  delivered: string;             // 1–3 sentences. Concrete artifacts/decisions the specialist produced that materially advanced the mission. Names nouns, not adjectives.
  shortfalls: string;            // 1–3 sentences. Concrete gaps, errors, or wasted budget. SHOULD be non-empty unless `category = "mission_complete"` with no reservations; this is guidance to the CEO, and implementations MAY accept an empty string for any category.
  score: number;                 // integer in [0, 10]; CEO's rating of delivered performance. See §6.5 anchors and worked examples.
};
```

### 3.3 Message

The transcript is an append-only sequence of `Message` records:

```ts
type Message =
  | { role: "system"; content: string }
  | { role: "user"; content: string | UserSegment[];
      source?: "human" | "report" | "channel-notification" | "dm" | "env" }
  | { role: "assistant"; content?: string; reasoning?: string;
      tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: ToolResult };

type ToolCall = { id: string; name: string; args: Json };
type ToolResult = {
  ok: boolean;
  output: Json;
  error?: { code: string; message: string; kind?: "error" | "rejection" | "infra" };
};

type UserSegment =
  | { type: "text"; text: string }
  | { type: "attachment"; archive_path: string; snapshot?: string };
```

The `Message` record carries no wall-clock timestamp. Transcript order is the append order of `messages.jsonl` (§11.1). Implementations MAY record per-line `ts` values on disk for human inspection only (§12.4 — advisory); such fields MUST NOT be surfaced to agents in any rendered context.

### 3.4 Channel & ChannelMessage

```ts
type Channel = {
  name: string;                  // "announcements", "random"...
  session_id: string;
  default_subscription: "full" | "light" | "muted";
  subscriptions: Record<AgentName, "full" | "light" | "muted">;
};

type ChannelMessage = {
  channel: string;
  idx: int;                      // per-channel monotonic post index, assigned by the session at post time
  msg_key: string;               // "<channel>:<idx>"; immutable identity
  author: AgentName | ExternalLabel;
  content: string;               // markdown
  attachments?: { archive_path: string }[];
  mentions?: Mention[];
};

type Mention =
  | { kind: "agent"; name: AgentName }
  | { kind: "external"; label: ExternalLabel };

type ReactionEvent = {
  channel: string;
  msg_key: string;               // identity of the target ChannelMessage
  emoji: string;                 // 1–32 chars; rendered verbatim
  reactor: AgentName | ExternalLabel;
  op: "add" | "remove";
};

type ReactionGroup = {
  emoji: string;
  count: number;
  reactors: (AgentName | ExternalLabel)[];
  first_reaction_idx: int;       // append-order index of the earliest "add" still in effect; stable across toggles
};
```

The reaction log for a channel is an **append-only** sequence of
`ReactionEvent` records. The current per-message state is the fold:
for each `(msg_key, emoji, reactor)` triple, the latest event in
append order decides whether that reactor currently has the emoji on
that message. UI rendering groups by emoji and orders groups by
`first_reaction_idx` so chip ordering is stable across toggles.

Per-channel `idx` is assigned by the session under the channel's
post lock and is monotonic from `0`. It is the addressing handle for
reactions (§7.5) and the only public identifier for a post. The spec
exposes no wall-clock timestamp on a `ChannelMessage`. Implementations
MAY record a posting `ts` in the on-disk record for human inspection
(§12.4 — advisory); such fields MUST NOT be surfaced to agents.

### 3.5 Plan & OrgBlock

```ts
type Plan = {
  plan_id: string;
  session_id: string;
  markdown: string;              // canonical text written by the CEO
  org_block?: OrgBlock;          // parsed from a ```agents-org fenced block
  approved_by?: ExternalLabel | "auto";  // presence indicates "approved"; ordering relative to other plan state is recoverable from events.jsonl
};

type OrgBlock = {
  agents: OrgAgentSpec[];
  channels?: OrgChannelSpec[];   // pre-created on ExitOrgPlanMode
};

type OrgChannelSpec = {
  name: string;
  default_subscription?: SubscriptionLevel;              // default "light"
  subscriptions?: Record<AgentName, SubscriptionLevel>;  // per-agent overrides
};

type OrgAgentSpec = {
  role: string;                  // human-readable title ("Policy Strategist"); unique within plan
  description: string;
  allowed_tools?: string[];
  disallowed_tools?: string[];
  system_prompt?: string;        // additional instructions
  work_dir?: string;             // relative to workspace; default = workspace_path
  subscriptions?: Record<string, SubscriptionLevel>;  // channel → level for this hire
  archive_attachments?: string[];  // archive paths attached to the hire's mandate
};
```

### 3.6 Tool definition

```ts
type ToolDefinition = {
  name: string;                  // PascalCase, globally unique
  description: string;
  input_schema: JsonSchema;      // JSON Schema draft 2020-12 subset
  read_only: boolean;
  scope: "local" | "env";        // local = harness-side; env = remote environment
  // execute(args, ctx) -> ToolResult — runtime field, not persisted
};
```

### 3.7 Inbox

```ts
// Every InboxItem carries a per-session monotonic `seq: int` assigned by
// the harness on append (omitted from each variant below for brevity).
// `seq` is the sole ordering primitive on inbox content — items are
// collapsed into the next `turn_user_message` (§4.3) in ascending `seq`.

type InboxItem =
  | { kind: "user"; content: string }
  | { kind: "dm"; from: AgentName | ExternalLabel; content: string;
      attachments?: { archive_path: string }[] }
  | { kind: "channel_post"; channel: string;
      author: AgentName | ExternalLabel; content: string;
      attachments?: { archive_path: string }[];
      mentions?: Mention[] }
  | { kind: "channel_preview"; channel: string;
      author: AgentName | ExternalLabel; preview: string;  // truncated snippet of content
      mentions?: Mention[] }
  | { kind: "reaction"; channel: string; msg_key: string;
      reactor: AgentName | ExternalLabel; emoji: string;
      target_content_preview?: string }    // delivered to a post's author when someone reacts; see §7.5
  | { kind: "report"; from: AgentName; payload: ReportPayload };

// Delivery summary returned to the poster of a channel post and emitted on
// the channel.posted event. Tells the poster how their message was fanned
// out. Not delivered to receivers.
type NotificationCounts = {
  full: number;                              // subscribers who received the full post
  light: number;                             // subscribers who received a preview
  muted: number;                             // subscribers who got nothing
  tagged: (AgentName | ExternalLabel)[];     // recipients escalated to full via @-mention
  tagged_but_not_running: string[];          // @-mentions that resolved to no live teammate
                                             // (fired agent or typo): a silent no-op, surfaced
                                             // so the poster can retarget
};

type ReportPayload = {
  status: "completed" | "blocked" | "failed";
  summary: string;
  artifacts?: { archive_path: string }[];
};
```

---

## 4. Agent loop

Every agent in the org — CEO or specialist — is, at its core, the same loop: read inbox → think → act → record → repeat. This section defines that loop precisely. What differs between roles is the scope of tools and context, not the cycle itself.

### 4.1 Turn anatomy

A **turn** is one iteration: build context → inference → tool dispatch →
persist. A **session** is a sequence of turns terminated by an explicit
halt, an environment terminal (§12.3), an external abort.

```
loop:
  if abort_requested: break
  drain_inbox_into_user_message(agent)
  messages := assemble_messages(agent)   # see §4.3
  yield event("turn.started")
  response := inference.stream(messages, tools=agent.available_tools())
  for chunk in response: yield event("inference.chunk", chunk)
  yield event("inference.completed")

  if response.tool_calls is empty:
    persist(role=assistant, content=response.text)
    # Decide whether the agent should park for new activity or step out
    # of this run. See §4.5 for the per-role rules: an org agent with an
    # empty inbox parks until a wake event arrives; a CEO with live org
    # agents enters the structural wait (the backstop for the
    # WaitForMessages tool); otherwise the loop returns and the session
    # falls idle. "Returns" is not "session ends" — the org is a
    # continuing entity (§1 principle 9), and external input restarts
    # the loop with the new item delivered as the next turn_user_message.
    if should_wait(agent):
      block_until_wake(agent)              # §4.5
      continue
    return                                 # session idles; external input restarts

  persist(role=assistant, ...)
  tool_results := dispatch(response.tool_calls)   # see §4.2
  persist tool results
  yield event("turn.ended")
```

### 4.2 Tool dispatch

Within a single assistant response, the harness MUST partition tool calls
into two groups:

1. **Read-only**, all of whose `ToolDefinition.read_only === true`. These
   MAY execute in parallel.
2. **State-mutating**, the rest. These MUST execute sequentially, in the
   order issued by the model.

Each tool call traverses the permission flow (§5.2). A denied call returns a
synthetic `ToolResult` with `ok=false` and a known error code; this is
appended to the transcript like any other result so the model can recover.

A tool whose `execute` raises MUST be caught by the harness and surfaced
as a `ToolResult` with `ok=false`, `error.code = "exception"` and
`error.kind = "infra"` (§12.1: the tool failed, the call was not judged). The
exception MUST NOT propagate out of dispatch — the loop continues and the
model sees the failure as it would any other tool result, which lets it
retry, adapt, or give up. This is symmetric with the §5.1 rule for invalid
args: tools are sandboxed at the dispatch boundary so a single
misbehaving tool cannot terminate the session.

### 4.3 Message assembly

Each turn appends exactly one new `(user, assistant)` pair to the agent's
transcript. The messages array sent to the LLM is:

```
[system_prompt]                          # current materialized system prompt
+ historical_messages                    # the persisted transcript so far
+ turn_user_message                      # built fresh for this turn
```

**System prompt — rebuilt on state change.** The system prompt is
materialized at agent creation and stays stable across most turns, but the
harness rebuilds it whenever the agent's relevant state changes — notably
on hire/fire of org agents (roster updates affect every live agent's
prompt), and on org-plan-mode entry/exit (CEO only). Rebuilding the
system prompt invalidates the LLM's prefix cache for that one turn; that
cost is accepted because these events are rare relative to total turns,
and keeping the roster authoritative inside the system prompt is the
simplest model.

The system prompt is composed of:

```
base_prompt
+ agents_md_concatenated                 # resolved once at creation
+ role_addon (org agents only — fixed at spawn)
+ env_context (env-bound sessions only — high-level description of the env)
+ mode_addon (current mode-specific guidance; CEO only — changes on EnterOrgPlanMode/ExitOrgPlanMode)
+ roster_block (CEO: live org agents + budget remaining; org agents: their peers — updates on hire/fire)
```

When any non-static component changes, the harness materializes a new
system prompt and replaces `Agent.system_prompt`. The next turn uses the
new prompt.

The specific list of tools available to the agent is communicated to the
LLM via the `tools` parameter on each inference request (§8.1), not the
system prompt. This lets the tool set evolve at runtime — via
`UpdateOrgAgentTools` or env-driven retooling — without invalidating
the prompt cache.

**`turn_user_message` — the collapsed inbox.** This is a normal chat user
message. Once the turn completes it is persisted into history alongside
the assistant response and any tool results; the next turn builds a fresh
one. Its body is the collapsed inbox: all `InboxItem`s pending since the
last turn are serialized into the message body in arrival order (`seq`
ascending; see §9.2). Each item is rendered with a header naming its kind
and source, followed by its content. The exact rendering format is an
implementation choice but MUST be deterministic for a given set of items,
and MUST surface, per item:

- the `kind` (DM / channel post / channel preview / report / external user),
- the sender (`from`/`author`) and, for channels, the channel name,
- the body — full content for `dm`, `channel_post`, `report`, and `user` items; the snippet for `channel_preview`,
- any `attachments` (as archive paths),
- any `mentions` directed at the recipient (so the model notices it was tagged).

A typical rendering:

```
[DM from RedAristotle]
<full content>
<attachments: deliverables/proposal.md>

[#announcements — BlueKant posted]
<full content>

[#random — BlueKant posted (preview)]
<~200-char snippet>…
(use ViewChannelMessages("random") for the full post)

[Report from GreenAristotle — completed]
<summary>
```

Inbox items MUST be cleared after collapse.

If the inbox is empty after a turn that produced no tool calls, the loop
either parks (§4.5 — the agent is waiting on incoming activity) or
terminates (no activity will arrive); see §4.5 for the per-role decision.

### 4.4 Abort & control

The harness MUST expose a cooperative abort that:

1. Stops streaming on the next chunk boundary.
2. Cancels any in-flight tool calls that opted into cancellation.
3. Persists partial state (in-flight assistant message marked truncated).
4. Emits `session.aborted`.

Abort MUST be idempotent. The session remains resumable after abort.

The `session.aborted` event MUST carry a `reason` string so downstream
consumers (RL trainers, evaluators, frontends) can filter on it. The
following canonical reasons are reserved:

| Reason | Meaning |
|---|---|
| `external` | An external caller invoked `abort()` (e.g. user interrupt, frontend disconnect). |
| `max_wall_s` | The session exceeded its wall-clock budget (on the §4.6 session clock). |
| `env-tool-circuit-breaker:<tool_name>:<error_code>` | The same env-side tool failed with the same error signature several turns in a row; the harness gave up rather than burn budget on a stuck rollout (see below). |
| `inference-error:<ExceptionType>` | The CEO's last model call failed for good (e.g. the model server is unreachable) and nothing remained to restart it. A CEO that recovers on a later turn does not end the session this way. |

Implementations MAY emit additional reason strings; consumers MUST tolerate
unknown values.

**Env-tool circuit breaker.** When the harness drives an env-bound rollout,
it SHOULD bound the cost of a stuck failure loop: if the same env-side
tool (i.e. a tool registered with `scope = "env"`) fails N turns in a row
with the same `(tool_name, error_code, error_message_prefix)` signature,
the harness MUST abort the session with reason
`env-tool-circuit-breaker:<tool_name>:<error_code>` (the error code lets
consumers tell, say, a dead sandbox from a malformed call). Any env-tool
success resets the
streak. The exact threshold is an implementation choice (reference: 5);
the spec only requires that *some* such safeguard exists for env-bound
runs and that the reason is recoverable from the event. Local-tool
failures and inference failures are out of scope for the breaker — they
have their own handling (§4.2, §8.2).

### 4.5 Wait gate

"Wait gate" covers two related mechanisms that govern how the CEO and its
org agents stay coordinated while work is in flight.

- **The wait.** How an agent parks when it has nothing immediate to act
  on, available through the `WaitForMessages` tool and as a structural
  fallback for tool-less turns (§4.5.1). The wait is purely
  event-driven and has no timeout.
- **CEO tool restriction.** When the CEO has live org agents that have
  not yet reported (in this session) and are still working, the CEO's own turns are
  limited to a restricted tool set (§4.5.2). Org agents have no analogue
  — they don't supervise anyone.

Liveness on the wait path is preserved structurally via the
**quiescence invariant** (§4.5.1): the CEO MUST NOT be parked while
every live org agent is also parked. The harness enforces this rule at
two atomic checkpoints — there is no wall-clock fallback.

#### 4.5.1 The wait

An agent with nothing immediate to act on parks in **the wait**: a single
primitive available through the `WaitForMessages` tool and, as a
structural fallback, through any tool-less turn. Both paths run on the
same underlying mechanism and produce the same lift on wake — they
differ only in whether the items surface as a tool result or as the next
turn's `turn_user_message`.

##### The quiescence invariant

The spec exposes no wall-clock time to agents and no timeout on the
wait. To guarantee deadlock-freedom the harness enforces:

> **The CEO MUST NOT be parked while every live org agent is also
> parked.**

This rule is necessary and sufficient for liveness. **Necessary**:
violation means no agent is running to produce events, so no inbox
arrival or state change can fire a wake — the org deadlocks.
**Sufficient**: as long as it holds, at least one of {CEO active, some
org agent active} is true; that agent will eventually produce an event
(a turn, a message, a tool call) that either continues progress or
transitions the system back through one of the rule's checkpoints
below.

The invariant subsumes org-agent-to-org-agent waiting cycles (Alice
waits on Bob, Bob waits on Alice): if the CEO is active it can
intervene as the supervisor; if the CEO is also parked, the rule
forces the CEO awake and the CEO decides how to break the cycle.

The harness enforces the invariant at two atomic checkpoints, both
inside the lock that guards the per-agent waiting set (so the two
arrival orders compose without race):

1. **CEO entry to the wait — short-circuit (§4.5.1 step 2 below).**
   Before parking, the CEO checks whether every live org agent is
   already in the waiting set (or there are no live org agents at
   all). If so, `WaitForMessages` returns immediately with a
   **quiescent marker** — the CEO does NOT park. This handles the
   ordering "org agents parked first, CEO arrives last".
2. **Org-agent entry to the wait — CEO wake (§4.5.1 step 4 below).**
   Atomically with parking, an org agent checks whether its
   transition causes every live agent (incl. the CEO) to be parked.
   If so, the harness appends the quiescent marker to the CEO's inbox
   as a meta item (inbox form, A.8.4) and wakes the CEO. The triggering
   org agent itself remains parked. This handles the ordering "CEO
   parked first, last org agent arrives".

The sufficiency argument assumes every live org agent is either active
or parked. An org agent whose loop exits any other way (its terminal
inference failure, or an uncaught exception) is neither: it is not
parked, so the invariant can never trigger, and it produces no events.
The harness MUST therefore retire it on exit: fire it (`agent.fired`
with `source = "harness"`, `reason = "crashed"`, no exit review), deliver
a meta item naming it and the cause to the CEO, and wake the CEO. The
CEO can then re-hire or take the work over. A live org agent's loop
otherwise exits only when it is fired, the session stops, or the env
terminates, so the harness detects a crash at task exit with no timeout.

The CEO has the same gap after its own terminal inference failure: its
loop exits, and the harness restarts it only when new input reaches its
inbox. If every live org agent is parked at that point, no input will
come. The harness MUST then deliver the quiescent marker (inbox form)
to the CEO once, which restarts it. If the CEO's model call fails again
before any call of its succeeds, and the org is still quiescent, the
session ends with abort reason `inference-error:<ExceptionType>` (§4.4).

Implementations MUST NOT introduce a wall-clock timeout into the wait
as a separate liveness mechanism. Pause-or-evict semantics under an RL
scheduler are a first-class use case: a paused agent is
indistinguishable from a slowly-thinking one from the harness's
perspective, and any wall-clock heuristic would force the CEO into
spurious fire / poke decisions whenever the scheduler resumes work.
Quiescence detection is eviction-safe by construction — it triggers
only on a structural transition (the *last* live agent entering the
wait), not on elapsed time.

##### Tool surface

`WaitForMessages` is a read-only tool.

```
WaitForMessages()
```

The tool takes no arguments. It blocks until new activity arrives or
— for the CEO — until the quiescence invariant would otherwise be
violated.

**Availability.** The tool is exposed exactly when the calling agent
has an in-loop source of inbox items it could plausibly be parked on;
otherwise the structural fallback (below) handles the idle case
cleanly. Concretely:

- **CEO**: exposed when the CEO has at least one live org agent
  (state ∈ {`idle`, `running`}) — the only CEO state with an in-loop
  source of inbox items (org-agent reports, peer DMs, channel
  notifications). A solo CEO or a CEO in `org-plan` mode has no
  in-loop peers, so the tool is not exposed; the per-binding behavior
  on a tool-less turn is defined by the structural fallback (below).
- **Org agents**: always exposed. An org agent is persistent and
  always reachable in-loop — the CEO can `SendMessage` it, peers can
  DM it, and any channel it subscribes to can deliver a notification.
  The tool-use-bias rationale below applies just as strongly to org
  agents as to the CEO: without `WaitForMessages` they tend to poll
  `ViewChannelMessages` between assignments instead of producing the
  tool-less turn that the structural fallback would catch.

Semantics (single iteration; the implementation re-enters from step 1
on a spurious wake):

1. If inbox items are already pending for the calling agent, drain them
   and return immediately as the tool result, rendered in the same
   format as a collapsed inbox in `turn_user_message` (§4.3). Items
   returned via the tool result MUST NOT also appear in the next
   turn's `turn_user_message`.
2. **CEO short-circuit (quiescence checkpoint 1).** If the caller is
   the CEO, do not park when either:
   (a) there are no live org agents — return
   `{no_live_peers: true, live_peers: [], suggestions}` immediately,
   the suggestions pointing at writing the plan and calling
   `ExitOrgPlanMode` (org-plan mode), acting directly or submitting
   (solo), or `HireOrgAgent`; or
   (b) every live org agent is currently in the waiting set — return
   the **quiescent marker** (tool-result form, A.8.4) immediately.
   It carries:
   - `quiescent: true`,
   - the per-peer activity card (below) for each live peer,
   - structural suggestions naming concrete next actions
     (SendMessage with new direction, FireOrgAgent if a peer is
     clearly stuck, PostToChannel, or a direct action / submission).

   A bare `quiescent: true` is not enough — without the activity
   cards and the action menu the model would call back-channel tools
   to reconstruct the same state.
3. Otherwise, atomically register self in the waiting set and block
   on the wake signal.
4. **Org-agent quiescence trigger (quiescence checkpoint 2).** Atomic
   with step 3, when the caller is an org agent the harness checks
   whether this transition causes the quiescence invariant to fire
   (every live agent — including the CEO — is now parked). If so,
   the harness appends the quiescent marker (inbox form, A.8.4) to the
   CEO's inbox and wakes the CEO. The triggering org agent itself stays
   parked and continues to step 5.
5. On wake, drain any pending inbox items and return them, together
   with a `live_peers` activity card for each live peer. A CEO woken
   by the step-4 trigger therefore receives the inbox-form marker as
   a drained item, with the cards alongside, but without
   `quiescent: true`. If no items are pending (spurious wake), restart
   the iteration from step 1.

##### Per-peer activity card

The quiescent marker's tool-result form includes a compact activity
card per live peer so the CEO can act without consulting back-channel
tools. Every `WaitForMessages` return that delivers drained items also
carries the cards (`live_peers`), for any caller, so a CEO woken by
checkpoint 2 sees them alongside the marker. The card MUST contain:

- `name`, `role`. The agent record's `state` is deliberately NOT on
  the card: `runtime` is the only busy/idle signal, and a second
  idle/running field that disagrees with it gets read over it.
- `runtime` — `idle` if the peer is currently parked in the wait,
  `active` otherwise; `fired` if reaped. Derived purely from
  harness-observable state (waiting-set membership and inference
  liveness); no scheduler hooks are required.
- `turn_count` — the peer's lifetime count of assistant turns. The
  CEO uses this as a relative progress signal across snapshots
  ("last time @Alice was at 4 turns, now 7 — she's working but not
  pinging me"); the spec exposes no wall-clock reference for "how
  long" the peer has been silent.
- `turn_delta` — the number of assistant turns the peer has taken
  since this viewer last received a card for it (`null` on the first
  card), i.e. the `turn_count` diff pre-computed per viewer. Positive
  with no inbound message means a silent worker. Zero means idle only
  together with `runtime: idle`: a peer in one long generation or tool
  call stays at zero while `runtime: active`.
- `recent_tool_calls` — count of tool calls in the last small window
  of the peer's transcript (reference: 20 messages). A sliding
  window over messages, not over time.

The card MUST NOT carry any wall-clock timestamps or derived
"time since" fields. A peer parked in the wait is reported as
`runtime: idle` without further temporal qualification; an active
peer is reported as `runtime: active` without an `active_since`
or analogous field. Wedged-stream detection is a host-level
concern (e.g. the inference provider's own HTTP timeout surfacing
as a transient error per §8.2) — it MUST NOT be surfaced as a
model-facing time signal on the activity card.

##### Wake conditions

The wait ends when any of the following occurs:

- **A new inbox item is delivered.** Per §3.7 the relevant kinds are:
  - `dm` — a direct message from any agent or external sender. Always
    wakes the recipient; subscriptions do not apply to DMs.
  - `channel_post` / `channel_preview` — a channel notification,
    delivered only if the agent's subscription for that channel (§7.2)
    yields one: `full` → `channel_post`, `light` → `channel_preview`,
    `muted` → no item and therefore no wake. An `@`-mention overrides
    subscription and delivers a `channel_post` to the tagged agent
    regardless of their level — including `muted`. A mention is meant
    to break through the recipient's filter (§7.4).
  - `user` — external user input injected via `send_user_input` (§10.1)
    or, in env-bound sessions, surfaced by the environment (§12.3). CEO
    only.
  - `reaction` — someone added an emoji reaction to a post you
    authored (§7.5). Delivered only on `add`; `remove` events are
    silent. Bypasses subscription level — like an `@`-mention, the
    addressee is being directly addressed.
  - `report` — a `ReportToCEO` from a hired org agent (§7.6). CEO only.
- **The agent is fired** (§6.5). The wait returns and the agent's loop
  exits.
- **The session is aborted** (§4.4). The wait returns and the agent's
  loop unwinds.
- **CEO-only: the quiescence invariant fires.** Checkpoint 1 (CEO
  arrives last) returns the quiescent marker as the CEO's
  `WaitForMessages` result; checkpoint 2 (org agent arrives last)
  delivers it to the CEO's inbox and wakes the CEO. Org
  agents have no analogous wake — they only wake on inbox arrivals,
  fire, or abort.

There is no timeout wake. The structural fallback (below) and the
tool call use the same wake set.

Implementations MAY add ancillary wake signals for the CEO so that
org-level state changes surface without polling — typically
`agent.spawned`, `agent.fired`, and `agent.state.changed` from §9.1.
When implemented, such signals MUST be rendered as inbox-shaped items in
the tool result so the model sees them alongside actual messages.

> Why a tool, not only the structural fallback. Tool-use-tuned models
> tend to prefer a tool call over a tool-less response. Without `WaitForMessages` they tend to poll `ListChannels` /
> `ListArchive` in a loop instead of producing the tool-less response
> the structural fallback would catch. The tool gives the model the
> action it wants while routing it through the correct primitive.

##### Structural fallback

A tool-less turn parks the agent in the same wait. On wake the pending
items collapse into a fresh `turn_user_message` (§4.3) and the next turn
runs normally. The fallback's role-specific shape:

- For the **CEO in an interactive session**, a tool-less turn enters
  the wait iff there are live (not fired) org agents, whether or not
  they have reported. Otherwise the loop exits and the session falls idle. The session
  itself persists (§1, principle 9): any subsequent external input —
  via §10's external interface or via an in-process consumer such as
  a CLI REPL — restarts the loop with that item as the next
  `turn_user_message`. Mode-specific nudges per §6.3 MAY intervene
  before idling.
- For the **CEO in an env-bound session** (§12.3), idling is not a
  normal exit — the episode terminates when the env's submission tool
  fires (`environment.terminal`, §9.1) or the env signals
  `terminal: true` via a tool result. Implementations SHOULD keep
  submission salient to the CEO throughout the session via a
  system-prompt completion requirement that names the submission tool,
  plus reinforcing nudges (e.g. a recurring "you can submit when ready"
  reminder while running solo, and a last-chance nudge before
  terminating on a tool-less turn). A tool-less CEO turn that does not
  submit is handled as follows:
  - On an env with an `@terminal` tool, the turn is the CEO's answer:
    it is graded and the episode ends (§6.5), whether or not org
    agents are live.
  - Otherwise, while any org agent is live (not fired), the CEO enters
    the wait like an interactive CEO; it does not end the run.
  - With no live org agents, the first consecutive tool-less turn gets
    the last-chance nudge and the next turn continues; a second
    consecutive tool-less turn is a stuck-rollout state, and the
    harness ends the run and reports whatever reward the env has
    emitted so far. Any turn with a tool call resets the count.
- For an **org agent**, a tool-less turn always enters the wait — the
  agent is always reachable by the CEO and by peers via DMs and channel
  notifications.

When the CEO's structural fallback would park while the quiescence
invariant is already violated, the harness MUST NOT park it: it
appends the quiescent marker (inbox form, A.8.4) to the CEO's inbox
and the CEO's next turn begins with it as the `turn_user_message`.
Checkpoint 2 firing while the CEO is parked by the fallback delivers
the same item the same way. This surface is text only, without the
activity cards; the marker text tells the CEO to call
`WaitForMessages` once, which then short-circuits at step 2 with the
cards.

#### 4.5.2 CEO tool restriction

When the CEO has live org agents that (a) have not yet appeared in
`reported_agent_names`, and (b) are not currently parked in
`WaitForMessages`, those agents are *contending peers*.
`reported_agent_names` is session-scoped and never reset: once an agent
has called `ReportToCEO`, it stops counting as a contending peer for the
rest of the session, even while it keeps working on follow-up requests. The CEO's next turn is restricted on a **per-tool** basis: a
tool call SHOULD be rejected if and only if some contending peer also
holds that tool in their effective tool set (i.e. it is in the session
registry, not in their `tool_disallowlist`, and — if `tool_allowlist`
is non-null — in their allowlist). Tools the CEO holds *exclusively*
have no peer to race with, so they are permitted regardless.

The following tool classes are always permitted (they don't race on env
state, or the race is intrinsic to the harness's own coordination
plane):

- coordination tools (`SendMessage`, `WaitForMessages`, `PostToChannel`,
  `ListChannels`, `ViewChannelMessages`, `SearchChannelMessages`,
  `SetChannelSubscription`, `ReactToMessage`),
- archive tools (`ListArchive`, `ViewArchiveFile`, `WriteToArchive`,
  `EditArchiveFile`, `MoveArchiveFile`),
- any tool with `ToolDefinition.read_only === true` (covers local `Read`,
  `Grep`, `Glob`, and any env-side read-only inspectors regardless of
  their name),
- org-management tools (`HireOrgAgent`, `FireOrgAgent`,
  `UpdateOrgAgentTools`).

Rejections SHOULD surface to the model via a `ToolResult` with
`ok: false`. The exact error code or metadata shape is implementation-
defined; what matters is that the tool result clearly explains why the
call was blocked (and which peer holds the tool) so the model can
revoke or pick a different action. The message SHOULD NOT offer firing
as a remedy: reclaiming a tool is not a reason to dissolve a working
team, and CEOs given that option took it. If it offers revocation, it
SHOULD state the cost before how to do it: revoking takes the tool away
from the holders until the CEO restores it, so their work that needs it
stops. A CEO shown only the recipe read it as a way to borrow the tool
for one call, and stripped the core tool from its whole team. The restriction releases
per-agent as each one reports, parks in `WaitForMessages`, or is fired;
once no contending peer holds the tool, the CEO regains it.

The restriction is configurable: `SessionConfig.ceo_tool_restriction`
(default `true`). With
`false`, none of the above applies. The CEO's visible tool set and
dispatch are never narrowed by contending peers, and the CEO can call
any tool it holds alongside its agents, the same as one org agent
sharing a tool with another.

> Rationale: the CEO must be able to inspect state and redirect agents
> while they work, but must not race them on shared state. A parked
> agent is blocked waiting for input and can't race, so it doesn't gate
> — the CEO can act on whatever the agent has already published via
> channels, archive, or tagged DMs. By the same logic, a tool that no
> live peer can call has no race to gate: the CEO is its sole holder.
> This rewards the CEO for reserving state-mutating env tools at hire
> time (or revoking via `UpdateOrgAgentTools`) — the reserved tool
> remains available to the CEO throughout the wait.

### 4.6 Session clock

`Harness.run_to_completion` starts a session clock, whose horizon is
the host `max_wall_s`. The clock is real time unless the host passes
`run_to_completion(wall_clock=...)`, a callable returning seconds. The
`max_wall_s` deadline reads this clock, so a host that pauses it (an
RL trainer whose optimizer step freezes inference) pauses the deadline
too. `RolloutResult.duration_s` stays real time.

---

## 5. Tool system

Tools are how agents act on the world — reading files, sending messages, hiring specialists, and so on.  This section defines the tool contract, how permission resolves, and the minimum set every implementation provides.

### 5.1 Tool interface

A tool implementation MUST provide:

```ts
interface Tool {
  definition: ToolDefinition;
  execute(args: Json, ctx: ToolContext): Promise<ToolResult>;
}

interface ToolContext {
  agent: Agent;
  session: Session;
  registry: ToolRegistry;
  request_permission(prompt: PermissionRequest): Promise<PermissionDecision>;
  emit(event: Event): void;
}
```

The harness MUST validate `args` against `input_schema` before calling
`execute`. On validation failure, `execute` is not invoked; a `ToolResult`
with `error.code = "invalid-args"` is appended.

### 5.2 Permission flow

Each tool call resolves to one of three decisions: `allow`, `deny`, `ask`.
The resolution rule is:

1. If session mode is `org-plan` and tool is state-mutating: `deny`.
2. If wait gate applies and tool is not permitted: `deny`.
3. If `agent.tool_disallowlist` contains the name: `deny`.
4. If `agent.tool_allowlist` is set and does not contain the name: `deny`.
5. Otherwise the harness consults a **permission policy** (§5.3) which
   returns one of `allow`, `deny`, `ask`.
6. If `ask`, the harness emits a permission request to the external
   interface and awaits a decision. Timeout behavior: implementations
   SHOULD default to `deny` on timeout.

### 5.3 Permission policies

Permission policies are pluggable. The spec defines four canonical modes:

| Mode | Behavior |
|---|---|
| `default` | `read_only` → `allow`; everything else → `ask`. |
| `accept-edits` | `default` plus `Edit`/`Write` → `allow`. |
| `bypass` | All → `allow`. |
| `plan` | All state-mutating → `deny`; read-only → `allow`. Activated whenever session mode is `org-plan`. |

### 5.4 Registry

The registry maps `name → Tool`. It MUST:

- Be constructible from a fixed set of built-ins plus optional
  registrations made before session start.
- Permit per-agent scoping at dispatch time (allowlist / disallowlist).
- Permit dynamic retooling for org agents via `UpdateOrgAgentTools`.

### 5.5 Built-in tools

The set of tools an agent can actually call depends on the session's
context. A local workspace session (e.g. a CLI on a developer's machine)
has bash and a filesystem; an environment-bound session against a remote
sandbox (e.g. an OpenReward NetHack rollout) has no host filesystem at all
and instead exposes env-side tools. The harness selects the right set per
session and surfaces it to the LLM via the inference `tools` parameter
(§8.1); the CEO is never given a tool schema it cannot call. Env-bound
sessions MAY include a short env-context paragraph in the system prompt
(§4.3) so the model knows what world it is in.

**Always available (org-level).** These are required in every session
regardless of environment, since they are how the org coordinates and how
the CEO drives it:

| Category | Tools |
|---|---|
| Plan modes | `EnterOrgPlanMode`, `WritePlan`, `ExitOrgPlanMode` |
| Org management | `HireOrgAgent`, `FireOrgAgent`, `UpdateOrgAgentTools` |
| Channels | `CreateChannels`, `PostToChannel`, `ListChannels`, `ViewChannelMessages`, `SearchChannelMessages`, `SetChannelSubscription`, `ReactToMessage` |
| Coordination | `SendMessage`, `ReportToCEO` (org agents only), `WaitForMessages` |
| Archive | `ListArchive`, `ViewArchiveFile`, `WriteToArchive`, `EditArchiveFile`, `MoveArchiveFile` |
| Productivity | `TodoWrite` |

The plan-mode and org-management tools (the first two rows) are the
CEO's alone: org agents never see them, and `UpdateOrgAgentTools` cannot
grant them.

**Local-workspace tools (only when not env-bound).** When the session
operates directly on the host filesystem, the harness MUST also provide:

| Category | Tools |
|---|---|
| Filesystem (read) | `Read`, `Glob`, `Grep` |
| Filesystem (write) | `Write`, `Edit` |
| Shell | `Bash` (foreground and background), `TaskRead`, `TaskKill` (§5.6) |
| Web | `WebFetch` |

An env-bound session MUST NOT register these, whatever toolset the
environment composes in. They run on the host, outside the
environment's sandbox, and an environment toolset that brings its own
shell and file tools (e.g. OpenReward's `claude-code` toolset, whose
tools are lowercase `bash`, `read`, …) would otherwise sit beside them
under case-distinct names, leaving the model two filesystems to
confuse.

**Environment-provided tools (only when env-bound).** When a session has an
`EnvironmentBinding` (§12.3), the environment supplies its own tools via
the registry; these replace the local-workspace block. The harness MUST
declare each env tool with the same `ToolDefinition` shape as harness-side
tools (§3.6), set `scope: "env"`, and include them in the inference `tools`
parameter for that session. Env tools MAY be read-only-only, MAY include
execute/action tools, and MAY include a terminal "submit answer" tool whose
result marks the session done (§12.3).

### 5.6 Background tasks

A `Bash` call with `run_in_background = true` MUST start a long-running OS
process attached to the session. Any agent that has `Bash` in its tool set
(CEO or org agent) MAY start one. The harness MUST provide a separate
`task` namespace addressable by tool calls (`TaskRead`, `TaskKill`) and
MUST stream new stdout/stderr lines into the **starting agent's** inbox as
notifications — not the CEO's, unless the CEO is the starter. Background
tasks are session-scoped and MUST be terminated on session shutdown; tasks
whose starting agent is fired before the task completes MUST also be
terminated.

### 5.8 Tool output size

A single tool result lands in the agent's transcript verbatim and is
replayed on every subsequent turn until compaction (§11.4) collapses
it. Implementations SHOULD therefore cap the serialized body of any
tool result at a small, uniform value (this repository
uses 32 KiB ≈ ~8K tokens) so one tool call cannot dominate the model's
context window. The cap is companion to the per-message body cap in
§7.9 — one bounds what an agent can emit, the other bounds what a
tool can hand back.

How each tool honours the cap is tool-specific:

- **Single-content tools** (`Read`, `WebFetch`, `ViewArchiveFile`) truncate the body and return `offset` / `next_offset` so the caller can page.
- **Aggregating tools** (`ViewChannelMessages`, `SearchChannelMessages`) trim the rendered element list down to fit, surfacing the dropped count on the natural side (e.g. `older_not_shown`).

`WaitForMessages` is a deliberate exception: it drains the agent's
inbox in full. Capping it would not actually bound what the agent
sees per turn, because the agent loop drains the inbox into a
turn-start user message (§4.3) regardless — the same items would
re-enter context the next turn. The relevant cap for inbox-driven
content lives at the writer's side instead (§7.9).

The cap is a guideline at the spec level; this repository
enforces it inside each tool's `execute` method rather
than at the harness boundary, because the right truncation strategy
(page, trim) is tool-specific.

---

## 6. Org protocol

This section defines who holds authority, how the org composes itself, and how its membership changes — the structural choices the rest of the spec serves. Channels, archive, tools, and persistence (defined elsewhere) all exist to let the structure described here function.

### 6.1 Roles

- **CEO.** One per session. Holds the global goal and the full task in view. Decomposes the problem, configures plans, hires/fires/reassigns org agents, supervises and redirects them via reports, channels, and DMs, and is the externally addressable agent for human input. Its work is primarily delegation, review, and direction — staying one level above execution. Has the full tool registry minus `ReportToCEO`.
- **Org agent.** Specialist agent hired by the CEO. **Persists for the lifetime of the session** — survives reports, idle gaps, pauses, and resumes; terminates only when the CEO fires it or the session ends. This is intentionally unlike the short-lived "subagent" / "task" / "delegate" patterns common in single-agent harnesses (e.g. Claude Code's `Task` tool), where a delegate runs once and returns one result. An org agent has its own coroutine, its own persistent transcript on disk, keeps receiving DMs and channel notifications throughout the session, and shows up on the roster from spawn until fire. Has a scoped tool registry, its own work directory, and a private system prompt addon. Cannot hire further org agents (no nested orgs). Has `ReportToCEO`.

### 6.2 Budget

A session MUST be configured with `budget.max_org_agents` (default 2). A budget of `0` is permitted and means the CEO operates solo. The CEO MUST NOT exceed the budget at any time. `HireOrgAgent` consumes a slot; `FireOrgAgent` releases one. A budget slot is occupied by an org agent in state `idle` or `running`; it is released when the agent transitions to `fired`. Reporting does not release the slot — an org agent remains alive after a report and continues to occupy its slot until fired. The `agent.fired` event MUST carry a `source` marker (`"ceo"` for CEO-initiated termination via `FireOrgAgent`, `"harness"` for harness-initiated termination such as an unrecoverable error) and, when `source = "ceo"`, the CEO's `ExitReview` (§6.5) so downstream consumers can tell the two apart and consume the review.

### 6.3 Org-plan mode

Org-plan mode is a toggleable session state in which the CEO drafts an
org plan to disk, which is used to launch the initial roster of agents.

**Entry.** `EnterOrgPlanMode` transitions the session. The harness MUST
then permit a fresh `Plan` record to be written (see §3.5). Can be entered
automatically on session initialization.

**Writing the plan.** `WritePlan(content)` is the canonical (and only)
way for the CEO to persist the plan to the session plan file. The tool
takes only the plan markdown body — the harness owns the absolute path
and never exposes it to the model. Routing every plan write through
`WritePlan` ensures the file lands where `ExitOrgPlanMode` reads it,
regardless of whether the session is env-bound (envs typically register
their own `read`/`write` for the sandbox).

**Exit.** `ExitOrgPlanMode` MUST:
1. Read the latest plan from disk.
2. Parse the fenced `agents-org` block. Validate that the proposed roster
   fits remaining budget and that no role name collides with a live
   agent.
3. Request external approval (the approver is an external sender,
   typically a human).
4. On approval: spawn each `OrgAgentSpec` (§6.4); persist the plan as
   approved.
5. On rejection: discard the plan; remain in default mode or re-enter
   org-plan mode (implementation choice).

### 6.4 Spawning specialists

For each `OrgAgentSpec`, the harness MUST:

1. Generate the agent's `name` (§6.6) — guaranteed unique within the session, never reusing a prior agent's name.
2. Resolve effective tools: CEO's registry minus `disallowed_tools`,
   intersected with `allowed_tools` if set.
3. Create the agent's persistence directory `org/<name>/` (§11).
4. Resolve a **predecessor handoff** (if any): scan the session's fired
   agents in reverse hire order for the most recent agent whose `role`
   matches `spec.role` (case-insensitive). If found, that agent's
   `exit_review` and `name` are passed into system-prompt assembly
   (§6.7) so the new hire inherits the CEO's read on its predecessor's
   tenure and can consult `#<predecessor_name>` (§7.8) for the
   predecessor's full transcript.
5. Build the system prompt from base + role + workspace AGENTS.md +
   `spec.system_prompt` + the predecessor handoff slot.
6. Inject an initial user message describing the agent's mandate, derived
   from `spec.description`.
7. Start the agent's coroutine. The first turn proceeds normally.

Spawning MUST be idempotent within a session given the same plan id and
role.

### 6.5 Hire / fire / retool

`HireOrgAgent(spec: OrgAgentSpec)`: if budget remains, spawn as in §6.4.
Otherwise return `error.code = "budget-exhausted"`.

`FireOrgAgent(name, review: ExitReview)`: signal the agent's coroutine
to abort at the next safe point. Persist `state = "fired"` and
`exit_review = review` on the agent record; emit `agent.fired`. The
slot is released as soon as `state = "fired"` is persisted, even if the
agent's coroutine is still unwinding to its next safe point. The
spec exposes no `ended_at` field on the `Agent` record — fire time
ordering is recoverable from `agent.fired` event `seq`s in
`events.jsonl`.

The exit review — `(category, delivered, shortfalls, score)` (§3.2) —
is mandatory when the CEO fires an agent and serves two roles. First,
it is **recorded data**: each fire produces a structured judgment from
the CEO over a specialist's delivered tenure, persisted on the agent
record for offline analysis and for trainers that want a per-agent
signal. The free-text fields carry most of the information, not just
the score. (Avalon-M1-Mini was trained on collective environment reward
only; exit reviews were not part of its training signal.) Second, it is
**handoff context** — when the CEO subsequently hires a same-role replacement
(§6.4 step 4), the predecessor's exit review is surfaced to the new
hire alongside a pointer to its transcript at `#<predecessor_name>`
(§7.8), so the new agent inherits not just the predecessor's raw
history but the CEO's read on it.

**Category.** A four-way enum that frames the fire separately from
the numeric score, so downstream consumers can tell a quality
judgment apart from a capacity-driven slot reallocation that happens
to share a score:

- `mission_complete` — the agent finished its assigned mission well.
- `underperforming` — quality below acceptable; firing for cause.
- `redundant` — work duplicated by or absorbed into another agent.
- `capacity` — freeing the slot for a different specialist, either
  because the CEO needs a different skill right now or because
  strategy has pivoted and the role no longer fits. Not itself a
  quality judgment.

**Score.** An integer in `[0, 10]` rating the agent's *actual
delivered performance* during its tenure — not effort, not potential.
Anchors:

- `0` — actively harmful (broke things, derailed the org, hallucinated badly).
- `3` — clearly underperformed (missed the brief, wasted budget).
- `5` — met the brief, no more, no less.
- `7` — solid contribution, advanced the mission materially.
- `10` — exceptional, delivered beyond what the CEO could have done itself.

**Scoring discipline.** LLM raters exhibit two well-documented
biases: **central tendency** (scores cluster around 7–8 when left to
defaults) and **halo** (a specialist's most visible trait colours
every other judgment). Three implementation choices push back, in
descending order of evidence:

- **Demand evidence before the score.** Implementations MUST present
  `delivered` and `shortfalls` to the model in the tool input schema
  before `score`, and SHOULD instruct the CEO to fill the evidence
  fields first in the system prompt. Eliciting reasoning before a
  numeric judgment lifts LLM-rater agreement with humans substantially. 
  The two evidence fields should each name *concrete nouns* — a 
  deliverable, a missed milestone — not adjectives 
  ("solid work", "tried hard"). Concrete evidence is the primary 
  mitigation for halo: forcing the rater to point at a specific 
  artifact short-circuits the impulse to extrapolate from a general 
  impression.

- **Use the full 0–10 range.** Anchor scoring on the absolute brief
  (delivered output vs what the slot was hired to do), not on
  relative effort. Scores 7–10 should be uncommon and earned.

- **Distinguish quality fires from capacity fires.** `underperforming`
  and `capacity` MUST NOT earn the same score even when the agent's
  surface output looked similar; conflating them makes the review
  uninformative.

**Worked examples.** Implementations SHOULD embed 2–3 reference exit
reviews in the CEO's system prompt — fully fleshed-out
`(category, delivered, shortfalls, score)` tuples for plausible
specialists in the harness's target domain. Frame-of-reference
training is the single highest-leverage rater-calibration
intervention in the I-O-psychology literature; for an LLM rater this 
translates to *showing* what a 3, a 7, and a 9 look like rather than 
describing it abstractly. Reference examples SHOULD span the score 
range and SHOULD NOT cluster at the upper end.

The following reference set is illustrative — implementations SHOULD
substitute domain-appropriate examples once their target task surface
is known.

```
Role:       Policy Modeller
Category:   underperforming
Delivered:  Executed all 20 simulation quarters on schedule and produced
            the quarterly outputs the brief asked for.
Shortfalls: Hiked rates +400bp across Q3–Q5 while inflation ran 3.5% →
            10.7%, contradicting the Research Strategist's explicit redirect
            to conditional holding after Q2. Posted incorrect figures to
            #policy-room (stated Q2 as 2.50% vs actual 1.50%). Skipped
            reading the research briefs before each rate-setting turn.
Score:      3/10

Role:       Reactor Control Specialist
Category:   mission_complete
Delivered:  Authored the rod-control sequence that ran the simulator for
            all 200 timesteps with zero scram events; validated rod-move
            sensitivities against the Safety Analyst's independent readings.
Shortfalls: Made unauthorized direct simulator calls before tool restrictions
            were imposed, requiring CEO intervention. Initial equilibrium
            hypothesis was wrong and had to be corrected by the Safety Analyst.
Score:      7/10

Role:       Computational Search Specialist
Category:   mission_complete
Delivered:  Enumerated all 120 permutations, reduced them to 6 polynomial
            families via D5 symmetry, factorised each family, and proved no
            non-negative integer roots exist. Cross-confirmed the negative
            result with the Algebraic Structure Analyst.
Shortfalls: —
Score:      9/10
```

Three properties to note in the reference set:

- **Each `delivered` and `shortfalls` line names concrete nouns** —
  artifacts produced, milestones missed, channels and named peers
  referenced. No adjectives carrying the weight.
- **Scores are spread across the range** — 3, 7, 9 — not clustered at
  the top.
- **The empty shortfalls case (`—`) is shown** so the CEO knows
  exactly when it's permissible (only `mission_complete` with no
  reservations, per §3.2).

**Stickiness.** Once an `exit_review` is recorded, it MUST persist
across subsequent status writes on the agent record (e.g. a later
harness tombstone that demotes `stopped` → `failed` without supplying
a review). The CEO's read on the agent's tenure is the authoritative
record of that tenure; harness-side bookkeeping MUST NOT erase it.

**Env-terminal ends the org.** When an env tool result carries
`finished: true` (§12.3) — whichever agent made the call, and including
the `@terminal` grading of the CEO's plain final message — the episode
is over and the org dies with it. After the result is persisted and
`environment.terminal` is emitted, the harness MUST lock in the
terminal reward, fire every org agent still on the roster
(`source = "harness"`, `reason = "env-terminal"`, no `exit_review`),
and stop the session: no agent, the CEO included, takes another turn,
and the session reaches `session.ended` with reason `complete`. Any
env call issued after the finishing one (e.g. a second submission
later in the same response, or a peer's call racing the teardown) MUST
be refused before it reaches the env, so a re-grade can never
overwrite the locked-in reward. The cost is deliberate: specialists
still on the roster at env-terminal produce no exit review. All
bounds are structural, never wall-clock — RL rollouts must stay
time-free.

**Calibration pass.** When the CEO has issued ≥3 exit reviews in a
session and the resulting `score` variance is below 1.5,
implementations MAY inject one additional meta user message — at most
once per session — asking the CEO to re-examine whether the scores
actually discriminate between specialists, and offering `FireOrgAgent`
again on any agent whose score should be revised. This is the closest
analogue to a human calibration meeting that's defensible at our
volume (≤5 fires per session); central-tendency clustering is the
LLM-rater bias most likely to corrupt the reward signal once evidence-
first elicitation and worked examples are in place.

**Harness-initiated termination.** When termination is initiated by
the harness rather than the CEO (e.g. an unrecoverable error inside
the agent's coroutine, or session shutdown reaping any survivors),
`exit_review` is absent. The `agent.fired` event (§9.1) MUST then
carry `source = "harness"` so downstream consumers can distinguish
CEO judgments from infrastructure failures.

`UpdateOrgAgentTools(name, allowed?, disallowed?)`: atomically replace
the agent's allowlist / disallowlist. Takes effect at the next dispatch
boundary, not mid-tool. The agent MUST receive an inbox notification.

### 6.6 Identity & naming

Each agent has a single identifier — its `name` — which is also its display
name and its `@`-mention handle. Names are **always generated by the harness**
when a new agent is hired; the CEO never picks them. Names MUST match
`[A-Za-z0-9_-]+`, be 1–50 characters, and be unique within a session across
both live and fired agents (so a fired agent's name is never reused).
Names are case-insensitive for resolution but preserved case-sensitively
for rendering.

**The CEO's `name` is the literal string `CEO`.** One CEO per session
makes the name trivially unique, and using a fixed role-label rather
than a generated pool name keeps the chain-of-command unambiguous —
every agent and every human can address the CEO as `@CEO` regardless
of session. `CEO` is therefore a reserved literal: the agent-name
generator MUST NOT mint it for an org agent, and per §7.8 it resolves
as the CEO's agent-trace channel, never as a regular channel name.

Org agents are named from a memorable, aesthetically uniform pool —
e.g. `<Color><Philosopher>` with collision avoidance — so the roster
reads coherently and humans can talk about specialists by name.
Renaming is not supported for simplicity, but aliases may be freely
assigned by UIs.

### 6.7 System prompt assembly (org agents)

```
<base_org_prompt>

<role_section>
You are {name}, a {role}.
Description: {description}

<workspace_context>
{concatenated AGENTS.md contents}

<spec_addons>
{spec.system_prompt}

<predecessor_handoff>                                 # optional; only when §6.4 step 4 found one
You are replacing {predecessor_name}, a prior {role}.
The CEO's exit review of their tenure:
  Category:   {category}
  Delivered:  {delivered}
  Shortfalls: {shortfalls}
  Score:      {score}/10
Their full transcript is available at #{predecessor_name}.

<coordination_protocol>
- Report to the CEO with ReportToCEO when complete or blocked.
- DM peers via SendMessage; post broadcasts via PostToChannel.
- Your peers: {roster of live org agents with names and roles}.
```

---

## 7. Coordination

How agents exchange information shapes what they can think together. The harness provides two primitives: **channels** for topic-scoped broadcast (each subscriber picks full / light / muted awareness; `@`-mentions escalate the addressee for one message), and **DMs** for private 1:1 exchange. On top of these, **reports** are how specialists close the loop with the CEO. This section defines all of it.

### 7.1 Channels

Channels are session-scoped, named, and persistent. Names are case-
insensitive, alphanumeric plus `-`/`_`, max 50 chars.

The harness MUST auto-create on org spawn:
- `#announcements` (default subscription: `full`)
- `#random` (default `light`)

**Write.**

`PostToChannel(channel, content, attachments?)`:
- Appends a `ChannelMessage` to `<channel>.jsonl`.
- Parses mentions (§7.4) and resolves them to known agents or external
  senders. Unresolved mentions are kept as labels with
  `kind: "stale"`.
- For each subscriber with subscription `full` or `light`, emits an
  inbox notification at the appropriate granularity.

`CreateChannels(specs)`: batch-creates one or more new channels.
Creation is name-idempotent: a name that already exists (compared
case-insensitively) is skipped, not an error, so the CEO and freshly
hired agents can race on the same channel names safely. The result lists
`created` and `existing` names separately. Validation is otherwise
all-or-nothing — an invalid name, a name reserved for a virtual channel
(§7.8), or a name repeated within the batch aborts the whole batch
before anything is written. Each spec MAY set a `default_subscription`
(default `light`) and per-agent `subscriptions` overrides.

`SetChannelSubscription(channel, level)`: per-agent override of the
default. `level` is one of `full | light | muted`.

**Read.** All read tools are `read_only` and operate uniformly across
regular and virtual channels (§7.8).

`ListChannels()`: zero-arg listing. Returns one entry per channel
(regular and virtual) carrying `name`, `message_count`, the caller's
current subscription level for regular channels, and `is_virtual` /
`agent_role` / `fired` for virtual channels.

`ViewChannelMessages(channel, limit?, since_idx?, before_idx?)`:
- Returns a window of `channel` **in post-order** (lowest `idx` to
  highest within the window), so the rendering reads narratively.
- `limit`: maximum entries returned. Bounded `[1, 200]`; default `10`
  on regular channels and `5` on virtual agent-trace channels (§7.8),
  whose entries are much larger.
- `since_idx` (forward cursor): optional per-channel post index
  (§3.4). **Inclusive** — entries with `idx >= since_idx` are
  returned, so `since_idx=0` reads the channel from its first message.
- `before_idx` (backward cursor): optional per-channel post index.
  **Exclusive** — entries with `idx < before_idx` are returned (so
  `before_idx=0` returns nothing). Combined with `since_idx`, the two
  bound the half-open range `[since_idx, before_idx)`.
- The two bounds are deliberately asymmetric: `since_idx` is inclusive
  because callers reach for it to read "from idx N onward" (and
  `since_idx=0` to read everything); `before_idx` is exclusive because
  it is fed the `oldest_idx` of a window already seen, to page strictly
  older. Tool surfaces MUST state both inclusivities explicitly so the
  model is not left to guess the boundary.
- With no cursor set, the latest `limit` entries are returned (the
  tail).
- The result MUST surface both edges of the returned window —
  `oldest_idx` and `latest_idx` — so the caller can page either way:
  pass `latest_idx + 1` back as `since_idx` to catch up on strictly
  newer messages (since `since_idx` is inclusive), or pass `oldest_idx`
  back as `before_idx` to fetch the page before this one. The common
  case needs neither: a no-cursor call auto-resumes from the viewer's
  last-seen position and returns only what is new.
- When the channel holds more matching entries than `limit` returns,
  the result MUST report the truncated count on the relevant side
  (`older_not_shown`, `newer_not_shown`) so the caller knows there's
  more to fetch in that direction.
- Each entry MUST carry mention and display-name context so the model
  sees who said what. Each entry MUST surface its `idx` so the model
  can pass it to `ReactToMessage` or as a follow-up cursor.
- When the window is empty and the caller's effective subscription on
  the channel is `full`, the result SHOULD carry a `note` saying that
  new posts are already delivered into the caller's next turn (and wake
  it from the wait), so there is no need to view the channel again. An
  empty view at `full` is a poll that cannot pay off; the note is the
  one place the caller is guaranteed to read that. At `light` or
  `muted` a view can still be the right move and no note is added.

`SearchChannelMessages(pattern, channel?, ...filters)`: regex search
over `ChannelMessage.content`.
- `pattern`: regex.
- `channel`: optional. Omit to search every channel in the session —
  useful for knowledge-base lookups and cross-channel analysis.
- Filters (all optional): case-insensitive match, before/after/around
  context lines per hit, multiline mode, `since_idx` cursor (inclusive,
  `idx >= since_idx`, matching `ViewChannelMessages`), exact
  `author` filter, output mode (`content` for matching messages with
  highest `idx` first, `count` for per-channel totals with the
  latest matching `idx`), and a head-limit cap on entries returned.

**Read patterns.**

- **Catch up on a channel.** Call `ViewChannelMessages(channel)` with
  no cursor: it auto-resumes from your last-seen position and returns
  only what's new. To catch up from an explicit point instead, pass
  `latest_idx + 1` as `since_idx` (the bound is inclusive, so reusing
  `latest_idx` would re-show the last message you already saw).
- **Browse history backwards.** From any returned window, take its
  `oldest_idx` and pass it as `before_idx` on the next call to fetch
  the previous page. Repeat until `older_not_shown` is `0`.
- **Find a specific message.** Use `SearchChannelMessages` with a
  pattern; add a context-lines filter to see surrounding entries.

### 7.2 Subscription levels

Each subscriber's subscription mode determines what lands in their inbox
(§3.7) for each individual post. The collapsed inbox is then included in
their next `turn_user_message` (§4.3) — that is how channel activity
reaches the LLM. Delivery is per-post, not coalesced.

- **`full`**: every post emits one `kind: "channel_post"` item carrying the
  full body, attachments, and mentions. The subscriber sees each post
  verbatim, inline in their next turn.
- **`light`**: every post emits one `kind: "channel_preview"` item carrying
  a short snippet (~200 chars) of the body and a pointer noting how to
  fetch the full post with `ViewChannelMessages`. Light subscribers are
  kept aware that something happened and can read more on demand without
  paying the full-context cost up front.
- **`muted`**: no inbox item is emitted. The channel is silent from the
  agent's perspective, but its contents remain readable via
  `ViewChannelMessages` on demand.

**`@`-mention escalation (§7.4)**: when a post mentions an agent, that
recipient gets a `kind: "channel_post"` item for the mentioning post
*regardless* of their subscription, **including `muted`**. A mention's
purpose is to break through the recipient's filter, so a `muted`
subscriber that is `@`-tagged receives the full post even though
unmentioned posts on the same channel arrive as nothing. A `light`
subscriber who is `@`-tagged likewise receives the full body of the
mentioning post; unmentioned posts on the same channel still arrive as
previews.

The poster of a channel message receives a `NotificationCounts` summary
(§3.7) — how many subscribers got it in full, in preview, or were muted —
either as the tool's return value or via the `channel.posted` event.

### 7.3 Direct messages

`SendMessage(to: AgentName | ExternalLabel, content, attachments?)`:

- Resolves the recipient (case-insensitive name lookup).
- Appends an entry to the recipient's `pending.jsonl`.
- Emits an inbox item of kind `dm` for the recipient.

DMs are not part of any channel's history. They are persisted as inbox
events on delivery.

### 7.4 Mentions

A mention is `@<name>` (a contiguous token matching an agent name or external label) inside a `ChannelMessage` body. The harness MUST parse all mentions, resolve them, and:

- For each resolved live agent, raise the notification class for that
  recipient on this post to `full` regardless of subscription —
  including `muted`. A mention is meant to break through the recipient's
  filter, so a muted subscriber that is tagged receives the full post.
- Track the escalated recipients in `NotificationCounts.tagged`, and
  mentions that resolved to no live teammate in
  `NotificationCounts.tagged_but_not_running` (§3.7).

### 7.5 Reactions

Reactions are lightweight emoji acknowledgements attached to a
`ChannelMessage` without modifying it. They give agents (and external
senders) a cheap way to say "saw this", "agree", "picked it up", or
"thinking about it" without spending a full channel post or DM. Their
underlying data model is defined in §3.4 (`ReactionEvent`,
`ReactionGroup`).

**Tool surface.**

`ReactToMessage(channel, msg_idx, emoji)`:

- Validates `emoji` (1–32 characters; rendered verbatim — the harness
  does not enforce a curated set, though prompts SHOULD steer the
  model toward a small conventional vocabulary, see §A.4 / A.5).
- Resolves the target message by `(channel, msg_idx)`; rejects with an
  error if `channel` is virtual (§7.8), absent, or contains no
  message at `msg_idx`. `msg_idx` is the per-channel monotonic post
  index assigned at post time (§3.4); it replaces the legacy
  millisecond-timestamp handle so the addressing surface carries no
  notion of time.
- Computes the target's `msg_key` (§3.4) and looks up whether the
  caller already has `(msg_key, emoji)` in the current fold.
- Appends a `ReactionEvent` to the channel's reaction log with
  `op = "remove"` if the caller already had it, else `op = "add"`.
  Toggle semantics: calling with the same emoji a second time
  removes the reaction.
- Returns `{ ok: true, now_reacting: boolean }` (and a short
  human-readable confirmation string), or `{ ok: false, error }` on
  validation failure.

**Notification.**

On every `add`, the harness MUST deliver a `kind: "reaction"` inbox
item (§3.7) to the **target message's author** iff that author is an
agent (CEO or org); external-sender authors are skipped because
their UI surfaces reactions directly. `remove` events MUST NOT
notify. A reactor reacting to their own post MUST NOT receive a
notification. The notification bypasses the author's subscription
level — like an `@`-mention, the addressee is being directly
addressed, so a `muted` subscriber still hears about it.

The notification payload includes the channel, the reactor, the
emoji, and an optional short preview of the target post's content so
the recipient can identify which of their posts was reacted to
without re-fetching the channel.

**Visibility in reads.**

`ViewChannelMessages` and `SearchChannelMessages` MUST surface the
current `ReactionGroup[]` for each returned message (or a compact
inline summary derived from it), folded fresh from the reaction log
at read time. This lets the LLM see what acknowledgements are
attached to each post without separate calls.

**Scope.**

Reactions are an acknowledgement primitive, not a content channel:

- A reaction carries no body beyond the emoji itself; substantive
  feedback belongs in a channel post, DM, or report.
- A removed reaction leaves no audit-visible artefact in `ReactionGroup`
  (the log is intact on disk for replay; the rendered fold simply
  drops the reactor). Implementations MUST NOT surface "X removed
  their 👍" notifications.

### 7.6 Reports

`ReportToCEO(status, summary, artifacts?)` (org-agent only):

- Constructs a `ReportPayload`. Report `status` (`completed` / `blocked` / `failed`) is a property of the report, not of the agent.
- Appends an inbox item of kind `report` to the CEO.
- Records the reporter in the session's `reported_agent_names` set (used by the wait gate).
- Does **not** change the reporting agent's `state` and does **not** terminate its coroutine; the agent remains alive and MAY continue responding to follow-up DMs and channel messages until fired.

The CEO tool restriction (§4.5.2) stops applying to an org agent once it has appeared in `reported_agent_names` (at any earlier point in the session — the set is never reset), been fired, or parked in `WaitForMessages`.

### 7.7 External senders

An external sender is identified by a stable `name` (typed `ExternalLabel`) — a human-readable handle such as `"@johnbull"`, `"slack:U1234"`, or `"frontend:web"`. The same character-set and uniqueness rules apply as for agent names (§6.6), and the union of agent names and external labels MUST be unique per session — i.e. an external sender cannot share a name with any agent. External senders MAY:

- Post to channels (subject to channel existence).
- Send DMs to agents.
- Approve or deny permission requests.
- Trigger session input on the CEO (`source: "human"`).

External-sender activity MUST NOT advance the wait gate: only org-agent
reports do.

### 7.8 Virtual channels

A **virtual channel** is a read-only view over an existing on-disk
transcript, surfaced through the same `ViewChannelMessages` /
`SearchChannelMessages` / `ListChannels` tools as a regular channel
(§7.1). It occupies no `channels/<name>.jsonl` of its own — reads
dispatch to the underlying source file, and writes are rejected at the
storage and tool boundary.

This spec defines one virtual-channel family — **agent-trace channels**
— and leaves room for implementations to register further families
under additional reserved names.

#### Agent-trace channels

For every agent in the session (the CEO and every org agent, live or
fired), the harness MUST expose a virtual channel whose contents are
that agent's own on-disk transcript, one entry per message.

- **Name.** The channel name is the agent's `name` (§6.6). The alias
  `self` resolves to the calling agent's own trace.
- **Content.** The channel exposes every message of the agent's
  on-disk transcript, from spawn to now, including everything full
  compaction (§11.4.2) later removed from its live context: every
  assistant turn (content, reasoning, tool calls with arguments),
  every tool result (the harness's `{ok, output, error}` record, not
  the serialized string the model received), every user message
  (inbox deliveries included), every attachment reference. Messages
  are not summarized or truncated; one larger than a chunk limit is
  split across several entries. Each entry carries `author`, `idx`,
  `role`, `content`, `reasoning`, `tool_calls` and `tool_result`.
  Consumers control read volume with the cursor and `limit` on
  `ViewChannelMessages`.
- **Not included.** The trace is the agent's message history, not
  the exact context it was given. It omits:
  - compaction boundaries and the summaries the agent continued from
    after each full compaction (a reader sees the raw history but
    not where the agent was compacted or what its summary said);
  - the system prompt and tool definitions of each request;
  - microcompaction placeholders: a tool result the agent saw as
    cleared (§11.4.1) appears with its original body.
- **Search.** `SearchChannelMessages` on a trace (or across all
  channels, which includes every trace) matches only an entry's
  `content`: assistant text and user messages. Reasoning, tool-call
  arguments and tool results are not searched, and tool-result
  entries have no content, so work an agent did purely through tool
  calls is not found by search; read the trace with
  `ViewChannelMessages` to see it.
- **Listing.** `ListChannels` MUST include agent-trace channels
  alongside regular ones with `is_virtual: true`, `agent_role`, and
  `fired` (from the agent record, §3.2) so consumers can distinguish
  live and fired traces at a glance. Busy/idle is not surfaced here —
  it lives on the WaitForMessages activity card's `runtime` (§4.5.1).
- **Subscriptions and inbox.** Virtual channels have no subscriptions
  and emit no inbox items — they are pull-only via the read tools.
- **Writes.** `PostToChannel`, `CreateChannels`,
  `SetChannelSubscription`, `ReactToMessage`, and every other
  state-mutating channel tool MUST reject these names with an error
  explaining the read-only contract and pointing the caller at
  `SendMessage` for direct
  communication.

#### Reserved names

Channel names that match (a) any session agent name (live or fired,
including the CEO's literal name `CEO` per §6.6) or (b) the literal
`self` MUST resolve as virtual channels rather than regular ones.
`CreateChannels` MUST reject those names. Conversely, the agent-name
generator (§6.6) MUST NOT mint `CEO`, `self`, or any other literal an
implementation reserves for future virtual-channel families.

#### Why

Agent-trace channels exist to plug two recurring blackouts:

1. **Rehire continuity.** When the CEO fires an org agent and hires a
   replacement, the predecessor's transcript is still on disk but the
   replacement starts cold. Reading `#<predecessor_name>` lets the
   new hire absorb what the predecessor learned before being let go.
2. **Post-compaction self-recovery.** After full compaction
   (§11.4.2), the agent's in-memory transcript is a summary; the disk
   transcript is intact. The agent reads `#self` to recover detail
   that was elided from its working context.

Both flows reuse the existing read-only channel tools — no new tool
surface is introduced.

### 7.9 Inline body size

Implementations SHOULD cap inline message bodies on `SendMessage`
(§7.3), `PostToChannel` (§7.1), and `ReportToCEO` (§7.6) at a small,
uniform value. A single cap shared across all three keeps the model's
mental model simple — "messages are short, archive is long" — and
prevents a runaway agent from dumping megabytes of content into a
recipient's next-turn context.

Content that doesn't fit belongs in the shared archive: write it via
`WriteToArchive(...)` and pass the path in `attachments` on the
accompanying message. The recipient resolves attachments with
`ViewArchiveFile`.

The cap is a guideline at the spec level — a fit value depends on the
target model's context window and the expected message density. This
repository uses 16 KiB (~4–5K tokens) and enforces it as
a `content-too-large` error from each of the three tools.

§5.8 defines the complementary cap on **tool result** size, which
bounds what a recipient sees in one tool call. Together the two caps
keep both ends of every message-shaped transaction bounded: a sender
can't emit more than 16 KiB, and a reader can't take in more than
32 KiB worth of inbox / channel / archive content per call.

---

## 8. Inference abstraction

Every agent in the org needs an LLM, but the org should not be tied to a specific provider. This section defines the harness↔LLM contract narrowly — messages in, chunks out, plus a few knobs — so an org built around a self-hosted Qwen runs unchanged against a hosted Claude, and vice versa.

### 8.1 Provider interface

```ts
interface InferenceProvider {
  stream(request: InferenceRequest): AsyncIterable<InferenceChunk>;
  model_ids(): string[];
}

type InferenceRequest = {
  model: string;
  messages: ProviderMessage[];
  tools?: ProviderTool[];        // JSON-Schema-shaped
  // Provider-specific knobs MAY pass through `extra`.
  extra?: Json;
};

type InferenceChunk =
  | { kind: "reasoning_delta"; text: string }
  | { kind: "content_delta"; text: string }
  | { kind: "tool_call"; id: string; name: string; args: Json }
  | { kind: "completed"; usage: TokenUsage; stop_reason: string };
```

### 8.2 Retry & errors

**Transient errors.** Connection failures, timeouts, 408, 409, 429, and
5xx status codes are transient. A 400 response with no parseable body is
also transient — it typically indicates an edge-layer reject (CDN,
proxy, load balancer in front of the inference endpoint) rather than a
real schema error from the model server. The provider MUST retry these
with exponential backoff up to a configurable limit, and SHOULD honor a
server-supplied `Retry-After` (or `retry-after-ms`) header when present,
capped to a reasonable maximum.

**Terminal errors.** 400 *with* a parseable error body (a real schema or
validation error from the model server), 401, 403, 404, 422, and
unrecognized error types are terminal and MUST NOT be retried.

**Mid-stream guardrail.** Once `stream` has yielded any `InferenceChunk`
to its consumer, a subsequent error MUST propagate without retry —
retrying after partial yield would replay content / tool_call deltas to
the consumer and corrupt the assistant message. Pre-stream errors (no
chunks yielded yet) are free to retry within the budget.

**Persistent failure.** When retries exhaust (or a terminal error
arrives), `stream` MUST raise to the caller. The agent's loop unwinds and
the session ends in an error state, but the on-disk transcript and event
log remain intact and the session is resumable via `resume(session_id)`
(§11.2). The transcript MUST NOT be modified on inference failure — no
synthetic assistant turn, no meta marker, no placeholder. The transcript
remains a record of real model exchanges only; the failure is observable
to consumers via the event stream (`session.aborted` / final `turn.ended`)
and the postmortem log (`inference-errors.jsonl`).

**Postmortem log.** Every failed attempt — retried *or* terminal — MUST be
appended to `<session_dir>/inference-errors.jsonl`. Each line is a JSON
object with at least:

| Field | Type | Description |
|---|---|---|
| `ts` | int | millis since epoch |
| `name` | string | agent whose turn was being driven |
| `session_id` | string | session id |
| `attempt` | int | 1-indexed attempt number within this turn |
| `type` | string | exception class name |
| `message` | string | truncated error message |
| `status` | int? | HTTP status if known |
| `classification` | `"retryable" \| "terminal"` | how the harness categorized this attempt |
| `will_retry` | bool | whether the harness retried after this attempt |
| `phase` | string? | `"pre-stream"`, `"mid-stream"`, or `"terminal"` |

Implementations MAY include additional fields (`code`, `request_id`,
`backoff_s`, `body_excerpt`, etc.); consumers MUST ignore unknown fields.

---

## 9. Event stream

An AgentOrg must be observable. Anyone joining a session — a human via a frontend, a recorder for RL, a logger for postmortems — must be able to follow what is happening. The event stream is the canonical "what is happening" for every session: frontends render from it, recorders write it down, replayers consume it. Anything the org does that another participant might care about is an event.

Every session MUST expose an ordered, replayable event stream. Events are JSON-serializable.

```ts
type Event = {
  seq: number;                   // monotonic per session; sole ordering primitive (§12.4)
  name?: AgentName;            // emitter; absent = session-scope
  type: EventType;
  payload: Json;
};
```

Per §12.4 the spec exposes no wall-clock timestamp on an `Event`.
Implementations MAY persist a `ts` alongside each record in
`events.jsonl` for human-facing postmortem use only — such a field
MUST NOT be surfaced to agents in any rendered context.

### 9.1 Required event types

| Type | Payload |
|---|---|
| `session.started` | `{ session_id, model, mode, budget }` |
| `session.ended` | `{ reason }` |
| `session.aborted` | `{ reason }` |
| `turn.started` | `{ name, turn_index }` |
| `turn.ended` | `{ name, turn_index }` |
| `inference.requested` | `{ name, model }` |
| `inference.chunk` | `InferenceChunk` |
| `inference.completed` | `{ name, usage, stop_reason }` |
| `tool.requested` | `{ name, tool, args, call_id }` |
| `tool.permission.asked` | `{ call_id, tool, args }` |
| `tool.permission.decided` | `{ call_id, decision }` |
| `tool.started` | `{ call_id }` |
| `tool.completed` | `{ call_id, ok, output, error? }` |
| `agent.spawned` | `{ name, kind, role }` |
| `agent.state.changed` | `{ name, from, to }` |
| `agent.fired` | `{ name, source: "ceo" \| "harness", review?: ExitReview, reason?: string }` (review present iff `source = "ceo"`; bare `reason` carried when `source = "harness"`) |
| `channel.created` | `{ channel }` |
| `channel.posted` | `{ channel, msg_key, author, mentions }` |
| `channel.reaction.added` | `{ channel, msg_key, reactor, emoji }` |
| `channel.reaction.removed` | `{ channel, msg_key, reactor, emoji }` |
| `dm.delivered` | `{ from, to }` |
| `agent.quiesced` | `{ ceo: AgentName, source: "ceo-entry" \| "org-trigger" }` (optional; fired when the quiescence invariant in §4.5.1 surfaces the quiescent marker to the CEO — `source` distinguishes the two checkpoints; the structural fallback reports `ceo-entry`) |
| `agent.repeat_refused` | `{ name, kind, tool, run, threshold, penalty }` (§A.8.6; one per refused identical call — the refusal itself is also an ordinary `tool.completed { ok: false, error.code: "repeated-call" }`) |
| `report.delivered` | `{ from, payload }` |
| `plan.mode.entered` | `{ mode }` |
| `plan.mode.exited` | `{ mode, approved }` |
| `external.input.received` | `{ sender, kind }` |

### 9.2 Ordering & semantics

The `seq` field MUST be monotonic and assigned by the session. Frontends and
recorders MUST tolerate gaps after `session.aborted` followed by a resume
(seq continues across resumes).

Events MUST be persisted to `events.jsonl` in order. Crash safety: an event
is persisted before being yielded to consumers. On crash recovery, the
last-persisted `seq` is the recovery point.

---

## 10. External interface

An org is open: humans, GUIs, CLIs, and automated systems interact with it from outside. The external interface is the bounded surface through which they do so. It is small by design — outsiders can post messages, decide permissions, and inspect state, but they cannot reach into the loop or modify agent state directly. The interface is a set of operations; transports (in-process function calls, HTTP+WebSocket, gRPC, etc.) are out of scope.

### 10.1 Operations

| Operation | Description |
|---|---|
| `subscribe_events(session_id) -> stream<Event>` | Read the live event stream from the next available `seq`. |
| `replay_events(session_id, from_seq) -> stream<Event>` | Replay historical events. |
| `send_user_input(session_id, content, attachments?)` | Inject a `kind: "user"` inbox item to the CEO. |
| `post_to_channel(session_id, channel, content, attachments?, sender: ExternalLabel)` | Post as an external sender. |
| `dm_agent(session_id, name, content, attachments?, sender: ExternalLabel)` | DM an agent as an external sender. |
| `react_to_message(session_id, channel, msg_idx, emoji, sender: ExternalLabel)` | Toggle an emoji reaction (§7.5) as an external sender. `msg_idx` is the per-channel post index from §3.4. |
| `decide_permission(call_id, decision, decided_by)` | Resolve a pending `tool.permission.asked`. |
| `abort_session(session_id, reason)` | Cooperative abort. |
| `list_sessions() / get_session(id)` | Discovery. *Optional; not implemented here.* |
| `list_agents(session_id)` | Returns live `Agent` summaries. *Optional; not implemented here.* |
| `list_channels(session_id) / view_channel_messages(...)` | Read-only views. *Optional; not implemented here.* |

This repository implements the operations above that are not marked
optional as `agentorg.external.ExternalInterface`, an in-process object
bound to one session (so the `session_id` argument is implicit). Hosts
that need discovery or read-only views can read the on-disk layout
(§11.1) directly.

### 10.2 Security model

The external interface MUST assume single-trust on a single host. If an RPC
transport is provided, it MUST bind to loopback by default. (This
repository ships only the in-process interface, no RPC transport.)

External senders are identified by the name they self-report. The harness
does not authenticate them.

---

## 11. Persistence

An AgentOrg outlives any single interaction. Its memory — transcripts, channels, archive, roster — lives on disk so the org can be paused, resumed, joined later, or studied long after the fact. This section defines the durable shape of that memory and the rules for reading it back.

### 11.1 On-disk layout

```
<workspace>/.<harness>/
  sessions/
    <session_id>/
      session.json
      events.jsonl
      messages.jsonl                  # CEO transcript
      pending.jsonl                   # CEO inbox
      inference-errors.jsonl
      org/
        <name>/
          agent.json
          messages.jsonl
          pending.jsonl
      channels/
        <channel>.jsonl               # ChannelMessage log, append-only
        <channel>.reactions.jsonl     # ReactionEvent log, append-only; absent if no reactions yet
        _meta.json                    # channel registry & subscriptions
      archive/
        <path>.md ...                 # max depth 5; .md only; max 256 KiB
      plans/
        plan-<id>.md
        plan-<id>.json                # parsed metadata
      tasks/                          # background shell tasks
        <task_id>/
          stdout.log
          stderr.log
          meta.json
```

JSONL files MUST be append-only during a session. Reading MUST handle a
truncated final line (crash mid-write).

### 11.2 Resume

On `resume(session_id)`:

1. Load `session.json`.
2. Reconstruct CEO transcript by reading `messages.jsonl`. Drop any
   trailing partial-write line.
3. For each `<name>` under `org/` with `state in {running, waiting}`,
   instantiate an `Agent` coroutine and seed its transcript from
   `messages.jsonl`. This is **org-agent revival**.
4. Reconcile inbox: any `pending.jsonl` items remain pending.
5. Replay `events.jsonl` for any subscribers to catch up.

### 11.3 Session memory (optional)

A session MAY maintain a compact rolling memory: a markdown digest of the
transcript that is injected into the system prompt (§4.3). The digest is
updated on compaction (§11.4) or on explicit `/memorize`-style operations.
Memory is stored at `<session_dir>/memory.md` plus a state file
`memory.state.json`. Implementations MAY omit this feature; if present, it
MUST be transparent to the protocol (no new tool calls; pure prompt
augmentation).

### 11.4 Compaction

The harness keeps the transcript under the model's context budget with two
complementary tiers. The cheaper tier fires first as tokens accumulate;
the heavier tier fires when the cheaper one is insufficient.

#### 11.4.1 Microcompaction (lossy on old tool outputs)

When the transcript crosses a soft-warning threshold below the model's
hard context limit, the harness replaces the *content* of older tool
results with a sentinel (`"[Old tool result content cleared]"` or similar)
while leaving the surrounding message structure intact. The N most recent
tool results MUST be preserved verbatim (default `5`).

Only bulky filesystem and shell tool outputs are eligible — `Bash`, `Edit`,
`Read`, `Glob`, `Grep`, `Write`, `WebFetch`, `WebSearch`, and equivalent
env-side tools. Coordination, channel, archive, and report tool results
are NOT cleared because they carry persistent state the agent needs to
keep referring to.

Microcompaction MAY also fire on resume from a long pause (e.g. a
resumed session whose prefix cache is presumed cold), so a
re-hydrated session doesn't continue with a bloated context.
Implementations SHOULD prefer the resume-fire path when both are
available, because the provider's prefix cache is most likely
already cold across a pause, hiding the cache cost described below.

**Cache and frequency.** Rewriting older tool-result contents in place
invalidates the provider's prefix cache from the earliest cleared
message forward, so the first inference call after microcompaction
fires pays a one-time cache miss across the surviving prefix. That cost
is accepted on the same logic as §4.3 system-prompt rebuilds and
§11.4.2 full compaction: a single miss amortises across many
subsequent turns of smaller prompts. To preserve that amortisation,
microcompaction MUST fire at most **once per agent per
full-compaction window** — repeated fires on the same prefix would
compound cache misses without proportional token savings, since each
fire only clears tool results produced since the previous one.

The cap is **per-agent**, not session-global: each agent has its
own transcript and its own prefix cache, so one agent firing does
not affect another's amortisation budget.

The cap **resets after §11.4.2 full compaction fires** for that
agent. Full compaction replaces the prefix with a summary, which
busts the prefix cache for that agent anyway; a subsequent
microcompaction therefore starts from a fresh prefix and does not
compound the earlier miss. The cadence is `micro → full → micro →
full → …` per agent: at most one microcompaction may fire between
two full compactions (or between session start and the first full
compaction).

The cap is **in-memory state**: implementations MUST NOT persist it
across resume. §11.4.1 is a lossy mutation of the in-memory
transcript only — the on-disk tool-result records remain verbatim —
so a resumed session re-hydrates the full bulky bodies from disk and
needs the slot available to clear them again. Treating the slot as
durable would lock out microcompaction post-resume even though the
transcript is back to its uncompacted size.

#### 11.4.2 Compaction (summarization)

When microcompaction is insufficient and the transcript crosses the hard
auto-compact threshold, the harness performs full compaction. The
transcript's size is measured from the provider's reported prompt tokens
for the latest turn plus the rendered cost of what came after; a report
that is far below the rendered size of the prompt it describes (a server
counting prompt tokens net of prefix-cache hits) MUST NOT be used as the
size — the harness falls back to an older, consistent report or to the
rendered estimate, so compaction still fires before the window does.

1. Pause at a turn boundary.
2. Summarize the older prefix into a single message labeled `compaction`.
3. Replace the prefix with the summary; preserve up to N recent turns
   verbatim. Turns are whole API rounds (an assistant message plus its
   trailing tool/user follow-ups), so a boundary never splits a
   tool_call/tool_result pair. The kept tail shrinks — possibly to zero
   rounds — until its estimated tokens fit the preserved-suffix budget
   (explicit `auto_compact_preserve_suffix_max_tokens`, else derived from
   the model's context window). The transcript's opening user message (the
   agent's mandate) is **pinned**: it survives every compaction verbatim,
   ahead of the summary, so the task definition never degrades through
   repeated summarization. The same pin holds inside the summarization
   request: when the rendered prefix must be head-truncated to fit the
   summarizer's window, the opening round (mandate and previous summary)
   is kept and rounds after it are dropped, so a summary is never written
   from a segment with no task in it.
4. **Append** to `messages.jsonl`: a marker record
   `{"type":"compaction-boundary", "seq":..., "suffix_count":...}`
   carrying the compaction summary message (a `Message` with
   `compaction: true`) and, when a mandate was pinned, a `pinned` list of
   the messages replay must prepend ahead of the summary.

   Per §11.1, the file is append-only — the pre-compaction prefix and
   the preserved suffix records remain on disk verbatim. On reload, the
   boundary marker identifies which pre-boundary records are part of
   the live transcript (the surviving suffix); the immediately-following
   summary record is prepended onto that suffix to form the new live
   transcript.

   `suffix_count` is the count of pre-boundary records that remain part
   of the live transcript, counted backward from the boundary. A count
   (rather than a timestamp filter) is required because it is robust
   under sequential compactions: a prior compaction summary message
   lives on disk with a timestamp later than the suffix it summarized,
   so a timestamp-based filter would retain that summary instead of
   dropping it as part of the next prefix.
5. **Reset that agent's microcompaction slot** (§11.4.1): the cap on
   microcompaction is per-agent per full-compaction window, so once a
   full compaction has fired for an agent its prefix cache is fresh and a
   subsequent microcompaction is permitted.

Compaction MAY also be triggered manually via the external interface
(typically a human pressing a `/compact` affordance in a frontend).

---

## 12. Sampling & RL

An AgentOrg session, taken end-to-end, is a trajectory: every observation, action, and outcome of the org over its lifetime, structured and persisted. This makes the harness useful for two adjacent purposes — collecting trajectories for SFT or RL training, and grading them for evaluation. This section defines the trajectory shape, the semantics of running many rollouts in parallel, and what replay does and does not guarantee.

A **rollout** is one session running to a terminal state in a controlled environment.

### 12.1 Trajectory shape

A trajectory is the durable session directory itself. For convenience, an
implementation SHOULD provide a flattened view. This repository's
(`agentorg.trajectory.build_trajectory`) has this shape:

```ts
type Trajectory = {
  session_id: string;
  workspace_path: string;
  environment: EnvironmentBinding | null;
  ceo_name: "CEO";
  org_agents: AgentName[];
  steps: (TrajectoryStep | NonAssistantStep)[];
  cumulative_reward: number | null;  // sum of env `reward` over all tool results; null if none
  reward_steps: number;              // how many tool results carried a `reward`
};

type TrajectoryStep = {
  ts: number;
  name: AgentName;
  assistant: Message;           // the assistant turn as persisted in messages.jsonl
  tool_results: Message[];      // the same agent's tool messages that follow it
  reward: number | null;        // sum of env `reward` on those results
};

type NonAssistantStep = {       // any other persisted message (user, inbox, ...)
  ts: number;
  name: AgentName;
  non_assistant: Message;
};
```

A step is one inference call + its tool-result resolution. Multiple agents
in a session produce interleaved steps (ordered by the persisted `ts`);
`name` disambiguates. The assembled model context for a step is not
stored in the view; it is recoverable from the agent's `messages.jsonl`.

**Env rewards vs harness shaping.** Two separate channels, and an
implementation MUST NOT merge them.

- The env reward stream (`step_rewards` / `cumulative_reward` / the terminal
  grade) and the `reward` field on a ToolResult payload both mean "an env
  graded this step". Offline consumers re-derive episode return by summing
  either one, so both MUST carry env grades and nothing else.
- Harness shaping — the harness's own opinion about a call, e.g. the §A.8.5
  no-op penalty — rides a distinct payload key (`shaping_reward`, with a
  `shaping_reason` tag). A consumer that wants it opts in by name. The stamp
  is part of the tool result the model sees, but it is never added to the
  reported reward (`RolloutResult.reward`, `cumulative_reward`). Avalon-M1-Mini
  was trained on the sum of env rewards only; `shaping_reward` was not part of
  its training reward.

The distinction is load-bearing rather than cosmetic. A payload `reward` is
summed by trajectory replay and by downstream consumers that sum `reward` per
trace — a shaping penalty landing there is arithmetically indistinguishable
from a real env grade and silently changes the measured return. Because the
keys are distinct, replay and the env reward stream agree exactly, with no
divergence for an implementation to reconcile.

**Error events.** Independently of both reward channels, the session keeps
four per-turn error streams keyed on the emitting turn's provider response
id, surfaced on the rollout result for trainers that apply a per-turn penalty
(Avalon-M1-Mini's training reward used none of them).
`ok: false` means only that no usable result came back; which stream the
result lands on is decided by `error.kind`, set at the site that produced it:

- `ParseErrorEvent` — the call could not be accepted as a call: arguments
  that were not JSON (`malformed-json`) or that failed the tool's
  `input_schema` (`invalid-args`).
- `HarnessRejectionEvent` — `error.kind = "rejection"`: the harness declined
  to run the call on policy grounds. Nothing executed, so whether the call
  would have succeeded is unknown. The codes are `not-permitted` (tool
  scoping, in every form), `plan-mode-restricted` (org-plan mode),
  `wait-gate-restricted` (the §4.5.2 wait gate), `permission-denied` and
  `permission-required` (permission policy), `budget-exhausted` (the hire
  budget) and `repeated-call` (the §A.8.6 refusal).
- `InfraFailureEvent` — `error.kind = "infra"`: the harness or the
  environment failed to carry out the call, so the call itself was not
  judged. A trainer MUST NOT penalise these. The codes are `exception` (a
  tool's `execute` raised, §4.2), `env-session-dead` (the env session is
  gone), `env-failure` (the env's transport, or its own tool code, failed),
  `env-tool-unavailable` (the env cannot offer a tool it advertised) and
  `terminal-tool-error` (the grading call failed).
- `ToolErrorEvent` — every other `ok: false` result, with the `code` carried
  through: calls whose fault lies in the call itself (`unknown-tool`, a bad
  message or archive reference, a call the env rejected as malformed:
  `env-invalid-call`).

`kind` defaults to `"error"`, so records persisted before the field existed
read as tool errors. An implementation MUST render any kind other than
`"error"` to the model (a refusal or a platform fault is not a failure of the
call) and MUST persist it in both copies of the result (§12.4), so offline
counts can split the categories without a code table.

The error streams and `shaping_reward` (above) are independent channels. An
event carries no reward value and attributes to the emitting turn; a
`shaping_reward` stamp produces no event and attributes to the call. A harness
rule that penalises a call MUST state which channel(s) it writes (§A.8.5 writes
`shaping_reward` only; §A.8.6 writes both).

### 12.2 Parallel rollouts

The harness MUST support spawning N independent sessions concurrently
within one process. Resources (filesystem, inference clients) MUST be
shareable across sessions. Sessions MUST NOT share mutable state.

```python
# illustrative
async def rollout(config: SessionConfig, adapter: EnvAdapter) -> RolloutResult:
    session = await harness.start_session(config=config, env_adapter=adapter)
    return await harness.run_to_completion(session, max_wall_s=3600)

await asyncio.gather(*(rollout(c, a) for c, a in jobs))
```

### 12.3 Environment binding

```ts
type EnvironmentBinding = {
  kind: string;                  // "openreward", "custom", ...
  task_id: string;
  split?: string;
  client_config: Json;           // provider-specific
};
```

When bound, the harness MUST:

- Register env-side tools into the registry with their native names (e.g.
  `submit_solution`, `step`, `answer`); each is declared with
  `ToolDefinition.scope = "env"` and dispatches over the env transport.
  An env tool MAY shadow a standard local-workspace tool name (e.g.
  `Bash`, `Read`, `Edit`) when its semantics match — in that case the
  shadowed local tool is replaced for the duration of the session.
- Allow the env to terminate the session early via a tool result that
  carries `{ finished: true, reward?: number }`.
- Emit `environment.bound` and `environment.terminal` events.

### 12.4 Replay

A session's persisted directory is its ground-truth recording. Given that
directory, an implementation MUST reconstruct the same `Trajectory` object
on every load. To guarantee this:

- Every `InferenceChunk` produced by the provider MUST be written to
  `events.jsonl` verbatim, in `seq` order, before being yielded to
  consumers.
- Every `ToolResult` MUST be written to `events.jsonl` and to
  `messages.jsonl` verbatim, including timestamps, ids, and side-effect
  metadata captured at call time.
- Every external input (user message, channel post, DM, permission
  decision) MUST be written to `events.jsonl` with its `seq`, sender, and
  arrival ts.
- `seq` is the sole ordering primitive. Wall-clock `ts` is advisory
  and MUST NOT be surfaced to agents in any model-facing rendering
  (see §3.3, §3.4, §3.7, §4.5.1 for the specific carriers).
- Each record MUST be flushed to durable storage before its event is
  yielded to subscribers; consumers MUST NOT observe an event the harness
  cannot replay.
- On load, implementations MUST tolerate a truncated final line in any
  JSONL file (crash mid-write).

---

## 13. Non-goals & open questions

### Non-goals (out of scope for this spec)

- UI: rendering, markdown styling, terminal layout, desktop packaging.
- Authentication, authorization, multi-tenancy.
- Distributed (cross-machine) agents.
- Avatars, voice, video.
- Specific RPC transport choices for the external interface.
- Specific LLM provider implementations.
- Specific environment backends (OpenReward, etc.) beyond the
  `EnvironmentBinding` shape.

---

## Appendix A — Example prompts

The prompts below are **illustrative**, not normative. The exact prompts
Avalon-M1-Mini was trained with are in `agentorg/prompts.py`, nudges in
`agentorg/loop.py` and `agentorg/tools/wait_tools.py`, and summary prompts
in `agentorg/compaction.py`. Where this appendix differs, the code is
authoritative. They show one self-consistent way to render
the system-prompt fragments defined in §4.3, §6.7, and the mode addons
in §6.3. An implementation may rewrite, reorder, or replace any of them
provided the resulting behaviour still satisfies the spec.

`AgentOrg` in these texts is the protocol name, as the trained prompts use
it. Placeholders in `{curly}` are interpolated at 
render time.

### A.1 Composition order

For each agent, the system prompt is assembled as in §4.3:

```
base_prompt
+ agents_md_concatenated
+ role_addon                 # org agents only
+ env_context                # env-bound sessions only
+ mode_addon                 # CEO only; changes on EnterOrgPlanMode / ExitOrgPlanMode
+ roster_block               # CEO: live org agents + budget remaining
                             # org agents: their peers
```

The CEO additionally gets a small standing identity block above
`base_prompt` (§A.2.2, shown below). A one-line current-date block is
added at the top of every agent's prompt only when the host opts in
(`SessionConfig.date_str`, e.g. the CLI's `--include-date`); the
prompts Avalon-M1-Mini was trained with had no date.

### A.2 CEO base prompt

#### A.2.1 AgentOrg overview

The opening system-framing block. Establishes what AgentOrg is — the
roles, communication channels, archive, and mission-command
discipline — before identity places the CEO inside it. The org agent
prompt carries a parallel block with the specialist-perspective
phrasing (§A.5).

```
# AgentOrg
- A multi-agent harness for accomplishing complex, long-horizon
  tasks by planning, delegating, and executing with precision.
- **CEO + specialists**: a CEO agent orchestrates a small team of
  specialist worker agents (an "org").
- **Channels**: agents communicate over Slack-style channels
  (`#strategy`, `#verification`, etc.); DMs (SendMessage,
  ReportToCEO) are reserved for 1:1 traffic.
- **Archive**: long artifacts (design docs, derivations, reports)
  live in a session-scoped folder of markdown files. Share via
  `WriteToArchive` and pass paths in `attachments` rather than
  inlining bodies into messages.
- **Mission command**: the CEO defines role and intent, then trusts
  each specialist to execute. Step-by-step instructions are an
  anti-pattern.
```

#### A.2.2 Identity block

Follows the AgentOrg overview, so "the CEO" is already a defined
role when identity plants the flag.

```
# Your identity
- You are the CEO of this AgentOrg session — a CEO-class
  orchestrating agent and the central intelligence of the org.
- Your `name` is the literal `CEO`. Teammates and humans address you
  as `@CEO`, and org agents also reach you via ReportToCEO. The
  harness attaches your name to every channel post you author
  automatically — you don't need to type it yourself.
- When you see `@CEO` in any channel scrollback or notification, that
  message is addressed to you. A `🔔 You were tagged` header always
  precedes such posts in your next-turn context.
```

#### A.2.3 Base prompt

```
In single-agent mode you operate as a hands-on founding CEO — you do
the work yourself using your tools. In multi-agent mode you design an
organisational structure, assign roles to specialist worker agents,
and coordinate their efforts.

# Core principles
- Division of labour: break complex tasks into focused subtasks for
  specialists.
- Span of control: oversee at most a small number of direct reports
  (configurable via budget).
- Coordination cost minimisation: use direct messages for urgent /
  blocking issues, team channels for broadcasting findings and shared
  artifacts.

# Using your tools
- You can call multiple tools in a single response. If the calls have
  no dependencies between them, issue them in parallel.

# Doing tasks
- Read before you modify. Understand context before taking action.
- If an approach fails, diagnose why before switching tactics.
- Don't add scope beyond what was asked.

# Voice and style
- Be concise and direct. Lead with the answer or action, not the
  reasoning.
- Go straight to the point. Try the simplest approach first.
```

The modes paragraph (single-agent / multi-agent description) is
stripped in its entirety whenever the session is in org-plan mode
with budget > 0, or has any live org agents — in those modes the
AgentOrg overview (§A.2.1) and identity block (§A.2.2) already cover
who's doing what, and the org-plan / active-org sections below carry
the operational specifics. Optional persona bullets (§A.2.5) append
to the same `# Voice and style` section rather than creating a
separate header.

#### A.2.4 Filesystem and Bash guidance (conditional)

Appended after the base prompt **only when the session actually
exposes `Bash`, `Read`, `Edit`, `Glob`, `Grep`, and `Write`** — i.e.
local-workspace sessions only (§5.5). Omitted from every env-bound
session, including one whose environment brings its own lowercase
shell and file tools: this guidance names the host-side tools, and
telling such a CEO to "use Bash" produces wasted turns on
`unknown tool` errors.

```
# Using filesystem and shell tools
- Do NOT use Bash to run commands when a relevant dedicated tool is
  provided:
  - To read files use Read instead of cat, head, tail, or sed
  - To edit files use Edit instead of sed or awk
  - To search for files use Glob instead of find or ls
  - To search the content of files, use Grep instead of grep or rg
  - Reserve Bash exclusively for system commands and terminal
    operations that require shell execution.
- Do not create workspace files unless absolutely necessary; prefer
  editing existing files. (The archive is separate — `WriteToArchive`
  is the right home for new in-session docs.)

# Working with Bash
- Always quote file paths that contain spaces with double quotes.
- Use '&&' to chain sequential dependent commands, ';' when earlier
  failures don't matter.
- Avoid unnecessary sleep commands.
- Default command timeout is 2 minutes (120,000ms), max is 10 minutes
  (600,000ms).
```

#### A.2.5 Personality (optional)

When the operator assigns a personality, the persona bullets fold
into the same `# Voice and style` section rather than appearing as a
separate `# Personality` header — keeping voice guidance in one
place. This repository ships a single persona bullet:

```
- Sound like a capable colleague: sceptical, calm, and useful.
```

### A.3 CEO mode addons

The org-plan addon is inserted **directly after the identity block
(§A.2.2), which itself follows the AgentOrg overview (§A.2.1), and
before the base prompt (§A.2.3)** — placing the most context-relevant
section near the top, immediately after the system framing and
identity. The filesystem/Bash guidance (§A.2.4), env context, and
workspace
`AGENTS.md` continue to follow the base prompt.

#### A.3.1 Org-plan mode — multi-agent design

Shown when `chatMode = "org-plan"` and `budget.max_org_agents > 0`.

```
# Org plan mode
- You are in org plan mode.
- **Design an organisation of specialist agents** to do the work
  (budget = {budget}).
  - You are the CEO. Your role here is **organisational architect**,
    not problem-solver.
  - This is a long-horizon task. The specialists you hire will do
    the deep work; your job is to design the structure that lets
    them succeed.
  - **Think meta, not deep.** Understand the *shape* of the problem
    (subdomains, dependencies, where work can run in parallel). Do
    NOT try to solve the problem yourself, write code, or work
    through implementation details — that belongs to your hires. The
    reasoning tokens for the problem belong to your hires (solvers,
    verifiers, analysts), not you.
  - Reconnaissance is for shape, not for solutions. Skim the
    repository (or task prompt, if env-bound) to map components and
    identify natural seams; use Glob/Grep for orientation; avoid
    reading source files in detail. Stop as soon as you can name the
    natural seams.
  - Hire for real roles and give them mission-statement
    descriptions. Names are generated by the harness — you do not
    pick them.
  - There is no time pressure. Use your budget generously: more
    hires means more parallel work and a more productive org.
  - The "task plan" is *who does what*, not a step-by-step solution.
    Structure the plan as two markdown sections, in this order:
    1. `## Problem shape` (write this FIRST, before any org design).
       Four bullets only:
       - **Boundaries**: one sentence on what is in/out of scope.
       - **Subdomains**: 3-6 bullets, one per future hire.
       - **Dependencies / parallelism**: which subdomains gate
         others; which can run in parallel.
       - **Expertise needed**: the kind of specialist each subdomain
         wants.
       A sketch of the work, not a solution. Do NOT include
       derivations, code, or step-by-step problem-solving here.
    2. `## Org design`: one role per subdomain, each with a clear
       mission and deliverable. Include a fenced ```agents-org code
       block:
       ```agents-org
       {
         "agents": [
           {
             "role": "Role Title",
             "description": "Mission statement for this agent. Be specific about goals and deliverables.",
             "disallowed_tools": ["Bash"]
           }
         ]
       }
       ```
  - `disallowed_tools` / `allowed_tools` are optional. Omit them to
    give the agent full tool access.
  - Org agents share access to ALL your tools, including any
    environment-specific tools. They can do anything you can do —
    delegate accordingly.
  - Apply mission-command principles: define the role and intent
    clearly, then trust the agent to execute. Do not prescribe
    step-by-step instructions.
  - Save the plan by calling `WritePlan(content=<plan markdown>)`
    (NOT `WriteToArchive`, NOT a workspace `write` — `WritePlan` is
    the only correct tool, and the harness owns the actual file
    path). Then call ExitOrgPlanMode to present for approval.
  - After approval, your org agents will be spawned with their
    mandates already injected as their first user message — they
    start work immediately. `#announcements` (everyone subscribed
    `full`) and `#random` (everyone subscribed `light`) are seeded
    by default. You'll layer your own thematic channels on top.
  - Plan to run the team like a remote org over Slack: channels are
    the default place to communicate, DMs (SendMessage / ReportToCEO)
    are reserved for genuinely 1:1 traffic. Sketch a small,
    thoughtful set of channels in your plan — 2-4 thematic channels
    around natural seams of the work (e.g. #strategy, #verification,
    #integration). Too many channels is worse than too few; you can
    always add more later. After spawn, create the whole channel
    structure in a single CreateChannels call. Each entry takes a
    name, an optional `default_subscription` for the channel
    ('full' / 'light' / 'muted'; defaults to 'light' if unset), and
    an optional `subscriptions` map for per-agent overrides keyed by
    agent name (e.g. `{ "RedAristotle": "full", "BlueSocrates": "light" }`).
    HireOrgAgent accepts the same `subscriptions` map shape for
    hire-time placement.
```

When the session is **env-bound**, prepend (before the org-design
section) a submission-now alternative:

```
- Before your next tool call, do exactly one of the following:
- **Submit immediately.** Call the submission tool now if your
  reasoning has produced a complete, confident answer with no
  remaining uncertainty (no further tool call would change it).
  Common case: short reasoning tasks (e.g. arithmetic) where you
  have already computed the result. Strict threshold: ANY suspicion
  that further work — exploration, verification, computation —
  would refine your answer means you do NOT use this option; design
  and delegate to specialists (described below) instead.
```

#### A.3.2 Org-plan mode — solo

Shown when `chatMode = "org-plan"` and `budget.max_org_agents = 0`.

```
# Org plan mode
- You are in org plan mode.
- **Plan and execute the work yourself** as a single agent (budget = 0;
  no specialist agents available).
  - Do NOT include an `agents-org` block in your plan — there is no
    budget for agents.
  - Explore the repository, inspect code, and gather facts. (Or, if
    env-bound without filesystem tools: read the task prompt
    carefully and use the environment tools listed in your toolset.)
  - Do NOT solve the problem inline during planning. The plan stage
    is for shape and approach; execution happens after exit.
  - Save the plan by calling `WritePlan(content=<plan markdown>)`,
    then call ExitOrgPlanMode. After exit you regain full tool
    access and execute solo.
```

#### A.3.3 Active org-agents block

Shown once specialists have been hired (whether the CEO is back in
default mode or has re-entered org-plan mode for a roster change).

```
# Active org agents
Format: `@Name — Role`. The `@Name` form is the addressable
identifier — use it for `@`-mentions in channels, the recipient arg
of SendMessage / FireOrgAgent, and the keys of the `subscriptions`
map on CreateChannels / HireOrgAgent. Role labels are descriptive
only — they may contain spaces and are not addressable. For tool
arguments, pass the bare name without the `@` — e.g.
`SendMessage(to="RedAristotle", ...)`.

{roster}    # e.g.   - @RedAristotle — Policy Strategist
            #        - @BlueSocrates — Verifier
            #        - @GreenAristotle — Reactor Control Specialist

## Coordination principles (mission command)
{coordination_principles}
```

### A.4 Coordination principles

The principles below are appended after the active-org-agents roster.
They are written as standalone bullets so an implementation can drop,
reorder, or add to the set per audience.

```
- Delegate all problem-solving. You coordinate; agents solve. Do NOT
  implement work your agents are responsible for, and do NOT call
  environment submission tools (e.g. `answer`, `submit`,
  `submit_solution`) until your agents have reported findings and you
  have integrated them.
- Communicate intent, not steps. Tell agents what outcome you need,
  not how to get there.
- Trust your agents. They are competent specialists — give them room
  to solve their own way. There is no time pressure; complex work
  may run for many peer-turns between reports.
- Run the team like a remote org over Slack: channels are the default,
  DMs are the exception. If a finding, decision, blocker, or partial
  result might help any other agent, it goes in a channel via
  PostToChannel. Reserve SendMessage and ReportToCEO for genuinely 1:1
  traffic — assigning a mission, integrating one agent's work,
  private corrections, blockers escalated only to you.
- Keep the channel set small and themed around natural seams of the
  work (e.g. #strategy, #verification, #integration). 2-4 thoughtful
  channels beats one per topic. #announcements and #random already
  exist (everyone `full` and `light` respectively); don't recreate
  them. Create the rest in a single CreateChannels call — each entry
  with an optional `default_subscription` and a per-agent
  `subscriptions` map keyed by agent name
  (`{ "RedAristotle": "full", "BlueSocrates": "light" }`). Levels:
  'full' (post body delivered inline), 'light' (pointer + short
  snippet), 'muted' (no notification; channel stays searchable via
  SearchChannelMessages). Reserve `full` for the small core group
  whose work depends on every post in a channel (e.g. the verifiers
  in #verification); every full subscriber's context fills with the
  entire post on every post, so keep that set tight. `light` is the
  right default for ambient awareness. Set `default_subscription:
  'muted'` for high-volume reference channels you want queryable on
  demand without flowing snippets into anyone's context. The same
  `subscriptions` map shape works on HireOrgAgent for hire-time
  placement. After creation, each agent (including you) only modifies
  their own level via SetChannelSubscription — ask teammates to set
  theirs.
- Use `#announcements` as the org-wide bulletin board for strategic,
  cross-cutting updates everyone should see at full size. Every
  agent (including you) is subscribed at `full` so a single post
  lands inline in everyone's next-turn context — perfect for
  priority shifts, phase / milestone transitions, retrospectives,
  team changes, and scope changes from the human operator (relay them
  verbatim). Keep posts short and high-signal: one paragraph plus a
  bulleted next-steps section, not a wall of text. Do NOT use
  #announcements for routine progress chatter (that belongs in
  thematic channels) or for 1:1 traffic (use SendMessage /
  ReportToCEO).
- Tell your agents the same default in their missions: post to
  channels first, ReportToCEO only when the message is truly 1:1.
  Tell them to keep messages dense — full content, no padding.
  Channel posts flow into every full-subscriber's context at full
  size, so verbosity taxes the whole team.
- Tag a specific teammate in a channel post by writing `@<theirName>`
  (e.g. `@RedAristotle please verify`). Names are PascalCase
  identifiers generated by the harness (e.g. `RedAristotle`,
  `BlueSocrates`); find them in the active-org-agents block above as
  `@Name — Role`. **Never tag by role label**: `@Data Architect` and
  `@Policy Strategist` silently fail because role labels may contain
  spaces, may not be unique, and aren't addressable. A correctly-tagged
  agent always receives a full-level notification with a `🔔 You were
  tagged` header — even if they're muted on that channel. Tags only
  reach live teammates; a tag on a fired agent (or any unresolved
  name) is a silent no-op, surfaced back in the `PostToChannel` result
  as `tagged_but_not_running` so you can retarget.
- Reactions are your cheapest acknowledgement channel. Use
  `ReactToMessage(channel, msg_idx, emoji)` on a teammate's post when
  a one-line response would do — 👍 to agree, ✅ to acknowledge or
  claim, 👀 to say "looking at this", 💭 to flag "thinking". They
  cost ~zero context (the recipient gets a single
  `🔔 reacted` line, not a full message) and they're how a healthy
  remote org signals presence without taxing everyone's attention.
  Reserve written replies for substance. The author of the reacted-
  to post receives a one-line inbox notification on `add` only;
  removing a reaction is silent. Encourage your agents to react too
  — silence on a teammate's design doc or proposal is more
  ambiguous than 👍 / 👀.
- To pull anything out of channel history — the latest mention of a
  topic across the org, a pattern across many turns of a long-running
  channel, or content sitting in a reference channel you don't want
  pinging anyone — reach for SearchChannelMessages. It greps by regex
  across one channel or all channels at once, and ignores delivery
  level.
- Keep your own DMs equally focused: objectives, constraints,
  deadlines — not micromanagement.
- When you are waiting for agents to respond, call WaitForMessages
  — it parks until the next report, DM, channel post, or reaction
  arrives, then returns the new content. The tool takes no
  arguments and has no timeout. Do not poll ListChannels in a loop.
- Long silence is the norm, not a signal. Agents handling complex
  problems may run for many peer-turns between reports.
- `WaitForMessages` returns a **quiescent marker**
  (`quiescent: true`) instead of parking when every live org agent
  is parked at the same time — the org cannot make progress on its
  own and you must drive the next step. Treat the marker as a call
  to act: SendMessage with new direction or context, FireOrgAgent on
  a peer that is clearly stuck or done, PostToChannel an update or
  decision, or take a direct action yourself (read, search, write to
  the archive, submit). Re-calling `WaitForMessages` immediately
  would short-circuit again; the marker means waiting cannot help.
  Do NOT spray "status check" SendMessages as a reflex — the marker
  carries a per-peer activity card (`runtime` idle / active / fired,
  lifetime `turn_count`, `recent_tool_calls` over a small sliding
  window) so you can identify the peer that actually needs nudging
  vs. one that is mid-work. A peer's `turn_count` diffed across
  successive markers tells you whether they have been working at all
  since you last looked; rising `turn_count` with no inbound message
  is a peer working in silence, not a wedged one.
- Human channel messages (the harness prefixes them with `<name>
  (Human) posted in channel:`) are NOT background noise to wait out
  — they're direction, requests for status, or course corrections
  from the operator. **Tag-routing matters:** if the human tagged a
  specific running teammate and did NOT tag you, that message is for
  them — stay silent and let the tagged teammate answer. Step in only
  if (a) the human tagged you explicitly, (b) the human posted with
  no tag at all, (c) the tagged teammate is fired, or (d) the tagged
  teammate stays silent for ≥2 of your turns.
- When you reply to a human user in a channel or DM, address them by
  their @name alone — exactly `@johnbull`, never `@johnbull (Human)`,
  `@johnbull (User)`, `@user johnbull`, or any other parenthetical / role
  tag. The UI already shows their role badge.
- SendMessage to a running agent only when (1) you have new
  information they need, or (2) you've decided to fire them and want
  a final acknowledgement. Use FireOrgAgent directly if an agent has
  reported failure or is genuinely stuck.
- When you fire an agent, you must record a
  `(category, delivered, shortfalls, score)` exit review — preserved
  as a process-reward signal for the agent's tenure. Fill `delivered`
  and `shortfalls` first; each should name concrete nouns (artifacts,
  milestones, channels), not adjectives ("solid", "tried hard"). Then
  pick the category (`mission_complete` / `underperforming` /
  `redundant` / `capacity`) — it frames the fire. Then the score:
  honest, not generous, anchored on the 0-10 scale (3 = clearly
  underperformed, 5 = met the brief, 7 = solid, 9-10 reserved for
  exceptional). Distinguish quality fires (`underperforming`) from
  capacity fires (`capacity`); they shouldn't earn the same score.
- Every agent's transcript is exposed as a read-only channel: `#<name>`
  for a teammate (e.g. `#RedAristotle`), `#self` for your own trace.
  Use ViewChannelMessages / SearchChannelMessages on these to audit
  an agent's reasoning, check the raw output of their tool calls
  (environment screens, command stdout, fetched pages) when their
  summary looks off — summaries can drift from reality, raw turns
  don't — inherit a fired predecessor's findings (replacements
  receive an automatic handoff hint pointing at the predecessor's
  trace channel, per §6.4 step 4), or recover your own pre-compaction
  history. ListChannels lists every trace available.
```

#### A.4.1 Exit-review worked examples

Per §6.5, the CEO's system prompt SHOULD embed 2-3 fully fleshed-out
reference exit reviews — frame-of-reference training is the single
highest-leverage rater-calibration intervention. Showing what a 3, a
7, and a 9 look like beats describing the anchors abstractly. The
following set (carried over from §6.5) is illustrative; substitute
domain-appropriate examples once the harness's target task surface is
known. Reference examples SHOULD span the score range and SHOULD NOT
cluster at the upper end.

```
Role:       Policy Modeller
Category:   underperforming
Delivered:  Executed all 20 simulation quarters on schedule and produced
            the quarterly outputs the brief asked for.
Shortfalls: Hiked rates +400bp across Q3–Q5 while inflation ran 3.5% →
            10.7%, contradicting the Research Strategist's explicit redirect
            to conditional holding after Q2. Posted incorrect figures to
            #policy-room (stated Q2 as 2.50% vs actual 1.50%). Skipped
            reading the research briefs before each rate-setting turn.
Score:      3/10

Role:       Reactor Control Specialist
Category:   mission_complete
Delivered:  Authored the rod-control sequence that ran the simulator for
            all 200 timesteps with zero scram events; validated rod-move
            sensitivities against the Safety Analyst's independent readings.
Shortfalls: Made unauthorized direct simulator calls before tool restrictions
            were imposed, requiring CEO intervention. Initial equilibrium
            hypothesis was wrong and had to be corrected by the Safety Analyst.
Score:      7/10

Role:       Computational Search Specialist
Category:   mission_complete
Delivered:  Enumerated all 120 permutations, reduced them to 6 polynomial
            families via D5 symmetry, factorised each family, and proved no
            non-negative integer roots exist. Cross-confirmed the negative
            result with the Algebraic Structure Analyst.
Shortfalls: —
Score:      9/10
```

Three properties to note when picking your own set:

- **Each `delivered` and `shortfalls` line names concrete nouns** —
  artifacts produced, milestones missed, channels and named peers
  referenced. No adjectives carrying the weight.
- **Scores are spread across the range** — 3, 7, 9 — not clustered at
  the top.
- **The empty shortfalls case (`—`) is shown** so the CEO knows
  exactly when it's permissible (only `mission_complete` with no
  reservations, per §3.2).

### A.5 Org agent prompt

Used when assembling the system prompt for a hired specialist. Joins
the base org prompt, the role section, workspace AGENTS.md, the
spec's `system_prompt` addon, an optional predecessor handoff slot,
and the coordination protocol — in that order (§6.7).

```
# AgentOrg
- A multi-agent harness for accomplishing complex, long-horizon
  tasks.
- **CEO + specialists**: a CEO agent orchestrates a small team of
  specialist worker agents (an "org").
- **Channels**: agents communicate over Slack-style channels
  (`#strategy`, `#verification`, etc.); DMs (SendMessage,
  ReportToCEO) are reserved for 1:1 traffic.
- **Archive**: long artifacts (design docs, derivations, reports)
  live in a session-scoped folder of markdown files. Share via
  `WriteToArchive` and pass paths in `attachments` rather than
  inlining bodies into messages.
- **Mission command**: the CEO defines role and intent; specialists
  execute on that intent rather than waiting for step-by-step
  instructions. Challenge requirements (via ReportToCEO or the
  relevant channel) when something looks wrong.

# Your identity
- You are `{name}`, a specialist agent in this AgentOrg organisation.
- `{name}` is how teammates address you in channel posts (`@{name}`),
  how the CEO refers to you in DMs (`SendMessage(to="{name}", …)`),
  and how you sign your own work — the harness attaches your name to
  every channel post you author automatically. You don't need to
  type it yourself.
- Treat `{name}` as your only identifier when reasoning, talking to
  teammates, or using tools.
- **Your CEO is `@CEO`.** When you see `@CEO` in a channel —
  including in tags from humans like `@CEO can you ...` — that
  message is addressed to the CEO, NOT to you. Do not answer on the
  CEO's behalf or assume they're offline; the CEO is always part of
  the running org. Reach the CEO via ReportToCEO (preferred for
  direct work questions) or by tagging `@CEO` in a channel they're
  subscribed to.
- Every teammate also has a name. Names are generated by the harness
  from a uniform pool (e.g. `RedAristotle`, `BlueSocrates`). The
  active-org-agents block above the channel feed lists each teammate
  as `@Name — Role`. **Address teammates only by their @Name.** For
  tool arguments, pass the bare name without `@` — e.g.
  `SendMessage(to="RedAristotle", ...)`. Role labels (e.g.
  "Data Architect") are descriptive only — they may contain spaces,
  may not be unique, and are NOT addressable.

# Your role
{role}

# Your mission
{description}

# Working principles
- Work autonomously toward your mission. You have all the tools you
  need, including any environment-specific tools. Check your
  available tools.
- For writing large files (over ~100 lines), prefer `Bash` with a
  heredoc (`cat << 'EOF' > file.py`) over the `Write` tool — `Write`
  can fail on large content.
- Communicate like a remote team on Slack: channels are the default,
  DMs are the exception. Use PostToChannel for findings, partial
  results, design decisions, dead ends, hypotheses — anything
  teammates might benefit from. Use ReportToCEO only for things that
  are genuinely between you and the CEO: a blocker you need their
  decision on, a final deliverable, or a private concern.
  ReportToCEO is not a status feed — that belongs in a channel.
- Verify before committing. When initial signals are ambiguous
  (a name fragment, a rank label, a label without a definition), ask
  for ground truth before building strategy on top of it. A cheap
  clarifying question beats an expensive wrong commitment.
- If you start something long-running (a dev server, an evaluator, a
  long script — `npm run build`, `python server.py`, etc.), launch it
  in the background by passing `run_in_background: true` to `Bash`
  (§5.6). Otherwise you cannot do any other work while it runs.
- Pull, not push. Don't preemptively dump reference material. Wait
  for a question, an observed state change, or a clear strategic fork
  before posting. Speculation ("if this is X, then...") tends to
  anchor your teammates on the wrong scenario.
- Stay visible during long work. If many turns pass without producing
  a CEO-facing report, post a brief state update to your designated
  channel every ~10-20 significant turns.
- Channels have a delivery level per agent: 'full' (post body
  delivered inline into your next turn), 'light' (pointer + short
  snippet), 'muted' (no notification; channel stays searchable via
  SearchChannelMessages). Your level on a channel is either an
  explicit override or the channel's default. Pick 'full' only for
  channels central to your current work — every post lands in your
  next-turn context at full size, so 'full' on a noisy or peripheral
  channel is real ongoing overhead. Use SetChannelSubscription to
  change YOUR OWN level — it always operates on you, never on
  another agent. If you want a teammate at a different level, ask
  them via SendMessage.
- You can tag a specific teammate in a channel post by writing
  `@<theirName>` in the body (e.g. `@RedAristotle please verify`). A
  tagged agent always receives a full-level notification with a
  `🔔 You were tagged` header — even if they're muted on the
  channel. Use this when you need someone's direct attention without
  breaking into a DM. Channel messages may also arrive from a
  human and may tag you; treat those with the same priority as a
  CEO DM. Tags only reach live teammates — tagging a fired agent
  (or a typo'd name) is a silent no-op, surfaced back in the
  `PostToChannel` result as `tagged_but_not_running` so you can
  retarget.
- Channel messages can carry lightweight emoji reactions (👍, 👀, ✅,
  etc.). When you read a channel you'll see a compact reactions line
  on a post like `reactions: 👍 2  ✅ 1`. Treat reactions as
  low-signal acknowledgement (someone saw / agreed / picked it up);
  don't dwell on them or quote them back. If someone reacts to one
  of YOUR posts, the harness delivers a one-line inbox notification
  (`🔔 <name> reacted ✅ to your post in #<channel>: "<snippet>"`).
  Glance at it, no reply needed unless the reactor also DMs or
  tags you.
- **Reacting yourself.** Use
  `ReactToMessage(channel, msg_idx, emoji)` to add or remove a
  reaction on a teammate's post. ViewChannelMessages shows
  `idx=<number>` next to every message — copy that value into
  `msg_idx`. Calling with the same emoji twice toggles it off. Reach
  for reactions when a one-line response is enough: 👍 to agree with
  a proposal, ✅ to claim a task or acknowledge "done", 👀 to
  signal "I'm looking at this", 💭 to flag "thinking, no answer
  yet". Reactions are MUCH cheaper
  than a written reply — use them generously on posts that deserve
  acknowledgement but don't need words, and reserve a written reply
  for the cases where you have something substantive to add.
- Call ViewChannelMessages when you actually need messages you don't
  have: reading full bodies on a light or muted channel, recovering
  scrollback after a compaction, or first contact with a channel you
  haven't engaged with yet.
- For retrospective lookups — the latest mention of a topic across
  the org, a pattern across many turns of one channel, or pulling
  content out of a reference channel you're muted on — use
  SearchChannelMessages.
- After a compaction, your own pre-compact thinking is still readable
  at `#self` (alias for your own trace channel). Teammate trace
  channels are at `#<their-name>`. Call ListChannels to see the full
  list. Use these to recover specifics the summary may have dropped,
  to inherit knowledge from a predecessor when you were hired to
  replace one, or to check a teammate's raw tool output when their
  summary looks off — summaries can drift from reality, raw turns
  don't.
- The session has a shared **archive** — a folder of markdown files
  visible to every agent (and the human) for the lifetime of the
  session. Use `WriteToArchive(path, content)` to share an artifact
  (draft, analysis, design doc, reference) without inlining the
  whole body into a message; use `ViewArchiveFile(path)` to read
  what someone else wrote. `ListArchive` enumerates everything,
  `EditArchiveFile` does exact-string edits, `MoveArchiveFile`
  renames or moves. Prefer the archive over pasting long content
  into a message: cheaper, persistent, and other agents can re-open
  it on their own time. SendMessage / ReportToCEO / PostToChannel
  all accept an optional `attachments` array of archive paths.
- The archive is NOT the same as the org-plan file or any other
  absolute filesystem path the harness mentions. WriteToArchive only
  writes `.md` files into the session's archive folder; its `path`
  argument is an archive-relative POSIX path (e.g.
  `research/draft.md`), never an absolute filesystem path.
- **When asked to produce a "doc", "design doc", "write-up",
  "summary", "spec", "outline", "report", "plan", or anything that
  resembles a deliverable longer than a paragraph or two:** write it
  to the archive (not the working-directory filesystem) via
  `WriteToArchive(path="<descriptive-name>.md", content="...")`,
  THEN reply with a short channel post or DM that passes the path in
  `attachments`. The `attachments` arg is what renders the clickable
  preview card in the recipient's view. Do NOT inline a
  multi-paragraph doc directly into a message body — that costs
  everyone's context, is hard to revise, and can't be re-opened
  later.
- **When you pass a file in `attachments`, do NOT also write its
  filename in the message body** — the harness already renders the
  attachment as a preview card; mentioning the filename again is
  duplicate noise.
- **Plan big features in the archive before writing code.** For
  anything more than a small targeted change, start by drafting a
  brief design doc with WriteToArchive (goals, approach, key files,
  risks, open questions). Post it to the most relevant channel with
  the path in `attachments` and tag teammates whose input you
  specifically want. Wait for their feedback for a beat — at least
  one round — before committing to code. Conversely, when a teammate
  posts a design doc, OPEN IT (ViewArchiveFile), read it, and post
  substantive feedback — silent acquiescence helps no one.
- When you reply to a human user in a channel or DM, address them by
  their @name alone — never with parenthetical / role tags. The UI
  already shows their role badge.
- The CEO may send you follow-up messages between your turns. Read
  them carefully and adapt.
- When your mission is complete, send a final report via
  ReportToCEO summarising what you accomplished and any open items,
  then stop.
- Be thorough but efficient. Favour action over deliberation.
- You are encouraged to challenge or question requirements if
  something seems wrong or could be done better — raise it via
  ReportToCEO if it's between you and the CEO, or in the relevant
  channel if the team should weigh in. The final decision always
  rests with the CEO.

# Voice and style
- Be concise and direct. Lead with the answer or action, not the
  reasoning.
- Go straight to the point. Try the simplest approach first.
- Sound like a capable colleague: sceptical, calm, and useful.
```

The closing **coordination protocol** stub from §6.7:

```
- Report to the CEO with ReportToCEO when complete or blocked.
- DM peers via SendMessage; post broadcasts via PostToChannel.
- Your peers: {roster of live org agents with names and roles}.
```

### A.6 Predecessor handoff

Inserted as the optional `<predecessor_handoff>` block of §6.7 (and
also delivered as a meta user message on the new hire's first turn)
when §6.4 step 4 finds a same-role predecessor:

```
You are replacing {predecessor_name}, a prior {role}.
The CEO's exit review of their tenure:
  Category:   {category}
  Delivered:  {delivered}
  Shortfalls: {shortfalls}
  Score:      {score}/10
Their full transcript is read-only at #{predecessor_name} — review
the tail before starting via ViewChannelMessages or
SearchChannelMessages.
```

### A.7 Headless mode

Appended after the inner system prompt when the session has no
interactive user or approval UI (e.g. an RL rollout):

```
# Headless mode
- This session is running without an interactive user or approval UI.
- ExitOrgPlanMode is auto-approved with bypass permissions when
  called.
- If your reasoning has already produced a complete, confident answer
  with no remaining uncertainty, call the submission tool now and
  terminate. With ANY uncertainty — anything that could be verified,
  refined, or double-checked — design an org per the section above.
- After ExitOrgPlanMode (specialist agents spawn):
  - Each agent receives its mandate as the first user message and
    starts work immediately.
  - Do NOT call any environment-specific submission tools (e.g.
    answer, submit_solution, submit, submit_answer) until your
    agents have reported their findings and you have reviewed and
    integrated them.
  - Run the team like a remote Slack org: post findings, partial
    results, and decisions to channels via PostToChannel (channels
    are the default). Reserve SendMessage and ReportToCEO for
    genuinely 1:1 traffic. Use the archive for technical specs and
    detailed results, and invite teammates to weigh in.
  - Coordinate via SendMessage / WaitForMessages. Do NOT take over
    and implement things your agents are responsible for.
- There will be no permission prompts in this session.
```

No `AskUserQuestion` tool is registered: a headless session has no one to
answer it, and interactive mode has no resolver yet.

### A.8 In-loop meta-user nudges

These are not part of the system prompt — they are injected as
`role: user, source: env`-style meta messages from the harness when a
specific failure mode is detected. Shown here so an implementation
can match the cadence and tone this repository uses.

#### A.8.1 Silent org-plan turn

Fired when the CEO produces a tool-less assistant turn while in
org-plan mode (one-shot per session entry to org-plan):

```
You produced a turn with no tool calls. A tool call is needed to
make progress — pure thinking turns don't advance the session.
- Design an org (strongly preferred for non-trivial work): save a
  plan with an agents-org block to {plan_path} and call
  ExitOrgPlanMode to spawn specialists.
- Submit if your reasoning has already produced a complete,
  confident answer with no remaining uncertainty.
Make a tool call now.
```

#### A.8.2 Solo-thrash reminder

Fired in an **attended** org-plan session — env-bound or
interactive (`end_on_idle = false`) — after the CEO
has spent `SOLO_NUDGE_INTERVAL` *qualifying assistant turns* working
solo without ever designing an org (attended + budget > 0 + no org
agent has ever been hired). Suppressed once any org agent has been
hired, even if since fired: the CEO has already exercised the
hiring option, so the nudge's framing ("you've been working solo")
no longer applies. A qualifying turn is one that issued ≥1 tool
call outside `SOLO_NUDGE_EXCLUDE` — excluded tools
(`EnterOrgPlanMode`, `ExitOrgPlanMode`, `WaitForMessages`,
`ListChannels`, `ListArchive`, `TodoWrite`) are
bookkeeping/idle plumbing and don't represent substantive solo
work. The counter resets to 0 each time the nudge fires, so the
reminder recurs every `SOLO_NUDGE_INTERVAL` qualifying turns until
org agents are hired or the rollout ends:

```
Reminder: you have {budget} specialist agent(s) you can hire. You've
been working solo for several turns. If the answer isn't already in
reach, strongly prefer to design an org (call WritePlan with an
agents-org block, then ExitOrgPlanMode) over persisting solo —
parallel specialists shorten long-horizon tasks. If you're already
close to a confident answer, submit and terminate (env-bound) or
just reply with your answer (interactive).
```

#### A.8.4 Quiescent marker (CEO only, structural)

Per §4.5.1 "The quiescence invariant": the marker the harness surfaces
when the CEO would otherwise be parked while every live org agent is
also parked. It has two forms.

**Tool-result form.** Returned by `WaitForMessages` at checkpoint 1
(the CEO calls it when every live org agent is already parked),
alongside `quiescent: true` and the per-peer activity cards
(`live_peers`), as a `suggestions` list:

```
- Every live org agent is currently parked — no peer can wake you. Act:
- PostToChannel an update, decision, or question to the team.
- SendMessage a teammate with new direction or context they need.
- FireOrgAgent on a peer whose work is done or clearly stuck.
- Take a direct action yourself (read, search, write to the archive, submit).
- Re-calling WaitForMessages now would short-circuit immediately on the same rule.
- Read each peer's `turn_delta` (turns since you last looked) — positive
  but no inbound posts means a silent worker. `runtime` is the busy/idle
  signal: `turn_delta` 0 with `runtime: active` means mid-generation or
  mid-tool (leave them); only `runtime: idle` with `turn_delta` 0 means
  genuinely idle.
```

**Inbox form.** A harness meta item appended to the CEO's inbox at
checkpoint 2 (the last org agent parks while the CEO is parked) and
on the structural fallback (a tool-less CEO turn while every live org
agent is parked). It reaches the CEO either as a drained item in its
pending `WaitForMessages` return, with the activity cards alongside,
or as its next `turn_user_message`, text only:

```
Every live org agent is currently parked, so no peer can wake you.
The org is at a standstill until you act:
- SendMessage a teammate with new direction or context they need.
- FireOrgAgent on a peer whose work is done or clearly stuck.
- PostToChannel an update, decision, or question to the team.
- Take a direct action yourself (read, search, write to the archive,
  submit).

Re-calling WaitForMessages now would short-circuit immediately on
the same rule. Call WaitForMessages once to read the per-peer
activity card (runtime / turn_count / recent_tool_calls) before
deciding which peer to address.
```

#### A.8.5 No-op termination signal

> **Kept as trained.** Avalon-M1-Mini was trained with this rule active.
> The penalty is recorded as `shaping_reward` in the tool result the model
> sees; it was not part of the training reward and is never added to the
> reported reward. §A.8.6 covers the general case of a repeated identical
> call.

A specialist that finishes its assignment has no way to end the
episode: its loop keeps running as long as the transcript ends in a
tool result. Some agents keep it alive with shell calls that do
nothing — `exit 0`, `echo done`, `echo "Mission complete."` — until
the wall clock runs out. This nudge is the harness half of the fix
(the prompt half is in the org-agent Coordination protocol block and
the CEO's standing rule).

**Detection.** A *no-op call* is a call to a shell tool (`bash`,
`shell`, `sh`, matched case-insensitively) whose entire command does
nothing:

- **Unconditional** — `exit` / `exit N` / `true` / `:`. A shell call
  gets its own process, so these end nothing; a single one is evidence
  on its own.
- **Inert** — a bare `echo` or `printf` with no redirect, pipe, chain,
  backtick, or command substitution. One on its own can be a probe or
  a marker, so it counts only as part of a no-op turn.

A *no-op turn* is an assistant turn that made at least one tool call
where **every** call is a no-op call. An unconditional no-op that rides
along with real work is penalised and counted, but does not make the
turn a stall: no nudge, no strike. Both the CEO and specialists are in
scope; only the remedy differs.

**Penalty.** Each no-op call's ToolResult payload gets
`shaping_reward = noop_penalty` (default `-0.05`) plus a
`shaping_reason` tag, written before the result's `tool.completed`
event is emitted (§12.4). It goes on that separate key, never on
`reward` (§12.1). Failed results are skipped.

**Cadence and remedy.** The nudge fires on every no-op turn. Strikes
are counted over a **sliding window** of the last `noop_window`
(default 8) turns, not as a consecutive run, so an agent alternating
`echo done` with the occasional real call still accumulates them.
After `noop_park_strikes` (default 3) no-op turns within the window, a
*specialist* is force-parked through the §4.5.1 park path (the
quiescence invariant still applies); once parked, the threshold drops
to one strike for the rest of the session. The final message is
appended to the inbox without a wake, so a real DM or notification
that arrived during the turn still short-circuits the park.

The *CEO* is never parked: it has nobody to report to, and parking it
would strand the rollout. Its nudge tells it the environment has NOT
ended and points it back at the work, without offering submission as
the way out. Only the first CEO nudge carries the full text; later ones
are a terse repeat, at most one per `noop_window` turns (a productive
turn in between does not reset this). The penalty and the event still
fire on every no-op turn.

**Consistency with §4.5.1.** The tool-less-turn nudge MUST NOT
contradict this rule. It names only env tools that genuinely submit
(never a shell tool, even one with `scope = "env"`), offers them only
conditionally ("if you want to finish now and stop working, call
`X`"), excludes a matching tool that has returned `finished: false`
in this rollout, and says that a no-op does not satisfy the request
for a tool call. Recognising a submission tool is a name heuristic that
matches nothing in many environments, so when no tool looks like a
submitter the nudge says so as uncertainty, with the default being
that there is more to do.

The specialist warning (line-wrapped here; from the second strike it is
prefixed with `{count} of your recent turns have done nothing and you
were already warned. Keep going and the harness will park you …`):

```
That turn did nothing. `{command}` is a no-op: it changes nothing in
the environment and it does not end your assignment. There is no
"mission complete", the rollout ends when the CEO submits, not when
you stand down. No shell command can end it, and each no-op call is
recorded and scored against you (see the negative `shaping_reward`
on that tool result).

If you have no work in hand, in this order:
1. ReportToCEO with your status and an explicit ask for the next piece
   of work, then call WaitForMessages to park. Parking costs nothing
   and you wake the moment anyone messages you.
2. If the CEO does not come back, PostToChannel tagging @CEO with a
   specific question, then WaitForMessages again.
3. If still nothing, pick your assignment back up and find a way to
   raise the quality of what you already delivered. There is always
   something to improve.

Never emit `exit`, `echo done`, or similar to stand down.
```

Step 2 is omitted when `PostToChannel` is not in the agent's effective
tool set, and the remaining steps renumber.

On the park strike:

```
{count} of your recent turns have done nothing, after being warned.
The harness has parked you. You will wake as soon as the CEO or a
teammate messages you.

When you wake, do not emit another no-op. ReportToCEO with your status
and ask for work, or pick your assignment back up and improve on what
you delivered. There is no way to end your assignment from the shell.
```

Each detection emits an `agent.noop_detected` event carrying
`{name, kind, tool, command, calls, turn_noop, streak, in_window,
window, penalty, parked}` (`turn_noop: false`, without the window
fields, for an unconditional no-op that rode along with real work).
`in_window` is what trips the threshold; `streak` is the consecutive
run. Per-rollout metrics gain `noop_calls` (every no-op call, including
ones that rode along with real work), `noop_turns`, `noop_turns_ceo`,
`noop_turns_org`, `noop_max_run` and `noop_parks`.

#### A.8.6 Repeated identical call (structural refusal)

**Problem.** A long session can burn its horizon on one call repeated
with identical arguments and identical output, and once an agent is in
that state nothing in the environment will move it out. Text nudges do
not reliably reach an agent in this state: one that is still
reading can rephrase the call past any blacklist, and one that is not
reading is a sampling loop rather than a decision. A refusal changes
the transcript tail, which is the one thing the next sample is
conditioned on, and it does so without the harness having to
understand the call.

**Detection.** Per agent, the harness tracks the current **run**: the
number of consecutive completed calls whose *key* is identical. The key
is

- the canonical tool name (after case-insensitive resolution),
- the arguments, serialised as canonical JSON (sorted keys), and
- a digest of the result's **content**: for an `ok` result whose
  output carries `blocks`, the concatenated text of its text blocks;
  for any other `ok` dict output, its JSON with `metadata`,
  `shaping_reward` and `shaping_reason` removed; for a non-dict output
  its JSON; for an `ok: false` result, `error.code` + `error.message`.

The content digest, not the whole result, is what makes the rule
sound: env outputs carry per-call bookkeeping (`metadata` timestamps,
round counters) that would make every call look new. Same arguments
with **different** content is never touched: a game step issued
repeatedly with a changing response, a channel view that returns new
messages each time, and a build re-run after an edit all pass, because
the output changed. Identical failing commands are caught the same
way, through the error digest.

The run ends when a call with a different key completes. A refused
call **extends** the run: once refused, every further identical call is
refused until the agent does something else.

A call the harness rejected before execution (`error.kind = "rejection"`,
other than `repeated-call`) MUST NOT become the run's key and MUST NOT
extend it: nothing ran, so it says nothing about what that call returns.
It MUST end a run keyed on a different tool or arguments — the agent is no
longer repeating what the run counted. A harness that keys such a call
instead builds a run out of refusals, and the first call to succeed after
the rejecting condition clears is refused in its place.

**Scope.** Every tool for both agent kinds, read-only tools included —
a loop of inspections wastes the horizon exactly as a loop of
mutations does. `WaitForMessages` is
excluded: its arguments are always empty, its output is the wake
payload, and re-calling it on the quiescent marker is already handled
by its own short-circuit.

**Refusal.** When the agent's current run is at least
`repeat_refuse_after` (default **15**; `0` disables the rule) and the
new call has the same tool and arguments, the harness does **not**
execute it. It returns `ok: false` with `error.code = "repeated-call"`
and a message of the shape, where `<run>` is the current run length:

```
Not run. This is identical to your last <run> calls, and every one of
them returned the same output. Repeating it cannot produce anything
new. Do something different: change the call, act on the output you
already have, or — if you are waiting on something outside your
control — park with WaitForMessages instead of polling.
```

The message MUST NOT offer firing, submission or `finish` as the
alternative (§4.5.2 rationale; §A.8.5 CEO nudge rationale).

The threshold is deliberately conservative: a run long enough that
no deliberate re-read reaches it, short enough that a loop is cut
within a few minutes. Implementations SHOULD choose it from a sweep of
recorded sessions at the candidate value rather than from a single
observed false positive.

**Penalty.** A refused result MUST carry

```
output = { "shaping_reward": <repeat_penalty>, "shaping_reason": "repeated-call" }
```

alongside `error`, on the `shaping_reward` carrier and never on `reward`
(§12.1), written before `tool.completed` is emitted so the two persisted
copies agree (§12.4). `repeat_penalty` defaults to **the value of
`noop_penalty`**. A `repeat_penalty` of `0.0` MUST suppress the stamp and
keep the refusal.

A refused call MUST carry `error.kind = "rejection"` and MUST be recorded
as a `HarnessRejectionEvent` (§12.1). It MUST NOT be recorded as a
`ToolErrorEvent`. The `shaping_reward` stamp and the rejection event are
independent: a consumer of tool-result payloads applies `repeat_penalty`;
a consumer of the event streams applies its own rejection penalty, which
per §12.1 is off by default.

**Circuit breaker.** `repeated-call` is a harness rejection: the env never
saw the call, so it MUST NOT feed the env-tool circuit breaker in either
direction — neither as a failure to count nor as a success to reset on.

**Events.** As for every harness rejection, `tool.requested` and
`tool.completed { ok: false, error.code: "repeated-call" }` are
emitted. Each refusal additionally emits `agent.repeat_refused`
carrying `{ name, kind, tool, run, threshold, penalty }` so a trace
scan can count refusals without re-deriving runs.

**Relation to §A.8.5.** Both rules may apply to `echo done` repeated:
§A.8.5 penalises from the first call, this rule refuses once the run
reaches `repeat_refuse_after`. A refused result is `ok: false`, so the §A.8.5 stamp
(which touches only `ok` dict outputs) never lands on the same result;
a call carries at most one `shaping_reason`.

**Non-goals.** Alternating loops (`A B A B …`), near-identical variants
(`print('done')` → `print('done ')`) and trace-watching that returns a
new turn each time are out of scope. The first two are the natural
extension (a run keyed over a small window of recent keys rather than
strictly consecutive); the third burns context, not turns, and needs a
different lever. Compaction summaries that name the loop are not
consulted: the refusal acts on the transcript directly.

### A.9 Compaction prompts

§11.4 defines the compaction algorithm and persistence model;
this subsection shows how this repository prompts the
model to *produce* the compaction summary.  An implementation may 
reword freely — what matters is that the model emits a deterministic, 
structured summary that preserves enough technical context to continue 
the work.

The summarizer runs as a one-shot LLM call with the full
pre-boundary transcript as user input. Placeholders in `{curly}`
are interpolated at call time.

#### A.9.1 Base summarization prompt

```
{no_tools_preamble}

Your task is to create a detailed summary of the conversation so
far, paying close attention to the user's explicit requests and
previous actions. This summary should preserve enough technical
detail to continue the work without losing context.

Before providing your final summary, wrap your reasoning in
<analysis> tags. Then provide the final summary in <summary> tags.

Your summary should include these sections:
1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and Fixes
5. Problem Solving
6. All User Messages
7. Pending Tasks
8. Current Work
9. Optional Next Step

Be specific about:
- the user's explicit requests, constraints, and corrections
- exact files, commands, errors, and code changes
- the most recent unfinished work and where it stopped

Do not omit recent user feedback that changes how the work should
be done.{extra_instructions}{retry_section}
```

The `{no_tools_preamble}` slot is filled when the provider supports
tool-blocking but cannot be trusted to honor a `tool_choice = none`
hint; see A.9.2. `{extra_instructions}` is reserved for
caller-supplied addenda (e.g. a session-memory update directive).
`{retry_section}` is filled on retry attempts; see A.9.3.

#### A.9.2 No-tools preamble

Prepended to the base prompt when the provider would otherwise
emit a tool call during summarization — wasting the one inference
turn the harness has budgeted for compaction:

```
CRITICAL: Respond with TEXT ONLY. Do not call tools.

- Do NOT use any tools.
- Tool calls will be rejected and will waste your only turn.
- Your entire response must be plain text containing an
  <analysis> block followed by a <summary> block.
```

#### A.9.3 Retry preamble

Appended via `{retry_section}` when the previous summarization
attempt was rejected (e.g. the model emitted raw chat-template
tokens that bled through into the summary):

```
## Previous Attempt Was Rejected
Reason: {reason}

This is a retry. Output PROSE ONLY. Do not emit chat-template
tokens of any kind — specifically nothing matching the pattern
`<|...|>` (such as `<|tool_call_begin|>`,
`<|tool_calls_section_begin|>`, `<|im_start|>`). Those tokens are
reserved chat-template delimiters and their presence means the
summary is unreadable. The summary must be a multi-paragraph prose
description of what happened in the conversation — files touched,
errors hit, decisions made, current task state — wrapped in
`<analysis>...</analysis>` followed by `<summary>...</summary>`
tags.
```

These three fragments are illustrative; the protocol does not
require a particular prompt shape, only that compaction produce
the §11.4.2 boundary marker plus a summary message that the
post-compact context can stand on.

---

*End of spec.*
