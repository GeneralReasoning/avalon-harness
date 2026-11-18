"""Org-management tools: plan mode, hire/fire/retool (§5.5, §6)."""

from __future__ import annotations

from agentorg.org import (
    BudgetExhausted,
    fire_org_agent,
    spawn_org_agent,
    update_org_agent_tools,
)
from agentorg.plan import PlanParseError, extract_org_block
from agentorg.tools.comms_tools import LIGHT_LEVEL_LITERAL
from agentorg.tools.context import (
    ToolContext,
    err_result,
    ok_result,
    rejected_result,
)
from agentorg.tools.org_tools import _require_ceo
from agentorg.types import (
    ExitReview,
    OrgAgentSpec,
    ToolDefinition,
    ToolResult,
)


class WritePlanTool:
    """Write the session plan file at the canonical harness-local path.
    """

    definition = ToolDefinition(
        name="WritePlan",
        description=(
            "Write the session org-plan file. Accepts a single `content` arg "
            "containing the plan markdown (including the `agents-org` fenced "
            "block). The file path is managed by the harness — do not pass "
            "a path. Call ExitOrgPlanMode after writing to spawn the roster."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Plan markdown body. Should include a fenced ```agents-org block.",
                },
            },
            "required": ["content"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "write the org plan")):
            return denied
        content = ctx.args.get("content")
        if not isinstance(content, str):
            return err_result(
                "invalid-args", "`content` must be a string (the plan markdown body)."
            )
        # Validate any agents-org fence in the same turn — surfacing a
        # malformed block at write time is far cheaper than a silent
        # spawned=[] from ExitOrgPlanMode later.
        try:
            block = extract_org_block(content)
        except PlanParseError as e:
            return err_result("plan-parse-error", str(e))
        # In multi-agent org-plan mode the contract (see ORG_PLAN_MULTI in
        # prompts.py) is that the plan MUST contain an agents-org block.
        # No-block is only legitimate in solo mode (max_org_agents == 0,
        # ORG_PLAN_SOLO). Reject the multi-agent no-block case so the CEO
        # can re-issue rather than silently proceeding to ExitOrgPlanMode
        # with nothing to spawn.
        budget = ctx.session.state.budget.max_org_agents
        if block is None and ctx.session.state.mode == "org-plan" and budget > 0:
            return err_result(
                "plan-missing-org-block",
                (
                    f"Plan in multi-agent org-plan mode (budget={budget}) must "
                    "include a fenced ```agents-org block listing the specialists "
                    "to hire. Org-plan mode exists to design a team; there is no "
                    "solo path out of it."
                ),
            )
        from pathlib import Path

        plan_path = Path(ctx.session.plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(content, encoding="utf-8")
        return ok_result({"bytes": len(content.encode("utf-8"))})


class EnterOrgPlanModeTool:
    definition = ToolDefinition(
        name="EnterOrgPlanMode",
        description=(
            "Enter org-plan mode. The CEO must then write its plan with WritePlan and call ExitOrgPlanMode."
        ),
        input_schema={"type": "object", "properties": {}},
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "enter org-plan mode")):
            return denied
        ctx.session.state.mode = "org-plan"
        await ctx.session.persist_state()
        # Rebuild CEO system prompt.
        from agentorg.org import refresh_ceo_system_prompt

        await refresh_ceo_system_prompt(ctx.session)
        await ctx.session.events.emit("plan.mode.entered", {"mode": "org-plan"})
        return ok_result(
            {
                "mode": "org-plan",
                "budget": ctx.session.state.budget.max_org_agents,
            }
        )


class ExitOrgPlanModeTool:
    definition = ToolDefinition(
        name="ExitOrgPlanMode",
        description=(
            "Exit org-plan mode and restore full tool access. Reads the plan written by WritePlan; "
            "if it declares an agents-org block, spawns that roster (after approval). "
            "In headless sessions, this is auto-approved."
        ),
        input_schema={"type": "object", "properties": {}},
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "exit org-plan mode")):
            return denied
        session = ctx.session
        from pathlib import Path

        plan_path = Path(session.plan_path)
        if not plan_path.exists():
            return err_result(
                "plan-missing",
                "No plan found. Call WritePlan(content=<plan markdown>) before calling ExitOrgPlanMode.",
            )
        markdown = plan_path.read_text(encoding="utf-8")
        try:
            block = extract_org_block(markdown)
        except PlanParseError as e:
            return err_result("plan-parse-error", str(e))

        spawned: list[str] = []
        if block is not None:
            # Enforce budget and role-uniqueness up front.
            if len(block.agents) > session.remaining_budget():
                return rejected_result(
                    "budget-exhausted",
                    (
                        f"Plan requests {len(block.agents)} agents but only "
                        f"{session.remaining_budget()} slots are available."
                    ),
                )
            live_role_lower = {
                (a.role or "").lower() for a in session.live_org_agents() if a.role
            }
            for spec in block.agents:
                if spec.role.lower() in live_role_lower:
                    return err_result(
                        "role-collision",
                        f"Role {spec.role!r} collides with a live agent",
                    )
        # Persist plan record.
        from agentorg.types import Plan
        from agentorg.types import new_id as _new_id

        plan = Plan(
            plan_id=_new_id(),
            session_id=session.session_id,
            markdown=markdown,
            org_block=block,
            approved_by="auto" if session.headless else None,
        )
        from agentorg.storage import write_json_async

        await write_json_async(session.paths.plan_json(plan.plan_id), plan.model_dump())
        session.paths.plan_md(plan.plan_id).write_text(markdown, encoding="utf-8")

        # Exit mode + spawn.
        session.state.mode = "default"
        await session.persist_state()
        from agentorg.org import refresh_ceo_system_prompt, spawn_org_agent

        await refresh_ceo_system_prompt(session)
        await session.events.emit("plan.mode.exited", {"mode": "default", "approved": True})

        # Seed channels BEFORE spawning agents so per-agent subscriptions can
        # resolve and specialists don't race the CEO on `CreateChannels`.
        await _seed_default_channels(session)
        plan_channels_created: list[str] = []
        if block is not None and block.channels:
            channel_specs = [
                {
                    "name": c.name,
                    "default_subscription": c.default_subscription,
                    "subscriptions": dict(c.subscriptions),
                }
                for c in block.channels
            ]
            try:
                created_chs, _existing = await session.channels.create_channels(
                    channel_specs,
                    live_agent_names=session.live_agent_names(),
                    external_labels=set(session.external_labels),
                )
            except ValueError as e:
                return err_result("invalid-args", f"Plan channel: {e}")
            for ch in created_chs:
                await session.events.emit("channel.created", {"channel": ch.name})
                plan_channels_created.append(ch.name)

        if block is not None:
            for spec in block.agents:
                try:
                    agent = await spawn_org_agent(session, spec)
                    spawned.append(agent.name)
                except BudgetExhausted as e:
                    # Pre-flight check should have caught this; reaching
                    # here means another caller raced for budget.
                    return rejected_result("budget-exhausted", str(e))
        return ok_result(
            {
                "spawned": spawned,
                "plan_id": plan.plan_id,
                "channels_created": plan_channels_created,
            }
        )


async def _seed_default_channels(session) -> None:
    await session.channels.load()
    if not session.channels.resolve("announcements"):
        await session.channels.create_channels(
            [{"name": "announcements", "default_subscription": "full"}],
            live_agent_names=session.live_agent_names(),
            external_labels=set(session.external_labels),
        )
        await session.events.emit("channel.created", {"channel": "announcements"})
    if not session.channels.resolve("random"):
        await session.channels.create_channels(
            [{"name": "random", "default_subscription": "light"}],
            live_agent_names=session.live_agent_names(),
            external_labels=set(session.external_labels),
        )
        await session.events.emit("channel.created", {"channel": "random"})


class HireOrgAgentTool:
    definition = ToolDefinition(
        name="HireOrgAgent",
        description=(
            "Hire a new org agent. Provide role, description, optional allowed/disallowed tools, "
            "system_prompt, work_dir, subscriptions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "description": {"type": "string"},
                "allowed_tools": {"type": "array", "items": {"type": "string"}},
                "disallowed_tools": {"type": "array", "items": {"type": "string"}},
                "system_prompt": {"type": "string"},
                "work_dir": {"type": "string"},
                "subscriptions": {
                    "type": "object",
                    "additionalProperties": {"type": "string", "enum": LIGHT_LEVEL_LITERAL},
                },
                "archive_attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Archive paths injected into the new hire's first user message as <attachments: …>.",
                },
            },
            "required": ["role", "description"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "hire org agents")):
            return denied
        spec = OrgAgentSpec(
            role=ctx.args["role"],
            description=ctx.args.get("description") or "",
            allowed_tools=ctx.args.get("allowed_tools"),
            disallowed_tools=ctx.args.get("disallowed_tools"),
            system_prompt=ctx.args.get("system_prompt"),
            work_dir=ctx.args.get("work_dir"),
            subscriptions=ctx.args.get("subscriptions") or {},
            archive_attachments=ctx.args.get("archive_attachments"),
        )
        try:
            agent = await spawn_org_agent(ctx.session, spec)
        except BudgetExhausted as e:
            return rejected_result("budget-exhausted", str(e))
        return ok_result({"name": agent.name, "role": agent.role})


class FireOrgAgentTool:
    definition = ToolDefinition(
        name="FireOrgAgent",
        description=(
            "Fire an org agent. Required: a structured exit review "
            "(delivered, shortfalls, category, score 0-10). "
            "Categories: mission_complete | underperforming | redundant | capacity. "
            "Fill delivered and shortfalls FIRST with concrete nouns, then category, then score."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "delivered": {
                    "type": "string",
                    "description": "1-3 sentences naming concrete artifacts the agent produced.",
                },
                "shortfalls": {
                    "type": "string",
                    "description": "1-3 sentences naming concrete gaps. Empty only for mission_complete.",
                },
                "category": {
                    "type": "string",
                    "enum": ["mission_complete", "underperforming", "redundant", "capacity"],
                },
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": ["name", "delivered", "shortfalls", "category", "score"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "fire org agents")):
            return denied
        name = ctx.args["name"].lstrip("@")
        try:
            review = ExitReview(
                category=ctx.args["category"],
                delivered=ctx.args["delivered"],
                shortfalls=ctx.args["shortfalls"],
                score=int(ctx.args["score"]),
            )
        except Exception as e:
            return err_result("invalid-args", f"Could not construct ExitReview: {e}")
        try:
            agent = await fire_org_agent(ctx.session, name, review, source="ceo")
        except KeyError as e:
            return err_result("unknown-agent", e.args[0])
        return ok_result({"name": agent.name, "exit_review": review.model_dump()})


class UpdateOrgAgentToolsTool:
    definition = ToolDefinition(
        name="UpdateOrgAgentTools",
        description="Replace an org agent's allow/disallow lists. Takes effect at next dispatch.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "allowed_tools": {"type": "array", "items": {"type": "string"}},
                "disallowed_tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        if (denied := _require_ceo(ctx, "retool org agents")):
            return denied
        allowed = ctx.args.get("allowed_tools")
        disallowed = ctx.args.get("disallowed_tools")
        # Neither list given is a no-op the caller almost never means: the
        # schema only requires `name`, so `{"name": "X"}` used to return ok
        # with the lists unchanged. A CEO trying to free a wait-gated tool sent
        # exactly that three times, read three successes, and kept hitting the
        # gate. Refuse instead, and say what a revoke has to carry. `[]` is a
        # real value (allow nothing / disallow nothing), hence `is None`.
        if allowed is None and disallowed is None:
            return err_result(
                "invalid-args",
                "UpdateOrgAgentTools changed nothing: pass allowed_tools and/or "
                "disallowed_tools. Each REPLACES that list wholesale, so to add "
                "one tool to the disallowlist, send the existing entries plus "
                "the new one.",
            )
        try:
            agent = await update_org_agent_tools(
                ctx.session,
                ctx.args["name"].lstrip("@"),
                allowed=allowed,
                disallowed=disallowed,
            )
        except KeyError as e:
            return err_result("unknown-agent", e.args[0])
        return ok_result({"name": agent.name, "allowed": agent.tool_allowlist, "disallowed": agent.tool_disallowlist})
