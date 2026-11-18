"""Org management primitives — spawn / hire / fire / retool / system-prompt refresh."""

from __future__ import annotations

from typing import Optional

from agentorg.inbox import render_attachment_marker
from agentorg.naming import generate_name
from agentorg.prompts import build_ceo_system_prompt, build_org_agent_system_prompt
from agentorg.session import Session
from agentorg.types import (
    Agent,
    ExitReview,
    InboxItem,
    Message,
    OrgAgentSpec,
)


def has_filesystem_tools(session: Session) -> bool:
    """True if local Bash/Read/Edit/etc. are registered."""
    expected = {"Bash", "Read", "Edit", "Glob", "Grep", "Write"}
    available = set(session.tools.names())
    return bool(expected & available)


def _agent_has_tool(
    session: Session,
    *,
    tool_name: str,
    allowed: Optional[list[str]],
    disallowed: Optional[list[str]],
) -> bool:
    """True if tool_name is in the session registry and not filtered out for this agent."""
    if tool_name not in session.tools.names():
        return False
    if allowed is not None and tool_name not in allowed:
        return False
    if disallowed and tool_name in disallowed:
        return False
    return True


async def refresh_ceo_system_prompt(session: Session) -> None:
    ceo = session.agent(session.ceo_name)
    if ceo is None:
        return
    new_prompt = build_ceo_system_prompt(
        mode=session.state.mode,
        budget=session.state.budget.max_org_agents,
        env=session.state.environment,
        toolset_name=session.config.toolset_name,
        live_org_agents=session.live_org_agents(),
        has_filesystem_tools=has_filesystem_tools(session),
        headless=session.headless,
        workspace_agents_md=session.workspace_agents_md(),
        date_str=session.config.date_str,
    )
    ceo.system_prompt = new_prompt
    await session.update_agent(ceo)


async def refresh_org_agent_system_prompts(session: Session) -> None:
    peers = session.live_org_agents()
    for agent in peers:
        agent_has_bash = _agent_has_tool(
            session,
            tool_name="Bash",
            allowed=agent.tool_allowlist,
            disallowed=agent.tool_disallowlist,
        )
        new_prompt = build_org_agent_system_prompt(
            name=agent.name,
            role=agent.role or "specialist",
            description=agent.description,
            peers=peers,
            spec_addon=None,  # spec system_prompt is baked at spawn time only
            predecessor_handoff=None,
            env=session.state.environment,
            toolset_name=session.config.toolset_name,
            has_filesystem_tools=has_filesystem_tools(session),
            agent_has_bash=agent_has_bash,
            workspace_agents_md=session.workspace_agents_md(),
            date_str=session.config.date_str,
            headless=session.headless,
        )
        # Preserve original mission addon: re-attach the original spec
        # system_prompt fragment if previously embedded — for simplicity we
        # only refresh peers section and identity here.
        if "# Mission addons" in agent.system_prompt and "# Mission addons" not in new_prompt:
            mission_block = _extract_block(agent.system_prompt, "# Mission addons")
            if mission_block:
                new_prompt = _insert_after(new_prompt, "# Your mission", mission_block)
        agent.system_prompt = new_prompt
        await session.update_agent(agent)


def _extract_block(text: str, header: str) -> Optional[str]:
    if header not in text:
        return None
    idx = text.index(header)
    # Block is from the header to the next "# " heading (or end).
    after = text[idx:]
    lines = after.splitlines()
    out_lines = [lines[0]]
    for line in lines[1:]:
        if line.startswith("# ") and line != lines[0]:
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def _insert_after(text: str, marker_header: str, block: str) -> str:
    if marker_header not in text:
        return text + "\n\n" + block
    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i] == marker_header and not inserted:
            # find end of marker section
            j = i + 1
            while j < len(lines) and not (lines[j].startswith("# ") and lines[j] != marker_header):
                out.append(lines[j])
                j += 1
            out.append("")
            out.append(block)
            i = j
            inserted = True
            continue
        i += 1
    if not inserted:
        return text + "\n\n" + block
    return "\n".join(out)


# Plan-mode and org-management tools only the CEO may hold. Structural, not a
# preference: an org agent that could hire, fire or re-tool its peers is no
# longer an org agent. WritePlan is here too: it already rejects non-CEO
# callers, so showing it to an org agent only invites a wasted call. Applied on spawn AND on every `UpdateOrgAgentTools` edit — the tool
# replaces a disallowlist wholesale, so without re-applying it here a CEO that
# sends `disallowed_tools=["bash"]` to reclaim one tool silently grants the
# peer all six.
ORG_AGENT_DENIED_TOOLS: tuple[str, ...] = (
    "EnterOrgPlanMode",
    "WritePlan",
    "ExitOrgPlanMode",
    "HireOrgAgent",
    "FireOrgAgent",
    "UpdateOrgAgentTools",
)


async def spawn_org_agent(
    session: Session,
    spec: OrgAgentSpec,
) -> Agent:
    """Create + register a new org agent per §6.4."""
    if session.remaining_budget() <= 0:
        raise BudgetExhausted("Cannot hire: max_org_agents reached")

    name = generate_name(used=session.all_agent_names())

    # Resolve effective tool registry (CEO's minus disallowed, intersected with allowed).
    ceo_tools = set(session.tools.names())
    allowed = spec.allowed_tools or None
    disallowed = list(spec.disallowed_tools or []) + list(ORG_AGENT_DENIED_TOOLS)
    if allowed is not None:
        allowed = [t for t in allowed if t in ceo_tools]
    # Disallowlist is consumed by `registry.visible` and `_agent_has_tool` —
    # both iterate registered tools and check membership, so unknown names
    # are inert. We dedupe + sort for stable rendering and persistence.
    effective_disallow = sorted(set(disallowed))
    work_dir = spec.work_dir or session.state.workspace_path

    # Predecessor handoff
    predecessor = session.predecessor_for_role(spec.role)
    handoff = None
    if predecessor and predecessor.exit_review is not None:
        handoff = {
            "name": predecessor.name,
            "role": predecessor.role or spec.role,
            "review": predecessor.exit_review.model_dump(),
        }

    peers = session.live_org_agents()
    agent_has_bash = _agent_has_tool(
        session,
        tool_name="Bash",
        allowed=allowed,
        disallowed=effective_disallow,
    )
    sys_prompt = build_org_agent_system_prompt(
        name=name,
        role=spec.role,
        description=spec.description,
        peers=peers,
        spec_addon=spec.system_prompt,
        predecessor_handoff=handoff,
        env=session.state.environment,
        toolset_name=session.config.toolset_name,
        has_filesystem_tools=has_filesystem_tools(session),
        agent_has_bash=agent_has_bash,
        workspace_agents_md=session.workspace_agents_md(),
        date_str=session.config.date_str,
        headless=session.headless,
    )

    agent = Agent(
        name=name,
        session_id=session.session_id,
        kind="org",
        role=spec.role,
        description=spec.description,
        parent_name=session.ceo_name,
        work_dir=work_dir,
        tool_allowlist=allowed,
        tool_disallowlist=effective_disallow,
        system_prompt=sys_prompt,
        state="idle",
    )
    await session.register_agent(agent)

    # Inject mandate as the first user message (§6.4 step 6).
    mandate = f"You are {name}. {spec.description}\n\nBegin work on your mission now."
    if spec.archive_attachments:
        mandate += "\n\n" + render_attachment_marker(spec.archive_attachments)
    mandate_msg = Message(role="user", source="env", content=mandate)
    await session.append_message(name, mandate_msg)

    if handoff is not None:
        handoff_text = (
            f"Predecessor handoff: you are replacing {handoff['name']}.\n"
            f"  Category:   {handoff['review']['category']}\n"
            f"  Delivered:  {handoff['review']['delivered']}\n"
            f"  Shortfalls: {handoff['review']['shortfalls']}\n"
            f"  Score:      {handoff['review']['score']}/10\n"
            f"Review the full transcript at #{handoff['name']} before starting."
        )
        await session.inbox.append(
            name,
            InboxItem(kind="predecessor_handoff", content=handoff_text),
        )

    # Apply per-spec subscriptions on existing channels.
    if spec.subscriptions:
        await session.channels.load()
        for ch_name, level in spec.subscriptions.items():
            await session.channels.set_subscription(ch_name, name, level)

    # Refresh other agents' rosters since membership changed.
    await refresh_org_agent_system_prompts(session)
    await refresh_ceo_system_prompt(session)

    await session.events.emit(
        "agent.spawned",
        {"name": name, "kind": "org", "role": spec.role},
    )
    session.wakes.wake(name)  # release agent to its first turn
    return agent


class BudgetExhausted(Exception):
    pass


async def fire_org_agent(
    session: Session,
    name: str,
    review: Optional[ExitReview] = None,
    *,
    source: str = "ceo",
    reason: Optional[str] = None,
) -> Agent:
    agent = session.agent(name)
    if agent is None:
        raise KeyError(f"No such agent: {name}")
    if agent.state == "fired":
        return agent
    agent.state = "fired"
    agent.exit_review = review
    agent.fired_source = source  # type: ignore[assignment]
    await session.update_agent(agent)
    payload = {"name": name, "source": source}
    if review is not None:
        payload["review"] = review.model_dump()
    if reason is not None:
        payload["reason"] = reason
    await session.events.emit("agent.fired", payload, name=name)
    # §5.6 — terminate every background task this agent started.
    # A fired agent can't act on task_update notifications anymore;
    # leaving its subprocesses running would orphan resources.
    await session.background_tasks.kill_agent_tasks(name)
    await refresh_ceo_system_prompt(session)
    await refresh_org_agent_system_prompts(session)
    session.wakes.wake(name)  # let the agent's loop unwind
    # §6.5: after env-terminal the harness fires every org agent and stops
    # the session (`loop._settle_env_terminal`). The fire that empties the
    # roster stops it here already.
    if session.terminal_event is not None and not session.live_org_agents():
        session.stop()
    return agent


async def retire_crashed_org_agent(session: Session, name: str, cause: str) -> None:
    """Fire an org agent whose loop died, and tell the CEO.

    A dead but live agent never parks, so the §4.5.1 quiescence check that
    releases a parked CEO never passes and the session hangs until
    max_wall_s. The fire drops it from the live roster; the meta item wakes
    the CEO, which can re-hire or take the work over.
    """
    await fire_org_agent(session, name, source="harness", reason="crashed")
    if session.stop_requested():
        return
    await session.inbox.append(
        session.ceo_name,
        InboxItem(
            kind="meta",
            content=(
                f"@{name} stopped ({cause}) and has left the org. "
                "Re-hire for its work or take it over."
            ),
        ),
    )
    session.wakes.wake(session.ceo_name)


async def update_org_agent_tools(
    session: Session,
    name: str,
    *,
    allowed: Optional[list[str]] = None,
    disallowed: Optional[list[str]] = None,
) -> Agent:
    agent = session.agent(name)
    if agent is None:
        raise KeyError(f"No such agent: {name}")
    if allowed is not None:
        agent.tool_allowlist = allowed
    if disallowed is not None:
        agent.tool_disallowlist = sorted(set(disallowed) | set(ORG_AGENT_DENIED_TOOLS))
    await session.update_agent(agent)
    await session.inbox.append(
        name,
        InboxItem(
            kind="tool_update",
            content=(
                f"Your tool allowlist was updated by the CEO. "
                f"allowed={agent.tool_allowlist} disallowed={agent.tool_disallowlist}"
            ),
        ),
    )
    session.wakes.wake(name)
    return agent
