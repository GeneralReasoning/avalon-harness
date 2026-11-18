"""Per-rollout metrics computed from a finished AgentOrg :class:`Session`.

These are aggregated across *every* agent in the org (the CEO and all hired
specialists) and returned on :class:`agentorg.harness.RolloutResult` so a
training harness can log them per rollout.

Token attribution
------------------
The only token signal AgentOrg records is ``MessageUsage`` on assistant turns
(``input_tokens``/``output_tokens``/...), and it covers the *whole* turn —
reasoning + content + every tool call in that turn. There is no per-tool-call
token count. So "tokens spent on tool X" is necessarily an estimate: for each
assistant turn we split that turn's ``output_tokens`` across its tool calls in
proportion to each call's serialized-argument size, and bucket the share by the
tool's category. Turns with no tool call contribute nothing to these buckets.
This is an *output*-token attribution (what the agent generated to invoke the
tool), which is what the requested metrics ask for.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from agentorg.channels import MENTION_RE

# --- tool name → category -------------------------------------------------
# Channel communication: real inter-agent channel traffic (posts + reads of
# regular channels). Reads of *virtual* (agent-trace) channels are split out
# below into the "mind-read" bucket, since those are peeking at another agent's
# trace rather than communicating.
_CHANNEL_POST = {"PostToChannel"}
_CHANNEL_READ = {"ViewChannelMessages", "SearchChannelMessages", "ListChannels"}
_ARCHIVE_WRITE = {"WriteToArchive", "EditArchiveFile", "MoveArchiveFile"}
_ARCHIVE_READ = {"ListArchive", "ViewArchiveFile"}

# Outbound message tools whose body we scan for @mentions.
_MENTION_TOOLS = {"PostToChannel", "SendMessage", "ReportToCEO"}


def _is_virtual_channel(session: Any, channel: Any) -> bool:
    """True if ``channel`` names a virtual (agent-trace / "mind") channel."""
    if not isinstance(channel, str) or not channel:
        return False
    try:
        from agentorg.tools.comms_tools import _is_virtual_channel_name

        return _is_virtual_channel_name(session, channel)
    except Exception:
        return False


def compute_rollout_metrics(session: Any) -> dict[str, Any]:
    """Aggregate org-wide activity metrics from a finished session.

    Returns a JSON-able dict. Safe to call on a partially-populated session;
    missing usage / malformed tool args are tolerated.
    """
    # token buckets (estimated output tokens, see module docstring)
    tok: dict[str, float] = defaultdict(float)
    # discrete counts
    cnt: dict[str, int] = defaultdict(int)
    roles: Counter[str] = Counter()

    # --- roster / role distribution --------------------------------------
    try:
        agents = list(session.all_agents())
    except Exception:
        agents = []
    for agent in agents:
        if getattr(agent, "kind", None) == "org":
            roles[getattr(agent, "role", None) or "(unspecified)"] += 1

    # --- walk every agent's transcript -----------------------------------
    for agent in agents:
        try:
            transcript = session.transcript(agent.name)
        except Exception:
            continue
        for msg in transcript:
            if getattr(msg, "role", None) != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                continue

            # output tokens for this whole turn (may be absent)
            usage = getattr(msg, "usage", None)
            out_tokens = float(getattr(usage, "output_tokens", 0) or 0) if usage else 0.0
            weights = [max(1, len(json.dumps(tc.args, default=str))) for tc in tool_calls]
            total_w = float(sum(weights))

            for tc, w in zip(tool_calls, weights):
                name = tc.name
                share = out_tokens * (w / total_w) if total_w > 0 else 0.0

                # ---- counts + token buckets by category -----------------
                if name in _CHANNEL_POST:
                    cnt["channel_posts"] += 1
                    tok["channel_comm_output"] += share
                elif name in _CHANNEL_READ:
                    if _is_virtual_channel(session, tc.args.get("channel")):
                        cnt["mind_reads"] += 1
                        tok["mind_read_output"] += share
                    else:
                        cnt["channel_reads"] += 1
                        tok["channel_comm_output"] += share
                elif name in _ARCHIVE_WRITE:
                    cnt["archive_writes"] += 1
                    tok["archive_write_output"] += share
                elif name in _ARCHIVE_READ:
                    cnt["archive_reads"] += 1
                    tok["archive_read_output"] += share
                elif name == "ReactToMessage":
                    cnt["reactions"] += 1
                elif name == "HireOrgAgent":
                    cnt["hires"] += 1
                elif name == "FireOrgAgent":
                    cnt["fires"] += 1
                elif name == "CreateChannels":
                    cnt["create_channel_calls"] += 1
                    # CreateChannels is batch — count the channels requested too.
                    specs = tc.args.get("channels")
                    cnt["channels_created"] += len(specs) if isinstance(specs, list) else 1

                # ---- @mention tagging count -----------------------------
                if name in _MENTION_TOOLS:
                    text = " ".join(v for v in tc.args.values() if isinstance(v, str))
                    cnt["mentions"] += len(MENTION_RE.findall(text))

    # §A.8.5 no-op ("doom loop") counters, aggregated across agents.
    # `noop_max_run` is the worst single agent's longest consecutive run, a
    # severity measure; `noop_parks` is
    # how many times the harness had to force-park a looping agent.
    noop_stats = getattr(session, "noop_stats", None) or {}
    noop_calls = sum(v.get("calls", 0) for v in noop_stats.values())
    noop_turns = sum(v.get("turns", 0) for v in noop_stats.values())
    noop_max_run = max((v.get("max_run", 0) for v in noop_stats.values()), default=0)
    noop_parks = sum(v.get("parks", 0) for v in noop_stats.values())
    # Split by kind: the CEO variant and the specialist variant have different
    # causes and different remedies (only a specialist can be parked), so a
    # comparison that pools them cannot tell which one moved.
    noop_turns_ceo = sum(
        v.get("turns", 0) for v in noop_stats.values() if v.get("kind") == "ceo"
    )
    noop_turns_org = noop_turns - noop_turns_ceo

    return {
        "tokens": {k: round(v, 1) for k, v in tok.items()},
        "counts": dict(cnt),
        "roles": dict(roles),
        "num_org_agents": sum(roles.values()),
        "noop_calls": noop_calls,
        "noop_turns": noop_turns,
        "noop_turns_ceo": noop_turns_ceo,
        "noop_turns_org": noop_turns_org,
        "noop_max_run": noop_max_run,
        "noop_parks": noop_parks,
    }
