"""Plan parsing — the `agents-org` fenced block inside the CEO's plan (§3.5)."""

from __future__ import annotations

import json
import re

from agentorg.types import OrgAgentSpec, OrgBlock, OrgChannelSpec

FENCE_RE = re.compile(
    r"```\s*(agents-org)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
# Matches just the opening fence. Used to distinguish "no block at all" (a
# legitimate solo plan) from "fence opened but never closed" (malformed).
OPEN_FENCE_RE = re.compile(
    r"```\s*(agents-org)\s*\n",
    re.IGNORECASE,
)


class PlanParseError(Exception):
    pass


def extract_org_block(markdown: str) -> OrgBlock | None:
    """Find and parse a fenced `agents-org` JSON block. Returns None if absent."""
    match = FENCE_RE.search(markdown)
    if not match:
        opener = OPEN_FENCE_RE.search(markdown)
        if opener:
            lang = opener.group(1)
            raise PlanParseError(
                f"Found an opening '{lang}' fence but no closing '```' — "
                f"the block is unterminated. Re-issue the plan with the "
                f"fence closed."
            )
        return None
    body = match.group(2).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise PlanParseError(f"agents-org block is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("agents"), list):
        raise PlanParseError("agents-org block must be a JSON object with an `agents` array")
    specs: list[OrgAgentSpec] = []
    seen_roles: set[str] = set()
    for raw in data["agents"]:
        if not isinstance(raw, dict):
            raise PlanParseError("Each agent in agents-org must be a JSON object")
        # Accept camelCase as well as the spec-style snake_case.
        spec_kwargs = {
            "role": raw.get("role"),
            "description": raw.get("description", ""),
            "allowed_tools": raw.get("allowed_tools") or raw.get("allowedTools"),
            "disallowed_tools": raw.get("disallowed_tools") or raw.get("disallowedTools"),
            "system_prompt": raw.get("system_prompt") or raw.get("systemPrompt"),
            "work_dir": raw.get("work_dir") or raw.get("workDir"),
            "subscriptions": raw.get("subscriptions", {}),
            "archive_attachments": raw.get("archive_attachments") or raw.get("archiveAttachments"),
        }
        if not spec_kwargs["role"]:
            raise PlanParseError("Each agent must have a `role`")
        role_lower = spec_kwargs["role"].lower()
        if role_lower in seen_roles:
            raise PlanParseError(f"Duplicate role {spec_kwargs['role']!r} in agents-org block")
        seen_roles.add(role_lower)
        specs.append(OrgAgentSpec(**{k: v for k, v in spec_kwargs.items() if v is not None or k == "subscriptions"}))

    channel_specs: list[OrgChannelSpec] = []
    raw_channels = data.get("channels")
    if raw_channels is not None:
        if not isinstance(raw_channels, list):
            raise PlanParseError("`channels` in agents-org must be an array")
        seen_channels: set[str] = set()
        for raw in raw_channels:
            if not isinstance(raw, dict):
                raise PlanParseError("Each channel in agents-org must be a JSON object")
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                raise PlanParseError("Each channel must have a non-empty `name`")
            lower = name.lower()
            if lower in seen_channels:
                raise PlanParseError(f"Duplicate channel {name!r} in agents-org block")
            seen_channels.add(lower)
            channel_kwargs: dict = {"name": name}
            if raw.get("default_subscription") is not None:
                channel_kwargs["default_subscription"] = raw["default_subscription"]
            elif raw.get("defaultSubscription") is not None:
                channel_kwargs["default_subscription"] = raw["defaultSubscription"]
            if raw.get("subscriptions") is not None:
                channel_kwargs["subscriptions"] = raw["subscriptions"]
            channel_specs.append(OrgChannelSpec(**channel_kwargs))

    return OrgBlock(agents=specs, channels=channel_specs)
