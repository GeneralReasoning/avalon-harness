"""Permission policies (§5.2/§5.3)."""

from __future__ import annotations

from typing import Protocol

from agentorg.tools.context import PermissionDecision, ToolContext


class PermissionPolicy(Protocol):
    name: str

    def decide(self, tool_name: str, ctx: ToolContext) -> PermissionDecision: ...


class _Policy:
    def __init__(self, name: str, decider):
        self.name = name
        self._decider = decider

    def decide(self, tool_name: str, ctx: ToolContext) -> PermissionDecision:
        return self._decider(tool_name, ctx)


def _is_read_only(tool_name: str, ctx: ToolContext) -> bool:
    tool = ctx.session.tools.get(tool_name)
    return bool(tool and tool.definition.read_only)


def _default_decide(tool_name: str, ctx: ToolContext) -> PermissionDecision:
    if _is_read_only(tool_name, ctx):
        return "allow"
    return "ask"


def _accept_edits_decide(tool_name: str, ctx: ToolContext) -> PermissionDecision:
    if _is_read_only(tool_name, ctx):
        return "allow"
    if tool_name in ("Edit", "Write", "WriteToArchive", "EditArchiveFile", "MoveArchiveFile"):
        return "allow"
    return "ask"


# Host filesystem / shell tools that mutate the machine — the non-read-only
# tools registered from tools/fs_tools.py. The `host-ask` policy gates exactly
# these behind human approval, while letting read-only tools AND every org /
# coordination tool (SendMessage, channels, Hire/Fire, archive, TodoWrite, …)
# run freely — so an interactive CEO is never blocked on a human just to talk
# to its org. A new host-mutating tool should be added here. Not one of §5.3's
# four canonical policies, which the spec states are pluggable.
_HOST_ASK_TOOLS = frozenset({"Write", "Edit", "Bash", "TaskKill"})


def _host_ask_decide(tool_name: str, ctx: ToolContext) -> PermissionDecision:
    if tool_name in _HOST_ASK_TOOLS:
        return "ask"
    return "allow"


def _bypass_decide(_tool_name: str, _ctx: ToolContext) -> PermissionDecision:
    return "allow"


def _plan_decide(tool_name: str, ctx: ToolContext) -> PermissionDecision:
    if _is_read_only(tool_name, ctx):
        return "allow"
    return "deny"


_POLICIES: dict[str, _Policy] = {
    "default": _Policy("default", _default_decide),
    "accept-edits": _Policy("accept-edits", _accept_edits_decide),
    "host-ask": _Policy("host-ask", _host_ask_decide),
    "bypass": _Policy("bypass", _bypass_decide),
    "plan": _Policy("plan", _plan_decide),
}


def get_policy(name: str) -> PermissionPolicy:
    return _POLICIES[name]


def default_permission_policy() -> PermissionPolicy:
    return _POLICIES["default"]
