"""Tool execution context plumbing (§5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

from agentorg.types import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from agentorg.session import Session
    from agentorg.types import Agent


# Soft cap on the rendered body of any single tool result. ~8K tokens at
# typical English/code densities — large enough for a meaningful single read
# (a source file, a channel scrollback) but small enough that one tool call
# can never dominate the model's context window. Each tool applies the cap
# on its own terms:
#
#   - Single-content tools (Read, WebFetch, ViewArchiveFile) truncate the
#     body and return an `offset` / `next_offset` so the agent can page.
#   - Aggregating tools (ViewChannelMessages, SearchChannelMessages) trim
#     the element list down to fit, surfacing the dropped count.
#
# `WaitForMessages` is intentionally NOT capped here: the agent loop's
# top-of-iteration inbox drain would re-feed any deferred items as a user
# message on the very next turn, so capping the tool would just split the
# same content across two messages without bounding what the agent sees.
# Inbox content is bounded at the writer's side instead (MAX_MESSAGE_BYTES,
# §7.9). See SPEC §5.8 for the full rationale.
MAX_TOOL_OUTPUT_BYTES = 32 * 1024


PermissionDecision = Literal["allow", "deny", "ask"]


@dataclass
class ToolContext:
    agent: "Agent"
    session: "Session"
    args: dict[str, Any]
    call_id: str
    # Provider completion id of the assistant turn that issued this call
    # (None when the provider reports none). Env tools pass it to
    # Session.record_reward so per-step rewards stay attributable to the
    # exact generation span — session-global "last turn" state would race
    # under concurrent multi-agent dispatch.
    turn_response_id: Optional[str] = None

    @property
    def workspace(self) -> str:
        return self.session.state.workspace_path


class ToolImpl(Protocol):
    definition: ToolDefinition

    async def execute(self, ctx: ToolContext) -> ToolResult: ...


PermissionRequester = Callable[[str, dict[str, Any], "ToolContext"], "PermissionDecision"]


def ok_result(output: Any) -> ToolResult:
    return ToolResult(ok=True, output=output)


def err_result(code: str, message: str) -> ToolResult:
    from agentorg.types import ToolResultError

    return ToolResult(ok=False, error=ToolResultError(code=code, message=message))


def rejected_result(code: str, message: str) -> ToolResult:
    """An ``ok=False`` result for a call the harness declined to run, so
    nothing executed. Differs from `err_result` only in ``error.kind = "rejection"``, which
    keeps it out of the tool-error stream (§12.1)."""
    from agentorg.types import ToolResultError

    return ToolResult(
        ok=False, error=ToolResultError(code=code, message=message, kind="rejection")
    )


def infra_result(code: str, message: str) -> ToolResult:
    """An ``ok=False`` result for a call the harness or the environment
    failed to carry out, so the call itself was not judged. Differs from
    `err_result` only in ``error.kind = "infra"``, which keeps it out of the
    tool-error stream (§12.1)."""
    from agentorg.types import ToolResultError

    return ToolResult(
        ok=False, error=ToolResultError(code=code, message=message, kind="infra")
    )
