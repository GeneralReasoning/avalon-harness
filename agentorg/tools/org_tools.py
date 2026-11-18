"""Org-level tools required in every session (§5.5)."""

from __future__ import annotations

from typing import Optional

from agentorg.tools.context import (
    err_result,
    ok_result,
    rejected_result,
    ToolContext,
)
from agentorg.types import (
    ToolDefinition,
    ToolResult,
)


# Single per-message body cap shared across SendMessage, PostToChannel, and
# ReportToCEO. 16 KiB ≈ 4–5K tokens — comfortably holds a structured status
# report or a multi-paragraph DM, but small enough that no single message can
# dominate a recipient's next-turn context. Anything bigger belongs in the
# archive (`WriteToArchive`); link it via `attachments` instead.
MAX_MESSAGE_BYTES = 16 * 1024

# Back-compat alias. PostToChannel originally allowed 64 KiB; tightening it
# is the same kind of "limit at source" change as adding the cap to SendMessage
# / ReportToCEO, just on an existing constant. Anything still importing the
# old name resolves to the new value.
MAX_POST_BYTES = MAX_MESSAGE_BYTES


def _too_long_error(tool_name: str, content_bytes: int, field: str = "content") -> ToolResult:
    """Shared 'content body too large' response for the message-emitting tools."""
    return err_result(
        "content-too-large",
        f"{tool_name} {field} is {content_bytes} bytes; cap is {MAX_MESSAGE_BYTES} "
        f"({MAX_MESSAGE_BYTES // 1024} KiB). Write the long content to the archive "
        "via WriteToArchive(...) and reference it via `attachments` instead.",
    )


def _require_ceo(ctx: ToolContext, action: str) -> Optional[ToolResult]:
    """Return a not-permitted ToolResult if caller is not the CEO, else None."""
    if ctx.agent.kind != "ceo":
        return rejected_result("not-permitted", f"Only the CEO can {action}.")
    return None


class TodoWriteTool:
    definition = ToolDefinition(
        name="TodoWrite",
        description="Replace the agent's local todo list. Each item: {content, status}.",
        input_schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        # Session-local todo storage.
        store = getattr(ctx.session, "_todos", None)
        if store is None:
            store = {}
            setattr(ctx.session, "_todos", store)
        store[ctx.agent.name] = ctx.args.get("todos") or []
        return ok_result({"todos": store[ctx.agent.name]})


class AskUserQuestionTool:
    definition = ToolDefinition(
        name="AskUserQuestion",
        description=(
            "Ask the external operator a question. In headless mode, the harness picks "
            "the first option automatically (per spec §A.7)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "options"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        options = ctx.args.get("options") or []
        if ctx.session.headless:
            choice = options[0] if options else None
            return ok_result({"answer": choice, "auto_chosen": True})
        # Interactive mode has no resolver wired through `ExternalInterface`
        # yet (spec §A.7). Returning `ok` with `answer=None` would silently
        # break the agent's tool-call expectations — it would believe the
        # call succeeded with no answer and likely re-ask in a loop.
        # Surface a clear error code instead so the caller can choose to
        # disable the tool or implement the resolver.
        return err_result(
            "headless-only",
            "AskUserQuestion currently requires headless mode; no interactive "
            "resolver is wired through the external interface yet.",
        )


# Re-exports — see comms_tools / archive_tools / wait_tools / org_mgmt_tools
# for the actual implementations. Kept here so external imports
# (`from agentorg.tools.org_tools import X`) keep working.
from agentorg.tools.comms_tools import (  # noqa: E402
    SendMessageTool,
    PostToChannelTool,
    CreateChannelsTool,
    ListChannelsTool,
    ViewChannelMessagesTool,
    SearchChannelMessagesTool,
    ViewMyHistoryTool,
    SearchMyHistoryTool,
    ReactToMessageTool,
    SetChannelSubscriptionTool,
    _fanout,  # noqa: F401  (imported by agentorg.external)
)
from agentorg.tools.archive_tools import (  # noqa: E402
    ListArchiveTool,
    ViewArchiveFileTool,
    WriteToArchiveTool,
    EditArchiveFileTool,
    MoveArchiveFileTool,
)
from agentorg.tools.wait_tools import (  # noqa: E402
    WaitForMessagesTool,
    ReportToCEOTool,
)
from agentorg.tools.org_mgmt_tools import (  # noqa: E402
    WritePlanTool,
    EnterOrgPlanModeTool,
    ExitOrgPlanModeTool,
    HireOrgAgentTool,
    FireOrgAgentTool,
    UpdateOrgAgentToolsTool,
)

__all__ = [
    # Constants / helpers
    "MAX_MESSAGE_BYTES",
    "MAX_POST_BYTES",
    # Productivity
    "TodoWriteTool",
    "AskUserQuestionTool",
    # Comms
    "SendMessageTool",
    "PostToChannelTool",
    "CreateChannelsTool",
    "ListChannelsTool",
    "ViewChannelMessagesTool",
    "SearchChannelMessagesTool",
    "ViewMyHistoryTool",
    "SearchMyHistoryTool",
    "ReactToMessageTool",
    "SetChannelSubscriptionTool",
    # Archive
    "ListArchiveTool",
    "ViewArchiveFileTool",
    "WriteToArchiveTool",
    "EditArchiveFileTool",
    "MoveArchiveFileTool",
    # Wait / report
    "WaitForMessagesTool",
    "ReportToCEOTool",
    # Org management
    "WritePlanTool",
    "EnterOrgPlanModeTool",
    "ExitOrgPlanModeTool",
    "HireOrgAgentTool",
    "FireOrgAgentTool",
    "UpdateOrgAgentToolsTool",
]
