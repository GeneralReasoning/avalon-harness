"""Archive tools: read/write/edit/move markdown files in the shared archive."""

from __future__ import annotations

from typing import Any

from agentorg.archive import ArchiveError
from agentorg.tools.context import (
    MAX_TOOL_OUTPUT_BYTES,
    ToolContext,
    err_result,
    ok_result,
)
from agentorg.types import (
    ToolDefinition,
    ToolResult,
)


class ListArchiveTool:
    definition = ToolDefinition(
        name="ListArchive",
        description="List markdown files in the session's shared archive.",
        input_schema={"type": "object", "properties": {}},
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        return ok_result({"files": ctx.session.archive.list()})


class ViewArchiveFileTool:
    definition = ToolDefinition(
        name="ViewArchiveFile",
        description=(
            "Read a markdown file from the archive by relative POSIX path. "
            "Files larger than the per-tool output cap are paged: pass `offset` "
            "(byte position) on the follow-up call to continue."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["path"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            content = ctx.session.archive.read(ctx.args["path"])
        except ArchiveError as e:
            return err_result("archive-error", str(e))
        offset = int(ctx.args.get("offset") or 0)
        # Byte-offset, not char-offset, so paging composes correctly even when
        # the file contains multi-byte UTF-8 sequences. Archive files are capped
        # at 256 KiB on write, so two or three calls cover any well-formed file.
        body = content.encode("utf-8")
        end = offset + MAX_TOOL_OUTPUT_BYTES
        window = body[offset:end].decode("utf-8", errors="replace")
        out: dict[str, Any] = {
            "path": ctx.args["path"],
            "content": window,
            "offset": offset,
            "total_bytes": len(body),
        }
        if end < len(body):
            out["truncated"] = True
            out["next_offset"] = end
        return ok_result(out)


class WriteToArchiveTool:
    definition = ToolDefinition(
        name="WriteToArchive",
        description="Write a markdown file into the shared archive. Path is archive-relative.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            await ctx.session.archive.write(ctx.args["path"], ctx.args["content"])
        except ArchiveError as e:
            return err_result("archive-error", str(e))
        return ok_result({"path": ctx.args["path"], "bytes": len(ctx.args["content"].encode("utf-8"))})


class EditArchiveFileTool:
    definition = ToolDefinition(
        name="EditArchiveFile",
        description=(
            "Exact-string edit of an archive file. Pass old_string and new_string; "
            "replace_all=false requires a unique match."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            n = await ctx.session.archive.edit(
                ctx.args["path"],
                ctx.args["old_string"],
                ctx.args["new_string"],
                bool(ctx.args.get("replace_all") or False),
            )
        except ArchiveError as e:
            return err_result("archive-error", str(e))
        return ok_result({"path": ctx.args["path"], "replacements": n})


class MoveArchiveFileTool:
    definition = ToolDefinition(
        name="MoveArchiveFile",
        description="Rename or move a markdown file inside the archive.",
        input_schema={
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            await ctx.session.archive.move(ctx.args["src"], ctx.args["dst"])
        except ArchiveError as e:
            return err_result("archive-error", str(e))
        return ok_result({"src": ctx.args["src"], "dst": ctx.args["dst"]})
