"""Local-workspace tools: Read / Write / Edit / Glob / Grep / Bash / WebFetch.

These mirror the standard Claude Code tool surface so the
prompts/examples in the spec compose unchanged. All paths are resolved
relative to the agent's `work_dir`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import itertools
import os
import re
from pathlib import Path

import httpx

from agentorg.tools.context import (
    MAX_TOOL_OUTPUT_BYTES,
    ToolContext,
    err_result,
    ok_result,
)
from agentorg.types import ToolDefinition, ToolResult

# `Read` uses the shared per-tool-result cap (see `MAX_TOOL_OUTPUT_BYTES`).
# Re-exported under the historical name so callers that imported the
# `Read`-specific constant keep working.
MAX_READ_OUTPUT_BYTES = MAX_TOOL_OUTPUT_BYTES

# Lines longer than this are hard-truncated. Real source files don't hit this;
# minified bundles, NDJSON, and accidental binary reads do — they would
# otherwise burn the entire output cap on a single line.
MAX_READ_LINE_CHARS = 2_000

# Probe size for binary-file detection. NUL bytes anywhere in the head are a
# robust signal that the file is binary (Python's `open(..., errors="replace")`
# happily decodes binary garbage into U+FFFD without raising, so the existing
# `except UnicodeDecodeError` branch in `ReadTool.execute` is unreachable).
_READ_BINARY_PROBE_BYTES = 8 * 1024

MAX_GREP_HITS = 200


def _resolve_workdir(ctx: ToolContext, path: str) -> Path:
    base = Path(ctx.agent.work_dir).expanduser()
    p = (base / path) if not os.path.isabs(path) else Path(path)
    return p


class ReadTool:
    definition = ToolDefinition(
        name="Read",
        description="Read a file from the agent's workspace (absolute or relative path).",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["file_path"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        p = _resolve_workdir(ctx, ctx.args["file_path"])
        if not p.exists():
            return err_result("not-found", f"No such file: {p}")
        if p.is_dir():
            return err_result("is-directory", f"{p} is a directory")
        # Reject binaries before decoding. `errors="replace"` masks decode
        # failures into U+FFFD, so a textual decode of an ELF / PNG / etc.
        # silently produces megabytes of garbage that pin the agent's
        # context above the model limit until the session is aborted.
        try:
            with p.open("rb") as fb:
                head = fb.read(_READ_BINARY_PROBE_BYTES)
        except OSError as e:
            return err_result("io-error", str(e))
        if b"\x00" in head:
            return err_result(
                "binary-file",
                f"{p} appears to be binary (NUL byte in first {_READ_BINARY_PROBE_BYTES} bytes).",
            )
        offset = int(ctx.args.get("offset") or 0)
        limit = int(ctx.args.get("limit") or 2000)
        # Stream the file via `islice` so a 10 GB log doesn't OOM the
        # harness when the model asks for a small offset/limit window.
        try:
            view: list[str] = []
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in itertools.islice(f, offset, offset + limit):
                    view.append(line)
        except UnicodeDecodeError:
            return err_result("binary-file", "File appears to be binary")
        numbered: list[str] = []
        bytes_used = 0
        line_truncations = 0
        last_emitted_offset = offset
        output_truncated = False
        for i, line in enumerate(view, start=offset + 1):
            text = line.rstrip()
            if len(text) > MAX_READ_LINE_CHARS:
                text = text[:MAX_READ_LINE_CHARS] + "…[line truncated]"
                line_truncations += 1
            formatted = f"{i:6d}\t{text}"
            # +1 accounts for the joining "\n" we'd add between lines.
            projected = bytes_used + len(formatted.encode("utf-8")) + (1 if numbered else 0)
            if projected > MAX_READ_OUTPUT_BYTES:
                output_truncated = True
                break
            numbered.append(formatted)
            bytes_used = projected
            last_emitted_offset = i
        output = {
            "path": str(p),
            "lines": "\n".join(numbered),
            "viewed_lines": len(numbered),
            "offset": offset,
            "limit": limit,
        }
        if line_truncations:
            output["line_truncations"] = line_truncations
        if output_truncated:
            output["truncated"] = True
            output["next_offset"] = last_emitted_offset
            output["note"] = (
                f"Output capped at {MAX_READ_OUTPUT_BYTES} bytes; "
                f"call Read again with offset={last_emitted_offset} to continue, "
                "or use Grep to locate a specific region."
            )
        return ok_result(output)


class WriteTool:
    definition = ToolDefinition(
        name="Write",
        description="Write content to a file in the workspace, overwriting if present.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        p = _resolve_workdir(ctx, ctx.args["file_path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_text, ctx.args["content"], encoding="utf-8")
        return ok_result({"path": str(p), "bytes": len(ctx.args["content"].encode("utf-8"))})


class EditTool:
    definition = ToolDefinition(
        name="Edit",
        description=(
            "Exact-string replacement in a workspace file. The old_string must be unique "
            "unless replace_all is true."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        p = _resolve_workdir(ctx, ctx.args["file_path"])
        if not p.exists():
            return err_result("not-found", f"No such file: {p}")
        text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        old = ctx.args["old_string"]
        new = ctx.args["new_string"]
        if old not in text:
            return err_result("no-match", "old_string not found in file")
        if not ctx.args.get("replace_all") and text.count(old) > 1:
            return err_result(
                "ambiguous",
                "old_string occurs multiple times — pass replace_all=true or expand context.",
            )
        if ctx.args.get("replace_all"):
            count = text.count(old)
            updated = text.replace(old, new)
        else:
            count = 1
            updated = text.replace(old, new, 1)
        await asyncio.to_thread(p.write_text, updated, encoding="utf-8")
        return ok_result({"path": str(p), "replacements": count})


class GlobTool:
    definition = ToolDefinition(
        name="Glob",
        description="Glob-match files in the workspace. Pattern uses fnmatch syntax; e.g. '**/*.py'.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        base = Path(ctx.args.get("path") or ctx.agent.work_dir).expanduser()
        if not base.exists():
            return err_result("not-found", str(base))
        pattern = ctx.args["pattern"]
        matches = []
        for path in base.rglob("*"):
            rel = path.relative_to(base).as_posix()
            if fnmatch.fnmatch(rel, pattern):
                matches.append(str(path))
        return ok_result({"matches": matches[:1000]})


class GrepTool:
    definition = ToolDefinition(
        name="Grep",
        description="Regex search across workspace files. Optional path/include filters.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string", "description": "Glob filter, e.g. '*.py'"},
                "ignore_case": {"type": "boolean"},
                "head_limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["pattern"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        base = Path(ctx.args.get("path") or ctx.agent.work_dir).expanduser()
        if not base.exists():
            return err_result("not-found", str(base))
        flags = re.MULTILINE
        if ctx.args.get("ignore_case"):
            flags |= re.IGNORECASE
        try:
            rx = re.compile(ctx.args["pattern"], flags)
        except re.error as e:
            return err_result("invalid-regex", str(e))
        include = ctx.args.get("include")
        head = int(ctx.args.get("head_limit") or 100)
        hits = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if include and not fnmatch.fnmatch(path.name, include):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"path": str(path), "line": i, "text": line.rstrip()})
                    if len(hits) >= head:
                        break
            if len(hits) >= head:
                break
        return ok_result({"hits": hits})


class BashTool:
    definition = ToolDefinition(
        name="Bash",
        description=(
            "Run a shell command in the agent's workspace. Default timeout 120s, max 600s. "
            "Set `run_in_background=true` to spawn a long-running task; the call returns "
            "immediately with `task_id` and `state=running`. Inbox notifications stream "
            "as the task produces output; use `TaskRead(task_id)` to inspect the full "
            "buffer and `TaskKill(task_id)` to terminate."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                "run_in_background": {"type": "boolean"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        cmd = ctx.args["command"]
        description = ctx.args.get("description")
        if ctx.args.get("run_in_background"):
            task = await ctx.session.background_tasks.start(
                command=cmd,
                cwd=ctx.agent.work_dir,
                starter=ctx.agent.name,
                description=description,
            )
            return ok_result(
                {
                    "task_id": task.task_id,
                    "state": task.state,
                    "started_at": task.started_at,
                    "stdout_log": str(task.stdout_path),
                    "stderr_log": str(task.stderr_path),
                    "note": (
                        "Task is running in the background. You'll receive "
                        "`task_update` notifications as it produces output and "
                        "a final notification when it exits. Use `TaskRead` to "
                        "inspect the full buffer, or `TaskKill` to terminate."
                    ),
                }
            )
        timeout = int(ctx.args.get("timeout") or 120)
        timeout = max(1, min(600, timeout))
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ctx.agent.work_dir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            # Reap the zombie so the kernel can release process resources.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            return err_result("timeout", f"Command exceeded {timeout}s")
        return ok_result(
            {
                "stdout": stdout.decode("utf-8", errors="replace")[-20000:],
                "stderr": stderr.decode("utf-8", errors="replace")[-20000:],
                "exit_code": proc.returncode,
            }
        )


class TaskReadTool:
    """Read accumulated output of a background `Bash` task (§5.6)."""

    definition = ToolDefinition(
        name="TaskRead",
        description=(
            "Read accumulated stdout/stderr lines of a background task started by "
            "`Bash(run_in_background=true)`. Use `since` to skip lines you've already "
            "seen. Returns the current state + exit code if the task has finished."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "since": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Skip the first N buffered lines.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                    "description": "Max lines to return (default 500).",
                },
            },
            "required": ["task_id"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        task_id = ctx.args["task_id"]
        task = ctx.session.background_tasks.get(task_id)
        if task is None:
            return err_result("unknown-task", f"No background task {task_id!r}")
        since = int(ctx.args.get("since") or 0)
        limit = int(ctx.args.get("limit") or 500)
        view = task.lines[since : since + limit]
        return ok_result(
            {
                "task_id": task.task_id,
                "state": task.state,
                "exit_code": task.exit_code,
                "starter_agent": task.starter_agent,
                "command": task.command,
                "since": since,
                "returned": len(view),
                "buffered_total": len(task.lines),
                "stdout_log": str(task.stdout_path),
                "stderr_log": str(task.stderr_path),
                "lines": [ln.to_jsonable() for ln in view],
            }
        )


class TaskKillTool:
    """Terminate a running background task (§5.6)."""

    definition = ToolDefinition(
        name="TaskKill",
        description=(
            "Terminate a background task started by `Bash(run_in_background=true)`. "
            "Sends SIGTERM to the task's process group, then SIGKILL after 2s if still "
            "alive. Idempotent on already-finished tasks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
            "required": ["task_id"],
        },
        read_only=False,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        task_id = ctx.args["task_id"]
        task = ctx.session.background_tasks.get(task_id)
        if task is None:
            return err_result("unknown-task", f"No background task {task_id!r}")
        prior_state = task.state
        ok = await ctx.session.background_tasks.kill(task_id)
        return ok_result(
            {
                "task_id": task_id,
                "ok": ok,
                "prior_state": prior_state,
                "state": task.state,
                "exit_code": task.exit_code,
            }
        )


class WebFetchTool:
    definition = ToolDefinition(
        name="WebFetch",
        description=(
            "Fetch a URL and return text content. Renders HTML to text. "
            "Provide an optional prompt for context."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["url"],
        },
        read_only=True,
        scope="local",
    )

    async def execute(self, ctx: ToolContext) -> ToolResult:
        url = ctx.args["url"]
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await client.get(url)
                content_type = r.headers.get("content-type", "")
                text = r.text
        except Exception as e:
            return err_result("fetch-error", str(e))
        if "html" in content_type.lower():
            # Cheap HTML -> text: strip tags.
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return ok_result({"url": url, "content": text[:MAX_TOOL_OUTPUT_BYTES]})


def all_local_tools() -> list:
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        BashTool(),
        TaskReadTool(),
        TaskKillTool(),
        WebFetchTool(),
    ]
