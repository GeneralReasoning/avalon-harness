"""Background-task subsystem (§5.6).

Owned by a `Session`. A background task is an OS subprocess started via
`Bash(run_in_background=true)` (§5.5). The registry holds the live
process, streams stdout/stderr lines into the starting agent's inbox as
batched `task_update` notifications, and persists the full output to
`tasks/<task_id>/{stdout.log,stderr.log}` for later inspection via the
plain `Read` tool. The agent inspects accumulated state via the
`TaskRead` / `TaskKill` tools.

Lifecycle (§5.6):
- Session shutdown: every running task is terminated (see `shutdown`).
- Fired agent: every running task started by that agent is terminated
  (see `kill_agent_tasks`).
- Subprocess natural exit: the task transitions to `completed` (rc==0)
  or `failed` (rc!=0) and a final `task_update` notification is sent.

Notifications are BATCHED — 2 s flush interval OR 50 buffered lines —
because a chatty subprocess (e.g. `tail -f`) would otherwise flood the
agent's inbox and crowd out actual coordination. The notification body
includes a tail snippet plus the literal hint to call `TaskRead` for
the full output.

Tasks are NOT persisted across `Session.resume`: the OS subprocess dies
with the harness process, and re-surfacing a dead task with no live
output is more confusing than helpful. Past tasks' stdout/stderr logs
remain on disk for offline analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agentorg.types import AgentName, InboxItem, new_id, now_ms

if TYPE_CHECKING:
    from agentorg.session import Session

log = logging.getLogger(__name__)


# Batch stdout/stderr lines into one inbox notification every NOTIFY_S
# OR when NOTIFY_LINES lines have accumulated, whichever comes first.
NOTIFY_INTERVAL_S = 2.0
NOTIFY_LINES = 50
# In-memory ring bound on per-task buffered lines. Older lines stay
# on disk in `stdout.log` / `stderr.log` — TaskRead returns from the
# in-memory ring, so very-long-running tasks lose oldest visibility
# at this granularity (but can still be `Read` directly).
MAX_BUFFERED_LINES = 5000
# Sample tail size attached to each notification body.
NOTIFY_TAIL_LINES = 20
# Graceful-shutdown signals + timeout. We try SIGTERM, wait briefly,
# then SIGKILL the process group.
GRACEFUL_KILL_TIMEOUT_S = 2.0


@dataclass
class TaskOutputLine:
    stream: str  # "stdout" | "stderr"
    line: str
    ts: int

    def to_jsonable(self) -> dict[str, Any]:
        return {"stream": self.stream, "line": self.line, "ts": self.ts}


@dataclass
class BackgroundTask:
    task_id: str
    command: str
    starter_agent: AgentName
    cwd: str
    started_at: int
    stdout_path: Path
    stderr_path: Path
    description: Optional[str] = None
    state: str = "running"  # "running" | "completed" | "failed" | "killed"
    exit_code: Optional[int] = None
    ended_at: Optional[int] = None

    # Runtime-only (not persisted):
    proc: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    lines: list[TaskOutputLine] = field(default_factory=list, repr=False)
    _pending: list[TaskOutputLine] = field(default_factory=list, repr=False)
    _reader_tasks: list[asyncio.Task] = field(default_factory=list, repr=False)
    _flush_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _finalize_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_meta(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "command": self.command,
            "description": self.description,
            "starter_agent": self.starter_agent,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "state": self.state,
            "exit_code": self.exit_code,
            "ended_at": self.ended_at,
            "stdout_log": str(self.stdout_path),
            "stderr_log": str(self.stderr_path),
        }


class BackgroundTaskRegistry:
    """Per-session registry. One instance per `Session`."""

    def __init__(self, session: "Session"):
        self._session = session
        self._tasks: dict[str, BackgroundTask] = {}

    def get(self, task_id: str) -> Optional[BackgroundTask]:
        return self._tasks.get(task_id)

    def list(self, *, owner: Optional[AgentName] = None) -> list[BackgroundTask]:
        if owner is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.starter_agent == owner]

    async def start(
        self,
        *,
        command: str,
        cwd: str,
        starter: AgentName,
        description: Optional[str] = None,
    ) -> BackgroundTask:
        task_id = new_id("bg_")
        task_dir = self._session.paths.tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        # Create the log files so even an instant exit leaves something
        # for TaskRead to surface (avoids "No such file" on read).
        stdout_path.touch()
        stderr_path.touch()

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,  # own pgroup so SIGTERM hits children too
        )

        task = BackgroundTask(
            task_id=task_id,
            command=command,
            description=description,
            starter_agent=starter,
            cwd=cwd,
            started_at=now_ms(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            proc=proc,
        )
        self._tasks[task_id] = task

        _write_meta(task_dir, task.to_meta())

        # Reader + flush + finalize coroutines. We don't keep strong
        # references to the finalize task elsewhere because it's
        # self-terminating, but we DO need to ensure stdout/stderr
        # readers + flush loop are cancellable on shutdown.
        assert proc.stdout is not None and proc.stderr is not None
        task._reader_tasks = [
            asyncio.create_task(
                self._reader(task, proc.stdout, "stdout", stdout_path),
                name=f"bg-stdout:{task_id}",
            ),
            asyncio.create_task(
                self._reader(task, proc.stderr, "stderr", stderr_path),
                name=f"bg-stderr:{task_id}",
            ),
        ]
        task._flush_task = asyncio.create_task(
            self._flush_loop(task), name=f"bg-flush:{task_id}"
        )
        task._finalize_task = asyncio.create_task(
            self._finalize(task), name=f"bg-finalize:{task_id}"
        )
        log.info(
            "background task %s started: pid=%s cmd=%r starter=@%s",
            task_id,
            getattr(proc, "pid", "?"),
            command[:80],
            starter,
        )
        return task

    async def _reader(
        self,
        task: BackgroundTask,
        stream: asyncio.StreamReader,
        stream_name: str,
        log_path: Path,
    ) -> None:
        try:
            while True:
                line_b = await stream.readline()
                if not line_b:
                    return  # EOF
                # Persist verbatim to disk (preserves trailing newline).
                try:
                    with open(log_path, "ab") as f:
                        f.write(line_b)
                except Exception:
                    log.exception("failed to persist task line to %s", log_path)
                line = line_b.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                rec = TaskOutputLine(stream=stream_name, line=line, ts=now_ms())
                # In-memory ring.
                task.lines.append(rec)
                if len(task.lines) > MAX_BUFFERED_LINES:
                    task.lines = task.lines[-MAX_BUFFERED_LINES:]
                task._pending.append(rec)
                if len(task._pending) >= NOTIFY_LINES:
                    await self._flush(task, reason="line-threshold")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("background reader (%s) failed for %s", stream_name, task.task_id)

    async def _flush_loop(self, task: BackgroundTask) -> None:
        try:
            while task.state == "running":
                await asyncio.sleep(NOTIFY_INTERVAL_S)
                await self._flush(task, reason="interval")
        except asyncio.CancelledError:
            raise

    async def _flush(self, task: BackgroundTask, *, reason: str) -> None:
        if not task._pending:
            return
        batch, task._pending = task._pending, []
        tail = batch[-NOTIFY_TAIL_LINES:]
        sample = "\n".join(f"[{ln.stream}] {ln.line}" for ln in tail)
        truncated_note = (
            f"(showing last {len(tail)} of {len(batch)} new line(s))"
            if len(batch) > len(tail)
            else f"({len(batch)} new line(s))"
        )
        body = (
            f"Background task {task.task_id} produced new output {truncated_note}:\n"
            f"{sample}\n\n"
            f"Use TaskRead(task_id={task.task_id!r}) for the full buffer."
        )
        item = InboxItem(
            kind="task_update",
            content=body,
            payload={
                "task_id": task.task_id,
                "new_lines": len(batch),
                "state": task.state,
            },
        )
        try:
            await self._session.inbox.append(task.starter_agent, item)
            self._session.wakes.wake(task.starter_agent)
        except Exception:
            log.exception("failed to deliver task_update notification for %s", task.task_id)

    async def _finalize(self, task: BackgroundTask) -> None:
        if task.proc is None:
            return
        try:
            rc = await task.proc.wait()
        except asyncio.CancelledError:
            raise
        # Drain readers (they exit at EOF naturally).
        for r in task._reader_tasks:
            try:
                await r
            except (asyncio.CancelledError, Exception):
                pass
        # Stop the flush loop and flush whatever is left.
        if task._flush_task and not task._flush_task.done():
            task._flush_task.cancel()
            try:
                await task._flush_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._flush(task, reason="exit-drain")
        # Update state (don't clobber an earlier kill->killed transition).
        if task.state == "running":
            task.state = "completed" if rc == 0 else "failed"
        task.exit_code = rc
        task.ended_at = now_ms()
        _write_meta(
            self._session.paths.tasks_dir / task.task_id,
            task.to_meta(),
        )
        # Final notification (always — the agent needs to know the task ended).
        final = InboxItem(
            kind="task_update",
            content=(
                f"Background task {task.task_id} ended with state={task.state}, "
                f"exit_code={task.exit_code}.\n"
                f"Use TaskRead(task_id={task.task_id!r}) for the full output, "
                f"or Read {task.stdout_path} / {task.stderr_path} directly."
            ),
            payload={
                "task_id": task.task_id,
                "state": task.state,
                "exit_code": task.exit_code,
                "final": True,
            },
        )
        try:
            await self._session.inbox.append(task.starter_agent, final)
            self._session.wakes.wake(task.starter_agent)
        except Exception:
            log.exception("failed to deliver final task_update for %s", task.task_id)

    async def kill(self, task_id: str) -> bool:
        """Terminate `task_id`. Returns True iff the task existed.

        Idempotent on already-terminated tasks (no-op).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.state != "running" or task.proc is None:
            return True
        proc = task.proc
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Process already exited, or pgroup gone — fall through to wait.
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=GRACEFUL_KILL_TIMEOUT_S)
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=GRACEFUL_KILL_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.error("background task %s did not die after SIGKILL", task_id)
        # Mark BEFORE awaiting finalize so the finalizer sees "killed" and
        # doesn't overwrite it with completed/failed.
        task.state = "killed"
        # Let the finalize coroutine clean up readers + flush + final
        # notification; we don't await it here to avoid recursive cancel
        # complications when kill is called from shutdown().
        return True

    async def shutdown(self) -> None:
        """Terminate every running task. Called at session end (§5.6)."""
        # Snapshot ids — kill() mutates state and we don't want to
        # iterate a mutating dict.
        ids = [tid for tid, t in self._tasks.items() if t.state == "running"]
        for tid in ids:
            try:
                await self.kill(tid)
            except Exception:
                log.exception("background-task shutdown failed for %s", tid)
        # Give finalizers a moment to flush their final notifications
        # before the session.ended event closes the bus.
        pending_finalizers = [
            t._finalize_task for t in self._tasks.values()
            if t._finalize_task is not None and not t._finalize_task.done()
        ]
        if pending_finalizers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_finalizers, return_exceptions=True),
                    timeout=GRACEFUL_KILL_TIMEOUT_S + 1.0,
                )
            except asyncio.TimeoutError:
                pass

    async def kill_agent_tasks(self, agent: AgentName) -> None:
        """Terminate every running task started by `agent` (§5.6 on fire)."""
        ids = [
            tid for tid, t in self._tasks.items()
            if t.starter_agent == agent and t.state == "running"
        ]
        for tid in ids:
            try:
                await self.kill(tid)
            except Exception:
                log.exception("kill_agent_tasks failed for %s", tid)


def _write_meta(task_dir: Path, meta: dict) -> None:
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2))
