"""Background task subsystem tests (SPEC §5.6)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

import pytest

from agentorg.background_tasks import (
    NOTIFY_INTERVAL_S,
)
from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    InferenceProvider,
    InferenceRequest,
    TokenUsage,
)
from agentorg.org import spawn_org_agent
from agentorg.session import Session, SessionConfig
from agentorg.tools.context import ToolContext
from agentorg.tools.fs_tools import BashTool, TaskKillTool, TaskReadTool
from agentorg.tools.registry import ToolRegistry
from agentorg.types import OrgAgentSpec


class _Stub(InferenceProvider):
    async def stream(self, request: InferenceRequest) -> AsyncIterator:
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


def _make_session(tmp_path: Path) -> Session:
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=2,
        headless=True,
        permission_policy="bypass",
    )
    return Session(config=config, inference=_Stub(), tools=ToolRegistry())


def _ceo_ctx(session: Session, args: dict, *, call_id: str = "c1") -> ToolContext:
    from agentorg.types import Agent, CEO_NAME

    ceo = session.agent(CEO_NAME)
    if ceo is None:
        ceo = Agent(
            name=CEO_NAME, session_id=session.session_id, kind="ceo", role="CEO",
            work_dir=session.state.workspace_path, system_prompt="",
        )
        session._agents[CEO_NAME] = ceo
    return ToolContext(agent=ceo, session=session, args=args, call_id=call_id)


# ---------------------------------------------------------------------------
# Bash background mode — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_background_returns_task_id_immediately(tmp_path):
    """`Bash(run_in_background=true)` returns ~instantly with task_id + state."""
    session = _make_session(tmp_path)
    bash = BashTool()

    t0 = time.monotonic()
    r = await bash.execute(
        _ceo_ctx(
            session,
            {"command": "sleep 0.5; echo done", "run_in_background": True},
        )
    )
    elapsed = time.monotonic() - t0

    assert r.ok is True
    assert "task_id" in r.output
    assert r.output["state"] == "running"
    assert elapsed < 0.3, f"Bash background should return immediately, took {elapsed:.2f}s"
    # Log files exist (even if empty).
    assert Path(r.output["stdout_log"]).exists()
    assert Path(r.output["stderr_log"]).exists()

    # Wait for the task to finish then ensure cleanup.
    await asyncio.sleep(1.5)
    task = session.background_tasks.get(r.output["task_id"])
    assert task is not None
    assert task.state == "completed"
    assert task.exit_code == 0


@pytest.mark.asyncio
async def test_background_task_output_flushes_to_inbox(tmp_path):
    """New stdout lines stream into the starting agent's inbox as task_update."""
    session = _make_session(tmp_path)
    bash = BashTool()

    # Emit enough lines fast to hit the line-threshold flush quickly.
    r = await bash.execute(
        _ceo_ctx(
            session,
            {
                "command": "for i in $(seq 1 60); do echo line-$i; done",
                "run_in_background": True,
            },
        )
    )
    assert r.output["task_id"]

    # Wait for the line-threshold flush + the final exit flush.
    await asyncio.sleep(NOTIFY_INTERVAL_S + 1.0)

    items = session.inbox.load(session.ceo_name)
    task_updates = [i for i in items if i.kind == "task_update"]
    assert task_updates, "no task_update notifications landed in the CEO inbox"

    # The final notification is the one marked `final=True`.
    finals = [i for i in task_updates if (i.payload or {}).get("final") is True]
    assert len(finals) == 1
    assert finals[0].payload["state"] == "completed"
    assert finals[0].payload["exit_code"] == 0


@pytest.mark.asyncio
async def test_task_read_returns_buffered_lines(tmp_path):
    """`TaskRead` returns the in-memory line buffer."""
    session = _make_session(tmp_path)
    bash = BashTool()
    read = TaskReadTool()

    r = await bash.execute(
        _ceo_ctx(
            session,
            {"command": "echo a; echo b; echo c", "run_in_background": True},
        )
    )
    task_id = r.output["task_id"]

    # Wait for the subprocess to exit (and the finalizer to flush).
    await asyncio.sleep(0.5)

    rr = await read.execute(_ceo_ctx(session, {"task_id": task_id}, call_id="c2"))
    assert rr.ok is True
    out = rr.output
    assert out["state"] == "completed"
    assert out["exit_code"] == 0
    assert out["buffered_total"] >= 3
    lines = [ln["line"] for ln in out["lines"]]
    assert "a" in lines and "b" in lines and "c" in lines


@pytest.mark.asyncio
async def test_task_read_since_offset(tmp_path):
    """`since` skips lines already read."""
    session = _make_session(tmp_path)
    bash = BashTool()
    read = TaskReadTool()

    r = await bash.execute(
        _ceo_ctx(
            session,
            {"command": "for i in 1 2 3 4 5; do echo $i; done", "run_in_background": True},
        )
    )
    task_id = r.output["task_id"]
    await asyncio.sleep(0.5)

    first = await read.execute(_ceo_ctx(session, {"task_id": task_id, "limit": 2}, call_id="c2"))
    assert first.output["returned"] == 2
    second = await read.execute(
        _ceo_ctx(session, {"task_id": task_id, "since": 2}, call_id="c3")
    )
    # All remaining buffered lines (>= 3).
    assert second.output["returned"] >= 3


@pytest.mark.asyncio
async def test_task_read_unknown_id(tmp_path):
    session = _make_session(tmp_path)
    read = TaskReadTool()
    r = await read.execute(_ceo_ctx(session, {"task_id": "bg_doesnotexist"}))
    assert r.ok is False
    assert r.error.code == "unknown-task"


# ---------------------------------------------------------------------------
# TaskKill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_kill_terminates_running_task(tmp_path):
    """`TaskKill` SIGTERMs the pgroup and transitions state to `killed`."""
    session = _make_session(tmp_path)
    bash = BashTool()
    kill = TaskKillTool()

    r = await bash.execute(
        _ceo_ctx(
            session,
            {"command": "sleep 30; echo never", "run_in_background": True},
        )
    )
    task_id = r.output["task_id"]
    # Make sure it actually started.
    await asyncio.sleep(0.1)
    assert session.background_tasks.get(task_id).state == "running"

    t0 = time.monotonic()
    kr = await kill.execute(_ceo_ctx(session, {"task_id": task_id}, call_id="c2"))
    elapsed = time.monotonic() - t0

    assert kr.ok is True
    assert kr.output["prior_state"] == "running"
    assert kr.output["state"] == "killed"
    # SIGTERM should suffice for a plain `sleep` — well under graceful timeout.
    assert elapsed < 2.5, f"TaskKill took too long: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_task_kill_is_idempotent(tmp_path):
    """Killing an already-finished task is a no-op (still returns ok)."""
    session = _make_session(tmp_path)
    bash = BashTool()
    kill = TaskKillTool()

    r = await bash.execute(
        _ceo_ctx(session, {"command": "echo fast", "run_in_background": True})
    )
    task_id = r.output["task_id"]
    await asyncio.sleep(0.3)
    # First kill (already exited): ok.
    k1 = await kill.execute(_ceo_ctx(session, {"task_id": task_id}, call_id="c2"))
    assert k1.ok is True
    # Second kill: also ok.
    k2 = await kill.execute(_ceo_ctx(session, {"task_id": task_id}, call_id="c3"))
    assert k2.ok is True


# ---------------------------------------------------------------------------
# Session shutdown + fired-agent lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_shutdown_terminates_running_tasks(tmp_path):
    """`background_tasks.shutdown()` kills every running task (§5.6)."""
    session = _make_session(tmp_path)
    bash = BashTool()

    # Start two long-runners.
    ra = await bash.execute(
        _ceo_ctx(session, {"command": "sleep 30", "run_in_background": True})
    )
    rb = await bash.execute(
        _ceo_ctx(
            session, {"command": "sleep 30", "run_in_background": True}, call_id="c2"
        )
    )
    await asyncio.sleep(0.1)
    assert session.background_tasks.get(ra.output["task_id"]).state == "running"
    assert session.background_tasks.get(rb.output["task_id"]).state == "running"

    await session.background_tasks.shutdown()

    assert session.background_tasks.get(ra.output["task_id"]).state == "killed"
    assert session.background_tasks.get(rb.output["task_id"]).state == "killed"


@pytest.mark.asyncio
async def test_firing_starter_agent_kills_owned_tasks(tmp_path):
    """When an org agent is fired, every task it started MUST be terminated."""
    from agentorg.types import ExitReview
    from agentorg.org import fire_org_agent

    harness = Harness(inference=_Stub())
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=2,
        headless=True,
        permission_policy="bypass",
    )
    session = await harness.start_session(config=config)
    org = await spawn_org_agent(
        session, OrgAgentSpec(role="Worker", description="x")
    )

    # Worker starts a long-running task.
    bash = BashTool()
    org_ctx = ToolContext(
        agent=org,
        session=session,
        args={"command": "sleep 30", "run_in_background": True},
        call_id="cw1",
    )
    r = await bash.execute(org_ctx)
    task_id = r.output["task_id"]
    await asyncio.sleep(0.1)
    assert session.background_tasks.get(task_id).state == "running"

    # Fire the worker.
    review = ExitReview(category="mission_complete", delivered="x", shortfalls="—", score=5)
    await fire_org_agent(session, org.name, review)

    # Wait briefly for kill propagation.
    await asyncio.sleep(0.3)
    assert session.background_tasks.get(task_id).state == "killed"


# ---------------------------------------------------------------------------
# On-disk artefacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_logs_and_meta_persisted(tmp_path):
    """stdout.log, stderr.log, and meta.json are written under tasks/<id>/."""
    session = _make_session(tmp_path)
    bash = BashTool()

    r = await bash.execute(
        _ceo_ctx(
            session,
            {
                "command": "echo to-stdout; echo to-stderr 1>&2",
                "run_in_background": True,
                "description": "trivial test",
            },
        )
    )
    task_id = r.output["task_id"]
    await asyncio.sleep(0.5)

    task_dir = session.paths.tasks_dir / task_id
    assert task_dir.exists()
    stdout = (task_dir / "stdout.log").read_text()
    stderr = (task_dir / "stderr.log").read_text()
    assert "to-stdout" in stdout
    assert "to-stderr" in stderr
    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["task_id"] == task_id
    assert meta["state"] in ("completed", "failed", "running")  # may not have flushed yet
    assert meta["command"].startswith("echo")
    assert meta["description"] == "trivial test"
