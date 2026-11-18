"""On-disk persistence for sessions, per SPEC.md §11.

All JSONL files are append-only during a session. Readers must tolerate a
truncated final line (crash mid-write). Layout:

    <workspace>/.agentorg/
      sessions/<session_id>/
        session.json
        events.jsonl
        messages.jsonl
        pending.jsonl
        inference-errors.jsonl
        org/<name>/{agent.json, messages.jsonl, pending.jsonl}
        channels/<channel>.jsonl
        channels/_meta.json
        archive/<path>.md
        plans/plan-<id>.{md,json}
        tasks/<task_id>/{stdout.log, stderr.log, meta.json}
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pydantic import BaseModel

HARNESS_DIRNAME = ".agentorg"


def workspace_dir(workspace: str) -> Path:
    return Path(workspace).resolve() / HARNESS_DIRNAME


def session_dir(workspace: str, session_id: str) -> Path:
    return workspace_dir(workspace) / "sessions" / session_id


class SessionPaths:
    """Filesystem layout for a single session."""

    def __init__(self, workspace_path: str, session_id: str) -> None:
        self.workspace = Path(workspace_path).resolve()
        self.session_id = session_id
        self.root = session_dir(workspace_path, session_id)
        self.session_json = self.root / "session.json"
        self.events_jsonl = self.root / "events.jsonl"
        self.messages_jsonl = self.root / "messages.jsonl"  # CEO transcript
        self.pending_jsonl = self.root / "pending.jsonl"  # CEO inbox
        self.inference_errors = self.root / "inference-errors.jsonl"
        self.org_dir = self.root / "org"
        self.channels_dir = self.root / "channels"
        self.channels_meta = self.channels_dir / "_meta.json"
        self.archive_dir = self.root / "archive"
        self.plans_dir = self.root / "plans"
        self.tasks_dir = self.root / "tasks"

    def ensure(self) -> None:
        for p in (
            self.root,
            self.org_dir,
            self.channels_dir,
            self.archive_dir,
            self.plans_dir,
            self.tasks_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def agent_dir(self, name: str) -> Path:
        return self.org_dir / name

    def agent_messages(self, name: str) -> Path:
        return self.agent_dir(name) / "messages.jsonl"

    def agent_pending(self, name: str) -> Path:
        return self.agent_dir(name) / "pending.jsonl"

    def agent_json(self, name: str) -> Path:
        return self.agent_dir(name) / "agent.json"

    def ensure_agent(self, name: str) -> None:
        self.agent_dir(name).mkdir(parents=True, exist_ok=True)

    def channel_path(self, channel: str) -> Path:
        return self.channels_dir / f"{channel}.jsonl"

    def channel_reactions_path(self, channel: str) -> Path:
        """Append-only reaction log alongside the channel's message log."""
        return self.channels_dir / f"{channel}.reactions.jsonl"

    def plan_md(self, plan_id: str) -> Path:
        return self.plans_dir / f"plan-{plan_id}.md"

    def plan_json(self, plan_id: str) -> Path:
        return self.plans_dir / f"plan-{plan_id}.json"


# ---------------------------------------------------------------------------
# Append-only JSONL with per-path lock — both threadsafe and asyncio-safe.
# ---------------------------------------------------------------------------


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def append_jsonl(path: Path, obj: Any) -> None:
    """Atomic-ish append + fsync. Per spec §11/§12.4: persist before yielding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json_encode(obj) + "\n"
    lock = _lock_for(path)
    with lock:
        with path.open("ab", buffering=0) as f:
            f.write(line.encode("utf-8"))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


async def append_jsonl_async(path: Path, obj: Any) -> None:
    """Async wrapper — runs the blocking write off the loop thread."""
    await asyncio.to_thread(append_jsonl, path, obj)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate JSONL entries, skipping a truncated final line."""
    if not path.exists():
        return
    with path.open("rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                # Trailing partial-write line — drop it.
                return
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    lock = _lock_for(path)
    with lock:
        with tmp.open("wb") as f:
            f.write(_json_encode(obj).encode("utf-8"))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        tmp.replace(path)


async def write_json_async(path: Path, obj: Any) -> None:
    await asyncio.to_thread(write_json, path, obj)


def read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _json_encode(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(by_alias=True, exclude_none=False)
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def _json_default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(by_alias=True)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def to_jsonable(obj: Any) -> Any:
    """Convert pydantic models / dataclasses to plain JSON-able dicts."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(by_alias=True)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def truncate_jsonl(path: Path) -> None:
    """Remove a truncated final line in place (resume cleanup helper)."""
    if not path.exists():
        return
    lines = read_jsonl(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        for entry in lines:
            f.write((_json_encode(entry) + "\n").encode("utf-8"))
    tmp.replace(path)


def listdir_safe(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return sorted(path.iterdir())
