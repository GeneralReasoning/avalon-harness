"""Trajectory shape (§12.1).

A trajectory IS the persisted session directory. This module produces a
flattened, JSON-friendly view by replaying messages.jsonl for every agent
in the session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from agentorg.storage import iter_jsonl, read_json
from agentorg.types import CEO_NAME, SessionState


def build_trajectory(session_dir: Path) -> dict[str, Any]:
    state_raw = read_json(session_dir / "session.json")
    if not state_raw:
        raise FileNotFoundError(session_dir / "session.json")
    state = SessionState.model_validate(state_raw)

    agent_dirs = {}
    ceo = CEO_NAME
    agent_dirs[ceo] = session_dir / "messages.jsonl"
    org_dir = session_dir / "org"
    if org_dir.exists():
        for sub in org_dir.iterdir():
            if sub.is_dir():
                agent_dirs[sub.name] = sub / "messages.jsonl"

    steps: list[dict[str, Any]] = []
    # Interleave by ts to produce a global stream.
    rows: list[tuple[int, str, dict[str, Any]]] = []
    for name, path in agent_dirs.items():
        for raw in iter_jsonl(path):
            rows.append((int(raw.get("ts") or 0), name, raw))
    rows.sort(key=lambda r: r[0])

    # Group by assistant turn — each step = one assistant turn + immediately following tool results.
    pending: Optional[dict[str, Any]] = None
    pending_owner: Optional[str] = None
    total_reward = 0.0
    reward_steps = 0
    for ts, name, raw in rows:
        role = raw.get("role")
        if role == "assistant":
            if pending is not None:
                steps.append(pending)
            pending = {
                "ts": ts,
                "name": name,
                "assistant": raw,
                "tool_results": [],
                "reward": None,
            }
            pending_owner = name
        elif role == "tool" and pending is not None and name == pending_owner:
            pending["tool_results"].append(raw)
            # Per-step env reward lives inside the env tool's output payload.
            tr = raw.get("tool_result") or {}
            output = tr.get("output") if isinstance(tr.get("output"), dict) else None
            r = output.get("reward") if output else None
            if r is not None:
                pending["reward"] = (pending["reward"] or 0.0) + float(r)
                total_reward += float(r)
                reward_steps += 1
        else:
            # Other roles attach to the latest preceding step as preamble.
            steps.append({"ts": ts, "name": name, "non_assistant": raw})
    if pending is not None:
        steps.append(pending)

    return {
        "session_id": state.session_id,
        "workspace_path": state.workspace_path,
        "environment": state.environment.model_dump() if state.environment else None,
        "ceo_name": CEO_NAME,
        "org_agents": state.org_agent_names,
        "steps": steps,
        "cumulative_reward": total_reward if reward_steps else None,
        "reward_steps": reward_steps,
    }


def dump_trajectory(session_dir: Path, out_path: Path) -> dict[str, Any]:
    traj = build_trajectory(session_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    return traj
