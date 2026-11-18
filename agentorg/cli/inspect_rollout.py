"""Inspect an AgentOrg session directory.

Usage:
    uv run avalon-inspect rollouts/task-0000
    uv run avalon-inspect rollouts/task-0000 --transcript CEO
    uv run avalon-inspect rollouts/task-0000 --channel announcements
    uv run avalon-inspect rollouts/task-0000 --events --tail 40
    uv run avalon-inspect rollouts/task-0000 --follow

Prints a high-level overview by default. Use the flags to dig in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _find_session_dir(root: Path) -> Path:
    """Resolve a session directory from any sensible parent path.

    Accepts:
      - the session dir itself (contains session.json)
      - a task dir (contains .agentorg/sessions/ses_*)
      - a run/env/root dir from the nested layout — picks the most
        recently modified session under it.
    """
    if (root / "session.json").exists():
        return root
    direct = sorted(root.glob(".agentorg/sessions/ses_*"))
    if direct:
        return max(direct, key=lambda p: p.stat().st_mtime)
    nested = sorted(root.rglob(".agentorg/sessions/ses_*"))
    if not nested:
        raise SystemExit(f"No .agentorg/sessions/ses_* under {root}")
    return max(nested, key=lambda p: p.stat().st_mtime)


def _iter_jsonl(p: Path):
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# C0/C1 control characters other than tab and newline: terminal escape
# sequences in persisted model or env text must not reach the terminal.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _out(*parts: Any, **kwargs: Any) -> None:
    """`print`, with control characters stripped from every part."""
    print(*(_CONTROL_RE.sub("", str(p)) for p in parts), **kwargs)


def _truncate(s: Any, n: int = 200) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def print_overview(sess_dir: Path) -> None:
    state = json.loads((sess_dir / "session.json").read_text())
    _out(f"\n=== Session {state['session_id']} ===")
    _out(f"  model:           {state['model']}")
    # The current permission mode: an org-plan session reads "default" once
    # its plan has been approved.
    _out(f"  current mode:    {state['mode']}")
    _out(f"  budget:          {state['budget']['max_org_agents']}")
    _out(f"  workspace:       {state['workspace_path']}")
    env = state.get("environment")
    if env:
        _out(f"  environment:     {env.get('kind')}/{env.get('task_id')} split={env.get('split')}")
    _out(f"  permission:      {state.get('permission_policy', 'default')}")
    reported = state.get("reported_agent_names") or []
    _out(f"  reported_agents: {reported}")

    # Roster.
    ceo_path = sess_dir / "ceo.json"
    if ceo_path.exists():
        ceo = json.loads(ceo_path.read_text())
        _out(f"\n  CEO @{ceo['name']} ({ceo['state']})")

    org_dir = sess_dir / "org"
    if org_dir.exists():
        rows = []
        for sub in sorted(org_dir.iterdir()):
            if not sub.is_dir():
                continue
            ag_path = sub / "agent.json"
            if not ag_path.exists():
                continue
            a = json.loads(ag_path.read_text())
            review = a.get("exit_review")
            review_s = ""
            if review:
                review_s = (
                    f"  [{review['category']} {review['score']}/10] "
                    f"delivered={_truncate(review['delivered'], 80)}"
                )
            rows.append(f"  @{a['name']} — {a.get('role','?')} ({a['state']}){review_s}")
        if rows:
            _out(f"\n  Org agents ({len(rows)}):")
            for r in rows:
                _out(r)

    # Channels.
    meta = sess_dir / "channels" / "_meta.json"
    if meta.exists():
        chs = json.loads(meta.read_text()).get("channels") or {}
        if chs:
            _out("\n  Channels:")
            for name, ch in chs.items():
                jsonl = sess_dir / "channels" / f"{name}.jsonl"
                count = sum(1 for _ in _iter_jsonl(jsonl)) if jsonl.exists() else 0
                rxs = sess_dir / "channels" / f"{name}.reactions.jsonl"
                rcount = sum(1 for _ in _iter_jsonl(rxs)) if rxs.exists() else 0
                subs = ch.get("subscriptions") or {}
                _out(
                    f"    #{name}: {count} posts, {rcount} reactions, "
                    f"default={ch.get('default_subscription')}, overrides={subs}"
                )

    # Plans.
    plans_dir = sess_dir / "plans"
    if plans_dir.exists():
        plans = sorted(plans_dir.glob("plan-*.md"))
        if plans:
            _out(f"\n  Plans: {[p.name for p in plans]}")

    # Archive.
    arch = sess_dir / "archive"
    if arch.exists():
        files = list(arch.rglob("*.md"))
        if files:
            _out("\n  Archive:")
            for f in sorted(files):
                rel = f.relative_to(arch)
                _out(f"    {rel} ({f.stat().st_size}B)")

    # Inference errors.
    err = sess_dir / "inference-errors.jsonl"
    if err.exists() and err.stat().st_size > 0:
        n = sum(1 for _ in _iter_jsonl(err))
        _out(f"\n  ⚠ inference-errors.jsonl: {n} entries")

    # Quick stats from events.
    events_path = sess_dir / "events.jsonl"
    if events_path.exists():
        by_type: dict[str, int] = {}
        last_seq = 0
        for ev in _iter_jsonl(events_path):
            t = ev.get("type", "?")
            by_type[t] = by_type.get(t, 0) + 1
            last_seq = max(last_seq, ev.get("seq", 0))
        _out(f"\n  Events: last_seq={last_seq}")
        for t, n in sorted(by_type.items(), key=lambda x: -x[1])[:12]:
            _out(f"    {t:30s} {n}")


def print_transcript(sess_dir: Path, name: str) -> None:
    if name == "CEO":
        path = sess_dir / "messages.jsonl"
    else:
        path = sess_dir / "org" / name / "messages.jsonl"
    if not path.exists():
        raise SystemExit(f"No transcript at {path}")
    _out(f"\n=== Transcript: {name} ({path}) ===")
    for i, m in enumerate(_iter_jsonl(path), 1):
        role = m.get("role")
        m.get("ts")
        if role == "user":
            src = m.get("source") or ""
            body = _truncate(m.get("content"), 600)
            _out(f"\n[{i}] USER {src}: {body}")
        elif role == "assistant":
            r = _truncate(m.get("reasoning"), 300)
            c = _truncate(m.get("content"), 600)
            tc = m.get("tool_calls") or []
            if r:
                _out(f"\n[{i}] ASSISTANT [reasoning]: {r}")
            if c:
                _out(f"[{i}] ASSISTANT [content]: {c}")
            for call in tc:
                args = _truncate(call.get("args"), 400)
                _out(f"[{i}] ASSISTANT → tool {call.get('name')}({args})")
        elif role == "tool":
            tr = m.get("tool_result") or {}
            ok = tr.get("ok")
            if ok:
                out = _truncate(tr.get("output"), 400)
                _out(f"[{i}] TOOL ok: {out}")
            else:
                err = tr.get("error") or {}
                _out(f"[{i}] TOOL err {err.get('code')}: {_truncate(err.get('message'), 300)}")
        elif role == "system":
            _out(f"\n[{i}] SYSTEM: {_truncate(m.get('content'), 200)}")


def print_channel(sess_dir: Path, name: str, *, limit: Optional[int] = None) -> None:
    path = sess_dir / "channels" / f"{name}.jsonl"
    if not path.exists():
        # A channel's log is created by its first post.
        meta = sess_dir / "channels" / "_meta.json"
        known = (json.loads(meta.read_text()).get("channels") or {}) if meta.exists() else {}
        if name in known:
            _out(f"\n=== Channel #{name} — 0 posts ===")
            return
        raise SystemExit(f"No channel #{name}; channels: {', '.join(known) or 'none'}")
    rx_path = sess_dir / "channels" / f"{name}.reactions.jsonl"
    rx_by_msg: dict[str, list[dict]] = {}
    if rx_path.exists():
        # Fold reaction events for inline display. The log is append-only,
        # so the last event per (post, emoji, reactor) is the current one.
        latest: dict[tuple, dict] = {}
        for ev in _iter_jsonl(rx_path):
            latest[(ev["msg_key"], ev["emoji"], ev["reactor"])] = ev
        for (mk, emoji, reactor), ev in latest.items():
            if ev["op"] == "add":
                rx_by_msg.setdefault(mk, []).append({"emoji": emoji, "reactor": reactor})

    msgs = list(_iter_jsonl(path))
    if limit:
        msgs = msgs[-limit:]
    _out(f"\n=== Channel #{name} — {len(msgs)} posts ===")
    for m in msgs:
        marker = "(Human) " if m.get("is_external") else ""
        _out(f"\n@{marker}{m['author']} [idx={m.get('idx')}]:")
        _out(f"  {_truncate(m.get('content'), 1200)}")
        atts = m.get("attachments") or []
        if atts:
            _out(f"  attachments: {[a['archive_path'] for a in atts]}")
        mentions = m.get("mentions") or []
        if mentions:
            _out(f"  mentions: {[(x.get('name') or x.get('label')) for x in mentions]}")
        rxs = rx_by_msg.get(m["msg_key"]) or []
        if rxs:
            grouped: dict[str, list[str]] = {}
            for r in rxs:
                grouped.setdefault(r["emoji"], []).append(r["reactor"])
            _out(
                "  reactions: "
                + "  ".join(f"{e} {len(rs)} ({','.join(rs)})" for e, rs in grouped.items())
            )


def print_events(sess_dir: Path, *, tail: int = 50, filt: Optional[str] = None) -> None:
    path = sess_dir / "events.jsonl"
    rows = list(_iter_jsonl(path))
    if filt:
        rows = [r for r in rows if filt in (r.get("type") or "")]
    rows = rows[-tail:]
    _out(f"\n=== Events (last {len(rows)}) ===")
    for ev in rows:
        payload = ev.get("payload") or {}
        name = ev.get("name") or "-"
        body = _truncate(json.dumps(payload, default=str), 200)
        _out(f"  seq={ev['seq']:5d} ts={ev['ts']} {name:18s} {ev.get('type'):28s} {body}")


def follow(sess_dir: Path) -> None:
    """tail -f the events stream + every transcript + every channel, interleaved."""
    paths: list[tuple[str, Path, int]] = []
    paths.append(("EVENT", sess_dir / "events.jsonl", 0))
    paths.append(("CEO", sess_dir / "messages.jsonl", 0))
    for sub in (sess_dir / "org").glob("*/messages.jsonl"):
        paths.append((f"@{sub.parent.name}", sub, 0))
    for chp in (sess_dir / "channels").glob("*.jsonl"):
        paths.append((f"#{chp.stem}", chp, 0))
    # Start at end-of-file so we only see new output.
    offsets = {p: (p.stat().st_size if p.exists() else 0) for _, p, _ in paths}
    _out(f"following {sess_dir} (Ctrl-C to stop)")
    try:
        while True:
            for label, p, _ in paths:
                if not p.exists():
                    continue
                cur = p.stat().st_size
                if cur > offsets[p]:
                    with p.open() as f:
                        f.seek(offsets[p])
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            _print_follow(label, obj)
                    offsets[p] = cur
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


def _print_follow(label: str, obj: dict) -> None:
    if label == "EVENT":
        body = _truncate(json.dumps(obj.get("payload") or {}, default=str), 140)
        _out(f"[EVENT seq={obj.get('seq')}] {obj.get('type')} {body}")
        return
    if label.startswith("#"):
        _out(
            f"[{label} @{obj.get('author')}] {_truncate(obj.get('content'), 220)}"
        )
        return
    # Transcript
    role = obj.get("role")
    if role == "assistant":
        if obj.get("reasoning"):
            _out(f"[{label} assistant.reasoning] {_truncate(obj['reasoning'], 220)}")
        if obj.get("content"):
            _out(f"[{label} assistant] {_truncate(obj['content'], 220)}")
        for tc in obj.get("tool_calls") or []:
            _out(f"[{label} → {tc.get('name')}] {_truncate(tc.get('args'), 220)}")
    elif role == "tool":
        tr = obj.get("tool_result") or {}
        if tr.get("ok"):
            _out(f"[{label} tool ok] {_truncate(tr.get('output'), 220)}")
        else:
            err = tr.get("error") or {}
            _out(f"[{label} tool err] {err.get('code')}: {_truncate(err.get('message'), 200)}")
    elif role == "user":
        _out(f"[{label} user] {_truncate(obj.get('content'), 220)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="avalon-inspect", description="Inspect an AgentOrg session directory.")
    ap.add_argument("root", help="Either the rollout dir or a session dir directly.")
    ap.add_argument("--transcript", help="Print a full transcript by name ('CEO' or an org agent).")
    ap.add_argument("--channel", help="Print a channel's posts (with reactions).")
    ap.add_argument("--events", action="store_true", help="Print recent events.")
    ap.add_argument("--filter", help="Substring filter for --events.")
    ap.add_argument("--tail", type=int, default=50, help="Tail size for --events.")
    ap.add_argument("--follow", action="store_true", help="tail -f everything as it lands.")
    args = ap.parse_args(argv or sys.argv[1:])

    sess_dir = _find_session_dir(Path(args.root))

    if args.follow:
        follow(sess_dir)
        return 0
    if args.transcript:
        print_transcript(sess_dir, args.transcript)
        return 0
    if args.channel:
        print_channel(sess_dir, args.channel)
        return 0
    if args.events:
        print_events(sess_dir, tail=args.tail, filt=args.filter)
        return 0
    print_overview(sess_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
