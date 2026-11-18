"""Live CLI tracer for AgentOrg sessions.

Subscribes to a `Session.events` stream and pretty-prints what's happening
as it happens. Designed for the `avalon` command's `--trace` mode, but can
be used programmatically against any session.

Output is grouped by turn for the CEO and condensed for org agents (peer
chatter shows up via channel.posted / report.delivered events rather than
per-turn). Use `--verbose` if you want every event type, including the
deeply granular inference chunks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Optional

from rich.console import Console
from rich.markup import escape

from agentorg.session import Session


log = logging.getLogger(__name__)

# C0/C1 control characters other than tab and newline: terminal escape
# sequences in model or env text must not reach the terminal.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _s(value: Any) -> str:
    """Untrusted value → text safe to embed in Rich markup."""
    if value is None:
        return ""
    return escape(_CONTROL_RE.sub("", str(value)))


def _truncate(value: Any, n: int = 240) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            s = str(value)
    else:
        s = str(value)
    s = _CONTROL_RE.sub("", s).replace("\n", " ⏎ ")
    return escape(s if len(s) <= n else s[: n - 1] + "…")


def _multiline(value: str, *, max_chars: int = 800, indent: str = "    ") -> str:
    """Render a multi-line block with indentation; cap total length."""
    if not value:
        return ""
    body = _CONTROL_RE.sub("", value).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return "\n".join(f"{indent}{escape(line)}" for line in body.splitlines() if line.strip())


class LiveRenderer:
    """Subscribes to a session's event bus and prints CEO-centric activity.

    Lifetime is bounded by `start()` / `stop()` — call `start()` before the
    session loop runs and `stop()` after `run_to_completion` returns.
    """

    def __init__(
        self,
        session: Session,
        *,
        console: Optional[Console] = None,
        prefix: Optional[str] = None,
        verbose: bool = False,
        max_chars: int = 800,
    ):
        self.session = session
        self.console = console or Console()
        self.prefix = prefix
        self.verbose = verbose
        self.max_chars = max_chars
        self._task: Optional[asyncio.Task] = None
        # Per-agent buffers for streaming reasoning/content deltas — emitted
        # on inference.completed so the output reads as a single block.
        self._reasoning: dict[str, list[str]] = {}
        self._content: dict[str, list[str]] = {}
        # Tracks the running turn index per agent for nicer headers.
        self._last_turn: dict[str, int] = {}

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="LiveRenderer")

    async def stop(self) -> None:
        if self._task is None:
            return
        # Give any buffered events a tick to drain.
        await asyncio.sleep(0.05)
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            # The trace is a view: a renderer failure must never cost the
            # caller its rollout result.
            log.exception("live trace renderer failed")
        self._task = None

    async def _run(self) -> None:
        try:
            async for ev in self.session.events.stream():
                try:
                    self._render(ev)
                except Exception:
                    log.exception("live trace: could not render %s event", ev.type)
        except asyncio.CancelledError:
            return

    # ---- rendering ---------------------------------------------------------

    def _line(self, body: str, *, style: Optional[str] = None) -> None:
        prefix = f"[dim]\\[{_s(self.prefix)}][/dim] " if self.prefix else ""
        if style:
            self.console.print(prefix + body, style=style, highlight=False)
        else:
            self.console.print(prefix + body, highlight=False)

    def _block(self, body: str) -> None:
        if not body:
            return
        prefix = f"[dim]\\[{_s(self.prefix)}][/dim] " if self.prefix else ""
        # Multi-line bodies: print first line with prefix, indent the rest.
        for i, line in enumerate(body.splitlines()):
            if i == 0:
                self.console.print(prefix + line, highlight=False)
            else:
                self.console.print(line, highlight=False)

    def _render(self, ev) -> None:
        t = ev.type
        name = ev.name or ""
        payload = ev.payload or {}

        if t == "session.started":
            model = payload.get("model")
            mode = payload.get("mode")
            self._line(
                f"[bold green]── session started[/bold green] "
                f"[dim]model={_s(model)} mode={_s(mode)}[/dim]"
            )
            return

        if t == "environment.bound":
            self._line(
                f"[bold]🌐 env bound[/bold] "
                f"[dim]{_s(payload.get('kind'))}/{_s(payload.get('task_id'))} "
                f"split={_s(payload.get('split'))}[/dim]"
            )
            return

        if t == "plan.mode.entered":
            self._line(
                "[bold yellow]── org-plan mode ──[/bold yellow]"
            )
            return

        if t == "plan.mode.exited":
            self._line(
                f"[bold yellow]── plan exited[/bold yellow] "
                f"[dim]approved={_s(payload.get('approved'))}[/dim]"
            )
            return

        if t == "agent.spawned":
            kind = payload.get("kind")
            role = payload.get("role") or ""
            icon = "👤" if kind == "ceo" else "🤝"
            self._line(
                f"{icon} [bold green]+ @{_s(payload.get('name'))}[/bold green] "
                f"[dim]({_s(role)})[/dim]"
            )
            return

        if t == "agent.fired":
            source = payload.get("source")
            tail = ""
            review = payload.get("review")
            if review:
                tail = (
                    f" [dim]{_s(review.get('category'))} "
                    f"{_s(review.get('score'))}/10[/dim]"
                )
            elif payload.get("reason"):
                tail = f" [dim]{_s(payload.get('reason'))}[/dim]"
            self._line(
                f"🔻 [bold red]- @{_s(payload.get('name'))}[/bold red] "
                f"[dim]fired by {_s(source)}[/dim]{tail}"
            )
            return

        if t == "turn.started":
            idx = payload.get("turn_index", -1)
            self._last_turn[name] = idx
            tag = "@CEO" if name == "CEO" else f"@{_s(name)}"
            self._line(
                f"\n[bold blue]── turn {_s(idx)} {tag} ──[/bold blue]"
            )
            return

        if t == "turn.ended":
            return  # turn.started covers headers; ended is redundant.

        if t == "inference.requested":
            return  # noisy

        if t == "inference.chunk":
            chunk = payload
            kind = chunk.get("kind")
            text = chunk.get("text", "")
            if kind == "reasoning_delta":
                self._reasoning.setdefault(name, []).append(text)
            elif kind == "content_delta":
                self._content.setdefault(name, []).append(text)
            # tool_call chunks are handled via tool.requested
            return

        if t == "inference.completed":
            r = "".join(self._reasoning.pop(name, []))
            c = "".join(self._content.pop(name, []))
            if r:
                tag = "@CEO" if name == "CEO" else f"@{_s(name)}"
                self._block(f"  💭 [italic dim]{tag} thinking[/italic dim]")
                self._block(_multiline(r, max_chars=self.max_chars, indent="     "))
            if c:
                tag = "@CEO" if name == "CEO" else f"@{_s(name)}"
                self._block(f"  💬 [italic]{tag}[/italic]")
                self._block(_multiline(c, max_chars=self.max_chars, indent="     "))
            return

        if t == "tool.requested":
            tool = payload.get("tool")
            args = _truncate(payload.get("args"), 220)
            self._line(f"  [bold cyan]→ {_s(tool)}[/bold cyan] [dim]{args}[/dim]")
            return

        if t == "tool.completed":
            if payload.get("ok"):
                out = _truncate(payload.get("output"), 220)
                self._line(f"  [green]✓[/green] [dim]{out}[/dim]")
            else:
                err = payload.get("error") or {}
                self._line(
                    f"  [red]✗ {_s(err.get('code', 'error'))}[/red]: "
                    f"[dim]{_truncate(err.get('message'), 220)}[/dim]"
                )
            return

        if t == "channel.created":
            self._line(f"  [magenta]#{_s(payload.get('channel'))}[/magenta] [dim]created[/dim]")
            return

        if t == "channel.posted":
            author = payload.get("author")
            ch = payload.get("channel")
            mentions = payload.get("mentions") or []
            mention_str = ""
            if mentions:
                tags = [m.get("name") or m.get("label") for m in mentions]
                mention_str = f" [yellow]@{_s(','.join(str(t) for t in tags if t))}[/yellow]"
            self._line(
                f"  [magenta]#{_s(ch)}[/magenta] [bold]@{_s(author)}[/bold]"
                f"{mention_str} [dim](msg posted)[/dim]"
            )
            return

        if t in ("channel.reaction.added", "channel.reaction.removed"):
            verb = "reacted" if t.endswith("added") else "un-reacted"
            self._line(
                f"  [magenta]#{_s(payload.get('channel'))}[/magenta] "
                f"[dim]{_s(payload.get('reactor'))} {verb} {_s(payload.get('emoji'))}[/dim]"
            )
            return

        if t == "dm.delivered":
            self._line(
                f"  [yellow]✉ {_s(payload.get('from'))} → @{_s(payload.get('to'))}[/yellow]"
            )
            return

        if t == "report.delivered":
            p = payload.get("payload") or {}
            self._line(
                f"  [bold magenta]📋 @{_s(payload.get('from'))} → CEO "
                f"\\[{_s(p.get('status'))}][/bold magenta] "
                f"[dim]{_truncate(p.get('summary'), 240)}[/dim]"
            )
            return

        if t == "environment.terminal":
            reward = payload.get("reward")
            self._line(
                f"[bold green]🏁 env terminal[/bold green] "
                f"[dim]reward={_s(reward)}[/dim]"
            )
            return

        if t == "session.ended":
            self._line(
                f"[bold green]── session ended[/bold green] "
                f"[dim]reason={_s(payload.get('reason'))}[/dim]"
            )
            return

        if t == "session.aborted":
            self._line(
                f"[bold red]── session aborted[/bold red] "
                f"[dim]reason={_s(payload.get('reason'))}[/dim]"
            )
            return

        if self.verbose:
            self._line(f"[dim]· {_s(t)} {_truncate(payload, 180)}[/dim]")


@asynccontextmanager
async def live_trace(session: Session, **kwargs):
    """Context manager that runs a LiveRenderer alongside the session loop."""
    renderer = LiveRenderer(session, **kwargs)
    renderer.start()
    try:
        yield renderer
    finally:
        await renderer.stop()
