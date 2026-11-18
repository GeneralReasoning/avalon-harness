"""Event stream (§9). Monotonic seq, persist-before-yield."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import AsyncIterator, Callable, Optional

from agentorg.storage import SessionPaths, append_jsonl_async
from agentorg.types import AgentName, Event, EventType

log = logging.getLogger(__name__)


class EventBus:
    """Single-session event bus.

    - Assigns monotonic `seq` atomically.
    - Persists every event to events.jsonl *before* yielding to subscribers.
    - Supports multiple async subscribers.
    """

    def __init__(self, paths: SessionPaths, starting_seq: int = 0):
        self._paths = paths
        self._seq = starting_seq
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[Optional[Event]]] = []
        self._closed = False
        # Per-subscriber drop counters — a chronically slow subscriber
        # could spam the log; suppress repeats once we've reported the
        # condition for a given queue identity.
        self._drop_warned: set[int] = set()

    @property
    def last_seq(self) -> int:
        return self._seq

    async def emit(
        self,
        type: EventType,
        payload: dict | None = None,
        name: Optional[AgentName] = None,
    ) -> Event:
        # The lock is held across `append_jsonl_async` deliberately —
        # this enforces that events arrive on disk in `seq` order
        # (SPEC §9.2). Releasing earlier would let the thread-pool
        # reorder concurrent writes. The throughput cost (one disk
        # round-trip per emit) is acceptable; typical emit rates are
        # 10-100/s with sub-millisecond appends.
        async with self._lock:
            self._seq += 1
            ev = Event(seq=self._seq, type=type, name=name, payload=payload or {})
            await append_jsonl_async(self._paths.events_jsonl, ev)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    qid = id(q)
                    if qid not in self._drop_warned:
                        self._drop_warned.add(qid)
                        log.warning(
                            "EventBus subscriber queue full; dropping events "
                            "(seq=%d type=%s). Further drops for this subscriber "
                            "will be suppressed.",
                            ev.seq,
                            type,
                        )
            return ev

    def subscribe(self) -> "asyncio.Queue[Optional[Event]]":
        q: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=10_000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Optional[Event]]") -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def stream(self) -> AsyncIterator[Event]:
        q = self.subscribe()
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ev
        finally:
            self.unsubscribe(q)

    async def close(self) -> None:
        self._closed = True
        for q in list(self._subscribers):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


class WakeBroker:
    """Per-agent wake signals for the wait gate (§4.5).

    The wait is purely event-driven — no timeout. An agent parks via
    `park_and_wait(name, on_parked=…)`; anyone delivering an inbox item
    or otherwise wanting the agent to re-run calls `wake(name)`.

    `park_and_wait` is the atomic primitive that supports the
    **quiescence invariant** (SPEC §4.5.1): the `on_parked` callback
    runs synchronously under the same lock that mutates `_waiting`,
    so an org agent observing "I'm the last to park" can race-freely
    wake the CEO before releasing the lock.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._waiting: set[AgentName] = set()
        # Guards `_waiting` mutations + `on_parked` callbacks. Held only
        # across the registration step — released before the actual
        # await so other agents can wake the parked one.
        self._lock = asyncio.Lock()

    def wake(self, name: AgentName) -> None:
        """Sync — sets the wake event. Safe from any context.

        Does NOT remove `name` from `_waiting`; the waiter's `finally`
        in `park_and_wait` handles that on return. This is deliberate:
        `wake` may be called before, during, or after a `park_and_wait`
        and must not race with the registration step.
        """
        ev = self._events[name]
        ev.set()

    async def park_and_wait(
        self,
        name: AgentName,
        *,
        on_parked: Optional[Callable[[], None]] = None,
    ) -> None:
        """Park `name` and block until the wake event fires.

        Sequence under `self._lock`:
          1. If the wake event is already set (a `wake()` raced in
             before we got here), clear it and return immediately —
             do NOT enter the waiting set.
          2. Add `name` to `_waiting`.
          3. Run `on_parked()` (sync). This is the quiescence trigger
             hook: an org-agent caller can check "every live agent is
             now parked" and `wake(CEO)` from inside it without race.

        Then release the lock and `await ev.wait()`. On return (any
        reason), remove `name` from `_waiting` under the lock.

        No timeout. Liveness is preserved by the quiescence invariant
        (callers ensure the CEO is never the last to park while peers
        are still parked); host-level guards like `max_wall_s` (§4.4)
        live elsewhere.
        """
        ev = self._events[name]
        async with self._lock:
            if ev.is_set():
                ev.clear()
                return
            self._waiting.add(name)
            if on_parked is not None:
                on_parked()
        try:
            await ev.wait()
            ev.clear()
        finally:
            async with self._lock:
                self._waiting.discard(name)

    def is_waiting(self, name: AgentName) -> bool:
        return name in self._waiting

    def waiting_agents(self) -> set[AgentName]:
        return set(self._waiting)

    def clear(self, name: AgentName) -> None:
        """Sync — drop a pending wake signal for `name`.

        Only for a caller that knows the signal is stale. A `wake()` delivered
        while the agent was running is never consumed (the turn was driven by
        the transcript, not by a park), so it survives until that agent's next
        `park_and_wait`, which then returns immediately. A caller forcing an
        agent to park (`loop._handle_noop_turn`) has to clear that residue
        first, and only once it has confirmed the agent has no genuinely
        pending input — otherwise it would park an agent that has a message
        waiting to be read.
        """
        self._events[name].clear()


class PermissionBroker:
    """Pending interactive tool-permission decisions, keyed by tool-call id (§5.2).

    A tool call that resolves to `ask` in an interactive session registers a
    Future via `register(call_id)` and awaits it; the external interface's
    `decide_permission(call_id, …)` resolves it with "allow"/"deny". Single
    event loop, so no lock. `resolve` is idempotent (first-decide-wins, per
    §10.1) — a duplicate or stale decision is a no-op, never an error.
    """

    def __init__(self) -> None:
        self._pending: dict[str, "asyncio.Future[str]"] = {}

    def register(self, call_id: str) -> "asyncio.Future[str]":
        fut: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        self._pending[call_id] = fut
        return fut

    def resolve(self, call_id: str, decision: str) -> bool:
        """Resolve a pending request. False if unknown / already decided."""
        fut = self._pending.get(call_id)
        if fut is None or fut.done():
            return False
        fut.set_result(decision)
        return True

    def discard(self, call_id: str) -> None:
        self._pending.pop(call_id, None)
