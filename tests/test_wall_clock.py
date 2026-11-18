"""`run_to_completion(wall_clock=...)`: the host's clock for `max_wall_s`.

An RL trainer that shares GPUs between inference and the optimizer step
freezes every session while it trains. On real time a session that spans a
train phase loses the whole phase from its budget; with a clock that stops
during it, the budget only counts time the session could make progress.
"""

from __future__ import annotations

import asyncio

from agentorg.harness import Harness
from tests.test_host_cancel import _config, _EndlessProvider


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def test_max_wall_s_reads_the_host_clock(tmp_path):
    clock = _ManualClock()
    harness = Harness(inference=_EndlessProvider())
    session = await harness.start_session(config=_config(tmp_path))
    task = asyncio.create_task(
        harness.run_to_completion(session, max_wall_s=0.2, wall_clock=clock)
    )
    # Three budgets of real time pass while the host clock stands still.
    await asyncio.sleep(0.6)
    assert not task.done()
    assert session.abort_reason is None

    clock.now += 0.3
    result = await asyncio.wait_for(task, timeout=5)
    assert session.abort_reason == "max_wall_s"
    # §4.6 — the session clock starts on the host clock, over the host wall.
    assert session.clock_start_s == 1000.0
    assert session.clock_total_s == 0.2
    # duration_s stays real time.
    assert result.duration_s >= 0.6


async def test_max_wall_s_defaults_to_real_time(tmp_path):
    harness = Harness(inference=_EndlessProvider())
    session = await harness.start_session(config=_config(tmp_path))
    result = await asyncio.wait_for(harness.run_to_completion(session, max_wall_s=0.2), timeout=5)
    assert session.abort_reason == "max_wall_s"
    assert result.end_reason == "max_wall_s"

