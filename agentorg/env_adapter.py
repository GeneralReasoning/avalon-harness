"""Environment adapter protocol (§12.3).

The harness binds a `Session` to an external environment via an
`EnvAdapter`. The adapter exposes the environment's tools to the harness
registry; the env-session lifecycle (open / close) is owned by the
caller, not the harness.

Concrete adapters implement this protocol — see `agentorg.openreward_env`
for the one this repository ships.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentorg.tools.context import ToolImpl


# Generic adapter→harness contract: an adapter returns a failed ToolResult with
# THIS error code to signal that the environment session is gone for good — the
# backing process/pod died, or the adapter's client gave up reaching it. It is
# provider-agnostic on purpose; each adapter maps its own transport/SDK
# exceptions to this code (the OpenReward adapter does so in
# `agentorg.openreward_env`, keying off the OpenReward-specific
# signatures in `agentorg.openreward_errors`). The harness reacts in
# `agentorg.loop._update_env_tool_breaker`: a single occurrence aborts the
# session immediately rather than letting the agent flail against a dead env —
# no retry can succeed once the session is gone.
ENV_SESSION_DEAD_CODE = "env-session-dead"


@runtime_checkable
class EnvAdapter(Protocol):
    """Protocol consumed by `Harness.start_session(env_adapter=...)`.

    Implementations expose env-specific tools to the harness registry.
    Exceptions raised from `build_tools` are session-fatal: an env that
    can't list its tools can't be driven.
    """

    async def build_tools(self) -> list[ToolImpl]:
        ...
