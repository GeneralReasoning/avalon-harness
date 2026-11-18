"""OpenReward error classification.

The OpenReward adapter renders a failed `call_tool` as `{ExcClass}: {exc}`
(see `openreward_env._OpenRewardTool.execute`). Some of those failures mean the
env session is gone for good — the pod crashed / was evicted, or the SDK
exhausted its retries reaching it. This module owns the OpenReward-specific
signatures for that condition so two consumers share ONE source of truth
without either having to import the (heavy) OpenReward SDK:

  * `openreward_env.OpenRewardEnvAdapter` maps a matching exception to the
    generic `agentorg.env_adapter.ENV_SESSION_DEAD_CODE`, which the harness
    treats as an immediate session abort.
  * Training-data builders can drop any transcript that contains one by
    string-matching the recorded message (`is_session_death_message`) rather
    than relying on the `env-session-dead` code alone.

These signatures are OpenReward-specific, but they live in `agentorg/`
alongside `openreward_env` so the adapter does not depend on the CLI
(`agentorg.cli`).
"""

from __future__ import annotations

from typing import Optional

# Rendered-exception substrings that mean the OpenReward env session is dead.
# Each matches a `{ExceptionClassName}: {exc}` rendering from the adapter.
ENV_SESSION_DEATH_SIGNS: tuple[str, ...] = (
    "SessionTerminatedError",
    "environment session terminated",
    "RuntimeError: (environment session",
    # The SDK retried the session pod and gave up — e.g. 502 "upstream request
    # failed", 503 "Pod unavailable: Container env-server crashed", or a socket
    # read timeout — all surface as a single MaxRetriesError.
    "MaxRetriesError: (environment session",
    # The env proxy no longer knows this session id, so every subsequent
    # `call_tool` on this adapter fails identically. Rendered by the adapter as
    # `SessionResponseError: 400, message='Session not found.', url=...`. The
    # SDK does not retry 4xx under either retry policy, so nothing upstream
    # turns this into a MaxRetriesError — without this sign it is misread as a
    # recoverable per-call error. Anchored on the class + status because the
    # message body after `message=` is the server's `detail` string, which for
    # a 500 can be arbitrary text from user env code.
    "SessionResponseError: 400, message='Session not found",
    # The env's own @tool raised (or returned a non-ToolOutput). The SDK calls
    # `_mark_dead` BEFORE raising this and eagerly POSTs `/delete`, so the
    # session is gone server-side and every later `call_tool` raises
    # SessionTerminatedError. Without this sign it reads as a recoverable
    # per-call error and the model is invited to retry a session the SDK has
    # already torn down. Plain substring like every other sign here: the tuple
    # is also scanned verbatim against whole message bodies by training-data
    # builders, so a sign must not carry match syntax.
    "ToolFailed: ",
)


def is_session_death_message(message: Optional[str]) -> bool:
    """True if a rendered OpenReward error message signals an unrecoverable
    env-session death (vs. a recoverable per-call error)."""
    return bool(message) and any(s in message for s in ENV_SESSION_DEATH_SIGNS)
