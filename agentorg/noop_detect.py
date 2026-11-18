"""No-op shell-call detection (§A.8.5).

Kept as trained: Avalon-M1-Mini was trained with this rule active. The
penalty is recorded as `shaping_reward` in the tool result the model sees; it
was not part of the training reward and is never added to the reported
reward. The repeated identical call refusal (SPEC §A.8.6) covers the general
case.

A specialist that finishes its assignment has no way to yield: its loop keeps
running as long as the transcript ends in a tool result. Some agents keep it
alive with shell calls that do nothing to the environment — `exit 0`, `echo
done`, `echo "Mission complete."` — until the wall clock runs out. This module
is the single home for recognising that shape, so the loop, the metrics, and
any offline analysis all agree on what counts.

A no-op is a shell call whose whole command changes nothing: `exit` / `exit N`
/ `true` / `:` / a bare `echo` or `printf` with no redirect, pipe, chain, or
command substitution. The exclusion class on `echo` / `printf` is what keeps
the detector honest — `echo "x" > f`, `echo $(ls)`, `echo a && make` and
`echo hi | tee log` all do real work and none of them match.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from agentorg.types import ToolCall

# Tool names treated as a shell. Matched case-insensitively, so this covers
# both the host-side `Bash` (`fs_tools.BashTool`) and the lowercase `bash` an
# OpenReward claude-code toolset registers. Extend if an env names its shell
# something else; a name alone is never enough to flag a call, the command
# string still has to match `_NOOP_RE`.
SHELL_TOOL_NAMES = frozenset({"bash", "shell", "sh"})

# Anchored ^...$ so the WHOLE command must be the no-op — `exit 0` inside a
# heredoc or at the end of a longer script does not match.
#
# Two deliberate choices:
#   * `sleep N` is NOT included: an agent may legitimately wait on a build.
#   * `\n` and `\r` are in the exclusion class. Without them the class happily
#     eats interior newlines and `$` anchors before a trailing one, so
#     `echo done\nmake build` matches the echo branch — a real false positive.
#
# The two classes below differ in how much a single call proves, not in whether
# the command does anything. Both do nothing.

# Can never accomplish anything, in any context. A shell call gets its own
# process — the host `BashTool` runs `create_subprocess_shell` per call, and
# the env toolsets behave the same way — so there is no session to end, no
# state to reset, and no hung process to escape. There is no legitimate use,
# so a single one is evidence on its own.
_ALWAYS_NOOP_RE = re.compile(
    r"^\s*(?:exit(?:\s+\d+)?|true|:)\s*;?\s*$",
    re.IGNORECASE,
)

# Does nothing to the environment either, but carries a message, so one on its
# own is weaker evidence: it can be a one-off probe (`echo hello`) or a marker
# the agent writes for itself. Only counted when the whole turn was nothing but
# calls like these.
_INERT_NOOP_RE = re.compile(
    r"^\s*(?:echo|printf)\b[^|>&;<`$()\n\r]*(?:;\s*)?$",
    re.IGNORECASE,
)


def extract_command(args: Any) -> Optional[str]:
    """Pull the command string out of a shell tool's arguments.

    The host `Bash` tool fixes the name as `command` (`fs_tools.py`), but the
    env-side `bash` schema comes from the environment, so fall back to `cmd`
    and then to the sole string value if the env spells it differently.
    """
    if not isinstance(args, dict):
        return None
    for key in ("command", "cmd"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    strings = [v for v in args.values() if isinstance(v, str)]
    if len(strings) == 1:
        return strings[0]
    return None


# No no-op is longer than this. A backstop: the regexes run on the event loop
# for every shell call, and a quantifier overlap (since removed from
# _INERT_NOOP_RE) once made them take seconds on ~1k chars of whitespace.
_MAX_NOOP_COMMAND_CHARS = 512


def is_always_noop_command(command: Optional[str]) -> bool:
    """True iff `command` can never accomplish anything (`exit`, `true`, `:`)."""
    if not isinstance(command, str) or not command.strip():
        return False
    if len(command) > _MAX_NOOP_COMMAND_CHARS:
        return False
    return _ALWAYS_NOOP_RE.match(command) is not None


def is_noop_command(command: Optional[str]) -> bool:
    """True iff `command` is a shell command that does nothing."""
    if not isinstance(command, str) or not command.strip():
        return False
    if len(command) > _MAX_NOOP_COMMAND_CHARS:
        return False
    return (
        _ALWAYS_NOOP_RE.match(command) is not None
        # rstrip: the pattern's class can't eat a trailing \r or \n run.
        or _INERT_NOOP_RE.match(command.rstrip()) is not None
    )


def is_shell_call(call: ToolCall) -> bool:
    return (call.name or "").lower() in SHELL_TOOL_NAMES


def is_noop_call(call: ToolCall) -> bool:
    """True iff `call` is a shell call whose command does nothing."""
    if not is_shell_call(call):
        return False
    return is_noop_command(extract_command(call.args))


def is_always_noop_call(call: ToolCall) -> bool:
    """True iff `call` is a shell call that could never have accomplished
    anything, whatever else the turn did."""
    if not is_shell_call(call):
        return False
    return is_always_noop_command(extract_command(call.args))


def turn_is_noop(tool_calls: Iterable[ToolCall]) -> bool:
    """True iff the turn made at least one tool call and EVERY one is a no-op.

    Turn-level gating on purpose: an `echo done` alongside a `ReportToCEO` is a
    stray line in a turn that did real work, not a stall. Only a turn that did
    nothing at all counts.
    """
    calls = list(tool_calls)
    if not calls:
        return False
    return all(is_noop_call(c) for c in calls)


def always_noop_calls(tool_calls: Iterable[ToolCall]) -> list[ToolCall]:
    """The calls in a turn that could never have accomplished anything.

    Used for the turns `turn_is_noop` spares. A stray `echo done` beside real
    work is a stray line in a productive turn; a stray `exit 0` is a wasted
    call with no legitimate reading, and pairing one with a real call is the
    obvious way to slip a termination signal past a whole-turn gate.
    """
    return [c for c in tool_calls if is_always_noop_call(c)]
