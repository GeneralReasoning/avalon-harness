"""Tool registry — name → tool, with per-agent scoping (§5.4)."""

from __future__ import annotations

from typing import Iterable, Optional

from agentorg.tools.context import ToolImpl
from agentorg.types import ToolDefinition


class ToolRegistry:
    """Maps tool name → implementation.

    Implementations can be registered ahead of session start; per-call
    scoping (allow / disallow / wait-gate restriction) happens in the loop.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolImpl] = {}

    def register(self, tool: ToolImpl) -> None:
        self._tools[tool.definition.name] = tool

    def register_many(self, tools: Iterable[ToolImpl]) -> None:
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolImpl]:
        return self._tools.get(name)

    def resolve(self, name: str) -> Optional[tuple[str, ToolImpl]]:
        """Strict name match, falling back to case-insensitive.

        Returns `(canonical_name, ToolImpl)` so the caller can rewrite the
        invocation to the registered spelling. Used by the dispatch path to
        rescue mis-cased tool calls — e.g. the model emits `Bash` reflexively
        but the env registered `bash`. Returns None if no case-insensitive
        match exists, or if multiple registered names match the lowered form
        (ambiguous — refuse rather than guess).
        """
        tool = self._tools.get(name)
        if tool is not None:
            return name, tool
        target = name.lower()
        matches = [(n, t) for n, t in self._tools.items() if n.lower() == target]
        if len(matches) == 1:
            return matches[0]
        return None

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def visible(
        self,
        *,
        allowed: Optional[list[str]] = None,
        disallowed: Optional[list[str]] = None,
        restrict_to: Optional[set[str]] = None,
    ) -> list[ToolDefinition]:
        """Return tool definitions visible to a given agent / mode.

        - `allowed`: if set, only names in this list are exposed.
        - `disallowed`: never exposed.
        - `restrict_to`: e.g. wait-gate restriction set; if set, intersect.
        """
        out: list[ToolDefinition] = []
        for name, tool in self._tools.items():
            if allowed is not None and name not in allowed:
                continue
            if disallowed is not None and name in disallowed:
                continue
            if restrict_to is not None and name not in restrict_to:
                continue
            out.append(tool.definition)
        return out
