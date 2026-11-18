"""Shared archive (§7.1, §11.1) — markdown documents at `archive/<path>.md`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentorg.storage import SessionPaths

ARCHIVE_MAX_DEPTH = 5
ARCHIVE_MAX_BYTES = 256 * 1024  # 256 KiB


class ArchiveError(Exception):
    pass


class ArchiveStore:
    def __init__(self, paths: SessionPaths) -> None:
        self._paths = paths

    @property
    def root(self) -> Path:
        return self._paths.archive_dir

    def _resolve(self, rel: str) -> Path:
        if not rel:
            raise ArchiveError("Archive path must not be empty")
        if rel.startswith("/") or "\\" in rel:
            raise ArchiveError("Archive path must be a relative POSIX path")
        rel = rel.lstrip("./")
        if not rel.endswith(".md"):
            raise ArchiveError("Archive paths must end with .md")
        parts = [p for p in rel.split("/") if p]
        if any(p in ("..", ".") for p in parts):
            raise ArchiveError("Archive paths must not contain '.' or '..' components")
        if len(parts) > ARCHIVE_MAX_DEPTH:
            raise ArchiveError(f"Archive paths are limited to {ARCHIVE_MAX_DEPTH} levels of nesting")
        full = self.root / Path(*parts)
        # Ensure final path lies under root.
        try:
            full.resolve().relative_to(self.root.resolve())
        except ValueError as e:
            raise ArchiveError(f"Archive path escapes the archive root: {rel}") from e
        return full

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        results: list[dict] = []
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            stat = path.stat()
            results.append({"path": rel, "size": stat.st_size, "mtime": int(stat.st_mtime * 1000)})
        return results

    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()

    def read(self, rel: str) -> str:
        full = self._resolve(rel)
        if not full.exists():
            raise ArchiveError(f"Archive file not found: {rel}")
        return full.read_text(encoding="utf-8")

    async def write(self, rel: str, content: str) -> None:
        await asyncio.to_thread(self._write_sync, rel, content)

    def _write_sync(self, rel: str, content: str) -> None:
        full = self._resolve(rel)
        data = content.encode("utf-8")
        if len(data) > ARCHIVE_MAX_BYTES:
            raise ArchiveError(
                f"Archive file exceeds {ARCHIVE_MAX_BYTES} bytes ({len(data)} bytes given)"
            )
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_suffix(full.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(full)

    async def edit(self, rel: str, old: str, new: str, replace_all: bool = False) -> int:
        return await asyncio.to_thread(self._edit_sync, rel, old, new, replace_all)

    def _edit_sync(self, rel: str, old: str, new: str, replace_all: bool) -> int:
        full = self._resolve(rel)
        if not full.exists():
            raise ArchiveError(f"Archive file not found: {rel}")
        current = full.read_text(encoding="utf-8")
        if old not in current:
            raise ArchiveError("Old string not found in archive file")
        if not replace_all and current.count(old) > 1:
            raise ArchiveError(
                "Old string is not unique — provide more context or set replace_all=True"
            )
        if replace_all:
            count = current.count(old)
            updated = current.replace(old, new)
        else:
            count = 1
            updated = current.replace(old, new, 1)
        self._write_sync(rel, updated)
        return count

    async def move(self, src: str, dst: str) -> None:
        await asyncio.to_thread(self._move_sync, src, dst)

    def _move_sync(self, src: str, dst: str) -> None:
        src_full = self._resolve(src)
        dst_full = self._resolve(dst)
        if not src_full.exists():
            raise ArchiveError(f"Archive file not found: {src}")
        if dst_full.exists():
            raise ArchiveError(f"Destination already exists: {dst}")
        dst_full.parent.mkdir(parents=True, exist_ok=True)
        src_full.replace(dst_full)
