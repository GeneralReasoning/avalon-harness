"""Parallel rollout helpers (§12.2).

`run_openreward_rollouts` drives N rollouts of one OpenReward environment in
parallel; `_run_one_rollout` holds the per-rollout pipeline (env session open,
adapter, completion, trajectory dump).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from openreward import AsyncOpenReward
from openreward.api.environments.types import Task as ORTask

from agentorg.compaction import CompactionConfig, avalon_compaction_config
from agentorg.harness import Harness, RolloutResult
from agentorg.inference.openai_compat import OpenAICompatProvider
from agentorg.session import SessionConfig
from agentorg.trajectory import dump_trajectory
from agentorg.types import EnvironmentBinding

from agentorg.openreward_env import OpenRewardEnvAdapter

log = logging.getLogger(__name__)


@dataclass
class OpenRewardSamplingConfig:
    environment: str
    split: str = "train"
    indices: Optional[list[int]] = None
    limit: Optional[int] = None
    workspace_root: str = "rollouts"
    model: str = "GeneralReasoning/Avalon-M1-Mini"
    budget: int = 2
    mode: str = "org-plan"  # plan-then-act default
    toolset: Optional[str] = None
    max_wall_s: Optional[float] = None  # None → no wall-clock cap.
    max_total_tokens: Optional[int] = None  # None → no token cap.
    # Compaction settings. The default is what Avalon-M1-Mini was trained with;
    # for another model, pass a CompactionConfig sized to its context window.
    compaction: CompactionConfig = field(default_factory=avalon_compaction_config)
    concurrency: int = 4
    headless: bool = True
    permission_policy: str = "bypass"
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = 1.0  # as in training
    parallel_tool_calls: bool = True
    # Optional override of the per-invocation run id. If unset, a fresh
    # `<timestamp>-<short-uuid>` is generated so reruns of the same env do
    # not overwrite each other.
    run_id: Optional[str] = None
    rollout_id: Optional[str] = None
    # Training prompts carried no date. Set to put today's date in the system
    # prompt anyway.
    include_date: bool = False
    # Secrets forwarded to the env at session-create time (e.g. an
    # `openai_api_key` required by the env's grader). Well-known names —
    # see `openreward.api.sandboxes.secrets.WELL_KNOWN_SECRETS` — auto-resolve
    # their allowed domains; pass `(value, [domains])` tuples for anything
    # else. The CLI populates this from env vars; never from argv values.
    secrets: Optional[Mapping[str, Union[str, tuple[str, list[str]]]]] = None


def parse_secret_flags(specs: list[str]) -> dict[str, str]:
    """Resolve `--secret KEY[=ENV_VAR]` flag values from `os.environ`.

    `KEY` is the name the env expects (e.g. `openai_api_key`). When
    `=ENV_VAR` is omitted, the source env var defaults to `KEY.upper()`
    (e.g. `OPENAI_API_KEY`). The actual secret never appears in argv —
    only the env-var name does — so command lines and process listings
    stay clean.

    Raises `ValueError` if a referenced env var is unset; callers
    surface that as a CLI-level error.
    """
    out: dict[str, str] = {}
    for spec in specs:
        key, _, env_var = spec.partition("=")
        if not env_var:
            env_var = key.upper()
        if not key:
            raise ValueError(f"--secret {spec!r}: empty KEY")
        value = os.environ.get(env_var)
        if value is None:
            raise ValueError(
                f"--secret {spec!r}: environment variable {env_var!r} is not set"
            )
        out[key] = value
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unknown"


def _default_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


async def _run_one_rollout(
    *,
    task: Optional[ORTask],
    task_index: int,
    session_cfg: SessionConfig,
    run_dir: Path,
    max_wall_s: Optional[float],
    max_total_tokens: Optional[int],
    harness: Harness,
    or_env: Any,
    or_client: AsyncOpenReward,
    on_session_started=None,
    secrets: Optional[Mapping[str, Union[str, tuple[str, list[str]]]]] = None,
) -> RolloutResult:
    """Drive a single OpenReward rollout end-to-end.

    Caller owns:
    - constructing `session_cfg` (env binding, workspace_path, model, budget, …),
    - creating `run_dir` (trajectory.json is written there),
    - concurrency control (no semaphore acquired here),
    - any `on_complete` reporting (this function only fires `on_session_started`).

    `deployment_name` / `split` / `toolset` are read from `session_cfg`
    rather than passed separately so the caller has a single source of
    truth.
    """
    assert session_cfg.environment is not None, "session_cfg must have env binding set"
    deployment_name = session_cfg.environment.task_id
    split = session_cfg.environment.split
    toolset = session_cfg.toolset_name
    async with or_env.session(  # type: ignore[arg-type]
        task=task,
        toolset=toolset,
        secrets=secrets,
    ) as session_ctx:
        adapter = OpenRewardEnvAdapter(
            client=or_client,
            session=session_ctx,
            deployment_name=deployment_name,
            split=split,
            task_index=task_index,
            toolset_name=toolset,
        )
        # Fetch the env's task prompt and feed into CEO.
        blocks = await session_ctx.get_prompt()
        preamble = "\n\n".join(
            b.text for b in blocks if getattr(b, "type", None) == "text"
        )
        session_cfg.initial_user_message = preamble or None

        session = await harness.start_session(config=session_cfg, env_adapter=adapter)
        hook_obj = None
        if on_session_started is not None:
            hook_obj = await on_session_started(session, task_index, task)
        try:
            result = await harness.run_to_completion(
                session, max_wall_s=max_wall_s, max_total_tokens=max_total_tokens
            )
            return result
        finally:
            # Dump the trajectory first, before the async teardown below. It's a
            # pure, synchronous function of the session dir (so it can't be
            # cancelled mid-unwind) and is independent of the hook. Doing
            # it in `finally` means failed/timed-out/interrupted rollouts still
            # leave a trajectory.json for debugging — exactly when it's wanted.
            try:
                dump_trajectory(
                    Path(session.paths.root),
                    run_dir / "trajectory.json",
                )
            except Exception:
                log.exception("trajectory dump failed for %s", run_dir)
            if hook_obj is not None and hasattr(hook_obj, "stop"):
                await hook_obj.stop()


async def run_openreward_rollouts(
    config: OpenRewardSamplingConfig,
    *,
    inference_base_url: Optional[str] = None,
    inference_api_key: Optional[str] = None,
    openreward_api_key: Optional[str] = None,
    on_complete=None,
    on_session_started=None,
    on_failed=None,
) -> list[RolloutResult]:
    """Run N OpenReward rollouts in parallel.

    `index` in every hook is the task's index in the split, which also names
    its `task-NNNN` directory.

    Hooks (all optional):

    - `on_complete(index, task, result)` — sync callback fired after each
      rollout finishes (used for progress prints).
    - `on_failed(index, task, exc)` — sync callback fired for each rollout
      that raised instead of producing a result.
    - `on_session_started(session, index, task)` — async callback fired
      *after* the Session is built and *before* `run_to_completion` blocks.
      May return an object with an async `stop()` method; that `stop` is
      awaited after the rollout completes. This is the seam UX adornments
      (live tracers, progress bars, recorders) attach to without the
      driver depending on them.
    """
    if config.indices is not None and len(set(config.indices)) != len(config.indices):
        # Each task's index names its run directory.
        raise ValueError(f"duplicate task indices: {config.indices}")
    provider = OpenAICompatProvider(api_key=inference_api_key, base_url=inference_base_url)
    harness = Harness(inference=provider)
    client = AsyncOpenReward(api_key=openreward_api_key)
    env = client.environments.get(config.environment)
    session_secrets = dict(config.secrets) if config.secrets else None

    # Resolve task list.
    if config.indices is not None:
        indices = list(config.indices)
        tasks = [await env.get_task(config.split, i) for i in indices]
    else:
        all_tasks = await env.list_tasks(config.split)
        n = config.limit if config.limit is not None else len(all_tasks)
        tasks = all_tasks[:n]
        indices = list(range(len(tasks)))

    run_id = config.run_id or _default_run_id()
    env_slug = _slug(config.environment) + "__" + _slug(config.split)
    workspace_root = Path(config.workspace_root) / env_slug / run_id
    workspace_root.mkdir(parents=True, exist_ok=True)
    log.info(
        "Running %d rollouts of %s (split=%s) into %s",
        len(tasks),
        config.environment,
        config.split,
        workspace_root,
    )
    sem = asyncio.Semaphore(max(1, config.concurrency))
    results: list[RolloutResult] = []

    async def one(task: ORTask, index: int) -> RolloutResult:
        async with sem:
            run_dir = workspace_root / f"task-{index:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            session_cfg = SessionConfig(
                workspace_path=str(run_dir),
                model=config.model,
                budget=config.budget,
                mode=config.mode,
                environment=EnvironmentBinding(
                    kind="openreward",
                    task_id=config.environment,
                    split=config.split,
                    client_config={"index": index},
                ),
                headless=config.headless,
                permission_policy=config.permission_policy,
                initial_user_message=None,
                toolset_name=config.toolset,
                reasoning_effort=config.reasoning_effort,
                temperature=config.temperature,
                parallel_tool_calls=config.parallel_tool_calls,
                date_str=time.strftime("%Y-%m-%d", time.gmtime()) if config.include_date else None,
                rollout_id=config.rollout_id,
                compaction=config.compaction,
            )
            result = await _run_one_rollout(
                task=task,
                task_index=index,
                session_cfg=session_cfg,
                run_dir=run_dir,
                max_wall_s=config.max_wall_s,
                max_total_tokens=config.max_total_tokens,
                harness=harness,
                or_env=env,
                or_client=client,
                on_session_started=on_session_started,
                secrets=session_secrets,
            )
            if on_complete is not None:
                on_complete(index, task, result)
            return result

    coros = [one(task, i) for i, task in zip(indices, tasks)]
    try:
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
    finally:
        # Also on cancellation (Ctrl-C), so the client's connections close.
        try:
            await client.environments.aclose()
        except Exception:
            pass
    for i, task, r in zip(indices, tasks, raw_results):
        if isinstance(r, BaseException):
            log.error("rollout failed: %s", r)
            if on_failed is not None:
                on_failed(i, task, r)
            continue
        results.append(r)
    return results
