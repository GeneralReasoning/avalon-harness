"""Sampler CLI — drives N OpenReward rollouts of an environment.

Defaults match how Avalon-M1-Mini was trained: org-plan mode, temperature 1.0,
no date in the system prompt, and the training compaction settings. The model
server defaults to http://localhost:30000/v1 (SGLang's default port; pass --port 30000 to vLLM).

Examples:

    uv run avalon --env GeneralReasoning/Chernobyl --limit 1
    uv run avalon --env GeneralReasoning/Chernobyl --indices 0,1,2 --budget 4

The CLI is a thin wrapper around `agentorg.cli.lib.sampling.run_openreward_rollouts`.
It owns the user experience (argument parsing, live tracer, progress output);
the driver underneath stays free of UX concerns.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from agentorg.cli.cli_view import _CONTROL_RE, LiveRenderer
from agentorg.compaction import CompactionConfig
from agentorg.cli.lib.sampling import (
    OpenRewardSamplingConfig,
    parse_secret_flags,
    run_openreward_rollouts,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="avalon", description="Run Avalon-M1-Mini organisations on OpenReward environment tasks.")
    p.add_argument("--env", required=True, help="e.g. GeneralReasoning/Chernobyl")
    p.add_argument("--split", default="train", help="Task split (default: train).")
    p.add_argument("--indices", help="Comma-separated task indices to roll out.")
    p.add_argument("--limit", type=int, default=1, help="Roll out the first N tasks of the split (default 1). Ignored with --indices.")
    p.add_argument(
        "--workspace-root",
        default="rollouts",
        help="Top-level rollouts dir. Layout: <root>/<env-slug>__<split>/<run-id>/task-NNNN/",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override the per-invocation run id (default: timestamp + short uuid).",
    )
    p.add_argument("--rollout-id", default=None, help="Optional rollout id sent as X-Rollout-Id header on inference calls.")
    p.add_argument("--model", default="GeneralReasoning/Avalon-M1-Mini", help="Model name as the server knows it.")
    p.add_argument("--budget", type=int, default=2, help="Maximum number of specialists alive at once (default 2; 0 runs a single agent).")
    p.add_argument("--mode", default="org-plan", choices=["default", "org-plan"], help="org-plan (default): the CEO designs the organisation in plan mode before acting. default: no plan phase.")
    p.add_argument("--toolset", default=None, help="Named toolset to request from the environment, if it offers one (e.g. claude-code). Training used none.")
    p.add_argument("--max-wall-s", type=float, default=None, help="Wall-clock cap per session in seconds. Omit for no cap.")
    p.add_argument("--max-total-tokens", type=int, default=None, help="Cap on cumulative total tokens (input+output, no double-counting of re-sent context) per session. Omit for no cap.")
    p.add_argument("--context-window-tokens", type=int, default=None, help="For models other than Avalon-M1-Mini: the model's context window in tokens, from which compaction thresholds are derived. Omit to use the settings Avalon-M1-Mini was trained with.")
    p.add_argument("--concurrency", type=int, default=4, help="Rollouts run in parallel (default 4).")
    p.add_argument("--temperature", type=float, default=1.0, help="Default 1.0, as in training.")
    p.add_argument("--include-date", action="store_true", help="Put today's date in the system prompt (training prompts had none).")
    p.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    p.add_argument("--non-parallel-tool-calls", action="store_true")
    tg = p.add_mutually_exclusive_group()
    tg.add_argument(
        "--trace",
        dest="trace",
        action="store_true",
        default=None,
        help="Stream a human-readable trace of CEO activity to stdout. "
        "Default: on when running a single rollout, off otherwise.",
    )
    tg.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="Suppress the live activity trace.",
    )
    p.add_argument(
        "--trace-verbose",
        action="store_true",
        help="With --trace, also print otherwise-suppressed event types.",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("AVALON_BASE_URL", "http://localhost:30000/v1"),
        help="OpenAI-compatible endpoint serving the model (default: $AVALON_BASE_URL, else http://localhost:30000/v1).",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("AVALON_API_KEY", "none"),
        help="API key for --base-url, if the server needs one (default: $AVALON_API_KEY).",
    )
    p.add_argument("--openreward-api-key", default=os.getenv("OPENREWARD_API_KEY"))
    p.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="KEY[=ENV_VAR]",
        help=(
            "Forward an env-side secret (e.g. `--secret openai_api_key` for "
            "graders that need an OpenAI key). KEY is the name the env "
            "expects; the value is read from $ENV_VAR (default: $KEY in "
            "uppercase). Repeatable. The actual key never appears in argv."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
        if len(set(indices)) != len(indices):
            print(f"error: --indices has duplicates: {args.indices}", file=sys.stderr)
            return 2

    n_tasks = len(indices) if indices is not None else args.limit
    if args.trace is None:
        # Default heuristic: on whenever output won't be hopelessly interleaved.
        # Single-task runs always trace; sequential runs (concurrency=1) trace
        # one rollout at a time. Disable for genuine batches.
        args.trace = (
            n_tasks is None or n_tasks <= 1 or args.concurrency == 1
        )

    # Quiet the chatty third-party libs unless the user asked for verbose;
    # the live tracer is the more useful signal at the default level.
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "openai", "openreward", "aiohttp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        secrets = parse_secret_flags(args.secret) or None
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cfg = OpenRewardSamplingConfig(
        environment=args.env,
        split=args.split,
        indices=indices,
        limit=args.limit,
        workspace_root=args.workspace_root,
        model=args.model,
        budget=args.budget,
        mode=args.mode,
        toolset=args.toolset,
        max_wall_s=args.max_wall_s,
        max_total_tokens=args.max_total_tokens,
        **(
            {"compaction": CompactionConfig(context_window_tokens=args.context_window_tokens)}
            if args.context_window_tokens is not None
            else {}
        ),
        concurrency=args.concurrency,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        parallel_tool_calls=not args.non_parallel_tool_calls,
        run_id=args.run_id,
        secrets=secrets,
        rollout_id=args.rollout_id,
        include_date=args.include_date,
    )

    trace_enabled = bool(args.trace)
    trace_verbose = bool(args.trace_verbose)
    multi = (n_tasks is not None and n_tasks > 1) and args.concurrency > 1

    async def on_session_started(session, index, _task):
        if not trace_enabled:
            return None
        prefix = f"task-{index:04d}" if multi else None
        renderer = LiveRenderer(session, prefix=prefix, verbose=trace_verbose)
        renderer.start()
        return renderer

    def on_complete(index, task, result):
        reward = f"{result.reward:.3f}" if result.reward is not None else "—"
        # end_reason can carry an env-supplied error code.
        end_reason = _CONTROL_RE.sub("", result.end_reason)
        line = (
            f"[task {index:04d}] reward={reward} ({end_reason}) "
            f"duration={result.duration_s:.1f}s turns={result.used_turns} "
            f"orgs={len(result.org_agents)}"
        )
        if result.reward_steps > 1:
            line += f" steps_w_reward={result.reward_steps}"
        print(line)

    raised: list[int] = []

    def on_failed(index, task, exc):
        raised.append(index)
        reason = _CONTROL_RE.sub("", f"{type(exc).__name__}: {exc}")[:500]
        print(f"[task {index:04d}] failed: {reason}")

    try:
        results = asyncio.run(run_openreward_rollouts(
            cfg,
            inference_base_url=args.base_url,
            inference_api_key=args.api_key,
            openreward_api_key=args.openreward_api_key,
            on_complete=on_complete,
            on_session_started=on_session_started,
            on_failed=on_failed,
        ))
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    # Sessions that lost the model server or the env session end with a
    # result, but they still failed.
    infra = ("inference-error", "env-session-dead")
    n_failed = len(raised) + sum(r.end_reason.startswith(infra) for r in results)
    print(f"finished {len(results)} rollouts" + (f", {n_failed} failed" if n_failed else ""))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
