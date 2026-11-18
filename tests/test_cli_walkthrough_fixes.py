"""Fixes from a fresh-user run of the CLI: a dead model server ends the
session as aborted (not "complete") and fails the command; rollouts are
labelled by their task index in the split, not their position in the batch;
and the inspector shows channels, which carry no timestamps."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import (
    CompletedChunk,
    ContentDeltaChunk,
    InferenceProvider,
    TokenUsage,
    ToolCallChunk,
)
from agentorg.session import SessionConfig
from agentorg.cli import inspect_rollout, sample
from agentorg.cli.lib import sampling


class _DeadServer(InferenceProvider):
    async def stream(self, request):
        if False:  # pragma: no cover — async-generator signature
            yield
        raise ConnectionError("server unreachable")

    def model_ids(self):
        return ["dead"]


async def test_ceo_inference_failure_aborts_the_session(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="dead", budget=0, headless=True,
        permission_policy="bypass", initial_user_message="hi",
    )
    harness = Harness(inference=_DeadServer())
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=10)
    assert result.end_reason == "inference-error:ConnectionError"
    assert session.abort_reason == "inference-error:ConnectionError"


async def test_graded_episode_keeps_its_ending_when_the_ceo_then_fails(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="dead", budget=0, headless=True,
        permission_policy="bypass", initial_user_message="hi",
    )
    harness = Harness(inference=_DeadServer())
    session = await harness.start_session(config=config)
    session.mark_terminal({"tool": "submit", "by": "CEO", "reward": 1.0}, reward=1.0, stop=False)
    result = await harness.run_to_completion(session, max_wall_s=10)
    assert session.abort_reason is None
    assert result.end_reason == "env-terminal"


class _CeoBlip(InferenceProvider):
    """The CEO hires, its next call fails once, the specialist reports, and
    the restarted CEO fires it and finishes."""

    def __init__(self):
        self.ceo_calls = 0
        self.org_name = None

    async def stream(self, request):
        who = (request.extra_headers or {}).get("X-Agent-Id")
        if who in (None, "CEO"):
            self.ceo_calls += 1
            n = self.ceo_calls
            if n == 1:
                yield ToolCallChunk(id="c1", name="HireOrgAgent", args={"role": "Tester", "description": "Do it."})
            elif n == 2:
                raise ConnectionError("transient blip")
            elif n == 3:
                yield ToolCallChunk(id="c3", name="FireOrgAgent", args={
                    "name": self.org_name, "category": "mission_complete",
                    "delivered": "x", "shortfalls": "-", "score": 8,
                })
            else:
                yield ContentDeltaChunk(text="done")
                yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")
                return
            yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")
            return
        self.org_name = who
        await asyncio.sleep(0.3)
        yield ToolCallChunk(id="a1", name="ReportToCEO", args={"status": "completed", "summary": "ok"})
        yield CompletedChunk(usage=TokenUsage(), stop_reason="tool_calls")

    def model_ids(self):
        return ["blip"]


async def test_ceo_recovers_from_a_failed_call_when_input_arrives(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path), model="blip", budget=2, mode="default",
        headless=True, permission_policy="bypass", initial_user_message="go",
    )
    provider = _CeoBlip()
    harness = Harness(inference=provider)
    session = await harness.start_session(config=config)
    result = await harness.run_to_completion(session, max_wall_s=20)
    assert result.end_reason == "complete"
    assert provider.ceo_calls >= 3
    assert all(a.state == "fired" for a in session.all_agents() if a.kind == "org")


class _FakeEnv:
    async def get_task(self, split, i):
        return f"task{i}"

    async def list_tasks(self, split):
        return [f"task{i}" for i in range(3)]


class _FakeClient:
    def __init__(self, api_key=None):
        self.environments = SimpleNamespace(get=lambda name: _FakeEnv(), aclose=self._aclose)
        self.rollout = None
        self.closed = False

    async def _aclose(self):
        self.closed = True


def _patch_driver(monkeypatch, fail_on=()):
    seen = []

    async def fake_run_one(*, task, task_index, run_dir, session_cfg, **_):
        seen.append((task, task_index, run_dir.name, session_cfg.environment.client_config))
        if task_index in fail_on:
            raise RuntimeError("env refused")
        return SimpleNamespace(end_reason="env-terminal")

    monkeypatch.setattr(sampling, "AsyncOpenReward", _FakeClient)
    monkeypatch.setattr(sampling, "_run_one_rollout", fake_run_one)
    return seen


def _cfg(tmp_path, **kw):
    return sampling.OpenRewardSamplingConfig(
        environment="x/y", workspace_root=str(tmp_path), **kw
    )


async def test_rollouts_are_labelled_by_task_index(tmp_path, monkeypatch):
    seen = _patch_driver(monkeypatch)
    done = []
    await sampling.run_openreward_rollouts(
        _cfg(tmp_path, indices=[5, 9]), inference_base_url="http://x/v1",
        on_complete=lambda i, t, r: done.append(i),
    )
    assert sorted(seen) == [
        ("task5", 5, "task-0005", {"index": 5}),
        ("task9", 9, "task-0009", {"index": 9}),
    ]
    assert sorted(done) == [5, 9]


async def test_duplicate_indices_are_rejected(tmp_path, monkeypatch):
    _patch_driver(monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        await sampling.run_openreward_rollouts(
            _cfg(tmp_path, indices=[1, 1]), inference_base_url="http://x/v1"
        )


async def test_failed_rollouts_reach_on_failed(tmp_path, monkeypatch):
    _patch_driver(monkeypatch, fail_on={1})
    failed = []
    results = await sampling.run_openreward_rollouts(
        _cfg(tmp_path, limit=3), inference_base_url="http://x/v1",
        on_failed=lambda i, t, e: failed.append((i, str(e))),
    )
    assert failed == [(1, "env refused")]
    assert len(results) == 2


@pytest.mark.parametrize("end_reasons, raised, code", [
    (["env-terminal"], [], 0),
    (["env-terminal"], [3], 1),
    (["inference-error:APIConnectionError"], [], 1),
    (["env-session-dead:bash"], [], 1),
])
def test_cli_exit_code_reports_failures(monkeypatch, end_reasons, raised, code):
    async def fake_run(cfg, *, on_failed, **_):
        for i in raised:
            on_failed(i, None, RuntimeError("boom"))
        return [SimpleNamespace(end_reason=r) for r in end_reasons]

    monkeypatch.setattr(sample, "run_openreward_rollouts", fake_run)
    assert sample.main(["--env", "x/y", "--no-trace"]) == code


def test_cli_failure_line_names_the_exception(monkeypatch, capsys):
    async def fake_run(cfg, *, on_failed, **_):
        on_failed(3, None, asyncio.TimeoutError())
        return []

    monkeypatch.setattr(sample, "run_openreward_rollouts", fake_run)
    sample.main(["--env", "x/y", "--no-trace"])
    assert "[task 0003] failed: TimeoutError" in capsys.readouterr().out


def test_cli_rejects_duplicate_indices(capsys):
    assert sample.main(["--env", "x/y", "--indices", "1,1"]) == 2
    assert "duplicates" in capsys.readouterr().err


async def test_cancelled_batch_closes_the_client(tmp_path, monkeypatch):
    clients = []

    class _Client(_FakeClient):
        def __init__(self, api_key=None):
            super().__init__(api_key)
            clients.append(self)

    async def hang(**_):
        await asyncio.sleep(3600)

    monkeypatch.setattr(sampling, "AsyncOpenReward", _Client)
    monkeypatch.setattr(sampling, "_run_one_rollout", hang)
    task = asyncio.create_task(sampling.run_openreward_rollouts(
        _cfg(tmp_path, indices=[0]), inference_base_url="http://x/v1"
    ))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clients[0].closed


def _channels(tmp_path, posts, reactions=()):
    ch = tmp_path / "channels"
    ch.mkdir()
    (ch / "_meta.json").write_text(json.dumps({"channels": {"ops": {}, "quiet": {}}}))
    (ch / "ops.jsonl").write_text("".join(json.dumps(p) + "\n" for p in posts))
    (ch / "ops.reactions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reactions))


def test_inspector_prints_channel_posts_and_reactions(tmp_path, capsys):
    _channels(
        tmp_path,
        [{"idx": 0, "msg_key": "ops:0", "author": "CEO", "content": "status?"}],
        [
            {"msg_key": "ops:0", "emoji": "👍", "reactor": "A", "op": "add"},
            {"msg_key": "ops:0", "emoji": "👀", "reactor": "B", "op": "add"},
            {"msg_key": "ops:0", "emoji": "👀", "reactor": "B", "op": "remove"},
        ],
    )
    inspect_rollout.print_channel(tmp_path, "ops")
    out = capsys.readouterr().out
    assert "@CEO [idx=0]" in out and "status?" in out
    assert "👍" in out and "👀" not in out


def test_inspector_empty_and_unknown_channels(tmp_path, capsys):
    _channels(tmp_path, [])
    inspect_rollout.print_channel(tmp_path, "quiet")
    assert "#quiet — 0 posts" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="channels: ops, quiet"):
        inspect_rollout.print_channel(tmp_path, "nope")
