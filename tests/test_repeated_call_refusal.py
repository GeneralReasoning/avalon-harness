"""SPEC §A.8.6 — refuse a call identical to the agent's current run.

A run is consecutive completed calls with the same (tool, canonical args,
content digest). At `repeat_refuse_after` (default 15) the next identical
call is not executed: `ok: false`, `error.code = "repeated-call"`, kind
"rejection", with `shaping_reward` / `shaping_reason` stamped on the payload
so the trace consumer sees the penalty against the call that earned it.
"""

from __future__ import annotations

import json

import pytest

from agentorg.harness import Harness
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.loop import _dispatch_tools, _serialize_tool_result
from agentorg.session import SessionConfig
from agentorg.tools.context import err_result, ok_result
from agentorg.types import Message, ToolCall, ToolDefinition


class _ConstTool:
    """Returns the same output every call (the loop fingerprint)."""

    def __init__(self, name="probe", scope="local", output=None):
        self.definition = ToolDefinition(
            name=name, description="const",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            read_only=False, scope=scope,
        )
        self._output = output if output is not None else {"v": 1}

    async def execute(self, ctx):
        return ok_result(dict(self._output))


class _CounterTool(_ConstTool):
    """Same args, different output every call: never a loop."""

    def __init__(self):
        super().__init__(name="counter")
        self.n = 0

    async def execute(self, ctx):
        self.n += 1
        return ok_result({"blocks": [{"type": "text", "text": f"tick {self.n}"}],
                          "metadata": {"round": self.n}})


class _EnvConstTool(_ConstTool):
    """Env-shaped output whose only per-call difference is `metadata`."""

    def __init__(self):
        super().__init__(name="place_bet", scope="env")
        self.n = 0

    async def execute(self, ctx):
        self.n += 1
        return ok_result({"blocks": [{"type": "text", "text": "insufficient funds"}],
                          "metadata": {"ts": self.n}, "reward": 0.0, "finished": False})


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["mock"]


async def _session(tmp_path, **overrides):
    cfg = {"workspace_path": str(tmp_path), "model": "mock", "budget": 0, "mode": "default",
           "headless": True, "permission_policy": "bypass"}
    cfg.update(overrides)
    session = await Harness(inference=_Stub()).start_session(config=SessionConfig(**cfg))
    for t in (_ConstTool(), _CounterTool(), _EnvConstTool()):
        session.tools.register(t)
    return session


async def _call(session, name, args=None, call_id="c"):
    agent = session.agent(session.ceo_name)
    call = ToolCall(id=call_id, name=name, args=args or {})
    asst = Message(role="assistant", tool_calls=[call])
    await session.append_message(agent.name, asst)
    await _dispatch_tools(session, agent, asst, [call])
    return [m for m in session.transcript(agent.name) if m.role == "tool"][-1]


def _events(session, event_type):
    rows = (json.loads(ln) for ln in session.paths.events_jsonl.read_text().splitlines() if ln.strip())
    return [r for r in rows if r.get("type") == event_type]


@pytest.mark.asyncio
async def test_sixteenth_identical_call_is_refused_with_stamp(tmp_path):
    session = await _session(tmp_path)
    for i in range(15):
        msg = await _call(session, "probe", {"q": "x"}, f"k{i}")
        assert msg.tool_result.ok, i

    msg = await _call(session, "probe", {"q": "x"}, "k15")
    r = msg.tool_result
    assert r.ok is False
    assert r.error.code == "repeated-call"
    assert r.error.kind == "rejection"
    assert r.error.message.startswith("Not run. This is identical to your last 15 calls")
    for banned in ("FireOrgAgent", "submit", "finish"):
        assert banned not in r.error.message
    assert r.output == {"shaping_reward": pytest.approx(session.config.noop_penalty),
                        "shaping_reason": "repeated-call"}

    # Streams: a rejection, never a tool error.
    assert [e.code for e in session.harness_rejection_events] == ["repeated-call"]
    assert session.tool_error_events == []
    # Event with the run and threshold.
    [ev] = _events(session, "agent.repeat_refused")
    assert ev["payload"] == {"name": session.ceo_name, "kind": "ceo", "tool": "probe",
                             "run": 15, "threshold": 15,
                             "penalty": pytest.approx(session.config.noop_penalty)}
    # §12.4: both persisted copies carry the stamp and the kind.
    [completed] = [e for e in _events(session, "tool.completed") if e["payload"]["call_id"] == "k15"]
    assert completed["payload"]["output"]["shaping_reason"] == "repeated-call"
    assert completed["payload"]["error"]["kind"] == "rejection"
    # Rendered to the model as a rejection; the stamp is not for its eyes.
    rendered = json.loads(_serialize_tool_result(r))
    assert rendered["error"]["kind"] == "rejection"
    assert rendered["error"]["code"] == "repeated-call"
    assert "output" not in rendered


@pytest.mark.asyncio
async def test_refused_call_extends_the_run(tmp_path):
    session = await _session(tmp_path)
    for i in range(15):
        await _call(session, "probe", {"q": "x"}, f"k{i}")
    runs = []
    for i in range(15, 19):
        msg = await _call(session, "probe", {"q": "x"}, f"k{i}")
        assert msg.tool_result.error.code == "repeated-call"
    runs = [e["payload"]["run"] for e in _events(session, "agent.repeat_refused")]
    assert runs == [15, 16, 17, 18]


@pytest.mark.asyncio
async def test_different_output_never_refused(tmp_path):
    session = await _session(tmp_path)
    for i in range(30):
        msg = await _call(session, "counter", {"q": "x"}, f"k{i}")
        assert msg.tool_result.ok, i
    assert session.harness_rejection_events == []


@pytest.mark.asyncio
async def test_metadata_only_difference_is_still_identical(tmp_path):
    """Env outputs carry per-call bookkeeping in `metadata`; the digest
    ignores it, so the same losing bet 15 times is a run."""
    session = await _session(tmp_path)
    for i in range(15):
        assert (await _call(session, "place_bet", {"q": "all"}, f"k{i}")).tool_result.ok
    msg = await _call(session, "place_bet", {"q": "all"}, "k15")
    assert msg.tool_result.error.code == "repeated-call"
    # Harness rejection, so the env-tool circuit breaker is untouched.
    assert not session.abort_requested


@pytest.mark.asyncio
async def test_different_args_reset_the_run(tmp_path):
    session = await _session(tmp_path)
    for i in range(15):
        await _call(session, "probe", {"q": "x"}, f"a{i}")
    assert (await _call(session, "probe", {"q": "y"}, "b0")).tool_result.ok
    # Back to x: a fresh run of one, so it executes.
    assert (await _call(session, "probe", {"q": "x"}, "a15")).tool_result.ok
    assert session.harness_rejection_events == []


@pytest.mark.asyncio
async def test_a_different_call_between_ends_the_run(tmp_path):
    session = await _session(tmp_path)
    for i in range(14):
        await _call(session, "probe", {"q": "x"}, f"a{i}")
    await _call(session, "counter", {"q": "x"}, "c0")
    for i in range(14, 29):
        assert (await _call(session, "probe", {"q": "x"}, f"a{i}")).tool_result.ok
    assert (await _call(session, "probe", {"q": "x"}, "a29")).tool_result.error.code == "repeated-call"


@pytest.mark.asyncio
async def test_wait_for_messages_is_exempt(tmp_path):
    """With no live peers the CEO's WaitForMessages returns the same marker
    every call; it must never be refused."""
    session = await _session(tmp_path, budget=4)
    for i in range(20):
        msg = await _call(session, "WaitForMessages", {}, f"w{i}")
        assert msg.tool_result.ok, msg.tool_result
        assert msg.tool_result.output.get("no_live_peers") is True
    assert session.harness_rejection_events == []


@pytest.mark.asyncio
async def test_zero_threshold_disables(tmp_path):
    session = await _session(tmp_path, repeat_refuse_after=0)
    for i in range(40):
        assert (await _call(session, "probe", {"q": "x"}, f"k{i}")).tool_result.ok
    assert session.harness_rejection_events == []


@pytest.mark.asyncio
async def test_penalty_configurable_and_zero_suppresses_stamp(tmp_path):
    session = await _session(tmp_path, repeat_penalty=-0.2)
    for i in range(15):
        await _call(session, "probe", {"q": "x"}, f"k{i}")
    r = (await _call(session, "probe", {"q": "x"}, "k15")).tool_result
    assert r.output["shaping_reward"] == pytest.approx(-0.2)

    session = await _session(tmp_path / "zero", repeat_penalty=0.0)
    for i in range(15):
        await _call(session, "probe", {"q": "x"}, f"k{i}")
    r = (await _call(session, "probe", {"q": "x"}, "k15")).tool_result
    assert r.error.code == "repeated-call"
    assert r.output is None
    [ev] = _events(session, "agent.repeat_refused")
    assert ev["payload"]["penalty"] == 0.0


class _FailingTool(_ConstTool):
    def __init__(self):
        super().__init__(name="broken")

    async def execute(self, ctx):
        return err_result("not-found", "No such file: /x")


@pytest.mark.asyncio
async def test_identical_failures_are_a_run_too(tmp_path):
    """The digest of a failure is its code and message, so the same failing
    call 15 times is a run and the 16th is refused."""
    session = await _session(tmp_path)
    session.tools.register(_FailingTool())
    for i in range(15):
        r = (await _call(session, "broken", {"q": "x"}, f"k{i}")).tool_result
        assert r.error.code == "not-found"
    r = (await _call(session, "broken", {"q": "x"}, "k15")).tool_result
    assert r.error.code == "repeated-call"
    assert [e.code for e in session.tool_error_events] == ["not-found"] * 15
    assert [e.code for e in session.harness_rejection_events] == ["repeated-call"]


@pytest.mark.asyncio
async def test_unknown_tool_is_not_keyed(tmp_path):
    """unknown-tool returns before the rule runs: there is no tool to key on."""
    session = await _session(tmp_path)
    for i in range(20):
        assert (await _call(session, "nosuch", {}, f"k{i}")).tool_result.error.code == "unknown-tool"
    assert session.harness_rejection_events == []


@pytest.mark.asyncio
async def test_transient_rejection_does_not_build_a_run(tmp_path):
    """A gated call never ran, so it is no evidence about what the call
    returns and must not key a run.

    Before this, fifteen identical `wait-gate-restricted` results built a run
    of 15; the moment the gate lifted, the first call that would finally have
    executed was refused as `repeated-call` instead — and stayed refused for
    good, because a refusal extends the run it belongs to. The CEO lost the
    tool permanently, and was charged `repeat_penalty` every turn for it.
    """
    from agentorg.org import fire_org_agent, spawn_org_agent
    from agentorg.types import ExitReview, OrgAgentSpec

    session = await _session(tmp_path, budget=4)
    session.tools.register(_ConstTool(name="bash", scope="env"))
    peer = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))

    for i in range(15):
        r = await _call(session, "bash", {}, f"g{i}")
        assert r.tool_result.error.code == "wait-gate-restricted", i

    # Nothing contends for `bash` any more, so the gate lifts.
    await fire_org_agent(
        session, peer.name,
        ExitReview(category="mission_complete", delivered="d", shortfalls="-", score=7),
        source="ceo",
    )
    after = await _call(session, "bash", {}, "after")
    assert after.tool_result.ok, after.tool_result.error
    assert session.harness_rejection_events == [] or all(
        e.code == "wait-gate-restricted" for e in session.harness_rejection_events
    )


@pytest.mark.asyncio
async def test_digest_ignores_reward_but_not_finished(tmp_path):
    """`reward` is per-call bookkeeping; `finished` is not.

    Some envs drift `reward` by ~1e-7 on every call, so nearly every payload
    carries a distinct value. Digesting it would hand every repeat
    a fresh key and switch §A.8.6 off for exactly the polling loops it exists
    to stop. `finished` is bounded and really does end the episode, so it
    stays in.
    """
    from agentorg.loop import _result_content_digest
    from agentorg.tools.context import ok_result

    def payload(**kw):
        base = {"blocks": [{"type": "text", "text": "Bet placed."}],
                "metadata": {"ts": 1}, "reward": 0.0, "finished": False}
        base.update(kw)
        return ok_result(base)

    base = payload()
    assert _result_content_digest(base) == _result_content_digest(
        payload(reward=0.8342397163609812, metadata={"ts": 9})
    )
    assert _result_content_digest(base) != _result_content_digest(payload(finished=True))
    assert _result_content_digest(base) != _result_content_digest(
        payload(blocks=[{"type": "text", "text": "Bet rejected."}])
    )


@pytest.mark.asyncio
async def test_digest_separates_distinct_image_blocks(tmp_path):
    """An image block carries no `text`. Digesting only the joined text made
    every screenshot identical, so a screenshotting env tool built a run on
    genuinely different frames and — since a refusal extends the run — was
    lost for the rest of the session."""
    from agentorg.loop import _result_content_digest
    from agentorg.tools.context import ok_result

    def shot(n):
        return ok_result({
            "blocks": [{"type": "image", "mime_type": "image/png", "data_bytes": n}],
            "metadata": {"step": n}, "reward": 0.0, "finished": False,
        })

    assert _result_content_digest(shot(48213)) != _result_content_digest(shot(91002))
    assert _result_content_digest(shot(48213)) == _result_content_digest(shot(48213))


@pytest.mark.asyncio
async def test_rejection_clears_a_run_on_a_different_call(tmp_path):
    """A rejection clears a run counting something else, without keying itself.

    Both halves matter. If a rejection did not clear, a CEO at threshold on
    one tool would stay refused across arbitrarily many attempts at other
    tools, each refusal extending the run further. If it keyed instead of
    clearing, fifteen gated attempts would build a run on the GATED tool and
    lose it the moment the gate lifted.
    """
    from agentorg.loop import _canonical_args
    from agentorg.org import fire_org_agent, spawn_org_agent
    from agentorg.types import ExitReview, OrgAgentSpec

    session = await _session(tmp_path, budget=4)
    session.tools.register(_ConstTool(name="notes", scope="env"))
    session.tools.register(_ConstTool(name="bash", scope="env"))
    ceo = session.ceo_name
    notes_key = _canonical_args({})

    # Build a real run while there is no peer to gate on.
    for i in range(3):
        assert (await _call(session, "notes", {}, f"n{i}")).tool_result.ok
    assert session.repeat_run(ceo, "notes", notes_key) == 3

    peer = await spawn_org_agent(session, OrgAgentSpec(role="Eng", description="x"))

    # A gated call to a DIFFERENT tool clears the run...
    assert (await _call(session, "bash", {}, "g0")).tool_result.error.code == "wait-gate-restricted"
    assert session.repeat_run(ceo, "notes", notes_key) == 0

    # ...but never builds a run of its own, however often it repeats.
    for i in range(20):
        assert (await _call(session, "bash", {}, f"g{i + 1}")).tool_result.error.code == "wait-gate-restricted"
    assert session.repeat_run(ceo, "bash", _canonical_args({})) == 0

    # So the first call after the gate lifts still executes.
    await fire_org_agent(
        session, peer.name,
        ExitReview(category="mission_complete", delivered="d", shortfalls="-", score=7),
        source="ceo",
    )
    assert (await _call(session, "bash", {}, "after")).tool_result.ok
