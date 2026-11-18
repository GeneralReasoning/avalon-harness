"""An empty ViewChannelMessages on a `full` channel tells the caller not to poll.

An agent subscribed at `full` to a channel can poll it repeatedly with
ViewChannelMessages, getting empty results each time, because nothing in the
empty response says the channel is already being delivered to it. The note is only added when it is true: an empty window AND a `full`
subscription. At `light` the caller gets a pointer, not the body, so a view
can still be the right move and the result stays as it was.
"""

from __future__ import annotations

from agentorg.harness import Harness
from agentorg.inference.base import CompletedChunk, InferenceProvider, TokenUsage
from agentorg.session import SessionConfig
from agentorg.tools.comms_tools import ViewChannelMessagesTool
from agentorg.tools.context import ToolContext
from agentorg.tools.org_tools import PostToChannelTool


class _Stub(InferenceProvider):
    async def stream(self, request):
        yield CompletedChunk(usage=TokenUsage(), stop_reason="stop")

    def model_ids(self):
        return ["stub"]


async def _session(tmp_path):
    config = SessionConfig(
        workspace_path=str(tmp_path),
        model="stub",
        budget=0,
        headless=True,
        permission_policy="bypass",
    )
    session = await Harness(inference=_Stub()).start_session(config=config)
    await session.channels.create_channels(
        [{"name": "general"}],
        live_agent_names=session.live_agent_names(),
        external_labels=set(),
    )
    return session


def _ctx(session, args, call_id="c"):
    return ToolContext(
        agent=session.agent(session.ceo_name), session=session, args=args, call_id=call_id
    )


async def _view(session):
    r = await ViewChannelMessagesTool().execute(_ctx(session, {"channel": "general"}))
    assert r.ok, r
    return r.output


async def test_empty_view_at_full_carries_the_note(tmp_path):
    session = await _session(tmp_path)
    await session.channels.set_subscription("general", session.ceo_name, "full")

    out = await _view(session)
    assert out["messages"] == []
    assert "no need to check this channel again" in out["note"]
    assert "'full'" in out["note"]


async def test_empty_view_at_light_is_unchanged(tmp_path):
    session = await _session(tmp_path)
    assert session.channels.effective_subscription(
        session.channels.resolve("general"), session.ceo_name
    ) == "light"

    out = await _view(session)
    assert out["messages"] == []
    assert "note" not in out


async def test_note_only_when_nothing_came_back(tmp_path):
    session = await _session(tmp_path)
    await session.channels.set_subscription("general", session.ceo_name, "full")
    r = await PostToChannelTool().execute(
        _ctx(session, {"channel": "general", "content": "hello"}, "p")
    )
    assert r.ok

    first = await _view(session)
    assert len(first["messages"]) == 1
    assert "note" not in first

    # Caught up now: the resumed view is empty and the note appears.
    second = await _view(session)
    assert second["messages"] == []
    assert second["resumed_from_cursor"] is True
    assert "note" in second


async def test_exhausted_history_page_is_not_a_poll(tmp_path):
    """`before_idx` past the start returns nothing because history ran out,
    not because the caller is caught up, so telling it never to look at the
    channel again is a non sequitur. Direction alone decides: `before_idx`
    looks backwards, everything else forwards.

    The counts follow the same direction. An exhausted backward page has
    nothing older and the whole channel ahead of it."""
    session = await _session(tmp_path)
    await session.channels.set_subscription("general", session.ceo_name, "full")
    for i in range(3):
        r = await PostToChannelTool().execute(
            _ctx(session, {"channel": "general", "content": f"msg {i}"}, f"p{i}")
        )
        assert r.ok

    r = await ViewChannelMessagesTool().execute(
        _ctx(session, {"channel": "general", "before_idx": 0}, "v1")
    )
    assert r.ok, r
    assert r.output["messages"] == []
    assert r.output["older_not_shown"] == 0, r.output
    assert r.output["newer_not_shown"] == 3, r.output
    assert "note" not in r.output, r.output.get("note")

    # A real poll on the same channel still gets it: the first drains the
    # three messages, the second is genuinely caught up.
    assert len((await _view(session))["messages"]) == 3
    assert "note" in (await _view(session))


async def test_explicit_forward_cursor_is_still_a_poll(tmp_path):
    """`since_idx` asks "anything at or after N?". Empty means caught up —
    the same situation as the resumed window, so the note belongs here too.
    Only `before_idx` looks backwards."""
    session = await _session(tmp_path)
    await session.channels.set_subscription("general", session.ceo_name, "full")
    for i in range(3):
        r = await PostToChannelTool().execute(
            _ctx(session, {"channel": "general", "content": f"m{i}"}, f"p{i}")
        )
        assert r.ok

    r = await ViewChannelMessagesTool().execute(
        _ctx(session, {"channel": "general", "since_idx": 3}, "s1")
    )
    assert r.ok, r
    assert r.output["messages"] == []
    assert "note" in r.output, r.output

    # A bounded window is a lookup, not a poll.
    r = await ViewChannelMessagesTool().execute(
        _ctx(session, {"channel": "general", "since_idx": 1, "before_idx": 1}, "s2")
    )
    assert r.output["messages"] == []
    assert "note" not in r.output, r.output.get("note")
