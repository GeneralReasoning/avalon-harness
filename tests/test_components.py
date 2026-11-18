"""Unit tests for harness components (pure-function code, no network)."""

from __future__ import annotations


import pytest

from agentorg.channels import ChannelStore, light_snippet, parse_mentions
from agentorg.inbox import render_inbox
from agentorg.naming import generate_name
from agentorg.plan import PlanParseError, extract_org_block
from agentorg.storage import SessionPaths, append_jsonl, iter_jsonl
from agentorg.types import InboxItem, Mention


def test_jsonl_handles_truncated_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})
    # Corrupt: truncate last line mid-write.
    raw = path.read_bytes()
    path.write_bytes(raw + b'{"c":')
    rows = list(iter_jsonl(path))
    assert rows == [{"a": 1}, {"b": 2}]


def test_name_generator_unique():
    used = set()
    for _ in range(20):
        name = generate_name(used)
        assert name not in used
        used.add(name)


def test_parse_mentions_resolves_and_marks_stale():
    mentions = parse_mentions(
        "Hi @RedAristotle and @ghost",
        live_agents={"RedAristotle"},
        external_labels=set(),
    )
    kinds = [m.kind for m in mentions]
    assert "agent" in kinds and "stale" in kinds


def test_light_snippet_truncates():
    body = "x" * 1000
    out = light_snippet(body, limit=20)
    assert len(out) <= 20
    assert out.endswith("…")


def test_render_inbox_dm(tmp_path):
    items = [
        InboxItem(kind="dm", content="hi there", **{"from": "Alice"}),
        InboxItem(
            kind="channel_post",
            channel="general",
            author="Alice",
            content="post body",
            mentions=[Mention(kind="agent", name="Bob")],
        ),
    ]
    rendered = render_inbox(items, recipient="Bob")
    assert "DM from @Alice" in rendered
    assert "#general" in rendered
    assert "🔔 You were tagged" in rendered


def test_extract_org_block_parses_snake_case():
    md = """## Org

```agents-org
{
  "agents": [
    {"role": "Researcher", "description": "Find facts"}
  ]
}
```
"""
    block = extract_org_block(md)
    assert block is not None
    assert block.agents[0].role == "Researcher"


def test_extract_org_block_rejects_duplicate_roles():
    md = """```agents-org
{
  "agents": [
    {"role": "Foo", "description": "x"},
    {"role": "foo", "description": "y"}
  ]
}
```
"""
    with pytest.raises(PlanParseError):
        extract_org_block(md)


def test_extract_org_block_returns_none_when_absent():
    assert extract_org_block("plain markdown, no org block") is None


def test_extract_org_block_raises_on_unterminated_fence():
    md = """## Org

```agents-org
{
  "agents": [
    {"role": "Researcher", "description": "Find facts"}
  ]
}
"""
    with pytest.raises(PlanParseError, match="unterminated"):
        extract_org_block(md)


@pytest.mark.asyncio
async def test_inbox_counter_init_from_disk_on_resume(tmp_path):
    """On resume, the inbox sequence counter must be bumped past the max
    seq on disk so freshly-appended items don't sort BEFORE pre-resume
    items in `render_inbox`."""
    from agentorg.inbox import InboxStore

    # Set up two agents' inbox files with pre-existing items at seq=7.
    inbox_dir = tmp_path / "inboxes"
    inbox_dir.mkdir()

    def paths_factory(name):
        return inbox_dir / f"{name}.jsonl"

    store = InboxStore("CEO", paths_factory)
    # Manually seed disk with items at higher seq numbers.
    await store.append("CEO", InboxItem(kind="meta", content="old-1", seq=5))
    await store.append("CEO", InboxItem(kind="meta", content="old-2", seq=7))
    await store.append("Alice", InboxItem(kind="meta", content="old-3", seq=6))

    # Fresh store (simulating resume): counter starts at 0.
    fresh = InboxStore("CEO", paths_factory)
    assert fresh._counter == 0

    # After init: counter should be at least 7.
    fresh.init_counter_from_disk(["CEO", "Alice"])
    assert fresh._counter >= 7

    # The next appended item gets seq > 7.
    await fresh.append("CEO", InboxItem(kind="meta", content="new"))
    items = fresh.load("CEO")
    new_items = [i for i in items if i.content == "new"]
    assert len(new_items) == 1
    assert new_items[0].seq > 7


@pytest.mark.asyncio
async def test_channel_store_create_and_post(tmp_path):
    paths = SessionPaths(str(tmp_path), "sess1")
    paths.ensure()
    store = ChannelStore(paths)
    await store.load()
    await store.create_channels(
        [{"name": "team", "default_subscription": "full"}],
        live_agent_names={"Alice"},
        external_labels=set(),
    )
    assert store.resolve("team") is not None
    msg, counts = await store.post(
        channel="team",
        author="Bob",
        is_external=False,
        content="hi @Alice",
        attachments=[],
        live_agents={"Alice", "Bob"},
        external_labels=set(),
    )
    assert msg.author == "Bob"
    assert any(m.kind == "agent" and m.name == "Alice" for m in msg.mentions)
    assert "Alice" in counts.tagged
