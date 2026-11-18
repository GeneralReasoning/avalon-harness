"""`_enforce_output_cap` must TRUNCATE an oversized env result, not blank it.

The cap is on the serialized payload, so trimming must account for JSON escaping
(newlines, quotes, backslashes, control characters). Text keeps its head and tail with
a marker giving the omitted line range. The stub is only for payloads whose non-text
parts alone exceed the cap.
"""

import copy
import json
import random
import re

from agentorg.openreward_env import _enforce_output_cap
from agentorg.tools.context import MAX_TOOL_OUTPUT_BYTES

STUB = "[env tool output exceeded cap before content rendering]"


def _size(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def _payload(text: str, metadata=None) -> dict:
    return {"blocks": [{"type": "text", "text": text}], "metadata": metadata,
            "reward": None, "finished": False}


def _text(out: dict) -> str:
    return "".join(b.get("text", "") for b in out["blocks"] if b.get("type") == "text")


MARKER = re.compile(
    r"\[\u2026 (?:line (\d+)|lines (\d+)\u2013(\d+)) omitted \((\d+) bytes\); "
    r"output was (\d+) lines / (\d+) bytes total \u2026\]"
)


def _head_tail(out: dict, original: str) -> tuple[str, str]:
    """Split the capped text at its marker and check the marker against `original`
    independently: head is a prefix, tail a suffix, and the stated line range, byte count
    and totals are those of the omitted middle. Returns (head, tail)."""
    text = _text(out)
    [m] = list(MARKER.finditer(text))
    raw = original.encode("utf-8")
    first = int(m.group(1) or m.group(2))
    last = int(m.group(1) or m.group(3))
    dropped, n_lines, n_bytes = int(m.group(4)), int(m.group(5)), int(m.group(6))
    after = text[m.end():]
    tail = after[1:] if after else after
    assert not after or after.startswith("\n")
    before = text[:m.start()]
    tail_len = len(tail.encode("utf-8"))
    head = next(h for h in (before, before[:-1])
                if len(h.encode("utf-8")) + dropped + tail_len == len(raw))
    assert original.startswith(head) and original.endswith(tail)
    head_end, tail_start = len(head.encode("utf-8")), len(raw) - tail_len
    assert dropped == out["dropped_bytes"] == tail_start - head_end > 0
    assert first == raw.count(b"\n", 0, head_end) + 1
    assert last == raw.count(b"\n", 0, tail_start - 1) + 1
    assert n_bytes == len(raw)
    assert n_lines == raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    return head, tail


def test_small_payload_is_untouched():
    p = _payload("short page")
    assert _enforce_output_cap(p) is p


def test_newline_heavy_page_is_truncated_not_blanked():
    """Plain prose with a newline every ~70 chars."""
    page = ("word " * 14 + "\n") * 800
    out = _enforce_output_cap(_payload(page))
    assert _text(out) != STUB
    head, tail = _head_tail(out, page)
    assert len(head) + len(tail) > 20_000
    assert out["truncated"] is True and out["dropped_bytes"] > 0
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_escape_heavy_latex_page_keeps_its_head_and_tail():
    """Quotes and backslashes on every line: roughly twice the escape overhead."""
    page = 'We study "noisy" labels in \\mathcal{X} and \\mathbb{R}^n.\n' * 1200
    out = _enforce_output_cap(_payload(page))
    assert _text(out) != STUB
    _head_tail(out, page)
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_control_characters_escape_to_six_bytes_and_still_fit():
    page = "a\x01b\x02" * 20_000
    out = _enforce_output_cap(_payload(page))
    assert _text(out) != STUB and _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_multibyte_text_stays_valid_utf8_and_fits():
    page = "日本語のテキスト\n" * 5_000
    out = _enforce_output_cap(_payload(page))
    _head_tail(out, page)
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_result_uses_most_of_the_budget():
    """Truncation should not over-shrink: the kept payload is close to the cap."""
    page = ("word " * 14 + "\n") * 800
    out = _enforce_output_cap(_payload(page))
    assert _size(out) > MAX_TOOL_OUTPUT_BYTES - 1024


def test_stub_still_fires_when_block_structure_alone_exceeds_the_cap():
    """Non-text parts too large for any text budget still collapse to the stub."""
    p = {"blocks": [{"type": "image", "mime_type": "image/png", "data_bytes": i} for i in range(5_000)],
         "metadata": None, "reward": None, "finished": False}
    out = _enforce_output_cap(p)
    assert _text(out) == STUB
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_multiple_blocks_keep_order_and_fit():
    p = {"blocks": [{"type": "text", "text": "HEADER\n" * 100},
                    {"type": "image", "mime_type": "image/png", "bytes": 1234},
                    {"type": "text", "text": ("body line\n") * 6_000}],
         "metadata": {"url": "https://example.org"}, "reward": None, "finished": False}
    out = _enforce_output_cap(p)
    assert out["blocks"][0]["text"].startswith("HEADER")
    assert any(b.get("type") == "image" for b in out["blocks"])
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def _bash_payload(output: str, exit_code: int = 0) -> dict:
    """The common env shape: the output as text, mirrored whole into metadata."""
    return {"blocks": [{"type": "text", "text": f"{output}\n\n(exit {exit_code})"}],
            "metadata": {"output": output, "exit_code": exit_code, "truncated": False},
            "reward": 0.0, "finished": False}


def test_metadata_mirror_of_long_output_keeps_the_text():
    output = "".join(f"line {i}: some build log output\n" for i in range(4_000))
    p = _bash_payload(output, exit_code=2)
    out = _enforce_output_cap(p)
    assert _text(out) != STUB
    head, tail = _head_tail(out, p["blocks"][0]["text"])
    assert len(head) + len(tail) > 25_000 and tail.endswith("(exit 2)")
    assert out["metadata"]["exit_code"] == 2 and out["metadata"]["truncated"] is False
    assert out["metadata"]["output"].startswith("[omitted: same as the tool output text")
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_mirror_that_fits_whole_once_deduplicated_is_not_cut():
    """20 KB of output, mirrored: over the cap together, under it once the mirror goes."""
    output = "x" * 20_000
    out = _enforce_output_cap(_bash_payload(output))
    assert _text(out) == output + "\n\n(exit 0)"
    assert out["dropped_bytes"] == 0 and "the text is complete" in out["note"]
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_large_metadata_value_is_cut_and_short_text_kept_whole():
    """A graded submit that echoes a large submission into metadata."""
    p = {"blocks": [{"type": "text", "text": "Graded: 3/5 checks passed."}],
         "metadata": {"submission": '{"rows": [1, 2, 3]}\n' * 6_000, "score": 0.6,
                      "passed": 3, "total": 5, "status": "graded"},
         "reward": 0.6, "finished": True}
    out = _enforce_output_cap(p)
    assert _text(out) == "Graded: 3/5 checks passed."
    meta = out["metadata"]
    assert (meta["score"], meta["passed"], meta["total"], meta["status"]) == (0.6, 3, 5, "graded")
    assert p["metadata"]["submission"].startswith(meta["submission"].split("...[truncated")[0])
    assert f"of {len(p['metadata']['submission'])} chars kept]" in meta["submission"]
    assert len(meta["submission"]) > 8_000
    assert out["reward"] == 0.6 and out["finished"] is True
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_nested_metadata_strings_are_cut_and_short_fields_kept():
    p = _payload("listing", metadata={"files": [{"path": f"src/m{i}.py", "content": "def f():\n    pass\n" * 600}
                                               for i in range(8)]})
    out = _enforce_output_cap(p)
    assert _text(out) == "listing"
    assert [f["path"] for f in out["metadata"]["files"]] == [f"src/m{i}.py" for i in range(8)]
    assert all("[truncated: first" in f["content"] for f in out["metadata"]["files"])
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_many_small_metadata_values_fall_back_to_top_level_scalars():
    p = _payload("ok " * 5_000, metadata={"rows": list(range(50_000)), "exit_code": 1, "name": "run"})
    out = _enforce_output_cap(p)
    assert _text(out).startswith("ok ok") and len(_text(out)) > 10_000
    assert out["metadata"]["exit_code"] == 1 and out["metadata"]["name"] == "run"
    assert out["metadata"]["rows"].startswith("[omitted: list")
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_metadata_with_too_many_keys_becomes_a_marker_and_text_is_kept():
    p = _payload("result text", metadata={f"k{i}": i for i in range(20_000)})
    out = _enforce_output_cap(p)
    assert _text(out) == "result text"
    assert out["metadata"] == "[omitted: metadata too large for the output cap]"
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_small_metadata_is_untouched_when_text_is_cut():
    meta = {"url": "https://example.org/page", "status": 200, "exit_code": 0, "ok": True}
    out = _enforce_output_cap(_payload("body line\n" * 6_000, metadata=dict(meta)))
    assert out["metadata"] == meta
    assert out["truncated"] is True and _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_non_json_metadata_value_is_rendered_as_its_str_and_cut():
    out = _enforce_output_cap(_payload("done", metadata={"raw": b"\x00\xff" * 40_000, "code": 0}))
    assert _text(out) == "done" and out["metadata"]["code"] == 0
    assert out["metadata"]["raw"].startswith("b'\\x00\\xff")
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_input_payload_is_not_mutated():
    p = _bash_payload("y" * 100_000)
    before = copy.deepcopy(p)
    _enforce_output_cap(p)
    assert p == before


def test_hard_cap_and_determinism_over_mixed_payloads():
    """Escapes, control characters, multibyte text and nested metadata in both parts."""
    rng = random.Random(0)
    alphabet = ['a', 'b', ' ', '\n', '"', '\\', '\x01', '日', '€', '{', '}']
    one_line = [c for c in alphabet if c != '\n']
    def rand_text(n: int) -> str:
        chars = rng.choice([alphabet, one_line, alphabet + ['x'] * 200])
        return "".join(rng.choice(chars) for _ in range(n))
    for _ in range(60):
        text = rand_text(rng.choice([10, 5_000, 40_000, 120_000]))
        meta = {"output": text if rng.random() < 0.5 else rand_text(rng.choice([10, 30_000, 90_000])),
                "nested": {"items": [rand_text(rng.choice([5, 3_000])) for _ in range(rng.randint(0, 20))]},
                "exit_code": rng.randint(0, 3)}
        p = _payload(text, metadata=meta)
        out = _enforce_output_cap(p)
        assert _size(out) <= MAX_TOOL_OUTPUT_BYTES
        assert _text(out) != STUB
        if out.get("dropped_bytes"):
            _head_tail(out, text)
        else:
            assert _text(out) == text
        assert out["metadata"]["exit_code"] == meta["exit_code"]
        assert json.dumps(_enforce_output_cap(copy.deepcopy(p))) == json.dumps(out)


def test_head_and_tail_are_kept_at_line_boundaries_with_exact_line_range():
    page = "".join(f"line {i}: test output\n" for i in range(1, 5_001)) + "FAILED 3 of 812 tests\n"
    out = _enforce_output_cap(_payload(page))
    head, tail = _head_tail(out, page)
    assert head.startswith("line 1: ") and head.endswith("\n")
    assert tail.startswith("line ") and tail.endswith("FAILED 3 of 812 tests\n")
    assert 0.8 < len(head) / len(tail) < 1.25
    first_omitted = head.count("\n") + 1
    last_omitted = 5_001 - tail.count("\n")
    assert f"lines {first_omitted}\u2013{last_omitted} omitted" in _text(out)
    assert "5002 lines" not in _text(out) and "output was 5001 lines" in _text(out)
    assert "a line range" in out["note"] and str(out["dropped_bytes"]) in out["note"]
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_single_giant_line_is_cut_mid_line_on_both_sides():
    page = "".join(chr(ord("a") + i % 26) for i in range(100_000))
    out = _enforce_output_cap(_payload(page))
    head, tail = _head_tail(out, page)
    assert "[\u2026 line 1 omitted" in _text(out)
    assert len(head) > 10_000 and len(tail) > 10_000
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_long_line_straddling_the_cut_is_cut_mid_line():
    """A line boundary that would discard over half the window is not used."""
    page = "short\n" + "z" * 60_000 + "\nend\n"
    out = _enforce_output_cap(_payload(page))
    head, tail = _head_tail(out, page)
    assert len(head) > 10_000 and head.startswith("short\nzzz")
    assert len(tail) > 10_000 and tail.endswith("zzz\nend\n")


def test_multibyte_single_line_cut_stays_valid_utf8():
    page = "日本語€" * 30_000
    out = _enforce_output_cap(_payload(page))
    head, tail = _head_tail(out, page)
    assert head and tail
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES


def test_multi_block_text_is_cut_as_one_string_in_block_order():
    a = "".join(f"a{i}\n" for i in range(6_000))
    b = "".join(f"b{i}\n" for i in range(6_000))
    c = "".join(f"c{i}\n" for i in range(6_000))
    p = {"blocks": [{"type": "text", "text": a},
                    {"type": "image", "mime_type": "image/png", "data_bytes": 99},
                    {"type": "text", "text": b},
                    {"type": "text", "text": c}],
         "metadata": None, "reward": None, "finished": False}
    out = _enforce_output_cap(p)
    head, tail = _head_tail(out, a + b + c)
    assert head.startswith("a0\n") and tail.endswith("c5999\n")
    assert out["dropped_blocks"] == 1
    types = [blk["type"] for blk in out["blocks"]]
    assert types == ["text", "image", "text"]
    assert MARKER.search(out["blocks"][0]["text"])
    assert out["blocks"][2]["text"] == tail
    assert _size(out) <= MAX_TOOL_OUTPUT_BYTES
