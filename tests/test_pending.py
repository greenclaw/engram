"""Pending-store: the Stop hook enqueues finished sessions for later (offline, gated) curation."""
import io
import json

import pytest

from engram import pending
from engram.cli import main


@pytest.fixture
def mem(tmp_path):
    d = tmp_path / "mem"
    d.mkdir()
    return d


def test_add_and_load_roundtrip(mem):
    pending.add(mem, "s1", "/tmp/t1.jsonl")
    q = pending.load(mem)
    assert q["s1"]["transcript_path"] == "/tmp/t1.jsonl"


def test_add_dedups_by_session_id(mem):
    # Stop fires after EVERY response — repeated adds for one session collapse to the latest
    pending.add(mem, "s1", "/tmp/a.jsonl")
    pending.add(mem, "s1", "/tmp/b.jsonl")
    q = pending.load(mem)
    assert len(q) == 1 and q["s1"]["transcript_path"] == "/tmp/b.jsonl"


def test_two_sessions_queue_independently(mem):
    pending.add(mem, "s1", "/tmp/a.jsonl")
    pending.add(mem, "s2", "/tmp/b.jsonl")
    assert set(pending.load(mem)) == {"s1", "s2"}


def test_corrupt_queue_file_reads_as_empty(mem):
    (mem / ".engram").mkdir()
    (mem / ".engram" / "pending.json").write_text("{not json")
    assert pending.load(mem) == {}


def test_clear_all_and_by_id(mem):
    pending.add(mem, "s1", "/a")
    pending.add(mem, "s2", "/b")
    assert pending.clear(mem, ["s1"]) == 1
    assert set(pending.load(mem)) == {"s2"}
    assert pending.clear(mem) == 1
    assert pending.load(mem) == {}


# --- CLI (hook semantics: never block) ---------------------------------------

def _stdin(monkeypatch, text):
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_cli_add_enqueues_from_hook_json(mem, monkeypatch):
    _stdin(monkeypatch, json.dumps({"session_id": "abc", "transcript_path": "/tmp/t.jsonl"}))
    assert main(["pending", "add", "--dir", str(mem)]) == 0
    assert pending.load(mem)["abc"]["transcript_path"] == "/tmp/t.jsonl"


def test_cli_add_malformed_stdin_exits_zero(mem, monkeypatch):
    _stdin(monkeypatch, "not json at all")
    assert main(["pending", "add", "--dir", str(mem)]) == 0
    assert pending.load(mem) == {}


def test_cli_add_missing_dir_exits_zero_no_ghost(tmp_path, monkeypatch):
    _stdin(monkeypatch, json.dumps({"session_id": "abc", "transcript_path": "/t"}))
    assert main(["pending", "add", "--dir", str(tmp_path / "nope")]) == 0
    assert not (tmp_path / "nope").exists()


def test_cli_list_and_clear(mem, capsys):
    pending.add(mem, "s1", "/a")
    assert main(["pending", "list", "--dir", str(mem)]) == 0
    assert json.loads(capsys.readouterr().out)["s1"]["transcript_path"] == "/a"
    assert main(["pending", "clear", "--dir", str(mem)]) == 0
    assert pending.load(mem) == {}
