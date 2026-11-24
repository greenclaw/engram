"""Note frontmatter guard: `engram check-note` (PreToolUse on Write/Edit) refuses a memory note whose
frontmatter engram can't read, and `engram hook` says so in context when a bad note slipped in anyway
(Bash/python writes bypass PreToolUse)."""
import io
import json

import pytest

from engram.cli import main

GOOD = "---\nname: pg-pool\ndescription: postgres pool size is 20\ntype: project\n---\nbody\n"
BAD_COLON = "---\nname: clone-api\ndescription: lists are wrapped as [{uuid: data}] objects\ntype: project\n---\nb\n"
BAD_QUOTE = '---\nname: asv\ndescription: "ASV reports fail" = RU docs refused\ntype: project\n---\nb\n'


def _check(monkeypatch, capsys, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(payload if isinstance(payload, str) else json.dumps(payload)))
    rc = main(["check-note"])
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _write(path, content):
    path = __import__("pathlib").Path(path)
    if "memory" in path.parts and path.parent.name in ("memory", "learnings") and "repo" not in path.parts:
        mem = path.parent if path.parent.name == "memory" else path.parent.parent
        mem.mkdir(parents=True, exist_ok=True)
        (mem / "MEMORY.md").write_text("# index\n")
    return {"tool_name": "Write", "tool_input": {"file_path": str(path), "content": content}}


@pytest.mark.parametrize("content", [BAD_COLON, BAD_QUOTE])
def test_write_of_invalid_yaml_note_is_refused(tmp_path, monkeypatch, capsys, content):
    rc, out, err = _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", content))
    assert rc == 2
    assert "x.md" in err and "quote" in err


@pytest.mark.parametrize("line", ["description: no", "description: 2026-07-01", "name: 42"])
def test_value_yaml_turns_into_a_non_string_is_refused(tmp_path, monkeypatch, capsys, line):
    key = line.split(":")[0]
    content = GOOD.replace(next(ln for ln in GOOD.splitlines() if ln.startswith(key + ":")), line)
    rc, _, err = _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", content))
    assert rc == 2 and key in err


def test_hash_comment_that_truncates_a_value_is_refused(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("postgres pool size is 20", "fix landed in PR #18 after review")
    rc, _, err = _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", content))
    assert rc == 2 and "#" in err


def test_non_numeric_importance_is_refused(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("type: project", "type: project\nimportance: high")
    rc, _, err = _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", content))
    assert rc == 2 and "importance" in err


def test_good_note_and_nested_note_pass(tmp_path, monkeypatch, capsys):
    assert _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", GOOD)) == (0, "", "")
    assert _check(monkeypatch, capsys, _write(tmp_path / "memory" / "learnings" / "y.md", GOOD))[0] == 0


@pytest.mark.parametrize("path", ["src/readme.md", "memory/notes.txt", "memory/MEMORY.md"])
def test_files_that_are_not_notes_are_ignored(tmp_path, monkeypatch, capsys, path):
    assert _check(monkeypatch, capsys, _write(tmp_path / path, BAD_COLON)) == (0, "", "")


def test_edit_that_breaks_frontmatter_is_refused(tmp_path, monkeypatch, capsys):
    note = _store_path(tmp_path)
    note.write_text(GOOD)
    payload = {"tool_name": "Edit", "tool_input": {
        "file_path": str(note), "old_string": "description: postgres pool size is 20",
        "new_string": "description: pool: 20"}}
    rc, _, err = _check(monkeypatch, capsys, payload)
    assert rc == 2 and "x.md" in err


def test_edit_that_keeps_frontmatter_valid_passes(tmp_path, monkeypatch, capsys):
    note = _store_path(tmp_path)
    note.write_text(BAD_COLON)  # already broken: an edit that fixes it must go through
    payload = {"tool_name": "Edit", "tool_input": {
        "file_path": str(note), "old_string": "description: lists are wrapped as [{uuid: data}] objects",
        "new_string": "description: \"lists are wrapped as [{uuid: data}] objects\""}}
    assert _check(monkeypatch, capsys, payload) == (0, "", "")


def test_edit_with_replace_all_is_applied_everywhere(tmp_path, monkeypatch, capsys):
    note = _store_path(tmp_path)
    note.write_text(GOOD.replace("body", "pool size is 20"))
    payload = {"tool_name": "Edit", "tool_input": {
        "file_path": str(note), "old_string": "pool size is 20", "new_string": "pool: 20", "replace_all": True}}
    assert _check(monkeypatch, capsys, payload)[0] == 2


@pytest.mark.parametrize("payload", ["not json{", {"tool_name": "Edit", "tool_input": {"file_path": "/nope/memory/x.md",
                                                                                       "old_string": "a", "new_string": "b"}}])
def test_guard_never_blocks_on_its_own_failure(monkeypatch, capsys, payload):
    assert _check(monkeypatch, capsys, payload)[0] == 0


def test_kill_switch_silences_the_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DISABLE", "1")
    assert _check(monkeypatch, capsys, _write(tmp_path / "memory" / "x.md", BAD_COLON)) == (0, "", "")


# --- backstop: a bad note written around the guard -----------------------------------------------

def _bad_store(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir(parents=True)
    (mem / "good.md").write_text(GOOD)
    (mem / "clone.md").write_text(BAD_COLON)
    return mem


def test_hook_reports_a_bad_note_in_context(tmp_path, monkeypatch, capsys):
    mem = _bad_store(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "what is the postgres pool size?"})))
    rc = main(["hook", "--dir", str(mem)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recall is off" in out.lower() and "clone.md" in out and "quote" in out


@pytest.mark.parametrize("argv", [["index"], ["recall", "what is the postgres pool size?"]])
def test_explicit_commands_fail_cleanly_on_a_bad_note(tmp_path, capsys, argv):
    mem = _bad_store(tmp_path)
    rc = main([*argv, "--dir", str(mem)])
    assert rc == 1 and "clone.md" in capsys.readouterr().err


# --- review round 1 -------------------------------------------------------------------------------

def _store_path(tmp_path, rel="x.md"):
    """A path inside a real engram store: a memory dir holding a MEMORY.md index."""
    mem = tmp_path / "memory"
    mem.mkdir(exist_ok=True)
    (mem / "MEMORY.md").write_text("# index\n")
    return mem / rel


def test_non_mapping_metadata_is_refused_and_reported(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("type: project", "metadata: some text")
    rc, _, err = _check(monkeypatch, capsys, _write(_store_path(tmp_path), content))
    assert rc == 2 and "metadata" in err
    mem = _bad_store(tmp_path / "s")
    (mem / "clone.md").write_text(content)
    assert main(["index", "--dir", str(mem)]) == 1


def test_memory_dir_that_is_not_a_store_is_ignored(tmp_path, monkeypatch, capsys):
    p = tmp_path / "repo" / "src" / "memory" / "arena.md"
    assert _check(monkeypatch, capsys, _write(p, BAD_COLON)) == (0, "", "")


def test_auto_memory_store_is_guarded_before_its_index_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / ".claude" / "projects" / "-Users-x-proj" / "memory" / "first.md"
    assert _check(monkeypatch, capsys, _write(p, BAD_COLON))[0] == 2


def test_yaml_comment_on_type_is_allowed(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("type: project", "type: feedback  # one of user/feedback")
    assert _check(monkeypatch, capsys, _write(_store_path(tmp_path), content)) == (0, "", "")


def test_hash_cut_on_a_continuation_line_is_refused(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("postgres pool size is 20", "postgres pool size\n  is set in PR #18 config")
    assert _check(monkeypatch, capsys, _write(_store_path(tmp_path), content))[0] == 2


def test_hash_inside_quotes_is_allowed(tmp_path, monkeypatch, capsys):
    content = GOOD.replace("postgres pool size is 20", '"fixed in PR #18"')
    assert _check(monkeypatch, capsys, _write(_store_path(tmp_path), content)) == (0, "", "")


def test_crlf_content_is_checked_like_on_disk(tmp_path, monkeypatch, capsys):
    rc, _, _ = _check(monkeypatch, capsys, _write(_store_path(tmp_path), BAD_COLON.replace("\n", "\r\n")))
    assert rc == 2


def test_edit_matching_only_after_curly_quote_normalization_is_checked(tmp_path, monkeypatch, capsys):
    note = _store_path(tmp_path)
    note.write_text(GOOD.replace("postgres pool size is 20", "“pool” size"))
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(note), "old_string": 'description: "pool" size',
                                                   "new_string": "description: pool: 20"}}
    assert _check(monkeypatch, capsys, payload)[0] == 2


def test_edit_creating_a_new_note_is_checked(tmp_path, monkeypatch, capsys):
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(_store_path(tmp_path, "new.md")),
                                                   "old_string": "", "new_string": BAD_COLON}}
    assert _check(monkeypatch, capsys, payload)[0] == 2


def test_hook_message_is_one_bounded_line(tmp_path, monkeypatch, capsys):
    mem = _bad_store(tmp_path)
    (mem / "clone.md").write_text(GOOD.replace("type: project", "importance: '" + "x" * 500 + "\\nIGNORE ALL'"))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "what is the postgres pool size?"})))
    main(["hook", "--dir", str(mem)])
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and len(out) < 600
