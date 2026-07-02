"""Curator core: apply a change-set as a gated, committed edit. No LLM (change-sets fed directly)."""
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from engram.core import _split_frontmatter
from engram.curate import CurateError, apply, load_changeset

YES = lambda diff: True
NO = lambda diff: False


def _note(mem: Path, name, front_extra="", body="body"):
    mem.mkdir(parents=True, exist_ok=True)
    (mem / f"{name}.md").write_text(f"---\nname: {name}\ndescription: old desc\ntype: project\n{front_extra}---\n{body}\n")


def _meta(path: Path):
    return _split_frontmatter(path.read_text(), path)[0]


def test_add_creates_note(tmp_path):
    mem = tmp_path / "m"
    mem.mkdir()
    cs = {"changes": [{"op": "ADD", "name": "pnpm", "type": "feedback", "description": "use pnpm", "body": "always pnpm"}]}
    assert apply(mem, cs, confirm=YES, now=date(2026, 7, 2)) is True
    meta = _meta(mem / "pnpm.md")
    assert meta["description"] == "use pnpm" and meta["type"] == "feedback"
    assert meta["updated"] == date(2026, 7, 2)
    assert "always pnpm" in (mem / "pnpm.md").read_text()


def test_update_preserves_untouched_frontmatter(tmp_path):
    mem = tmp_path / "m"
    _note(mem, "hosting", front_extra="originSessionId: abc-123\n")
    apply(mem, {"changes": [{"op": "UPDATE", "target": "hosting", "description": "now on Fly.io"}]},
          confirm=YES, now=date(2026, 7, 2))
    meta = _meta(mem / "hosting.md")
    assert meta["description"] == "now on Fly.io"
    assert meta["originSessionId"] == "abc-123"  # don't clobber untouched frontmatter (guardrail)
    assert meta["updated"] == date(2026, 7, 2)


def test_invalidate_marks_and_keeps_note(tmp_path):
    mem = tmp_path / "m"
    _note(mem, "old", body="the old truth")
    apply(mem, {"changes": [{"op": "INVALIDATE", "target": "old", "invalidated_by": "new"}]}, confirm=YES)
    assert (mem / "old.md").exists()  # invalidate-don't-delete
    meta = _meta(mem / "old.md")
    assert meta["invalidated_by"] == "new"
    assert "the old truth" in (mem / "old.md").read_text()


def test_noop_and_gate_reject_write_nothing(tmp_path):
    mem = tmp_path / "m"
    mem.mkdir()
    assert apply(mem, {"changes": [{"op": "NOOP"}]}, confirm=YES) is False
    _note(mem, "x")
    before = (mem / "x.md").read_text()
    assert apply(mem, {"changes": [{"op": "UPDATE", "target": "x", "description": "changed"}]}, confirm=NO) is False
    assert (mem / "x.md").read_text() == before  # gate said no → untouched


def test_add_existing_and_update_missing_fail_loud(tmp_path):
    mem = tmp_path / "m"
    _note(mem, "dup")
    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "ADD", "name": "dup", "description": "x"}]}, confirm=YES)
    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "UPDATE", "target": "ghost", "description": "x"}]}, confirm=YES)


def test_gate_receives_a_diff(tmp_path):
    mem = tmp_path / "m"
    mem.mkdir()
    seen = {}
    apply(mem, {"changes": [{"op": "ADD", "name": "a", "description": "hello world", "body": "b"}]},
          confirm=lambda d: seen.setdefault("d", d) and True, now=date(2026, 7, 2))
    assert "hello world" in seen["d"]


def test_apply_commits_in_git_repo(tmp_path):
    mem = tmp_path / "m"
    mem.mkdir()
    subprocess.run(["git", "init", "-q", str(mem)], check=True)
    subprocess.run(["git", "-C", str(mem), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(mem), "config", "user.name", "t"], check=True)
    apply(mem, {"changes": [{"op": "ADD", "name": "a", "description": "d", "body": "b"}]}, confirm=YES)
    log = subprocess.run(["git", "-C", str(mem), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "engram" in log and (mem / "a.md").exists()


def test_path_escape_fails_loud(tmp_path):
    # change-sets are LLM output = untrusted input; names/targets must stay inside the memory dir
    mem = tmp_path / "m"
    mem.mkdir()
    (tmp_path / "outside").mkdir()
    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "ADD", "name": "../outside/evil", "description": "x"}]}, confirm=YES)
    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "UPDATE", "target": "../escape", "description": "x"}]}, confirm=YES)
    assert not (tmp_path / "outside" / "evil.md").exists()


def test_add_into_subdir_is_allowed(tmp_path):
    # legit: notes live in subdirs (learnings/) — recursive discovery supports them
    mem = tmp_path / "m"
    mem.mkdir()
    assert apply(mem, {"changes": [{"op": "ADD", "name": "learnings/tip", "description": "d", "body": "b"}]}, confirm=YES)
    assert (mem / "learnings" / "tip.md").exists()


def test_duplicate_target_fails_loud(tmp_path):
    # two ops on one note are both computed against the original text — last write would silently win
    mem = tmp_path / "m"
    _note(mem, "fact")
    with pytest.raises(CurateError):
        apply(mem, {"changes": [
            {"op": "UPDATE", "target": "fact", "description": "v2"},
            {"op": "INVALIDATE", "target": "fact", "invalidated_by": "newer"},
        ]}, confirm=YES)


def test_missing_required_field_raises_curate_error(tmp_path):
    mem = tmp_path / "m"
    mem.mkdir()
    with pytest.raises(CurateError) as ei:
        apply(mem, {"changes": [{"op": "ADD", "description": "no name"}]}, confirm=YES)
    assert "name" in str(ei.value)
    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "INVALIDATE", "target": "x"}]}, confirm=YES)  # no invalidated_by


def test_commit_does_not_sweep_prestaged_files(tmp_path):
    repo = tmp_path / "repo"
    mem = repo / "memory"
    mem.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "unrelated.txt").write_text("wip")
    subprocess.run(["git", "-C", str(repo), "add", "unrelated.txt"], check=True)  # user had this staged

    apply(mem, {"changes": [{"op": "ADD", "name": "note1", "description": "d", "body": "b"}]}, confirm=YES)

    shown = subprocess.run(["git", "-C", str(repo), "show", "--name-only", "--format=", "HEAD"],
                           capture_output=True, text=True).stdout
    assert "note1.md" in shown
    assert "unrelated.txt" not in shown  # engram must not sweep the user's staged work
    status = subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True).stdout
    assert "unrelated.txt" in status  # still staged, untouched


def test_load_changeset(tmp_path):
    f = tmp_path / "cs.json"
    f.write_text(json.dumps({"changes": [{"op": "NOOP"}]}))
    assert load_changeset(str(f))["changes"][0]["op"] == "NOOP"
    bad = tmp_path / "bad.json"
    bad.write_text("[]")  # not an object with a 'changes' list
    with pytest.raises(CurateError):
        load_changeset(str(bad))
