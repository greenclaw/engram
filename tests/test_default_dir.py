"""--dir defaulting: derive the project's Claude auto-memory store from cwd (global-hook mode).
Worktree sessions map to the MAIN project's slug; no store → ambient commands stay silent,
explicit commands fail loud."""
import io
import json

from engram.cli import default_memory_dir, main


def _store(home, project_path):
    import re

    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project_path))
    d = home / ".claude" / "projects" / slug / "memory"
    d.mkdir(parents=True)
    return d


def test_derives_store_from_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    proj = tmp_path / "projects" / "my.app_v2"
    d = _store(tmp_path, proj)
    assert default_memory_dir(proj) == d


def test_worktree_maps_to_main_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    proj = tmp_path / "projects" / "engram"
    d = _store(tmp_path, proj)
    assert default_memory_dir(proj / ".worktrees" / "feature-x" / "sub") == d


def test_no_store_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_memory_dir(tmp_path / "elsewhere") is None


def test_ambient_commands_silent_without_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "a long enough prompt here"})))
    assert main(["hook"]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": "s", "transcript_path": "/t"})))
    assert main(["pending", "add"]) == 0
    assert capsys.readouterr().out == ""


def test_explicit_command_fails_loud_without_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["recall", "anything"]) == 2
    assert "no --dir" in capsys.readouterr().err


def test_explicit_command_uses_derived_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    proj = tmp_path / "projects" / "p"
    proj.mkdir(parents=True)
    _store(tmp_path, proj)
    monkeypatch.chdir(proj)
    assert main(["pending", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == {}
