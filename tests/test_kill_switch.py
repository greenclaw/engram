"""ENGRAM_DISABLE kill switch: instantly muzzles the ambient hook entrypoints (hook, pending add)
without touching explicit CLI commands — rollback without editing settings or restarting sessions."""
import io
import json

from engram.cli import main
from engram.pending import load
from engram.store import Hit


def _fake_recall(*a, **k):
    return [Hit(name="n", description="d", type="project", path="p", score=1.0, relevance=1.0)]


def _hook(monkeypatch, capsys, mem_dir):
    monkeypatch.setattr("engram.store.recall", _fake_recall)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "a long enough prompt to recall"})))
    rc = main(["hook", "--dir", str(mem_dir)])
    return rc, capsys.readouterr().out


def test_disable_silences_hook(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DISABLE", "1")
    rc, out = _hook(monkeypatch, capsys, tmp_path)
    assert rc == 0 and out == ""


def test_disable_zero_means_enabled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DISABLE", "0")
    rc, out = _hook(monkeypatch, capsys, tmp_path)
    assert rc == 0 and "[project] n" in out


def test_disable_skips_pending_add(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DISABLE", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", "transcript_path": "/t"})))
    assert main(["pending", "add", "--dir", str(tmp_path)]) == 0
    assert load(tmp_path) == {}


def test_explicit_commands_stay_live(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DISABLE", "1")
    assert main(["pending", "list", "--dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {}
