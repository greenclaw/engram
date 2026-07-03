"""UserPromptSubmit hook: semantic recall over the prompt, never blocking it."""
import io
import json

import pytest

from engram.cli import main
from engram.embed import model_available
from engram.store import build_index


def _run(monkeypatch, capsys, payload, mem_dir, k=3):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload) if isinstance(payload, dict) else payload))
    rc = main(["hook", "--dir", str(mem_dir), "-k", str(k)])
    return rc, capsys.readouterr().out


def test_short_prompt_is_skipped(tmp_path, monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, {"prompt": "hi"}, tmp_path / "m")
    assert rc == 0 and out == ""


def test_slash_command_is_skipped(tmp_path, monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, {"prompt": "/retro please summarize the session"}, tmp_path / "m")
    assert rc == 0 and out == ""


def test_invalid_stdin_never_blocks(tmp_path, monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, "not json{", tmp_path / "m")
    assert rc == 0 and out == ""


def test_missing_store_never_blocks(tmp_path, monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, {"prompt": "where do we deploy the application to?"}, tmp_path / "nope")
    assert rc == 0 and out == ""


@pytest.mark.skipif(not model_available(), reason="bge-m3 not in local HF cache")
def test_relevant_prompt_surfaces_note(tmp_path, monkeypatch, capsys):
    mem = tmp_path / "m"
    mem.mkdir()
    (mem / "hosting.md").write_text(
        "---\nname: hosting\ndescription: Production runs on Coolify on Hetzner, not Vercel.\ntype: project\n---\nb\n"
    )
    build_index(mem)
    rc, out = _run(monkeypatch, capsys, {"prompt": "how is the app deployed to production?"}, mem)
    assert rc == 0
    assert "hosting" in out and "Coolify" in out


@pytest.mark.skipif(not model_available(), reason="bge-m3 not in local HF cache")
def test_irrelevant_prompt_stays_silent(tmp_path, monkeypatch, capsys):
    mem = tmp_path / "m"
    mem.mkdir()
    (mem / "hosting.md").write_text(
        "---\nname: hosting\ndescription: Production runs on Coolify on Hetzner.\ntype: project\n---\nb\n"
    )
    build_index(mem)
    rc, out = _run(monkeypatch, capsys, {"prompt": "what is the capital city of France called?"}, mem)
    assert rc == 0 and out == ""  # abstention floor keeps noise out of context
