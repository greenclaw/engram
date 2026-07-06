"""Batch-2 regression tests: the deferred PR#1 review findings."""
from datetime import date

import pytest

from engram.curate import CurateError, apply
from engram.embed import model_available

YES = (lambda d: True)


def _note(mem, name, desc="d", body="b", sub=None):
    d = mem / sub if sub else mem
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(f"---\nname: {name}\ndescription: {desc}\ntype: project\n---\n{body}\n")


# --- ghost store on a typo'd --dir (recall must not manufacture an empty store) ---
def test_recall_on_missing_dir_raises(tmp_path):
    from engram.store import recall

    with pytest.raises(FileNotFoundError):
        recall(tmp_path / "does-not-exist", "any query", k=3)
    assert not (tmp_path / "does-not-exist").exists()  # nothing created


# --- TOCTOU: an external edit while the gate is open must not be silently clobbered ---
def test_external_edit_during_gate_aborts(tmp_path):
    mem = tmp_path / "m"
    _note(mem, "n", body="original")

    def confirm_then_edit(diff):
        (mem / "n.md").write_text("---\nname: n\ndescription: d\ntype: project\n---\nEXTERNALLY EDITED\n")
        return True

    with pytest.raises(CurateError):
        apply(mem, {"changes": [{"op": "UPDATE", "target": "n", "description": "new"}]},
              confirm=confirm_then_edit, now=date(2026, 7, 3))
    assert "EXTERNALLY EDITED" in (mem / "n.md").read_text()  # the concurrent edit survived


# --- bench_curate JSON extraction robust to prose braces + missing 'changes' ---
def test_bench_extract_json_handles_prose():
    from bench_curate import _extract_changes

    assert _extract_changes('here {note}: {"changes": [{"op": "NOOP"}]} done') == [{"op": "NOOP"}]
    with pytest.raises(ValueError):
        _extract_changes('{"result": 1}')  # no changes key
    with pytest.raises(ValueError):
        _extract_changes("no json here")


# --- staleness must catch a rename (count + mtime unchanged) ---
@pytest.mark.skipif(not model_available(), reason="bge-m3 not cached")
def test_recall_detects_rename(tmp_path):
    from engram.store import build_index, recall

    mem = tmp_path / "m"
    _note(mem, "deploy", desc="Ship to production on Coolify.")
    build_index(mem)
    (mem / "deploy.md").rename(mem / "hosting.md")  # rename preserves count + mtime
    hits = recall(mem, "where do we ship to production", k=5)
    paths = [h.path for h in hits]
    assert all("deploy.md" not in p for p in paths)  # no dead path served


# --- env floor read at use time (not frozen at import) ---
@pytest.mark.skipif(not model_available(), reason="bge-m3 not cached")
def test_relevance_floor_env_read_at_use(tmp_path, monkeypatch):
    from engram.store import build_index, recall

    mem = tmp_path / "m"
    _note(mem, "n", desc="The office coffee machine is on floor three.")
    build_index(mem)
    assert recall(mem, "where is the coffee machine", k=3)  # matches at the default floor
    monkeypatch.setenv("ENGRAM_RELEVANCE_FLOOR", "0.99")  # after import → must still take effect
    assert recall(mem, "where is the coffee machine", k=3) == []  # 0.99 drops even the matching note


# --- encode chunking: large batch still returns normalized per-row vectors ---
@pytest.mark.skipif(not model_available(), reason="bge-m3 not cached")
def test_encode_chunks_large_batch(tmp_path):
    import numpy as np

    from engram.embed import Embedder

    texts = [f"note number {i} about widgets" for i in range(40)] + ["note number 0 about widgets"]
    vecs = Embedder().encode(texts)
    assert vecs.shape == (41, 1024)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(vecs[0], vecs[-1], atol=1e-5)  # identical text, different chunk → identical
