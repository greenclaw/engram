"""Incremental rebuild: a stale index re-embeds only new/changed notes (a full 300-note embed takes
~40 s — longer than a UserPromptSubmit hook may run). Fake embedder — no model needed."""
import hashlib
import os

import numpy as np
import pytest

from engram import store


class RecordingEmbedder:
    """Deterministic per-text vector; records every text it was asked to embed."""

    seen: list = []

    def encode(self, texts):
        texts = list(texts)
        RecordingEmbedder.seen.extend(texts)
        return np.stack([_vec(t) for t in texts]) if texts else np.zeros((0, 8), dtype=np.float32)


def _vec(text):
    v = np.frombuffer(hashlib.sha256(text.encode()).digest()[:8], dtype=np.uint8).astype(np.float32) + 1
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def recording(monkeypatch):
    RecordingEmbedder.seen = []
    monkeypatch.setattr(store, "Embedder", RecordingEmbedder)


def _note(mem, name, body="body"):
    mem.mkdir(parents=True, exist_ok=True)
    p = mem / f"{name}.md"
    p.write_text(f"---\nname: {name}\ndescription: d-{name}\ntype: project\n---\n{body}\n")
    return p


def _touch_later(p):
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def _assert_rows_aligned(mem):
    mat, meta = store._load(mem)
    assert len(mat) == len(meta)
    for row, m in zip(mat, meta):
        from pathlib import Path

        from engram.core import parse_note

        assert np.allclose(row, _vec(store._embed_text(parse_note(Path(m["path"])))))


def test_edit_reembeds_only_the_changed_note(tmp_path):
    mem = tmp_path / "memory"
    for n in ("a", "b", "c"):
        _note(mem, n)
    store.build_index(mem)
    RecordingEmbedder.seen = []

    _note(mem, "b", body="new body")
    _touch_later(mem / "b.md")
    store._ensure_fresh(mem)

    assert RecordingEmbedder.seen == ["b. d-b\nnew body"]
    _assert_rows_aligned(mem)


def test_add_and_delete_keep_rows_aligned(tmp_path):
    mem = tmp_path / "memory"
    for n in ("a", "b", "c"):
        _note(mem, n)
    store.build_index(mem)
    RecordingEmbedder.seen = []

    (mem / "a.md").unlink()
    _note(mem, "d")
    store._ensure_fresh(mem)

    assert RecordingEmbedder.seen == ["d. d-d\nbody"]
    _assert_rows_aligned(mem)
    assert [m["name"] for m in store._load(mem)[1]] == ["b", "c", "d"]


def test_delete_only_does_not_embed(tmp_path):
    mem = tmp_path / "memory"
    for n in ("a", "b"):
        _note(mem, n)
    store.build_index(mem)
    RecordingEmbedder.seen = []

    (mem / "a.md").unlink()
    store._ensure_fresh(mem)

    assert RecordingEmbedder.seen == []
    _assert_rows_aligned(mem)


def test_explicit_index_is_a_full_rebuild(tmp_path):
    mem = tmp_path / "memory"
    for n in ("a", "b"):
        _note(mem, n)
    store.build_index(mem)
    RecordingEmbedder.seen = []

    store.build_index(mem)  # `engram index`: the escape hatch (e.g. after an embedder change)

    assert len(RecordingEmbedder.seen) == 2


def test_embedder_repo_change_invalidates_reuse(tmp_path, monkeypatch):
    mem = tmp_path / "memory"
    _note(mem, "a")
    _note(mem, "b")
    store.build_index(mem)
    RecordingEmbedder.seen = []

    monkeypatch.setenv("ENGRAM_EMBED_REPO", "other/model")
    _note(mem, "c")
    store._ensure_fresh(mem)

    assert len(RecordingEmbedder.seen) == 3  # vectors from another model are never mixed in


def test_torn_write_never_pairs_a_note_with_another_notes_vector(tmp_path, monkeypatch):
    # index.npy replaced, meta.json not (hook cancelled at its timeout between the two writes):
    # the next rebuild must not reuse rows by position from the mismatched pair.
    mem = tmp_path / "memory"
    for n in ("a", "b", "c"):
        _note(mem, n)
    store.build_index(mem)
    (mem / "a.md").unlink()
    _note(mem, "d")
    real_write = store._atomic_write

    def die_on_meta(path, write):
        if path.name == "meta.json":
            raise KeyboardInterrupt
        real_write(path, write)

    monkeypatch.setattr(store, "_atomic_write", die_on_meta)
    with pytest.raises(KeyboardInterrupt):
        store._ensure_fresh(mem)
    monkeypatch.setattr(store, "_atomic_write", real_write)

    store._ensure_fresh(mem)
    _assert_rows_aligned(mem)


def test_empty_index_file_falls_back_to_full_rebuild(tmp_path):
    mem = tmp_path / "memory"
    for n in ("a", "b"):
        _note(mem, n)
    store.build_index(mem)
    (mem / ".engram" / "index.npy").write_bytes(b"")  # power loss after rename, before data hit disk
    _note(mem, "c")

    store._ensure_fresh(mem)
    _assert_rows_aligned(mem)
