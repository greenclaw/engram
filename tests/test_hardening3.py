"""Regression tests for the accepted-remaining PR#1 findings: edit-during-rebuild race and
concurrent double rebuild (double model-load). Both use a fake Embedder — no model needed."""
import threading

import numpy as np
import pytest

from engram import store


def _note(mem, name, body="body"):
    mem.mkdir(parents=True, exist_ok=True)
    p = mem / f"{name}.md"
    p.write_text(f"---\nname: {name}\ndescription: d-{name}\ntype: project\n---\n{body}\n")
    return p


class FakeEmbedder:
    """Deterministic, model-free. Optional on_encode callback fires mid-encode (to simulate an edit
    landing while the real multi-second embed runs)."""

    on_encode = None
    batch_calls = 0
    lock = threading.Lock()

    def encode(self, texts):
        texts = list(texts)
        if len(texts) > 1:
            with FakeEmbedder.lock:
                FakeEmbedder.batch_calls += 1
            if FakeEmbedder.on_encode:
                FakeEmbedder.on_encode()
        v = np.ones((len(texts), 4), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    FakeEmbedder.on_encode = None
    FakeEmbedder.batch_calls = 0
    monkeypatch.setattr(store, "Embedder", FakeEmbedder)


# --- edit-during-rebuild race ------------------------------------------------

def test_edit_during_embed_is_caught_by_next_staleness_check(tmp_path):
    mem = tmp_path / "mem"
    _note(mem, "a")
    b = _note(mem, "b")

    def edit_mid_embed():  # lands after the notes were read but before the index is saved
        b.write_text("---\nname: b\ndescription: EDITED\ntype: project\n---\nnew body\n")

    FakeEmbedder.on_encode = edit_mid_embed
    store.build_index(mem)
    # the index embedded the OLD text of b — the store must read as stale so the edit isn't lost
    assert store._is_stale(mem)


# --- concurrent rebuild serialization ---------------------------------------

def test_concurrent_recalls_rebuild_once(tmp_path):
    mem = tmp_path / "mem"
    _note(mem, "a")
    _note(mem, "b")

    started = threading.Barrier(2)

    def slow_encode():
        import time
        time.sleep(0.3)

    FakeEmbedder.on_encode = slow_encode
    errors = []

    def run():
        started.wait()
        try:
            store.recall(mem, "anything about a", k=2, floor=0.0)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=run) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    # the second recall must wait on the lock, re-check staleness and SKIP its own rebuild
    assert FakeEmbedder.batch_calls == 1


def test_missing_dir_still_errors_not_ghost_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.recall(tmp_path / "nope", "q")
    assert not (tmp_path / "nope").exists()
