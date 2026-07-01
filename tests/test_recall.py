"""Integration tests: real bge-m3 embedder + index + semantic recall.

Skipped if the bge-m3 model isn't in the local HF cache (no network downloads in tests).
"""
from datetime import date

import numpy as np
import pytest

from engram.embed import Embedder, model_available
from engram.store import build_index, recall

pytestmark = pytest.mark.skipif(
    not model_available(), reason="bge-m3 not in local HF cache"
)


def _write(dir, name, desc, type_, body=""):
    (dir / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\ntype: {type_}\n---\n{body}\n"
    )


def test_embedder_returns_normalized_1024(tmp_path):
    vecs = Embedder().encode(["hello world", "hello world"])
    assert vecs.shape == (2, 1024)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    # identical inputs → identical rows
    np.testing.assert_allclose(vecs[0], vecs[1], atol=1e-5)


def test_semantic_recall_beats_lexical_distractor(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    _write(mem, "hosting", "The production app runs on Coolify on a Hetzner server, not Vercel.", "project")
    _write(mem, "testing_pref", "The user prefers pytest over unittest for Python tests.", "feedback")
    _write(mem, "coffee", "The office coffee machine is on the third floor by the window.", "reference")

    build_index(mem)
    # query shares NO keywords with the hosting note ("deploy"/"where") — lexical match would miss it
    hits = recall(mem, "where do we ship the site to in production?", k=3)

    assert hits[0].name == "hosting"
    # recall returns L2 (description), not the raw body
    assert "Coolify" in hits[0].description


def test_recall_respects_top_k(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    for i in range(5):
        _write(mem, f"note{i}", f"topic number {i} about widgets and gadgets", "reference")
    build_index(mem)
    assert len(recall(mem, "widgets", k=2)) == 2


def test_recall_auto_rebuilds_when_note_added(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    _write(mem, "deploy", "Push to main triggers a Coolify rebuild.", "project")
    build_index(mem)
    # add a note AFTER indexing, then recall without an explicit reindex
    _write(mem, "billing", "Stripe handles subscriptions and recurring invoices.", "project")
    hits = recall(mem, "how are recurring payments processed?", k=3)
    assert "billing" in [h.name for h in hits]


def test_recall_auto_rebuilds_after_delete(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    _write(mem, "kube", "Kubernetes autoscaling of worker pods.", "reference")
    _write(mem, "pg", "Postgres database tuning and vacuum.", "reference")
    build_index(mem)
    (mem / "pg.md").unlink()
    assert "pg" not in [h.name for h in recall(mem, "database", k=5)]


def test_recall_abstains_on_unrelated_query(tmp_path):
    # nothing in the store is even loosely related → recall returns nothing (abstention),
    # rather than a confidently-wrong top hit.
    mem = tmp_path / "memory"
    mem.mkdir()
    _write(mem, "deploy", "Push to main triggers a Coolify rebuild.", "project")
    _write(mem, "secrets", "Use the secret CLI keyring; never print a value.", "feedback")
    assert recall(mem, "what is the capital of France?", k=3) == []


def test_recall_does_not_over_abstain_on_related_query(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    _write(mem, "deploy", "Push to main triggers a Coolify rebuild.", "project")
    _write(mem, "secrets", "Use the secret CLI keyring; never print a value.", "feedback")
    hits = recall(mem, "how do we redeploy the app?", k=3)
    assert hits and hits[0].name == "deploy"
