"""Increment-1 gate: semantic recall must beat the lexical baseline on the labelled A/B set."""
import json
from pathlib import Path

import pytest

from engram.embed import model_available
from engram.eval import hit_rate_ab

DATASET = json.loads((Path(__file__).parent / "fixtures" / "recall_dataset.json").read_text())

pytestmark = pytest.mark.skipif(not model_available(), reason="bge-m3 not in local HF cache")


def test_semantic_recall_beats_lexical_at_k5(tmp_path):
    r = hit_rate_ab(DATASET, tmp_path / "mem", k=5)
    assert r["semantic"] > r["lexical"], r
    assert r["semantic"] >= 0.75, r  # absolute floor — recall must actually work, not just win


def test_semantic_recall_beats_lexical_at_k1(tmp_path):
    r = hit_rate_ab(DATASET, tmp_path / "mem", k=1)
    assert r["semantic"] > r["lexical"], r
