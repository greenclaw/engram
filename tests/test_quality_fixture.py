"""Well-formedness of the recall-quality fixture + the scale-dim distractor generator (pure logic)."""
import json
from pathlib import Path

from bench_quality import distractors

FIX = Path(__file__).parent / "fixtures" / "recall_quality.json"


def test_every_expect_names_a_note_in_its_section():
    q = json.loads(FIX.read_text())
    for dim in ("confusables", "negation", "keyword", "cross-lingual"):
        names = {n["name"] for n in q[dim]["notes"]}
        assert len(names) == len(q[dim]["notes"]), f"{dim}: duplicate note names"
        for item in q[dim]["queries"]:
            assert item["expect"] in names, f"{dim}: {item['expect']} not in notes"


def test_distractors_deterministic_unique_and_disjoint_from_base():
    d = distractors(100)
    assert d == distractors(100)
    names = {x["name"] for x in d}
    assert len(names) == 100
    base = json.loads((FIX.parent / "recall_dataset.json").read_text())
    assert names.isdisjoint({n["name"] for n in base["notes"]})
