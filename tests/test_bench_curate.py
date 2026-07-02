"""Deterministic half of the mem_curate bench: scoring a change-set against labels."""
from bench_curate import score_changeset

LABELS = [
    {"op": "ADD"},
    {"op": "NOOP", "target": "packages"},
    {"op": "UPDATE", "target": "testing"},
    {"op": "INVALIDATE", "target": "hosting"},
]


def test_perfect_prediction_scores_full():
    changes = [
        {"candidate": 0, "op": "ADD", "name": "tailwind"},
        {"candidate": 1, "op": "NOOP"},
        {"candidate": 2, "op": "UPDATE", "target": "testing"},
        {"candidate": 3, "op": "INVALIDATE", "target": "hosting", "invalidated_by": "hosting-fly"},
        {"candidate": 3, "op": "ADD", "name": "hosting-fly"},
    ]
    r = score_changeset(LABELS, changes)
    assert r["op_accuracy"] == 1.0
    assert r["contradiction_catch"] == 1.0
    assert r["false_invalidate"] == 0


def test_wrong_target_update_is_not_correct():
    changes = [
        {"candidate": 0, "op": "ADD"},
        {"candidate": 1, "op": "NOOP"},
        {"candidate": 2, "op": "UPDATE", "target": "email"},  # wrong note
        {"candidate": 3, "op": "INVALIDATE", "target": "hosting"},
    ]
    r = score_changeset(LABELS, changes)
    assert r["op_accuracy"] == 0.75


def test_missed_contradiction_and_false_invalidate():
    changes = [
        {"candidate": 0, "op": "ADD"},
        {"candidate": 1, "op": "INVALIDATE", "target": "packages"},  # false invalidate (label NOOP)
        {"candidate": 2, "op": "UPDATE", "target": "testing"},
        {"candidate": 3, "op": "ADD", "name": "hosting-fly"},  # missed the INVALIDATE
    ]
    r = score_changeset(LABELS, changes)
    assert r["contradiction_catch"] == 0.0
    assert r["false_invalidate"] == 1
    assert r["op_accuracy"] == 0.5


def test_unmentioned_candidate_counts_as_wrong():
    r = score_changeset(LABELS, [{"candidate": 0, "op": "ADD"}])
    assert r["op_accuracy"] == 0.25
