"""Unit tests for engram core scoring + note parsing (no model needed)."""
from datetime import date

import pytest

from engram.core import (
    DEFAULT_WEIGHTS,
    MemoryNoteError,
    default_importance,
    note_importance,
    parse_note,
    recency_decay,
    score,
)
from engram.store import _iter_notes

# --- importance ---------------------------------------------------------

def test_default_importance_ranks_feedback_above_reference():
    assert default_importance("feedback") > default_importance("reference")


def test_default_importance_unknown_type_is_middling():
    imp = default_importance("something-weird")
    assert 0.0 < imp < 1.0


def test_frontmatter_importance_overrides_type_default(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("---\nname: x\ntype: feedback\nimportance: 0.2\n---\nbody\n")
    note = parse_note(f)
    # feedback defaults high, but explicit 0.2 must win
    assert note_importance(note) == 0.2


# --- recency ------------------------------------------------------------

def test_recency_is_one_today():
    now = date(2026, 7, 1)
    assert recency_decay(now, now, half_life_days=90) == 1.0


def test_recency_halves_at_half_life():
    now = date(2026, 7, 1)
    updated = date(2026, 4, 2)  # 90 days earlier
    assert abs(recency_decay(updated, now, half_life_days=90) - 0.5) < 0.02


def test_recency_monotonic_decreasing():
    now = date(2026, 7, 1)
    recent = recency_decay(date(2026, 6, 1), now)
    old = recency_decay(date(2025, 6, 1), now)
    assert recent > old


# --- score --------------------------------------------------------------

def test_relevance_dominates_score():
    # a barely-relevant but fresh+important note must lose to a highly-relevant one
    high_rel = score(relevance=0.9, importance=0.2, recency=0.2)
    low_rel = score(relevance=0.2, importance=1.0, recency=1.0)
    assert high_rel > low_rel


def test_importance_breaks_ties_at_equal_relevance():
    a = score(relevance=0.7, importance=0.9, recency=0.5)
    b = score(relevance=0.7, importance=0.3, recency=0.5)
    assert a > b


def test_weights_are_relevance_first():
    assert DEFAULT_WEIGHTS["relevance"] > DEFAULT_WEIGHTS["importance"] >= DEFAULT_WEIGHTS["recency"]


# --- parsing ------------------------------------------------------------

def test_parse_reads_frontmatter_and_body(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("---\nname: Hosting\ndescription: uses Coolify\ntype: project\n---\nBody line one.\n")
    note = parse_note(f)
    assert note.name == "Hosting"
    assert note.description == "uses Coolify"
    assert note.type == "project"
    assert "Body line one." in note.body


def test_parse_supports_nested_metadata_type(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("---\nname: x\nmetadata:\n  type: feedback\n---\nb\n")
    note = parse_note(f)
    assert note.type == "feedback"


def test_parse_no_frontmatter_is_graceful(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("just a plain note, no fences\n")
    note = parse_note(f)
    assert "plain note" in note.body
    assert note_importance(note) > 0  # falls back to a default


# --- fail-loud on bad notes (with an actionable, file-named message) --------

def test_malformed_yaml_raises_named_error(tmp_path):
    # a colon in an unquoted description → invalid YAML mapping
    f = tmp_path / "auth.md"
    f.write_text("---\nname: x\ndescription: Auth: use JWT\ntype: reference\n---\nbody\n")
    with pytest.raises(MemoryNoteError) as ei:
        parse_note(f)
    assert "auth.md" in str(ei.value)  # message must name the offending file


def test_non_numeric_importance_raises_named_error(tmp_path):
    f = tmp_path / "hosting.md"
    f.write_text("---\nname: x\ntype: project\nimportance: high\n---\nb\n")
    with pytest.raises(MemoryNoteError) as ei:
        parse_note(f)
    assert "hosting.md" in str(ei.value)
    assert "importance" in str(ei.value).lower()


# --- recursive discovery, excluding MEMORY.md and the index dir -------------

def test_iter_notes_is_recursive_and_skips_index_and_memory(tmp_path):
    (tmp_path / "top.md").write_text("---\nname: t\n---\nx\n")
    sub = tmp_path / "learnings"
    sub.mkdir()
    (sub / "deep.md").write_text("---\nname: d\n---\ny\n")
    (tmp_path / "MEMORY.md").write_text("# index\n")
    (sub / "MEMORY.md").write_text("# nested index\n")
    eng = tmp_path / ".engram"
    eng.mkdir()
    (eng / "stray.md").write_text("---\nname: s\n---\nz\n")

    names = sorted(p.name for p in _iter_notes(tmp_path))
    assert names == ["deep.md", "top.md"]  # recursive; no MEMORY.md at any level; nothing under .engram
