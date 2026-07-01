"""Local index (numpy .npy + meta.json) and semantic recall over memory/*.md.

Markdown stays the source of truth; this index is a rebuildable secondary (guardrail). For a
few-hundred-note store a dense matrix + one matmul is plenty — no sqlite-vec, no service.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from engram.core import note_importance, parse_note, recency_decay, score
from engram.embed import Embedder

INDEX_DIR = ".engram"
BODY_HEAD = 500  # chars of body embedded alongside the description

# Abstention floor on RAW cosine: drop hits below it so an unrelated query returns nothing rather than
# a confidently-wrong top hit. Conservative by design — bge-m3's relevant/irrelevant cosine bands
# overlap (~0.37–0.45), and false abstention (hiding a real memory) is worse than a weak match, so the
# default sits safely below observed real-hit cosines. A bench-calibrated knob (RESEARCH.md §6).
RELEVANCE_FLOOR = float(os.environ.get("ENGRAM_RELEVANCE_FLOOR", "0.35"))


@dataclass
class Hit:
    name: str
    description: str
    type: str
    path: str
    score: float
    relevance: float


def _paths(mem_dir: Path) -> tuple[Path, Path, Path]:
    d = Path(mem_dir) / INDEX_DIR
    return d, d / "index.npy", d / "meta.json"


def _iter_notes(mem_dir: Path):
    """Recursive: real stores nest notes (e.g. learnings/). Skip the MEMORY.md index at any level and
    anything under the rebuildable `.engram/` index dir."""
    mem_dir = Path(mem_dir)
    for p in sorted(mem_dir.rglob("*.md")):
        if p.name == "MEMORY.md" or INDEX_DIR in p.relative_to(mem_dir).parts:
            continue
        yield p


def _embed_text(note) -> str:
    return f"{note.name}. {note.description}\n{note.body[:BODY_HEAD]}".strip()


def build_index(mem_dir) -> int:
    """(Re)build the index over `mem_dir`. Returns note count. ponytail: full rebuild, no incremental
    — add mtime-skip only when a few hundred notes becomes a few thousand."""
    mem_dir = Path(mem_dir)
    notes = [parse_note(p) for p in _iter_notes(mem_dir)]
    d, npy, metaf = _paths(mem_dir)
    d.mkdir(parents=True, exist_ok=True)

    if notes:
        vecs = Embedder().encode([_embed_text(n) for n in notes])
    else:
        vecs = np.zeros((0, 1024), dtype=np.float32)

    meta = []
    for n in notes:
        updated = n.updated or date.fromtimestamp(n.path.stat().st_mtime)
        meta.append({
            "path": str(n.path),
            "name": n.name,
            "description": n.description,
            "type": n.type,
            "importance": note_importance(n),
            "updated": updated.isoformat(),
        })
    np.save(npy, vecs)
    metaf.write_text(json.dumps(meta, ensure_ascii=False, indent=0))
    return len(notes)


def _load(mem_dir: Path):
    _, npy, metaf = _paths(mem_dir)
    if not npy.exists() or not metaf.exists():
        raise FileNotFoundError(f"No engram index under {mem_dir}; run `engram index --dir {mem_dir}` first.")
    return np.load(npy), json.loads(metaf.read_text())


def _is_stale(mem_dir: Path) -> bool:
    """The index is a rebuildable secondary; detect when the markdown source has drifted from it."""
    _, npy, metaf = _paths(mem_dir)
    if not npy.exists() or not metaf.exists():
        return True
    try:
        meta_count = len(json.loads(metaf.read_text()))
    except (ValueError, OSError):
        return True
    notes = list(_iter_notes(mem_dir))
    if len(notes) != meta_count:  # a note was added or deleted
        return True
    idx_mtime = npy.stat().st_mtime
    return any(p.stat().st_mtime > idx_mtime for p in notes)  # a note was edited


def recall(mem_dir, query: str, k: int = 5, now: date | None = None, floor: float | None = None) -> list[Hit]:
    """Return top-k L2 hits (id + description, NOT bodies) ranked by recency×importance×relevance.
    Auto-rebuilds the index first if the markdown source has drifted (added/edited/deleted notes).
    Abstains (drops hits below `floor` raw cosine) so an unrelated query returns [] not a wrong hit."""
    now = now or date.today()
    floor = RELEVANCE_FLOOR if floor is None else floor
    mem_dir = Path(mem_dir)
    if _is_stale(mem_dir):
        build_index(mem_dir)
    mat, meta = _load(mem_dir)
    if not meta:
        return []

    q = Embedder().encode([query])[0]
    cos = mat @ q  # both L2-normalized → dot product is cosine

    # Generative Agents min-max normalize each component to [0,1] before the weighted sum. Raw cosine
    # for related text sits in a compressed band (~0.4–0.7), so without this the importance/recency
    # terms would dominate ranking (note-type would outrank query match). Normalizing gives relevance
    # its full dynamic range; importance/recency stay honest tiebreakers. `relevance` on the Hit keeps
    # the raw cosine (interpretable); scoring uses the normalized value.
    lo, hi = float(cos.min()), float(cos.max())
    rng = hi - lo

    hits = []
    for i, m in enumerate(meta):
        rel_norm = (float(cos[i]) - lo) / rng if rng > 1e-9 else 1.0
        rec = recency_decay(date.fromisoformat(m["updated"]), now)
        hits.append(
            Hit(
                name=m["name"],
                description=m["description"],
                type=m["type"],
                path=m["path"],
                relevance=float(cos[i]),
                score=score(rel_norm, m["importance"], rec),
            )
        )
    hits = [h for h in hits if h.relevance >= floor]  # abstention: drop weakly-related notes
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]
