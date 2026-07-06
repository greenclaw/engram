"""Local index (numpy .npy + meta.json) and semantic recall over memory/*.md.

Markdown stays the source of truth; this index is a rebuildable secondary (guardrail). For a
few-hundred-note store a dense matrix + one matmul is plenty — no sqlite-vec, no service.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from engram.core import INDEX_FILE, note_importance, parse_note, recency_decay, score
from engram.embed import Embedder

INDEX_DIR = ".engram"
BODY_HEAD = 500  # chars of body embedded alongside the description

# Abstention floor on RAW cosine: drop hits below it so an unrelated query returns nothing rather than
# a confidently-wrong top hit. Conservative by design — bge-m3's relevant/irrelevant cosine bands
# overlap (~0.37–0.45), and false abstention (hiding a real memory) is worse than a weak match, so the
# default sits safely below observed real-hit cosines. A bench-calibrated knob (RESEARCH.md §6).
DEFAULT_FLOOR = "0.35"


def _env_floor() -> float:
    raw = os.environ.get("ENGRAM_RELEVANCE_FLOOR", DEFAULT_FLOOR)  # read at use, not frozen at import
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"ENGRAM_RELEVANCE_FLOOR must be a number, got {raw!r}") from e


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
        if p.name == INDEX_FILE or INDEX_DIR in p.relative_to(mem_dir).parts:
            continue
        yield p


def _embed_text(note) -> str:
    return f"{note.name}. {note.description}\n{note.body[:BODY_HEAD]}".strip()


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file + os.replace so a concurrent reader never sees a torn/half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


@contextmanager
def _lock(mem_dir: Path):
    """Exclusive flock serializing rebuilds — two concurrent first-recall rebuilds would otherwise
    each load the (heavy) model and race their writes. POSIX-only (fcntl); Windows isn't a target."""
    d = Path(mem_dir) / INDEX_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_index(mem_dir) -> int:
    """(Re)build the index over `mem_dir` (full rebuild, atomic writes). Raises if the dir is missing."""
    mem_dir = Path(mem_dir)
    if not mem_dir.is_dir():  # a typo'd --dir must error, not auto-create an empty ghost store
        raise FileNotFoundError(f"memory dir does not exist: {mem_dir}")
    with _lock(mem_dir):
        return _build(mem_dir)


def _ensure_fresh(mem_dir: Path) -> None:
    """Rebuild iff the source drifted. Re-checks staleness after acquiring the lock, so a recall that
    waited behind a concurrent rebuild skips its own (no double model-load, no duplicate work)."""
    if not _is_stale(mem_dir):
        return
    if not mem_dir.is_dir():  # before _lock — it mkdirs .engram (would ghost-create the store)
        raise FileNotFoundError(f"memory dir does not exist: {mem_dir}")
    with _lock(mem_dir):
        if _is_stale(mem_dir):
            _build(mem_dir)


def _build(mem_dir: Path) -> int:
    # stat() BEFORE reading/embedding: an edit landing during the multi-second embed then differs
    # from the recorded fingerprint, so the next staleness check catches it (edit-during-rebuild race).
    snap = [(p.stat(), parse_note(p)) for p in _iter_notes(mem_dir)]
    notes = [n for _, n in snap]
    d, npy, metaf = _paths(mem_dir)
    d.mkdir(parents=True, exist_ok=True)

    vecs = Embedder().encode([_embed_text(n) for n in notes]) if notes else np.zeros((0, 1024), dtype=np.float32)

    meta = []
    for st, n in snap:
        updated = n.updated or date.fromtimestamp(st.st_mtime)
        meta.append({
            "path": str(n.path),
            "name": n.name,
            "description": n.description,
            "type": n.type,
            "importance": note_importance(n),
            "updated": updated.isoformat(),
            "invalidated": bool(n.invalidated_by),
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
        })
    _atomic_write(npy, lambda f: np.save(f, vecs))
    _atomic_write(metaf, lambda f: f.write(json.dumps(meta, ensure_ascii=False, indent=0).encode("utf-8")))
    return len(notes)


def _load(mem_dir: Path):
    _, npy, metaf = _paths(mem_dir)
    if not npy.exists() or not metaf.exists():
        raise FileNotFoundError(f"No engram index under {mem_dir}; run `engram index --dir {mem_dir}` first.")
    return np.load(npy), json.loads(metaf.read_text())


def _fingerprint(mem_dir: Path) -> dict:
    fp = {}
    for p in _iter_notes(mem_dir):
        st = p.stat()
        fp[str(p)] = [st.st_mtime_ns, st.st_size]
    return fp


def _is_stale(mem_dir: Path) -> bool:
    """Rebuild when the source drifts. Compares each note's (mtime_ns, size) to the index, so adds,
    deletes, edits AND renames (which preserve count + mtime, missed by an mtime>index check) all catch."""
    _, npy, metaf = _paths(mem_dir)
    if not npy.exists() or not metaf.exists():
        return True
    try:
        meta = json.loads(metaf.read_text())
    except (ValueError, OSError):
        return True
    recorded = {m["path"]: [m.get("mtime_ns"), m.get("size")] for m in meta}
    return recorded != _fingerprint(mem_dir)


def recall(mem_dir, query: str, k: int = 5, now: date | None = None, floor: float | None = None) -> list[Hit]:
    """Return top-k L2 hits (id + description, NOT bodies) ranked by recency×importance×relevance.
    Auto-rebuilds the index first if the markdown source has drifted (added/edited/deleted notes).
    Abstains (drops hits below `floor` raw cosine) so an unrelated query returns [] not a wrong hit."""
    now = now or date.today()
    floor = _env_floor() if floor is None else floor
    mem_dir = Path(mem_dir)
    _ensure_fresh(mem_dir)  # raises FileNotFoundError on a nonexistent dir — no ghost store
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
        if m.get("invalidated"):  # superseded by a curator INVALIDATE — never surface it
            continue
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
