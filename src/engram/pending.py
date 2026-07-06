"""Pending-store: the Stop hook enqueues finished sessions; the /engram-curate skill drains the queue.

The queue is derived state under `.engram/` (like the index) — never a note, never committed. The hook
only RECORDS the session (fast, no LLM call); adjudication stays in the skill, offline and gated —
this is the "strict superset of the manual flow": same curation path, auto-fed instead of hand-fed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _file(mem_dir) -> Path:
    return Path(mem_dir) / ".engram" / "pending.json"


def load(mem_dir) -> dict:
    """{session_id: {transcript_path, ts}}. A corrupt queue reads as empty — hook resilience beats
    strictness here (the queue is rederivable convenience, not curated data)."""
    f = _file(mem_dir)
    try:
        q = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (ValueError, OSError):
        return {}
    return q if isinstance(q, dict) else {}


def add(mem_dir, session_id: str, transcript_path: str) -> None:
    """Enqueue (dedup by session_id: Stop fires after every response — the latest entry wins)."""
    if not Path(mem_dir).is_dir():  # a typo'd --dir must not ghost-create a store
        raise FileNotFoundError(f"memory dir does not exist: {mem_dir}")
    q = load(mem_dir)
    q[str(session_id)] = {"transcript_path": str(transcript_path), "ts": time.time()}
    f = _file(mem_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(q, ensure_ascii=False, indent=0), encoding="utf-8")


def clear(mem_dir, session_ids=None) -> int:
    """Drop the given session ids (or everything). Returns how many were removed."""
    q = load(mem_dir)
    keep = {} if session_ids is None else {k: v for k, v in q.items() if k not in set(session_ids)}
    removed = len(q) - len(keep)
    if removed:
        _file(mem_dir).write_text(json.dumps(keep, ensure_ascii=False, indent=0), encoding="utf-8")
    return removed
