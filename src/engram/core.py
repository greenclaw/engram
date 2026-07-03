"""Note parsing + Generative-Agents-style scoring (recency / importance / relevance).

Pure logic, no model. The embedder lives in embed.py so this stays fast to import + test.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

INDEX_FILE = "MEMORY.md"  # the L1 index — a note store's one non-note file
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)

# Weighted SUM, not product. PLAN.md's shorthand said "recency×importance×relevance", but a pure
# product zeroes an old-but-critical fact ("we use Coolify not Vercel" doesn't decay). Generative
# Agents used a weighted sum precisely so one low component can't annihilate a note. These weights
# are the L1 auto-tune knobs (RESEARCH.md §6).
DEFAULT_WEIGHTS = {"relevance": 1.0, "importance": 0.5, "recency": 0.3}

# importance ∈ [0,1] by note type (PLAN: gotcha/feedback high, reference low).
_IMPORTANCE_BY_TYPE = {
    "feedback": 1.0,
    "gotcha": 1.0,
    "user": 0.9,
    "decision": 0.7,
    "project": 0.7,
    "reference": 0.4,
}
_DEFAULT_IMPORTANCE = 0.6
_DEFAULT_TYPE = "reference"


class MemoryNoteError(Exception):
    """A note has invalid frontmatter. Fail-loud policy: corrupt curated data must surface with the
    offending filename, never be silently skipped nor crash with a cryptic yaml/float traceback."""


@dataclass
class Note:
    path: Path
    name: str
    description: str
    type: str
    body: str
    importance: float | None = None  # explicit frontmatter override
    updated: date | None = None      # explicit frontmatter date; index falls back to mtime
    invalidated_by: str | None = None  # set by a curator INVALIDATE; suppresses the note from recall


def default_importance(note_type: str) -> float:
    return _IMPORTANCE_BY_TYPE.get(note_type, _DEFAULT_IMPORTANCE)


def note_importance(note: Note) -> float:
    return note.importance if note.importance is not None else default_importance(note.type)


def recency_decay(updated: date, now: date, half_life_days: float = 90.0) -> float:
    """Ebbinghaus-style: 1.0 today, halves every `half_life_days`. Never negative."""
    dt = max((now - updated).days, 0)
    return math.pow(0.5, dt / half_life_days)


def score(relevance: float, importance: float, recency: float, weights=DEFAULT_WEIGHTS) -> float:
    return (
        weights["relevance"] * relevance
        + weights["importance"] * importance
        + weights["recency"] * recency
    )


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _split_frontmatter(text: str, path: Path) -> tuple[dict, str]:
    # The closing `---` must be line-anchored (\n---\n), so a '---' inside a frontmatter value can't
    # truncate it. Frontmatter that parses to a non-dict (e.g. a leading '---' horizontal rule) is NOT
    # frontmatter — keep the whole text as body rather than silently discarding it.
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        raise MemoryNoteError(f"{path.name}: invalid YAML frontmatter — {str(e).splitlines()[0]}") from e
    if not isinstance(meta, dict):
        return {}, text
    return meta, m.group(2)


def parse_note(path: Path) -> Note:
    path = Path(path)
    meta, body = _split_frontmatter(path.read_text(encoding="utf-8"), path)
    note_type = meta.get("type") or (meta.get("metadata") or {}).get("type") or _DEFAULT_TYPE

    importance = meta.get("importance")
    if importance is not None:
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            raise MemoryNoteError(f"{path.name}: 'importance' must be a number 0–1, got {importance!r}") from None

    # str-coerce name/type/description: yaml turns `name: 2026-07-01`→date, `description: no`→False, which
    # otherwise crash json.dumps in the index or print as 'False' in context.
    invalidated_by = meta.get("invalidated_by")
    return Note(
        path=path,
        name=str(meta.get("name", path.stem)),
        description=str(meta.get("description", "")),
        type=str(note_type),
        body=body,
        importance=importance,
        updated=_coerce_date(meta.get("updated")),
        invalidated_by=str(invalidated_by) if invalidated_by is not None else None,
    )
