"""Curator core: apply a Claude-produced change-set as a gated, git-committed edit.

Claude (via the /engram-curate skill) does the adjudication and emits a change-set; this module is the
deterministic half — it never decides, only applies ADD/UPDATE/INVALIDATE/NOOP behind a human gate.
Guardrails: never clobber untouched frontmatter (UPDATE), invalidate-don't-delete, commit for provenance.
"""
from __future__ import annotations

import difflib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

from engram import core


class CurateError(Exception):
    """A change-set can't be applied (bad shape, unknown op, missing target, ADD over an existing note)."""


def load_changeset(source: str) -> dict:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    try:
        cs = json.loads(text)
    except json.JSONDecodeError as e:
        raise CurateError(f"invalid change-set JSON: {e}") from e
    if not isinstance(cs, dict) or not isinstance(cs.get("changes"), list):
        raise CurateError("change-set must be an object with a 'changes' list")
    return cs


def _serialize(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
    return f"---\n{front}---\n{body}" + ("" if body.endswith("\n") else "\n")


def _require(ch: dict, field: str) -> str:
    if field not in ch:
        raise CurateError(f"{ch.get('op', '?')}: missing required field {field!r}")
    return ch[field]


def _note_path(mem_dir: Path, name: str, op: str) -> Path:
    """Change-sets are LLM output — untrusted. Subdir notes are fine; escaping the store is not."""
    if not isinstance(name, str):
        raise CurateError(f"{op}: 'name' must be a string, got {type(name).__name__}")
    path = (mem_dir / f"{name}.md").resolve()
    if not path.is_relative_to(mem_dir.resolve()):
        raise CurateError(f"{op} {name}: path escapes the memory dir")
    if path.name == core.INDEX_FILE:
        raise CurateError(f"{op} {name}: refusing to write the {core.INDEX_FILE} index as a note")
    return path


def _resolve_target(mem_dir: Path, target: str, op: str) -> Path:
    """UPDATE/INVALIDATE reference a note by its recall id (frontmatter name), which may not match the
    filename for nested notes. Look it up; the index (MEMORY.md) is never a target (not in _iter_notes)."""
    from engram.store import _iter_notes

    for p in _iter_notes(mem_dir):
        meta, _ = core._split_frontmatter(p.read_text(encoding="utf-8"), p)
        rel = str(p.relative_to(mem_dir).with_suffix(""))
        if target in (str(meta.get("name", p.stem)), p.stem, rel):
            return p
    raise CurateError(f"{op} {target}: no such note")


def _as_body(body) -> str:
    if not isinstance(body, str):
        raise CurateError(f"'body' must be a string, got {type(body).__name__}")
    return body


def _edit_for(mem_dir: Path, ch, today: date):
    """Return (path, new_text) for a change, or None for NOOP."""
    if not isinstance(ch, dict):
        raise CurateError(f"each change must be an object, got {type(ch).__name__}")
    op = ch.get("op")
    if op == "NOOP":
        return None
    if op == "ADD":
        name = _require(ch, "name")
        path = _note_path(mem_dir, name, op)
        if path.exists():
            raise CurateError(f"ADD {name}: note already exists (use UPDATE)")
        meta = {"name": name, "description": str(ch.get("description", "")),
                "type": ch.get("type", "reference"), "updated": today}
        return path, _serialize(meta, _as_body(ch.get("body", "")))
    if op in ("UPDATE", "INVALIDATE"):
        path = _resolve_target(mem_dir, _require(ch, "target"), op)
        meta, body = core._split_frontmatter(path.read_text(encoding="utf-8"), path)
        if op == "UPDATE":
            meta["updated"] = today  # content changed → refresh recency
            if "description" in ch:
                meta["description"] = str(ch["description"])
            if "type" in ch:
                meta["type"] = ch["type"]
            if "importance" in ch:
                meta["importance"] = ch["importance"]
            if "body" in ch:
                body = _as_body(ch["body"])
        else:  # INVALIDATE — mark superseded, keep note+body, do NOT refresh recency (Zep)
            meta["invalidated_by"] = _require(ch, "invalidated_by")
        return path, _serialize(meta, body)
    raise CurateError(f"unknown op {op!r}")


def _diff(edits, mem_dir: Path) -> str:
    out = []
    for path, new, old in edits:
        rel = path.relative_to(mem_dir) if path.is_relative_to(mem_dir) else path
        out += difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(out)


def _commit(mem_dir: Path, paths, changeset: dict) -> None:
    try:
        top = subprocess.run(["git", "-C", str(mem_dir), "rev-parse", "--show-toplevel"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("(not a git repo — files written, no provenance commit)")
        return
    # Only commit if the memory dir IS the repo root — never land a curate commit on an enclosing repo's
    # branch (a memory dir nested inside another project / dotfiles repo).
    if Path(top).resolve() != Path(mem_dir).resolve():
        print(f"(memory dir is not its own git repo — nested in {top}; files written, no provenance commit)")
        return
    rel = [str(p) for p in paths]
    ops = ", ".join(sorted({c.get("op") for c in changeset["changes"]
                            if isinstance(c, dict) and c.get("op") != "NOOP"}))
    try:  # a git failure (gitignored, nothing-to-commit, hook) must not crash after files were written
        subprocess.run(["git", "-C", str(mem_dir), "add", *rel], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(mem_dir), "commit", "-q", "-m", f"engram: curate ({ops})", "--", *rel],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode(errors="replace").strip() or "git error"
        print(f"(files written; git commit skipped: {detail})")


def apply(mem_dir, changeset: dict, confirm, now: date | None = None) -> bool:
    """Apply a change-set behind `confirm(diff)->bool`. Returns True iff changes were written (+committed)."""
    mem_dir = Path(mem_dir)
    today = now or date.today()
    edits = []
    seen: set[Path] = set()
    for ch in changeset["changes"]:
        e = _edit_for(mem_dir, ch, today)
        if e is None:
            continue
        path, new = e
        if path in seen:  # edits are computed against the original text — a second op would silently win
            raise CurateError(f"multiple ops target {path.name}; merge them into one change")
        seen.add(path)
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        edits.append((path, new, old))
    if not edits:
        return False
    if not confirm(_diff(edits, mem_dir)):
        return False
    for path, new, _ in edits:
        path.parent.mkdir(parents=True, exist_ok=True)  # subdir notes (learnings/...)
        path.write_text(new, encoding="utf-8")
    _commit(mem_dir, [p for p, _, _ in edits], changeset)
    return True
