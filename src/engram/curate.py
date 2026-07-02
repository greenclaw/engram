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
    cs = json.loads(text)
    if not isinstance(cs, dict) or not isinstance(cs.get("changes"), list):
        raise CurateError("change-set must be an object with a 'changes' list")
    return cs


def _serialize(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
    return f"---\n{front}---\n{body}" + ("" if body.endswith("\n") else "\n")


def _edit_for(mem_dir: Path, ch: dict, today: date):
    """Return (path, new_text) for a change, or None for NOOP."""
    op = ch.get("op")
    if op == "NOOP":
        return None
    if op == "ADD":
        name = ch["name"]
        path = mem_dir / f"{name}.md"
        if path.exists():
            raise CurateError(f"ADD {name}: note already exists (use UPDATE)")
        meta = {"name": name, "description": ch.get("description", ""),
                "type": ch.get("type", "reference"), "updated": today}
        return path, _serialize(meta, ch.get("body", ""))
    if op in ("UPDATE", "INVALIDATE"):
        target = ch["target"]
        path = mem_dir / f"{target}.md"
        if not path.exists():
            raise CurateError(f"{op} {target}: no such note")
        meta, body = core._split_frontmatter(path.read_text(encoding="utf-8"), path)
        meta["updated"] = today
        if op == "UPDATE":
            if "description" in ch:
                meta["description"] = ch["description"]
            if "body" in ch:
                body = ch["body"]
        else:  # INVALIDATE — keep the note + body, mark superseded (Zep: invalidate-don't-delete)
            meta["invalidated_by"] = ch["invalidated_by"]
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
        subprocess.run(["git", "-C", str(mem_dir), "rev-parse", "--is-inside-work-tree"],
                       check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("(not a git repo — files written, no provenance commit)")
        return
    subprocess.run(["git", "-C", str(mem_dir), "add", *[str(p) for p in paths]], check=True)
    ops = ", ".join(sorted({c.get("op") for c in changeset["changes"] if c.get("op") != "NOOP"}))
    subprocess.run(["git", "-C", str(mem_dir), "commit", "-q", "-m", f"engram: curate ({ops})"], check=True)


def apply(mem_dir, changeset: dict, confirm, now: date | None = None) -> bool:
    """Apply a change-set behind `confirm(diff)->bool`. Returns True iff changes were written (+committed)."""
    mem_dir = Path(mem_dir)
    today = now or date.today()
    edits = []
    for ch in changeset["changes"]:
        e = _edit_for(mem_dir, ch, today)
        if e is None:
            continue
        path, new = e
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        edits.append((path, new, old))
    if not edits:
        return False
    if not confirm(_diff(edits, mem_dir)):
        return False
    for path, new, _ in edits:
        path.write_text(new, encoding="utf-8")
    _commit(mem_dir, [p for p, _, _ in edits], changeset)
    return True
