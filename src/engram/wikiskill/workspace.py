"""The three-layer workspace (raw/ wiki/ skills/) and the harness-side mutations the paper
assigns to code, not to an LLM: patch ops, proposal apply, skill-impact.md, git audit trail."""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from pathlib import Path

INDEX_HEADER = "# Wiki Index\n\n"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ProposalError(ValueError):
    """Untrusted LLM proposal that cannot be applied (bad name, unknown skill, bad action)."""


def init_workspace(ws: Path) -> None:
    for rel in ("raw", "wiki/patterns", "skills", "dataset"):
        (ws / rel).mkdir(parents=True, exist_ok=True)
    for rel, txt in (("wiki/index.md", INDEX_HEADER), ("wiki/log.md", "# Evolution Log\n\n"),
                     ("wiki/skill-impact.md", "# Skill Impact\n\n")):
        p = ws / rel
        if not p.exists():
            p.write_text(txt)
    for rel in ("raw", "skills", "dataset", "wiki/patterns"):
        (ws / rel / ".gitkeep").touch()
    if not (ws / ".git").exists():
        git(ws, "init", "-q")
    commit_all(ws, "wikiskill: init workspace")


def apply_edits(text: str, edits: list[dict]) -> tuple[str, list[str]]:
    """The three patch ops of Appendix E (append / replace / insert_after). `target` must be an
    exact substring — a miss is skipped and reported, never fuzzy-matched."""
    skipped: list[str] = []
    for e in edits:
        op, content = e.get("op"), str(e.get("content", ""))
        if op == "append":
            text = text + ("" if text.endswith("\n") or not text else "\n") + content + "\n"
        elif op in ("replace", "insert_after"):
            target = str(e.get("target", ""))
            i = text.find(target) if target else -1
            if i < 0:
                skipped.append(f"{op}: target not found: {target[:80]!r}")
                continue
            j = i + len(target)
            text = text[:i] + content + text[j:] if op == "replace" else text[:j] + "\n" + content + text[j:]
        else:
            skipped.append(f"unknown op: {op!r}")
    return text, skipped


def skills_dir_section(skills: Path) -> str:
    """Full injection (§3.2.1): every SKILL.md verbatim, or "" when the skill set is empty."""
    parts = [f"### {d.name}\n{(d / 'SKILL.md').read_text()}"
             for d in sorted(skills.iterdir()) if (d / "SKILL.md").is_file()]
    return "## Skills\n\n" + "\n\n".join(parts) + "\n" if parts else ""


def skill_section(ws: Path) -> str:
    return skills_dir_section(ws / "skills")


def read_wiki(ws: Path) -> str:
    wiki = ws / "wiki"
    parts = [f"=== wiki/index.md ===\n{(wiki / 'index.md').read_text()}",
             f"=== wiki/log.md ===\n{(wiki / 'log.md').read_text()}"]
    for p in sorted((wiki / "patterns").glob("*.md")):
        parts.append(f"=== wiki/patterns/{p.name} ===\n{p.read_text()}")
    return "\n\n".join(parts)


def _safe_md_name(name: str) -> str:
    base = name[:-3] if name.endswith(".md") else name
    if not _NAME.match(base):
        raise ProposalError(f"bad name: {name!r}")
    return base + ".md"


def apply_maintainer(ws: Path, out: dict, k: int) -> list[str]:
    """Apply one Maintainer output (Eq. 2): create/patch patterns, rewrite index.md, append log.md.
    Returns the skipped patches; they are also recorded in log.md so nothing is lost silently."""
    patterns, skipped = ws / "wiki/patterns", []
    for c in out.get("create_patterns") or []:
        try:
            (patterns / _safe_md_name(str(c["name"]))).write_text(str(c["content"]))
        except (ProposalError, KeyError) as e:
            skipped.append(f"create_patterns: {e}")
    for u in out.get("update_patterns") or []:
        try:
            p = patterns / _safe_md_name(str(u["name"]))
        except (ProposalError, KeyError) as e:
            skipped.append(f"update_patterns: {e}")
            continue
        if not p.is_file():
            skipped.append(f"update_patterns: no such pattern {p.name}")
            continue
        text, sk = apply_edits(p.read_text(), u.get("edits") or [])
        p.write_text(text)
        skipped += [f"{p.name}: {s}" for s in sk]
    if out.get("update_index"):
        (ws / "wiki/index.md").write_text(str(out["update_index"]).rstrip() + "\n")
    entry = f"## iter {k}\n\n{str(out.get('append_log', '')).strip()}\n"
    if skipped:
        entry += "\nSkipped patches:\n" + "".join(f"- {s}\n" for s in skipped)
    with (ws / "wiki/log.md").open("a") as f:
        f.write(entry + "\n")
    return skipped


def _skill_text(d: Path) -> str:
    return (d / "SKILL.md").read_text() if (d / "SKILL.md").is_file() else ""


def apply_proposal(ws: Path, p: dict) -> tuple[str, list[str]]:
    """Apply one atomic Proposer output (Eq. 3) to skills/. Returns (unified diff, skipped patches)."""
    action = p.get("action")
    if action not in ("create", "patch"):
        raise ProposalError(f"bad action: {action!r}")
    name = str(p.get("name", ""))
    if not _NAME.match(name):
        raise ProposalError(f"bad skill name: {name!r}")
    d = ws / "skills" / name
    before, skipped = _skill_text(d), []
    if action == "create":
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(str(p.get("skill_md", "")))
        (d / "PURPOSE.md").write_text(str(p.get("purpose_md", "")))
    else:
        if not (d / "SKILL.md").is_file():
            raise ProposalError(f"patch: no such skill {name!r}")
        text, skipped = apply_edits(before, p.get("edits") or [])
        (d / "SKILL.md").write_text(text)
    after = _skill_text(d)
    rel = f"skills/{name}/SKILL.md"
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return diff, skipped


def append_impact(ws: Path, k: int, p: dict, diff: str, r_val: float | None, r_best: float, outcome: str) -> None:
    """The harness-only audit entry of §3.2.4: proposal metadata, full diff, val score, outcome."""
    action, name = p.get("action", "?"), p.get("name", "")
    val = f"{r_val:.3f}" if r_val is not None else "n/a"
    head = f"{action} {name}".strip()
    entry = (f"## iter {k} — {head} — val {val} (best {r_best:.3f}) — {outcome}\n\n"
             f"- action: {action}\n- skill: {name}\n- outcome: {outcome}\n")
    if diff:
        entry += f"\n```diff\n{diff.rstrip()}\n```\n"
    with (ws / "wiki/skill-impact.md").open("a") as f:
        f.write(entry + "\n")


def git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True, text=True).stdout


def commit_all(ws: Path, msg: str) -> None:
    git(ws, "add", "-A")
    if git(ws, "status", "--porcelain").strip():
        git(ws, "-c", "user.name=wikiskill", "-c", "user.email=wikiskill@engram", "commit", "-q", "-m", msg)


def tag(ws: Path, name: str) -> None:
    git(ws, "tag", "-f", name)


def restore_skills(ws: Path, ref: str) -> None:
    """Roll skills/ back to `ref` exactly: files from the ref, and nothing the ref lacks."""
    shutil.rmtree(ws / "skills")
    (ws / "skills").mkdir()
    git(ws, "checkout", ref, "--", "skills")
