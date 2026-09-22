"""Algorithm 1 of WikiSkill, line by line. The harness owns gating, rollback, skill-impact.md and the
git audit trail; the wiki is never rolled back. state.json makes every step resumable."""
from __future__ import annotations

import json
from pathlib import Path

from engram.wikiskill.bench import Bench, read_split
from engram.wikiskill.roles import (
    RoleError,
    maintain,
    mean_score,
    propose,
    rollout,
    sample_traces,
)
from engram.wikiskill.workspace import (
    ProposalError,
    append_impact,
    apply_maintainer,
    apply_proposal,
    commit_all,
    restore_skills,
    skill_section,
    skills_dir_section,
    tag,
)


def load_state(ws: Path) -> dict:
    p = ws / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"iteration": 0, "r_best": None, "history": [], "stopped": False}


def save_state(ws: Path, st: dict) -> None:
    (ws / "state.json").write_text(json.dumps(st, indent=1))


def _last_accepted(st: dict) -> str:
    acc = [h["k"] for h in st["history"] if h["outcome"] == "Accepted"]
    return f"accepted-{acc[-1]}" if acc else "accepted-0"


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def evolve(ws: Path, bench: Bench, *, model: str, iters: int, parallel: int, log=print) -> dict:
    train, val = read_split(ws / "dataset/train.jsonl"), read_split(ws / "dataset/val.jsonl")
    st = load_state(ws)
    if st["r_best"] is None:  # line 3: baseline validation with S_0 = ∅
        st["r_best"] = mean_score(rollout(bench, ws, val, skills_text="", model=model, parallel=parallel,
                                          out_dir=ws / "raw/val-0"))
        save_state(ws, st)
        commit_all(ws, f"iter 0: baseline val {st['r_best']:.3f}")
        tag(ws, "accepted-0")  # after the commit: a tag names the tree it must restore
        log(f"baseline R_best={st['r_best']:.3f}")
    for k in range(st["iteration"] + 1, iters + 1):
        if st["r_best"] >= 1.0:  # line 5: early stop
            st["stopped"] = True
            save_state(ws, st)
            log("R_best = 1.0 — early stop")
            break
        restore_skills(ws, _last_accepted(st))  # S_{k-1} exactly, even after a crash mid-iteration
        traces = rollout(bench, ws, train, skills_text=skill_section(ws), model=model, parallel=parallel,
                         out_dir=ws / f"raw/iter-{k}")  # line 8
        log(f"iter {k}: train R={mean_score(traces):.3f}")
        skipped = apply_maintainer(ws, maintain(ws, sample_traces(traces), model=model), k)  # lines 9–10
        commit_all(ws, f"iter {k}: wiki" + (f" ({len(skipped)} patches skipped)" if skipped else ""))
        entry = {"k": k, "action": None, "name": None, "r_val": None, "r_best": st["r_best"], "outcome": None}
        p, diff, r_val = {"action": "invalid", "name": ""}, "", None
        try:
            p = propose(ws, k, traces, model=model)  # line 11
            entry["action"], entry["name"] = p.get("action"), p.get("name")
            if p["action"] == "no_action":
                entry["outcome"] = "NoAction"
            else:
                diff, _ = apply_proposal(ws, p)  # line 12
        except (RoleError, ProposalError) as e:  # only the Proposer's own failures; rollout errors propagate
            entry["outcome"] = "Invalid"
            log(f"iter {k}: invalid proposal: {e}")
            restore_skills(ws, _last_accepted(st))
        if entry["outcome"] is None:
            r_val = mean_score(rollout(bench, ws, val, skills_text=skill_section(ws), model=model,
                                       parallel=parallel, out_dir=ws / f"raw/val-{k}"))  # line 13
            if r_val > st["r_best"]:  # line 14: strict improvement
                st["r_best"] = r_val
                entry["outcome"] = "Accepted"
            else:
                restore_skills(ws, _last_accepted(st))  # line 17: skills only, wiki retained
                entry["outcome"] = "Rejected"
        entry["r_val"], entry["r_best"] = r_val, st["r_best"]
        append_impact(ws, k, p, diff, r_val, st["r_best"], entry["outcome"])  # line 19
        st["history"].append(entry)
        st["iteration"] = k
        save_state(ws, st)
        commit_all(ws, f"iter {k}: {entry['outcome']} val={_fmt(r_val)} best={st['r_best']:.3f}")
        if entry["outcome"] == "Accepted":
            tag(ws, f"accepted-{k}")  # after the commit: the tag must name a tree that holds the skill
        log(f"iter {k}: {entry['outcome']} val={_fmt(r_val)} best={st['r_best']:.3f}")
    return st


def evaluate(ws: Path, bench: Bench, *, split: str, model: str, parallel: int,
             skills_dir: Path | None = None) -> float:
    """R(T_split) for the active skills, or for another workspace's skills/ (Table 2 transfer)."""
    tasks = read_split(ws / f"dataset/{split}.jsonl")
    label = skills_dir.parent.name if skills_dir else "self"
    skills = skills_dir_section(skills_dir) if skills_dir else skill_section(ws)
    return mean_score(rollout(bench, ws, tasks, skills_text=skills, model=model, parallel=parallel,
                              out_dir=ws / f"raw/eval-{split}-{label}"))
