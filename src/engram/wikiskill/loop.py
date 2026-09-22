"""Algorithm 1 of WikiSkill, line by line. The harness owns gating, rollback, skill-impact.md and the
git audit trail; the wiki is never rolled back. state.json makes every step resumable."""
from __future__ import annotations

import hashlib
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


class EvolveError(RuntimeError):
    """A run that would corrupt the experiment (e.g. resuming with a different model)."""


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


def iteration_cost(ws: Path, k: int) -> float:
    """USD spent in iteration k: train + val traces and the Maintainer/Proposer role logs.
    Error diagnostics are excluded (a failed call re-rolls; its retry is the counted one)."""
    files = [*(ws / f"raw/iter-{k}").glob("*.json"), *(ws / f"raw/val-{k}").glob("*.json"),
             *(ws / "raw/roles").glob(f"iter-{k}-*.json")]
    return sum(float(json.loads(f.read_text()).get("cost_usd", 0.0))
               for f in files if not f.name.endswith(".error.json"))


def _baseline(ws: Path, bench: Bench, st: dict, val: list, *, model: str, parallel: int, log) -> None:
    """Algorithm 1 line 3: R_best = R(val) with S_0 = ∅. Tag before recording, so a state.json that
    names r_best always has its accepted-0 tag (a crash in between just re-reads the traces)."""
    r = mean_score(rollout(bench, ws, val, skills_text="", model=model, parallel=parallel, out_dir=ws / "raw/val-0"))
    commit_all(ws, "iter 0: baseline traces")
    tag(ws, "accepted-0")
    st["r_best"] = r
    save_state(ws, st)
    commit_all(ws, f"iter 0: baseline val {r:.3f}")
    log(f"baseline R_best={r:.3f}")


def evolve(ws: Path, bench: Bench, *, model: str, iters: int, parallel: int, log=print) -> dict:
    """Run Algorithm 1 up to iteration `iters`. Resumable at every step: traces resume from raw/,
    and st["pending"] records a finished Maintainer step and the stored proposal of the running
    iteration, so a crash (e.g. a failed val rollout) never re-applies the wiki or re-proposes."""
    train, val = read_split(ws / "dataset/train.jsonl"), read_split(ws / "dataset/val.jsonl")
    st = load_state(ws)
    if st.setdefault("model", model) != model:  # one evolution = one model (self-evolution, §4.1)
        raise EvolveError(f"workspace was evolved with --model {st['model']}; resuming with {model} would "
                          "mix models in one run — use the same --model or a new --ws")
    if st["r_best"] is None:
        _baseline(ws, bench, st, val, model=model, parallel=parallel, log=log)
    roles = ws / "raw/roles"
    for k in range(st["iteration"] + 1, iters + 1):
        if st["r_best"] >= 1.0:  # line 5: early stop
            st["stopped"] = True
            save_state(ws, st)
            log("R_best = 1.0 — early stop")
            break
        pend = st.get("pending") if (st.get("pending") or {}).get("k") == k else None
        restore_skills(ws, _last_accepted(st))  # S_{k-1} exactly, even after a crash mid-iteration
        traces = rollout(bench, ws, train, skills_text=skill_section(ws), model=model, parallel=parallel,
                         out_dir=ws / f"raw/iter-{k}")  # line 8
        log(f"iter {k}: train R={mean_score(traces):.3f}")
        if pend is None:
            out = maintain(ws, sample_traces(traces), model=model, log_to=roles / f"iter-{k}-maintainer.json")
            skipped = apply_maintainer(ws, out, k)  # lines 9–10
            pend = st["pending"] = {"k": k, "proposal": None}
            save_state(ws, st)
            commit_all(ws, f"iter {k}: wiki" + (f" ({len(skipped)} patches skipped)" if skipped else ""))
        entry = {"k": k, "action": None, "name": None, "r_val": None, "r_best": st["r_best"], "outcome": None}
        p, diff, r_val = {"action": "invalid", "name": ""}, "", None
        try:
            p = pend["proposal"] or propose(ws, k, traces, model=model, log_to=roles / f"iter-{k}-proposer.json")
            pend["proposal"] = p  # line 11 — stored, so a crash below never re-proposes
            save_state(ws, st)
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
                commit_all(ws, f"iter {k}: accept {p.get('name')}")
                tag(ws, f"accepted-{k}")  # before state.json records it: the tag must exist and hold the skill
                st["r_best"] = r_val
                entry["outcome"] = "Accepted"
            else:
                restore_skills(ws, _last_accepted(st))  # line 17: skills only, wiki retained
                entry["outcome"] = "Rejected"
        entry["r_val"], entry["r_best"], entry["cost_usd"] = r_val, st["r_best"], iteration_cost(ws, k)
        append_impact(ws, k, p, diff, r_val, st["r_best"], entry["outcome"])  # line 19
        st.pop("pending", None)
        st["history"].append(entry)
        st["iteration"] = k
        save_state(ws, st)
        commit_all(ws, f"iter {k}: {entry['outcome']} val={_fmt(r_val)} best={st['r_best']:.3f}")
        log(f"iter {k}: {entry['outcome']} val={_fmt(r_val)} best={st['r_best']:.3f}")
    return st


def evaluate(ws: Path, bench: Bench, *, split: str, model: str, parallel: int,
             skills_dir: Path | None = None) -> float:
    """R(T_split) for the active skills, or for another workspace's skills/ (Table 2 transfer).
    The trace cache is keyed by skill CONTENT: a later eval with evolved skills never reuses the
    no-skill traces of an earlier eval of the same split."""
    tasks = read_split(ws / f"dataset/{split}.jsonl")
    source = skills_dir.parent.name if skills_dir else "self"
    skills = skills_dir_section(skills_dir) if skills_dir else skill_section(ws)
    key = hashlib.sha1(skills.encode()).hexdigest()[:10] if skills else "noskill"
    return mean_score(rollout(bench, ws, tasks, skills_text=skills, model=model, parallel=parallel,
                              out_dir=ws / f"raw/eval-{split}-{source}-{key}"))
