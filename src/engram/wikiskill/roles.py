"""The three LLM roles of §3.2 as thin `run_claude` calls: Inference Agent (rollout), Wiki
Maintainer (one call, JSON), Skill Proposer (ReAct over Read, JSON final). Plus Appendix C sampling."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from engram.wikiskill.bench import Bench, Task
from engram.wikiskill.claude import ClaudeResult, run_claude
from engram.wikiskill.workspace import read_wiki, skill_section

PROMPTS = Path(__file__).parent / "prompts"
_EDIT = {"type": "object", "properties": {"op": {"type": "string", "enum": ["append", "replace", "insert_after"]},
                                          "target": {"type": "string"}, "content": {"type": "string"}},
         "required": ["op", "content"]}
MAINTAINER_SCHEMA = {
    "type": "object",
    "properties": {
        "create_patterns": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "content": {"type": "string"}}, "required": ["name", "content"]}},
        "update_patterns": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "edits": {"type": "array", "items": _EDIT}}, "required": ["name", "edits"]}},
        "update_index": {"type": "string"},
        "append_log": {"type": "string"},
    },
    "required": ["update_index", "append_log"],
}
PROPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "patch", "no_action"]},
        "name": {"type": "string"},
        "skill_md": {"type": "string"},
        "purpose_md": {"type": "string"},
        "edits": {"type": "array", "items": _EDIT},
    },
    "required": ["action"],
}
_PRED = re.compile(r"<answer>\s*([^<]*?)\s*</answer>")


class RoleError(RuntimeError):
    """A role returned no usable structured output."""


class Trace(TypedDict, total=False):
    id: str
    split: str
    prompt: str
    response: str
    answer: str
    gold: str
    score: float
    cost_usd: float
    error: str


def _split_of(out_dir: Path) -> str:
    """iter-<k> → train; val-<k> → val; eval-<split>-<label> → <split>."""
    name = out_dir.name
    if name.startswith("iter-"):
        return "train"
    if name.startswith("eval-"):
        return name.split("-")[1]
    return name.split("-")[0]


def _infer(bench: Bench, ws: Path, task: Task, system: str, model: str, out_dir: Path, split: str) -> Trace:
    f = out_dir / f"{task['id']}.json"
    if f.exists():  # resume: a trace on disk is immutable (Raw Layer), never re-rolled
        return json.loads(f.read_text())
    prompt = bench.user_prompt(task)
    res = run_claude(prompt, system=system, model=model, cwd=ws, tools=list(bench.tools))
    if res.is_error:  # not a measurement: keep a diagnostic, never a scorable trace (resume re-rolls it)
        f.with_suffix(".error.json").write_text(json.dumps({"id": task["id"], "error": res.text, "raw": res.raw},
                                                           ensure_ascii=False, indent=1))
        return {"id": task["id"], "error": res.text}
    hits = _PRED.findall(res.text)
    tr: Trace = {"id": task["id"], "split": split, "prompt": prompt, "response": res.text,
                 "answer": hits[-1] if hits else "", "gold": task["answer"],
                 "score": bench.score(task, res.text), "cost_usd": res.cost_usd}
    f.write_text(json.dumps({**tr, "raw": res.raw}, ensure_ascii=False, indent=1))
    return tr


def rollout(bench: Bench, ws: Path, tasks: list[Task], *, skills_text: str, model: str,
            parallel: int, out_dir: Path) -> list[Trace]:
    """Eq. 1: every task once, with the active skills fully injected and no wiki access.
    Any failed call fails the whole rollout loud — a score with holes would corrupt the gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split, system = _split_of(out_dir), bench.system_prompt(skills_text)
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        traces = list(ex.map(lambda t: _infer(bench, ws, t, system, model, out_dir, split), tasks))
    failed = [t["id"] for t in traces if "error" in t]
    if failed:
        raise RoleError(f"{len(failed)} of {len(tasks)} rollouts failed in {out_dir.name} "
                        f"({failed[0]}: {traces[[t['id'] for t in traces].index(failed[0])]['error'][:120]!r}); "
                        "re-run to retry only the failed tasks")
    return traces


def mean_score(traces: list[Trace]) -> float:
    return sum(t["score"] for t in traces) / len(traces) if traces else 0.0


def sample_traces(traces: list[Trace], *, max_fail: int = 5, max_pass: int = 3, cap_chars: int = 15000) -> list[Trace]:
    """Appendix C: ≤5 failing + ≤3 passing, each log capped, stable by id."""
    ordered = sorted(traces, key=lambda t: t["id"])
    fails = [t for t in ordered if t["score"] < 1.0][:max_fail]
    passes = [t for t in ordered if t["score"] >= 1.0][:max_pass]
    return [{**t, "response": t["response"][:cap_chars]} for t in fails + passes]


def _trace_block(t: Trace) -> str:
    status = "PASS" if t["score"] >= 1.0 else "FAIL"
    return (f"### Task {t['id']} — {status} (pred={t['answer']!r}, gold={t['gold']!r})\n"
            f"#### Prompt\n{t['prompt']}\n#### Agent output\n{t['response']}\n")


def _log_role(log_to: Path | None, res: ClaudeResult) -> None:
    """Raw Layer for the optimizer roles: what they returned, turns and the raw usage (audit + budget)."""
    if log_to is not None:
        log_to.parent.mkdir(parents=True, exist_ok=True)
        log_to.write_text(json.dumps({"cost_usd": res.cost_usd, "turns": res.turns, "is_error": res.is_error,
                                      "structured": res.structured, "text": res.text, "raw": res.raw},
                                     ensure_ascii=False, indent=1))


def maintain(ws: Path, sample: list[Trace], *, model: str, log_to: Path | None = None) -> dict:
    """Eq. 2: one call over the sampled traces + the full current wiki."""
    prompt = ("# Execution traces\n\n" + "\n".join(_trace_block(t) for t in sample)
              + "\n\n# Current wiki\n\n" + read_wiki(ws))
    res = run_claude(prompt, system=(PROMPTS / "maintainer.md").read_text(), model=model, cwd=ws,
                     tools=[], json_schema=MAINTAINER_SCHEMA)
    _log_role(log_to, res)
    out = res.structured
    if not isinstance(out, dict) or "update_index" not in out:
        raise RoleError(f"maintainer returned no structured output: {res.text[:200]!r}")
    out.setdefault("create_patterns", [])
    out.setdefault("update_patterns", [])
    return out


def outcome_summary(traces: list[Trace]) -> str:
    return "\n".join(f"{t['id']}\t{'PASS' if t['score'] >= 1.0 else 'FAIL'}\tpred={t['answer']}\tgold={t['gold']}"
                     for t in sorted(traces, key=lambda t: t["id"]))


def propose(ws: Path, k: int, traces: list[Trace], *, model: str, max_turns: int = 25,
            task_desc: str = "multiple-choice mathematics questions", log_to: Path | None = None) -> dict:
    """Eq. 3: a ReAct agent over the wiki index, the impact tracker and the train outcomes; it
    reads pattern pages and raw traces itself and ends with one atomic proposal."""
    system = (PROMPTS / "proposer.md").read_text().replace("{iter}", str(k)).replace("{task_desc}", task_desc)
    active = skill_section(ws) or "(none — the skill set is empty; only `create` or `no_action` apply)\n"
    prompt = (f"# wiki/index.md\n\n{(ws / 'wiki/index.md').read_text()}\n\n"
              f"# wiki/skill-impact.md\n\n{(ws / 'wiki/skill-impact.md').read_text()}\n\n"
              f"# Active skills (S_{{k-1}})\n\n{active}\n"
              f"# Training outcomes (iteration {k})\n\n{outcome_summary(traces)}\n")
    res = run_claude(prompt, system=system, model=model, cwd=ws, tools=["Read"],
                     max_turns=max_turns, json_schema=PROPOSER_SCHEMA)
    _log_role(log_to, res)
    if not isinstance(res.structured, dict) or "action" not in res.structured:
        raise RoleError(f"proposer returned no structured output: {res.text[:200]!r}")
    return res.structured
