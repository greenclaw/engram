import json

import pytest

from engram.wikiskill import roles as r
from engram.wikiskill import workspace as w
from engram.wikiskill.bench import LiveMath
from engram.wikiskill.claude import ClaudeResult


def _task(i, ans="B"):
    return {"id": f"t{i}", "question": f"q{i}", "choices": {"A": "a", "B": "b"}, "answer": ans}


def _trace(i, score, resp="x"):
    return {"id": f"t{i}", "split": "train", "prompt": "p", "response": resp, "answer": "A",
            "gold": "B", "score": score, "cost_usd": 0.0}


def test_rollout_writes_traces_and_resumes(monkeypatch, tmp_path):
    calls = []

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900, **kw):
        calls.append((prompt, system, tools))
        return ClaudeResult(text="<answer>B</answer>", structured=None, turns=1, cost_usd=0.001, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    out = tmp_path / "raw/iter-1"
    tasks = [_task(0), _task(1, "A")]
    traces = r.rollout(LiveMath(), tmp_path, tasks, skills_text="## Skills\nS", model="haiku", parallel=2, out_dir=out)
    assert [t["score"] for t in traces] == [1.0, 0.0]
    assert [t["split"] for t in traces] == ["train", "train"]
    assert (out / "t0.json").exists() and json.loads((out / "t1.json").read_text())["gold"] == "A"
    assert all("## Skills\nS" in s and tools == [] for _, s, tools in calls)
    assert r.mean_score(traces) == 0.5
    calls.clear()
    traces2 = r.rollout(LiveMath(), tmp_path, tasks, skills_text="", model="haiku", parallel=2, out_dir=out)
    assert calls == [] and [t["id"] for t in traces2] == ["t0", "t1"]  # resumed from disk, same order


def test_traces_and_role_logs_carry_no_derived_usd(monkeypatch, tmp_path):
    """Consumption is tokens (usage_of reads raw.usage); the harness must not write its own USD field.
    The raw claude JSON keeps whatever claude returned, untouched."""
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    res = ClaudeResult(text="<answer>B</answer>", structured={"update_index": "# i", "append_log": "l"},
                       turns=1, cost_usd=0.05, is_error=False, raw={"total_cost_usd": 0.05, "usage": {"output_tokens": 9}})
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: res)
    r.rollout(LiveMath(), ws, [_task(0)], skills_text="", model="m", parallel=1, out_dir=ws / "raw/iter-1")
    r.maintain(ws, [], model="m", log_to=ws / "raw/roles/iter-1-maintainer.json")
    for f in (ws / "raw/iter-1/t0.json", ws / "raw/roles/iter-1-maintainer.json"):
        d = json.loads(f.read_text())
        assert "cost_usd" not in d and d["raw"]["usage"]["output_tokens"] == 9


class ToolBench:
    """A bench with a per-task workdir, tools and a file-based grade (the SpreadsheetBench shape)."""
    name, task_desc = "toolbench", "file tasks"

    def system_prompt(self, s):
        return "SYS" + s

    def prepare(self, ws, t, workdir):
        (workdir / "input.txt").write_text(t["question"])

    def user_prompt(self, t, workdir):
        return f"edit {workdir}/input.txt"

    def claude_opts(self, ws, workdir):
        return {"tools": ["Bash"], "allowed_tools": ["Bash"], "stream": True, "max_turns": 30,
                "settings": {"sandbox": {"filesystem": {"allowRead": [str(workdir)]}}}}

    def score(self, ws, t, resp, workdir):
        ok = (workdir / "output.txt").exists()
        return (1.0 if ok else 0.0), ("ok" if ok else "File not exist")


def test_rollout_runs_each_task_in_its_own_prepared_workdir(monkeypatch, tmp_path):
    seen = []

    def fake(prompt, **kw):
        seen.append(kw)
        (kw["cwd"] / "output.txt").write_text("done")
        return ClaudeResult(text="Saved", structured=None, turns=5, cost_usd=0, is_error=False,
                            raw={"usage": {"output_tokens": 7}}, transcript="$ ls\ninput.txt\nSaved")

    monkeypatch.setattr(r, "run_claude", fake)
    tr = r.rollout(ToolBench(), tmp_path, [_task(0)], skills_text="", model="m", parallel=1,
                   out_dir=tmp_path / "raw/iter-1")
    wd = tmp_path / "work/iter-1/t0"
    assert seen[0]["cwd"] == wd and (wd / "input.txt").read_text() == "q0"
    assert seen[0]["tools"] == ["Bash"] and seen[0]["stream"] is True and seen[0]["max_turns"] == 30
    assert seen[0]["settings"]["sandbox"]["filesystem"]["allowRead"] == [str(wd)]
    assert tr[0]["score"] == 1.0 and tr[0]["answer"] == "ok" and tr[0]["response"].startswith("$ ls")
    assert tr[0]["prompt"] == f"edit {wd}/input.txt"


def test_rollout_split_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="", structured=None, turns=1, cost_usd=0, is_error=False))
    for name, split in (("val-3", "val"), ("eval-test-self", "test")):
        tr = r.rollout(LiveMath(), tmp_path, [_task(0)], skills_text="", model="m", parallel=1, out_dir=tmp_path / name)
        assert tr[0]["split"] == split


def test_rollout_error_fails_loud_and_is_not_persisted(monkeypatch, tmp_path):
    """A failed claude call is not a measurement: no <id>.json (so resume re-rolls it), a diagnostic
    <id>.error.json instead, and the rollout raises so no gate decision is made on a bogus 0."""
    def flaky(prompt, **kw):
        if prompt.startswith("q0"):
            return ClaudeResult(text="Not logged in", structured=None, turns=0, cost_usd=0, is_error=True)
        return ClaudeResult(text="<answer>A</answer>", structured=None, turns=1, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", flaky)
    with pytest.raises(r.RoleError, match="1 of 2"):
        r.rollout(LiveMath(), tmp_path, [_task(0), _task(1)], skills_text="", model="haiku", parallel=1,
                  out_dir=tmp_path / "o")
    assert not (tmp_path / "o/t0.json").exists()
    assert "Not logged in" in (tmp_path / "o/t0.error.json").read_text()
    # the healthy sibling was still persisted — a retry only re-rolls the failed one
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="<answer>B</answer>", structured=None, turns=1, cost_usd=0, is_error=False))
    tr = r.rollout(LiveMath(), tmp_path, [_task(0), _task(1)], skills_text="", model="haiku", parallel=1,
                   out_dir=tmp_path / "o")
    assert [t["score"] for t in tr] == [1.0, 0.0]


def test_sample_traces_budget_and_cap():
    traces = [_trace(i, 0.0, resp="f" * 20000) for i in range(9)] + [_trace(i, 1.0) for i in range(9, 15)]
    s = r.sample_traces(traces)
    fails = [t for t in s if t["score"] < 1.0]
    passes = [t for t in s if t["score"] == 1.0]
    assert len(fails) == 5 and len(passes) == 3
    assert all(len(t["response"]) == 15000 for t in fails)
    assert [t["id"] for t in fails] == ["t0", "t1", "t2", "t3", "t4"]


def test_maintain_builds_prompt_and_validates(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "wiki/patterns/p.md").write_text("pat")
    seen = {}

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900, **kw):
        seen.update(prompt=prompt, system=system, tools=tools, schema=json_schema, cwd=cwd)
        return ClaudeResult(text="", structured={"update_index": "# i", "append_log": "l"},
                            turns=1, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    out = r.maintain(ws, [_trace(0, 0.0)], model="haiku")
    assert out["update_index"] == "# i" and out["create_patterns"] == [] and out["update_patterns"] == []
    assert "Wiki Maintainer Agent" in seen["system"] and seen["tools"] == [] and seen["cwd"] == ws
    assert "wiki/patterns/p.md" in seen["prompt"] and "t0" in seen["prompt"]
    assert seen["schema"] is r.MAINTAINER_SCHEMA


def test_roles_log_raw_result(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    outs = iter([{"update_index": "# i", "append_log": "l"}, {"action": "no_action"}])
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="t", structured=next(outs), turns=4, cost_usd=0.02, is_error=False, raw={"total_cost_usd": 0.02}))
    r.maintain(ws, [], model="m", log_to=ws / "raw/roles/iter-1-maintainer.json")
    r.propose(ws, 1, [], model="m", log_to=ws / "raw/roles/iter-1-proposer.json")
    for role in ("maintainer", "proposer"):
        d = json.loads((ws / f"raw/roles/iter-1-{role}.json").read_text())
        assert d["turns"] == 4 and d["structured"] and d["raw"] == {"total_cost_usd": 0.02}


def test_maintain_missing_structured_raises(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="prose", structured=None, turns=1, cost_usd=0, is_error=False))
    with pytest.raises(r.RoleError):
        r.maintain(ws, [], model="haiku")


def test_propose_prompt_and_result(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "wiki/skill-impact.md").write_text("# Skill Impact\n\n## iter 1 — create x — Rejected\n")
    seen = {}

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900, **kw):
        seen.update(prompt=prompt, system=system, tools=tools, max_turns=max_turns, schema=json_schema)
        return ClaudeResult(text="", structured={"action": "no_action"}, turns=5, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    p = r.propose(ws, 2, [_trace(0, 0.0), _trace(1, 1.0)], model="haiku")
    assert p == {"action": "no_action"}
    assert seen["tools"] == ["Read"] and seen["max_turns"] == 25 and seen["schema"] is r.PROPOSER_SCHEMA
    assert "raw/iter-2/" in seen["system"] and "{iter}" not in seen["system"] and "{task_desc}" not in seen["system"]
    assert "Rejected" in seen["prompt"] and "t0\tFAIL\tpred=A\tgold=B" in seen["prompt"] and "t1\tPASS" in seen["prompt"]


def test_propose_is_conditioned_on_active_skills(monkeypatch, tmp_path):
    """Eq. 3: P_k ← M_P(W'_k, S_{k-1}, T_train). Live smoke: without S_{k-1} in the prompt the Proposer
    inferred the skill set from skill-impact.md and patched a skill that had been rolled back."""
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    seen = []

    def fake(prompt, **kw):
        seen.append(prompt)
        return ClaudeResult(text="", structured={"action": "no_action"}, turns=1, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    r.propose(ws, 1, [], model="m")
    assert "# Active skills (S_{k-1})" in seen[0] and "none — the skill set is empty" in seen[0]
    (ws / "skills/kept").mkdir()
    (ws / "skills/kept/SKILL.md").write_text("KEEP-RULE")
    r.propose(ws, 2, [], model="m")
    assert "### kept" in seen[1] and "KEEP-RULE" in seen[1] and "none — the skill set is empty" not in seen[1]


def test_propose_missing_structured_raises(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="ran out of turns", structured=None, turns=25, cost_usd=0, is_error=False))
    with pytest.raises(r.RoleError):
        r.propose(ws, 1, [], model="haiku")


def test_proposer_cannot_read_validation_test_or_gold_and_its_reads_are_logged(monkeypatch, tmp_path):
    """§3.1: the Raw Layer the Proposer inspects holds TRAINING traces. Its Read tool is not sandboxed,
    so validation/eval traces, the splits (LiveMath test answers) and the data dir (golden files) are
    denied explicitly; the session is streamed so every file it read is in the role log."""
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return ClaudeResult(text="", structured={"action": "no_action"}, turns=3, cost_usd=0, is_error=False,
                            raw={}, transcript="[Read] {\"file_path\": \"wiki/index.md\"}")

    monkeypatch.setattr(r, "run_claude", fake)
    r.propose(ws, 1, [], model="m", log_to=ws / "raw/roles/iter-1-proposer.json")
    deny = seen["settings"]["permissions"]["deny"]
    for rel in ("raw/val-*", "raw/eval-*", ".data", "work", "dataset", ".hf-cache", ".venv"):
        assert f"Read(/{ws / rel}/**)" in deny
    assert not any("raw/iter" in d or "wiki" in d or "skills" in d for d in deny)
    assert seen["stream"] is True and seen["tools"] == ["Read"]
    assert "wiki/index.md" in json.loads((ws / "raw/roles/iter-1-proposer.json").read_text())["transcript"]


def test_rerolled_task_starts_from_a_clean_workdir(monkeypatch, tmp_path):
    """A crash after the agent wrote output.xlsx but before the trace was saved re-rolls the task; the
    stale output must not survive into the new attempt, or the grader credits the previous run."""
    stale = tmp_path / "work/iter-1/t0"
    stale.mkdir(parents=True)
    (stale / "output.txt").write_text("from a crashed attempt")

    def fake(prompt, **kw):
        return ClaudeResult(text="gave up", structured=None, turns=2, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    tr = r.rollout(ToolBench(), tmp_path, [_task(0)], skills_text="", model="m", parallel=1,
                   out_dir=tmp_path / "raw/iter-1")
    assert tr[0]["score"] == 0.0 and tr[0]["answer"] == "File not exist"
    assert sorted(p.name for p in stale.iterdir()) == ["input.txt"]
