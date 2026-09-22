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

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900):
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

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900):
        seen.update(prompt=prompt, system=system, tools=tools, schema=json_schema, cwd=cwd)
        return ClaudeResult(text="", structured={"update_index": "# i", "append_log": "l"},
                            turns=1, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    out = r.maintain(ws, [_trace(0, 0.0)], model="haiku")
    assert out["update_index"] == "# i" and out["create_patterns"] == [] and out["update_patterns"] == []
    assert "Wiki Maintainer Agent" in seen["system"] and seen["tools"] == [] and seen["cwd"] == ws
    assert "wiki/patterns/p.md" in seen["prompt"] and "t0" in seen["prompt"]
    assert seen["schema"] is r.MAINTAINER_SCHEMA


def test_roles_log_raw_result_with_cost(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    outs = iter([{"update_index": "# i", "append_log": "l"}, {"action": "no_action"}])
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="t", structured=next(outs), turns=4, cost_usd=0.02, is_error=False, raw={"total_cost_usd": 0.02}))
    r.maintain(ws, [], model="m", log_to=ws / "raw/roles/iter-1-maintainer.json")
    r.propose(ws, 1, [], model="m", log_to=ws / "raw/roles/iter-1-proposer.json")
    for role in ("maintainer", "proposer"):
        d = json.loads((ws / f"raw/roles/iter-1-{role}.json").read_text())
        assert d["cost_usd"] == 0.02 and d["turns"] == 4 and d["structured"]


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

    def fake(prompt, *, system, model, cwd, tools, max_turns=None, json_schema=None, timeout=900):
        seen.update(prompt=prompt, system=system, tools=tools, max_turns=max_turns, schema=json_schema)
        return ClaudeResult(text="", structured={"action": "no_action"}, turns=5, cost_usd=0, is_error=False)

    monkeypatch.setattr(r, "run_claude", fake)
    p = r.propose(ws, 2, [_trace(0, 0.0), _trace(1, 1.0)], model="haiku")
    assert p == {"action": "no_action"}
    assert seen["tools"] == ["Read"] and seen["max_turns"] == 25 and seen["schema"] is r.PROPOSER_SCHEMA
    assert "raw/iter-2/" in seen["system"] and "{iter}" not in seen["system"] and "{task_desc}" not in seen["system"]
    assert "Rejected" in seen["prompt"] and "t0\tFAIL\tpred=A\tgold=B" in seen["prompt"] and "t1\tPASS" in seen["prompt"]


def test_propose_missing_structured_raises(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="ran out of turns", structured=None, turns=25, cost_usd=0, is_error=False))
    with pytest.raises(r.RoleError):
        r.propose(ws, 1, [], model="haiku")
