import json

import pytest

from engram.wikiskill import loop as L
from engram.wikiskill import roles as r
from engram.wikiskill import workspace as w
from engram.wikiskill.bench import write_split


class FakeBench:
    name, tools = "fake", []

    def system_prompt(self, s):
        return "SYS\n" + s

    def user_prompt(self, t):
        return t["question"]

    def score(self, t, resp):
        return 1.0 if resp == t["answer"] else 0.0


def _ws(tmp_path, n_train=4, n_val=2):
    ws = tmp_path / "ws"
    w.init_workspace(ws)

    def mk(i):
        return {"id": f"t{i}", "question": f"q{i}", "choices": {}, "answer": "A"}

    write_split(ws / "dataset/train.jsonl", [mk(i) for i in range(n_train)])
    write_split(ws / "dataset/val.jsonl", [mk(100 + i) for i in range(n_val)])
    write_split(ws / "dataset/test.jsonl", [mk(200 + i) for i in range(3)])
    return ws


def _wire(monkeypatch, val_scores, proposals, maint=None):
    """val_scores: per validation rollout (baseline first) fraction correct; proposals: per iteration."""
    it = {"val": 0, "prop": 0}

    def fake_rollout(bench, ws, tasks, *, skills_text, model, parallel, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir.name.startswith("val"):
            frac = val_scores[it["val"]]
            it["val"] += 1
        else:
            frac = 0.5
        n_ok = round(frac * len(tasks))
        trs = []
        for i, t in enumerate(tasks):
            resp = "A" if i < n_ok else "B"
            trs.append({"id": t["id"], "split": "x", "prompt": "p", "response": resp, "answer": resp,
                        "gold": "A", "score": bench.score(t, resp), "cost_usd": 0.0})
        return trs

    def fake_propose(ws, k, traces, *, model, **kw):
        p = proposals[it["prop"]]
        it["prop"] += 1
        return p

    monkeypatch.setattr(L, "rollout", fake_rollout)
    monkeypatch.setattr(L, "propose", fake_propose)
    monkeypatch.setattr(L, "maintain", maint or (lambda ws, sample, *, model, **kw: {
        "create_patterns": [{"name": "p.md", "content": "pat"}], "update_patterns": [],
        "update_index": "# idx\n- p", "append_log": "found p"}))
    return it


CREATE = {"action": "create", "name": "s1", "skill_md": "---\nname: s1\n---\nrule\n", "purpose_md": "why"}
PATCH = {"action": "patch", "name": "s1", "edits": [{"op": "append", "content": "rule2"}]}
QUIET = {"log": lambda *a: None}


def test_gate_accept_and_early_stop(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 1.0, 0.5], proposals=[CREATE, PATCH])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, **QUIET)
    # iter 1: val 1.0 > 0.5 → Accepted and R_best hits 1.0 → the loop stops before iter 2
    assert st["r_best"] == 1.0 and st["stopped"] is True
    assert [h["outcome"] for h in st["history"]] == ["Accepted"]
    impact = (ws / "wiki/skill-impact.md").read_text()
    assert "— Accepted" in impact and "+rule" in impact
    assert (ws / "skills/s1/SKILL.md").read_text().endswith("rule\n")
    assert "accepted-1" in w.git(ws, "tag")
    # the tag names a tree that actually holds the accepted skill (a rollback restores it)
    assert "skills/s1/SKILL.md" in w.git(ws, "ls-tree", "-r", "--name-only", "accepted-1")
    assert (ws / "wiki/patterns/p.md").exists() and "found p" in (ws / "wiki/log.md").read_text()
    assert (ws / "raw/val-0").is_dir() and (ws / "raw/iter-1").is_dir() and (ws / "raw/val-1").is_dir()
    assert st["history"][0]["cost_usd"] == 0.0  # fakes cost nothing; real runs sum traces + role logs


def test_iteration_cost_sums_traces_and_role_logs(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    for rel, cost in (("raw/iter-2/t0.json", 0.01), ("raw/iter-2/t1.json", 0.02), ("raw/val-2/v0.json", 0.03),
                      ("raw/roles/iter-2-maintainer.json", 0.1), ("raw/roles/iter-2-proposer.json", 0.2),
                      ("raw/iter-3/t0.json", 9.0), ("raw/iter-2/t2.error.json", 9.0)):
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(json.dumps({"cost_usd": cost}))
    assert round(L.iteration_cost(ws, 2), 6) == 0.36


def test_reject_rolls_back_skills_keeps_wiki(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 0.5, 1.0], proposals=[CREATE, CREATE])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, **QUIET)
    assert [h["outcome"] for h in st["history"]] == ["Rejected", "Accepted"]
    impact = (ws / "wiki/skill-impact.md").read_text()
    assert impact.count("## iter") == 2 and "— Rejected" in impact and "+rule" in impact
    assert (ws / "wiki/patterns/p.md").exists()  # wiki never rolled back
    assert "accepted-0" in w.git(ws, "tag")
    # the rejected iter-1 commit carries an empty skills/ (rolled back before the commit)
    assert "SKILL.md" not in w.git(ws, "ls-tree", "-r", "--name-only", "accepted-0")
    assert "SKILL.md" not in w.git(ws, "ls-tree", "-r", "--name-only", "HEAD~1")


def test_strict_gate_equal_is_rejected(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 0.5], proposals=[CREATE])
    st = L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    assert st["history"][0]["outcome"] == "Rejected" and not (ws / "skills/s1").exists()


def test_no_action_and_invalid_proposal(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    it = _wire(monkeypatch, val_scores=[0.5],
               proposals=[{"action": "no_action"}, {"action": "patch", "name": "ghost", "edits": []}])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, **QUIET)
    assert [h["outcome"] for h in st["history"]] == ["NoAction", "Invalid"]
    assert it["val"] == 1  # no validation rollouts spent
    impact = (ws / "wiki/skill-impact.md").read_text()
    assert "— NoAction" in impact and "— Invalid" in impact


def test_resume_from_state(monkeypatch, tmp_path):
    ws = _ws(tmp_path, n_val=4)  # 4 val tasks: fractions 0.5 / 0.75 / 1.0 are exactly representable
    it = _wire(monkeypatch, val_scores=[0.5, 0.75, 1.0], proposals=[CREATE, PATCH])
    L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    st = L.load_state(ws)
    assert st["iteration"] == 1 and st["r_best"] == 0.75
    L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, **QUIET)
    st = L.load_state(ws)
    assert st["iteration"] == 2 and st["r_best"] == 1.0 and it["val"] == 3
    assert (ws / "skills/s1/SKILL.md").read_text().endswith("rule\nrule2\n")
    assert "accepted-2" in w.git(ws, "tag")


def test_role_error_is_invalid_not_crash(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    _wire(monkeypatch, val_scores=[0.5], proposals=[])

    def boom(ws, k, traces, *, model, **kw):
        raise r.RoleError("no output")

    monkeypatch.setattr(L, "propose", boom)
    st = L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    assert st["history"][0]["outcome"] == "Invalid"


def test_maintainer_error_aborts_iteration_cleanly(monkeypatch, tmp_path):
    ws = _ws(tmp_path)

    def boom(ws, sample, *, model, **kw):
        raise r.RoleError("no wiki output")

    _wire(monkeypatch, val_scores=[0.5], proposals=[CREATE], maint=boom)
    with pytest.raises(r.RoleError):
        L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    st = L.load_state(ws)
    assert st["iteration"] == 0 and st["r_best"] == 0.5  # baseline persisted; iteration 1 not recorded


def test_crash_after_apply_does_not_leak_candidate_into_next_run(monkeypatch, tmp_path):
    """Val rollout dies after the candidate skill was written: on resume the iteration restarts
    from S_{k-1} (skills restored from the last accepted tag), not from the un-gated candidate."""
    ws = _ws(tmp_path)
    it = _wire(monkeypatch, val_scores=[0.5, 0.5], proposals=[CREATE, {"action": "no_action"}])
    real_rollout = L.rollout
    seen_train_skills = []

    def dying_rollout(bench, ws_, tasks, *, skills_text, model, parallel, out_dir):
        if out_dir.name == "val-1":
            raise r.RoleError("2 of 2 rollouts failed")
        if out_dir.name.startswith("iter-"):
            seen_train_skills.append(skills_text)
        return real_rollout(bench, ws_, tasks, skills_text=skills_text, model=model, parallel=parallel,
                            out_dir=out_dir)

    monkeypatch.setattr(L, "rollout", dying_rollout)
    with pytest.raises(r.RoleError):
        L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    assert (ws / "skills/s1/SKILL.md").exists()  # the crash left the candidate on disk
    assert L.load_state(ws)["iteration"] == 0
    L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, **QUIET)
    assert seen_train_skills == ["", ""]  # both attempts trained on S_0, never on the candidate
    assert not (ws / "skills/s1").exists() and it["prop"] == 2


def test_evaluate_self_and_transfer(monkeypatch, tmp_path):
    ws = _ws(tmp_path)
    seen = {}

    def fake_rollout(bench, ws_, tasks, *, skills_text, model, parallel, out_dir):
        seen["skills"], seen["out"] = skills_text, out_dir
        return [{"id": t["id"], "split": "test", "prompt": "", "response": "A", "answer": "A",
                 "gold": "A", "score": 1.0, "cost_usd": 0} for t in tasks]

    monkeypatch.setattr(L, "rollout", fake_rollout)
    assert L.evaluate(ws, FakeBench(), split="test", model="m", parallel=1) == 1.0
    assert seen["skills"] == "" and seen["out"].name == "eval-test-self"
    other = tmp_path / "other/skills/z"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("Z")
    L.evaluate(ws, FakeBench(), split="test", model="m", parallel=1, skills_dir=other.parent)
    assert "Z" in seen["skills"] and seen["out"].name == "eval-test-other"
