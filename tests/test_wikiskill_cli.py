import argparse
import json

from engram import cli
from engram.wikiskill import cli as wcli
from engram.wikiskill import workspace as w


def test_parser_wired():
    p = argparse.ArgumentParser()
    wcli.add_evolve_parser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["evolve", "run", "--ws", "x", "--iters", "3"])
    assert a.evolve_cmd == "run" and a.iters == 3 and a.model == "haiku" and a.parallel == 8
    a = p.parse_args(["evolve", "eval", "--ws", "x", "--skills", "y"])
    assert a.split == "test" and a.skills == "y"


def test_engram_main_dispatches(monkeypatch):
    called = {}
    monkeypatch.setattr(wcli, "run_evolve", lambda args: called.update(cmd=args.evolve_cmd) or 0)
    assert cli.main(["evolve", "status", "--ws", "x"]) == 0  # no --dir, no memory store needed
    assert called["cmd"] == "status"


def test_init_writes_splits(monkeypatch, tmp_path):
    recs = [{"no": i, "month": "202606", "mcq": {"question": f"q{i}", "correct_choice": {"label": "A", "text": "r"},
             "choices": [{"label": lab, "text": "d"} for lab in "BCDE"]}} for i in range(200)]
    monkeypatch.setattr(wcli.LiveMath, "download_records", lambda self, cache_dir: recs)
    ws = tmp_path / "ws"
    rc = wcli.run_evolve(argparse.Namespace(evolve_cmd="init", ws=str(ws), bench="livemath", seed=0))
    assert rc == 0
    assert len((ws / "dataset/train.jsonl").read_text().splitlines()) == 35
    assert len((ws / "dataset/test.jsonl").read_text().splitlines()) == 124
    assert json.loads((ws / "dataset/meta.json").read_text())["bench"] == "livemath"
    assert ".hf-cache" in (ws / ".gitignore").read_text()
    assert w.git(ws, "status", "--porcelain").strip() == ""


def test_status_prints_history(tmp_path, capsys):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "dataset/meta.json").write_text(json.dumps({"bench": "livemath", "seed": 0}))
    (ws / "state.json").write_text(json.dumps({"iteration": 1, "r_best": 0.5, "stopped": False,
        "history": [{"k": 1, "action": "create", "name": "s", "r_val": 0.5, "r_best": 0.5, "outcome": "Accepted"}]}))
    assert wcli.run_evolve(argparse.Namespace(evolve_cmd="status", ws=str(ws))) == 0
    out = capsys.readouterr().out
    assert "Accepted" in out and "0.500" in out


def test_eval_resolves_skills_dir(monkeypatch, tmp_path, capsys):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "dataset/meta.json").write_text(json.dumps({"bench": "livemath", "seed": 0}))
    other = tmp_path / "other"
    (other / "skills").mkdir(parents=True)
    seen = {}

    def fake_eval(ws_, bench, *, split, model, parallel, skills_dir=None):
        seen["skills_dir"] = skills_dir
        return 0.25

    monkeypatch.setattr(wcli, "evaluate", fake_eval)
    rc = wcli.run_evolve(argparse.Namespace(evolve_cmd="eval", ws=str(ws), split="test", model="haiku",
                                            parallel=2, skills=str(other)))
    assert rc == 0 and seen["skills_dir"] == other / "skills"
    assert "R(test) = 0.250 (skills: other)" in capsys.readouterr().out
