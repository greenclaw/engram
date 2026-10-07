import argparse
import json
from pathlib import Path

from engram.wikiskill import cli as wcli
from engram.wikiskill import probe as pr
from engram.wikiskill import workspace as w
from engram.wikiskill.bench import LiveMath
from engram.wikiskill.claude import ClaudeResult


class ToolBench:
    name, task_desc = "toolbench", "t"
    probe_commands = ["python3 -c 'import json'"]

    def claude_opts(self, ws, workdir):
        return {"tools": ["Bash"], "allowed_tools": ["Bash"], "stream": True, "max_turns": 30,
                "settings": {"sandbox": {"enabled": True}}}


GOOD = {"must:0": 0, "allow_write_workdir": 0, "read_control": 0, "tools_present": 0, "deny_read_outside": 1,
        "deny_write_outside": 1, "deny_network": 56, "deny_applications": 1}


def test_script_runs_every_check_and_records_exit_codes(tmp_path):
    """The script itself is plain bash; run it here unsandboxed just to check it records every check."""
    import subprocess
    wd, outside = tmp_path / "run", tmp_path / "outside"
    wd.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    (wd / "probe.sh").write_text(pr.probe_script(["true", "false"], outside / "secret.txt", outside / "w.txt"))
    subprocess.run(["bash", "probe.sh"], cwd=wd, check=True, capture_output=True, timeout=60)
    res = pr.read_results(wd)
    assert res["must:0"] == 0 and res["must:1"] != 0
    assert set(res) >= {"allow_write_workdir", "deny_read_outside", "deny_write_outside", "deny_network",
                        "deny_applications"}


def test_evaluate_expectations(tmp_path):
    wd = tmp_path / "run"
    wd.mkdir()
    (wd / "ok.txt").write_text("ok")
    checks = pr.evaluate(GOOD, ["python3 -c 'import json'"], wd, tmp_path / "w.txt")
    assert all(c.ok for c in checks) and len(checks) == 6, [c for c in checks if not c.ok]
    # an allowed denial, a missing result and a write that landed despite rc != 0 all fail
    (tmp_path / "w.txt").write_text("x")
    bad = {**GOOD, "deny_network": 0}
    bad.pop("deny_applications")
    by = {c.name: c for c in pr.evaluate(bad, ["python3 -c 'import json'"], wd, tmp_path / "w.txt")}
    assert not by["deny_network"].ok and not by["deny_applications"].ok and "not run" in by["deny_applications"].detail
    assert not by["deny_write_outside"].ok and "file was written" in by["deny_write_outside"].detail
    assert by["must:0"].ok and "import json" in by["must:0"].detail


def test_run_probe_uses_the_bench_opts_and_cleans_up(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    seen = {}

    def fake_claude(prompt, **kw):
        seen.update(prompt=prompt, **kw)
        wd = kw["cwd"]
        assert (wd / "probe.sh").exists()
        (wd / "ok.txt").write_text("ok")
        (wd / pr.RESULTS).write_text("".join(f"{k}\t{v}\n" for k, v in GOOD.items()))
        return ClaudeResult(text="DONE", structured=None, turns=2, cost_usd=0, is_error=False,
                            transcript="$ bash probe.sh\nDONE", tool_calls=["$ bash probe.sh"])

    monkeypatch.setattr(pr, "run_claude", fake_claude)
    monkeypatch.setattr(pr, "_host_network", lambda: True)
    monkeypatch.setattr(pr, "_host_apps", lambda: True)  # CI is Linux: no /Applications there
    checks = pr.run_probe(ws, ToolBench(), model="haiku")
    assert checks and all(c.ok for c in checks)
    assert seen["tools"] == ["Bash"] and seen["settings"] == {"sandbox": {"enabled": True}}
    assert seen["max_turns"] == 4 and "bash probe.sh" in seen["prompt"] and seen["model"] == "haiku"
    assert seen["cwd"].name == "run" and not (ws / pr.PROBE_DIR).exists()  # sentinel + workdir removed


def test_run_probe_reports_a_failed_session(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    monkeypatch.setattr(pr, "run_claude", lambda prompt, **kw: ClaudeResult(
        text="Not logged in", structured=None, turns=0, cost_usd=0, is_error=True))
    checks = pr.run_probe(ws, ToolBench(), model="haiku")
    assert checks[0].name == "session" and not checks[0].ok and "Not logged in" in checks[0].detail


def test_toolless_bench_has_nothing_to_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(pr, "run_claude", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    assert pr.run_probe(tmp_path, LiveMath(), model="haiku") == []


def test_cli_probe_exit_code(monkeypatch, tmp_path, capsys):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "dataset/meta.json").write_text(json.dumps({"bench": "livemath", "seed": 0}))
    ok = [pr.Check("deny_network", True, "rc=7")]
    monkeypatch.setattr(wcli, "run_probe", lambda ws_, bench, *, model: ok)
    assert wcli.run_evolve(argparse.Namespace(evolve_cmd="probe", ws=str(ws), model="haiku")) == 0
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setattr(wcli, "run_probe", lambda ws_, bench, *, model: [pr.Check("deny_network", False, "rc=0")])
    assert wcli.run_evolve(argparse.Namespace(evolve_cmd="probe", ws=str(ws), model="haiku")) == 1
    p = argparse.ArgumentParser()
    wcli.add_evolve_parser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["evolve", "probe", "--ws", "x"])
    assert a.evolve_cmd == "probe" and a.model == "haiku"


def test_spreadsheet_bench_declares_its_probe_commands():
    from engram.wikiskill.spreadsheet import SpreadsheetBench
    assert any("openpyxl" in c for c in SpreadsheetBench.probe_commands)


def test_results_parser_ignores_garbage(tmp_path):
    (tmp_path / pr.RESULTS).write_text("deny_network\t7\ngarbage line\n\nmust:0\tx\n")
    assert pr.read_results(tmp_path) == {"deny_network": 7}
    assert pr.read_results(Path(tmp_path / "missing")) == {}


# --- review fixes: no vacuous PASS --------------------------------------------------------

def _wd(tmp_path):
    wd = tmp_path / "run"
    wd.mkdir(exist_ok=True)
    (wd / "ok.txt").write_text("ok")
    return wd


def test_missing_command_is_a_fail_not_a_denial(tmp_path):
    by = {c.name: c for c in pr.evaluate({**GOOD, "deny_network": 127}, ["true"], _wd(tmp_path), tmp_path / "w")}
    assert not by["deny_network"].ok and "command not found" in by["deny_network"].detail


def test_denial_without_a_working_control_is_inconclusive(tmp_path):
    wd = _wd(tmp_path)
    no_read_ctl = {**GOOD, "read_control": 1}
    c = {x.name: x for x in pr.evaluate(no_read_ctl, ["true"], wd, tmp_path / "w")}["deny_read_outside"]
    assert not c.ok and c.inconclusive and "control" in c.detail
    offline = {x.name: x for x in pr.evaluate(GOOD, ["true"], wd, tmp_path / "w", host_net=False)}["deny_network"]
    assert not offline.ok and offline.inconclusive and "offline" in offline.detail


def test_deny_applications_is_skipped_on_a_host_without_applications(tmp_path, monkeypatch, capsys):
    """Linux has no /Applications: nothing there to deny, so the check is SKIP (not a failure)."""
    c = {x.name: x for x in pr.evaluate(GOOD, ["true"], _wd(tmp_path), tmp_path / "w", host_apps=False)}
    assert c["deny_applications"].ok and c["deny_applications"].skipped and "no /Applications" in c["deny_applications"].detail
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "dataset/meta.json").write_text(json.dumps({"bench": "livemath", "seed": 0}))
    monkeypatch.setattr(wcli, "run_probe", lambda ws_, bench, *, model: [c["deny_applications"]])
    assert wcli.run_evolve(argparse.Namespace(evolve_cmd="probe", ws=str(ws), model="haiku")) == 0
    assert capsys.readouterr().out.startswith("SKIP\tdeny_applications")


def test_script_records_the_controls(tmp_path):
    s = pr.probe_script([], tmp_path / "secret", tmp_path / "w")
    assert "rec read_control" in s and "rec tools_present" in s


def test_integrity_fails_when_the_agent_did_not_run_the_untouched_script(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)

    def tamper(prompt, **kw):
        wd = kw["cwd"]
        (wd / "probe.sh").write_text("echo forged\n")
        (wd / "ok.txt").write_text("ok")
        (wd / pr.RESULTS).write_text("".join(f"{k}\t{v}\n" for k, v in GOOD.items()))
        return ClaudeResult(text="DONE", structured=None, turns=2, cost_usd=0, is_error=False,
                            transcript="$ echo forged", tool_calls=["$ echo forged"])

    monkeypatch.setattr(pr, "run_claude", tamper)
    monkeypatch.setattr(pr, "_host_network", lambda: True)
    monkeypatch.setattr(pr, "_host_apps", lambda: True)  # CI is Linux: no /Applications there
    by = {c.name: c for c in pr.run_probe(ws, ToolBench(), model="m")}
    assert not by["script_integrity"].ok and "modified" in by["script_integrity"].detail
    assert "not run" in by["script_integrity"].detail or "bash probe.sh" in by["script_integrity"].detail


def test_probe_dir_cannot_collide_with_a_rollout_dir():
    assert pr.PROBE_DIR == "work/.probe"


def test_integrity_fails_when_the_agent_ran_anything_besides_the_script(monkeypatch, tmp_path):
    """Re-review: `bash probe.sh` followed by an append to the results file kept probe.sh's hash and the
    transcript line, yet forged the verdict. The session must contain exactly one call: the script."""
    ws = tmp_path / "ws"
    w.init_workspace(ws)

    def forge_after(prompt, **kw):
        wd = kw["cwd"]
        (wd / "ok.txt").write_text("ok")
        (wd / pr.RESULTS).write_text("".join(f"{k}\t{v}\n" for k, v in GOOD.items()))
        return ClaudeResult(text="DONE", structured=None, turns=3, cost_usd=0, is_error=False,
                            transcript="$ bash probe.sh\n$ printf x >> probe-results.txt",
                            tool_calls=["$ bash probe.sh", "$ printf 'deny_network\\t7' >> probe-results.txt"])

    monkeypatch.setattr(pr, "run_claude", forge_after)
    monkeypatch.setattr(pr, "_host_network", lambda: True)
    monkeypatch.setattr(pr, "_host_apps", lambda: True)  # CI is Linux: no /Applications there
    by = {c.name: c for c in pr.run_probe(ws, ToolBench(), model="m")}
    assert not by["script_integrity"].ok and "other" in by["script_integrity"].detail
