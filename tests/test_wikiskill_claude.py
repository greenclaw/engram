import json

from engram.wikiskill import claude as c


def test_argv_no_tools_no_schema(monkeypatch, tmp_path):
    seen = {}

    def fake_exec(argv, cwd, timeout):
        seen["argv"], seen["cwd"] = argv, cwd
        return json.dumps({"result": "ok", "num_turns": 1, "total_cost_usd": 0.001, "is_error": False})

    monkeypatch.setattr(c, "_exec", fake_exec)
    r = c.run_claude("hi", system="SYS", model="haiku", cwd=tmp_path, tools=[])
    a = seen["argv"]
    assert a[:2] == ["claude", "-p"]
    assert "--setting-sources" in a and a[a.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in a
    assert a[a.index("--model") + 1] == "haiku"
    assert a[a.index("--system-prompt") + 1] == "SYS"
    assert a[a.index("--tools") + 1] == ""
    assert a[a.index("--output-format") + 1] == "json"
    assert "--max-turns" not in a and "--json-schema" not in a
    assert a[-1] == "hi"
    assert seen["cwd"] == tmp_path
    assert r.text == "ok" and r.structured is None and r.turns == 1 and r.cost_usd == 0.001


def test_argv_tools_schema_turns(monkeypatch, tmp_path):
    seen = {}
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    def fake_exec(argv, cwd, timeout):
        seen["argv"] = argv
        return json.dumps({"result": '{"action":"x"}', "structured_output": {"action": "x"},
                           "num_turns": 3, "total_cost_usd": 0.0, "is_error": False})

    monkeypatch.setattr(c, "_exec", fake_exec)
    r = c.run_claude("go", system="S", model="haiku", cwd=tmp_path, tools=["Read"],
                     max_turns=25, json_schema=schema)
    a = seen["argv"]
    assert a[a.index("--tools") + 1] == "Read"
    assert a[a.index("--max-turns") + 1] == "25"
    assert json.loads(a[a.index("--json-schema") + 1]) == schema
    assert r.structured == {"action": "x"}


def test_timeout_is_error_not_exception(monkeypatch, tmp_path):
    import subprocess

    def slow(argv, cwd, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(c, "_exec", slow)
    r = c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[], timeout=7)
    assert r.is_error and "timeout after 7s" in r.text


def test_non_json_stdout_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout: "Not logged in")
    r = c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[])
    assert r.is_error and r.text == "Not logged in" and r.structured is None
