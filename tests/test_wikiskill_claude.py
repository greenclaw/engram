import json

from engram.wikiskill import claude as c


def test_argv_no_tools_no_schema(monkeypatch, tmp_path):
    seen = {}

    def fake_exec(argv, cwd, timeout, env=None):
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

    def fake_exec(argv, cwd, timeout, env=None):
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

    def slow(argv, cwd, timeout, env=None):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(c, "_exec", slow)
    r = c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[], timeout=7)
    assert r.is_error and "timeout after 7s" in r.text


def test_non_json_stdout_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout, env=None: "Not logged in")
    r = c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[])
    assert r.is_error and r.text == "Not logged in" and r.structured is None


def _stream(*events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_always_disables_session_persistence(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout, env=None: seen.update(argv=argv) or json.dumps(
        {"result": "ok", "num_turns": 1, "is_error": False}))
    c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[])
    assert "--no-session-persistence" in seen["argv"]


def test_sandbox_settings_env_and_allowed_tools(monkeypatch, tmp_path):
    seen = {}

    def fake_exec(argv, cwd, timeout, env=None):
        seen.update(argv=argv, env=env)
        return json.dumps({"result": "ok", "num_turns": 1, "is_error": False})

    monkeypatch.setattr(c, "_exec", fake_exec)
    settings = {"sandbox": {"enabled": True, "filesystem": {"denyRead": ["~/"]}}}
    c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=["Bash"], allowed_tools=["Bash"],
                 settings=settings, env={"PATH": "/venv/bin:/usr/bin"})
    a = seen["argv"]
    assert json.loads(a[a.index("--settings") + 1]) == settings
    assert a[a.index("--allowedTools") + 1] == "Bash"
    assert seen["env"]["PATH"] == "/venv/bin:/usr/bin" and "HOME" in seen["env"]  # merged over os.environ


def test_stream_mode_builds_a_transcript_of_commands_and_outputs(monkeypatch, tmp_path):
    long_out = "x" * 5000
    out = _stream(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "hmm"},
                                                      {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "rate_limit_event", "rate_limit_info": {}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "input.xlsx"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash",
                                                       "input": {"command": "python solve.py"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True,
                                                  "content": [{"type": "text", "text": long_out}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Saved output.xlsx"}]}},
        {"type": "result", "subtype": "success", "result": "Saved output.xlsx", "num_turns": 3, "is_error": False,
         "total_cost_usd": 0.1, "usage": {"output_tokens": 50}},
    )
    seen = {}
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout, env=None: seen.update(argv=argv) or out)
    r = c.run_claude("go", system="S", model="haiku", cwd=tmp_path, tools=["Bash"], stream=True)
    a = seen["argv"]
    assert a[a.index("--output-format") + 1] == "stream-json" and "--verbose" in a
    assert r.text == "Saved output.xlsx" and r.turns == 3 and r.raw["usage"]["output_tokens"] == 50
    t = r.transcript
    assert "$ ls" in t and "input.xlsx" in t and "$ python solve.py" in t and "[error]" in t
    assert "x" * 2000 in t and "x" * 2001 not in t and "…[truncated 3000 chars]" in t
    assert t.rstrip().endswith("Saved output.xlsx") and "hmm" not in t


def test_stream_mode_without_result_event_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout, env=None: _stream({"type": "system"}))
    r = c.run_claude("go", system="S", model="haiku", cwd=tmp_path, tools=["Bash"], stream=True)
    assert r.is_error and "no result event" in r.text
