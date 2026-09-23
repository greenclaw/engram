"""The one subprocess wrapper every WikiSkill role uses: headless `claude -p`, no user settings,
no hooks, no MCP, no persisted session, JSON result. Roles differ only in system prompt, tools,
schema, turn cap — and, for tool-using benches, a sandbox and a streamed transcript."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TOOL_OUTPUT_CAP = 2000  # per tool result in the transcript; Appendix C caps the whole trace at 15k later


@dataclass
class ClaudeResult:
    text: str
    structured: dict | None
    turns: int
    cost_usd: float
    is_error: bool
    raw: dict = field(default_factory=dict)
    transcript: str = ""  # stream mode: the session's commands, outputs and final text


def _exec(argv: list[str], cwd: Path, timeout: int, env: dict | None = None) -> str:
    return subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=timeout, env=env).stdout


def _cap(s: str) -> str:
    return s if len(s) <= TOOL_OUTPUT_CAP else f"{s[:TOOL_OUTPUT_CAP]}…[truncated {len(s) - TOOL_OUTPUT_CAP} chars]"


def _tool_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def _parse_stream(out: str) -> tuple[dict | None, str]:
    """(final result event, transcript) from `--output-format stream-json --verbose` lines."""
    result, lines = None, []
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "result":
            result = ev
        elif kind in ("assistant", "user"):
            for c in (ev.get("message") or {}).get("content") or []:
                t = c.get("type")
                if t == "tool_use":
                    inp = c.get("input") or {}
                    lines.append(f"$ {inp['command']}" if "command" in inp else f"[{c.get('name')}] {json.dumps(inp)}")
                elif t == "tool_result":
                    lines.append(("[error] " if c.get("is_error") else "") + _cap(_tool_text(c.get("content"))))
                elif t == "text" and kind == "assistant":
                    lines.append(c.get("text", ""))
    return result, "\n".join(lines)


def run_claude(prompt: str, *, system: str, model: str, cwd: Path, tools: list[str],
               max_turns: int | None = None, json_schema: dict | None = None, timeout: int = 900,
               allowed_tools: list[str] | None = None, settings: dict | None = None,
               env: dict | None = None, stream: bool = False) -> ClaudeResult:
    argv = ["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
            "--model", model, "--system-prompt", system, "--tools", " ".join(tools),
            "--output-format", "stream-json" if stream else "json"]
    if stream:
        argv.append("--verbose")  # stream-json requires it in print mode
    if allowed_tools:
        argv += ["--allowedTools", " ".join(allowed_tools)]
    if settings is not None:
        argv += ["--settings", json.dumps(settings)]
    if max_turns is not None:
        argv += ["--max-turns", str(max_turns)]
    if json_schema is not None:
        argv += ["--json-schema", json.dumps(json_schema)]
    argv.append(prompt)
    try:
        out = _exec(argv, cwd, timeout, {**os.environ, **env} if env else None)
    except subprocess.TimeoutExpired:  # a hung call is a retryable failure, not a crash of the whole rollout
        return ClaudeResult(text=f"timeout after {timeout}s", structured=None, turns=0, cost_usd=0.0, is_error=True)
    transcript = ""
    if stream:
        d, transcript = _parse_stream(out)
        if d is None:
            return ClaudeResult(text=f"no result event in stream: {out[-200:].strip()!r}", structured=None,
                                turns=0, cost_usd=0.0, is_error=True, transcript=transcript)
    else:
        try:
            d = json.loads(out)
        except json.JSONDecodeError:  # not-logged-in / crash banners are plain text
            return ClaudeResult(text=out.strip(), structured=None, turns=0, cost_usd=0.0, is_error=True)
    return ClaudeResult(text=str(d.get("result", "")), structured=d.get("structured_output"),
                        turns=int(d.get("num_turns", 0)), cost_usd=float(d.get("total_cost_usd", 0.0)),
                        is_error=bool(d.get("is_error", False)), raw=d, transcript=transcript)
