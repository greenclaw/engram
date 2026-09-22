"""The one subprocess wrapper every WikiSkill role uses: headless `claude -p`, no user settings,
no hooks, no MCP, JSON result. Roles differ only in system prompt, tools, schema, turn cap."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClaudeResult:
    text: str
    structured: dict | None
    turns: int
    cost_usd: float
    is_error: bool
    raw: dict = field(default_factory=dict)


def _exec(argv: list[str], cwd: Path, timeout: int) -> str:
    return subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=timeout).stdout


def run_claude(prompt: str, *, system: str, model: str, cwd: Path, tools: list[str],
               max_turns: int | None = None, json_schema: dict | None = None,
               timeout: int = 900) -> ClaudeResult:
    argv = ["claude", "-p", "--setting-sources", "", "--strict-mcp-config",
            "--model", model, "--system-prompt", system, "--tools", " ".join(tools),
            "--output-format", "json"]
    if max_turns is not None:
        argv += ["--max-turns", str(max_turns)]
    if json_schema is not None:
        argv += ["--json-schema", json.dumps(json_schema)]
    argv.append(prompt)
    out = _exec(argv, cwd, timeout)
    try:
        d = json.loads(out)
    except json.JSONDecodeError:  # not-logged-in / crash banners are plain text
        return ClaudeResult(text=out.strip(), structured=None, turns=0, cost_usd=0.0, is_error=True)
    return ClaudeResult(text=str(d.get("result", "")), structured=d.get("structured_output"),
                        turns=int(d.get("num_turns", 0)), cost_usd=float(d.get("total_cost_usd", 0.0)),
                        is_error=bool(d.get("is_error", False)), raw=d)
