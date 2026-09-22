# WikiSkill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `engram evolve` runs the WikiSkill loop (Algorithm 1 of arXiv:2608.27454) on LiveMathematicianBench with headless `claude -p` as every LLM role, unattended, with a metric gate and a git audit trail.

**Architecture:** A deterministic Python harness (`src/engram/wikiskill/`) owns the three-layer workspace (`raw/ wiki/ skills/`), the patch ops, the strict validation gate and `skill-impact.md`. Each LLM role is one `claude -p` subprocess with the paper's Appendix E prompt as `--system-prompt`, no user settings/hooks/MCP, structured output via `--json-schema`. A `Bench` protocol isolates dataset loading and scoring so SpreadsheetBench slots in later.

**Tech Stack:** Python 3.11+, stdlib (`subprocess`, `json`, `difflib`, `concurrent.futures`, `random`), `huggingface_hub` (already a dep), `claude` CLI 2.1.278, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-wikiskill-design.md`

## Global Constraints

- Python via `uv` only; run tests with `uv run pytest tests/test_wikiskill_*.py -q`.
- No new dependencies. No API key: every LLM call is `claude -p`.
- Every `claude -p` call: `--setting-sources '' --strict-mcp-config --output-format json`, `stdin=subprocess.DEVNULL`, `cwd=<workspace>`; the Inference Agent NEVER receives wiki content (spec §Roles).
- `raw/` and `wiki/` are never rolled back; only `skills/` is (spec §Gating).
- `wiki/skill-impact.md` is written only by the harness, never by a role.
- Prompts live as files in `src/engram/wikiskill/prompts/`, verbatim from Appendix E except the path alias (spec §Deviations).
- Files ≤ ~200 LOC; ruff config from `pyproject.toml` (`E4,E7,E9,F,I,UP`) must pass: `uv run ruff check src tests` if ruff is available, else skip.
- Conventional commits, no `!`, no "BREAKING CHANGE" text. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Git tests use the `git_repo` fixture from `conftest.py` (sets a committer identity).
- Verified `claude` JSON result fields: `result` (str), `structured_output` (dict|None), `is_error`, `num_turns`, `total_cost_usd`, `stop_reason`.

---

## File structure

| path | responsibility |
|---|---|
| `src/engram/wikiskill/__init__.py` | empty |
| `src/engram/wikiskill/claude.py` | `run_claude()` — the one subprocess wrapper every role uses |
| `src/engram/wikiskill/bench.py` | `Task`, `Bench` protocol, `LiveMath` (label shuffle, prompts, score), split I/O |
| `src/engram/wikiskill/workspace.py` | layout, patch ops, `skill_section`, `apply_proposal`, impact entry, git helpers |
| `src/engram/wikiskill/roles.py` | `rollout`, `sample_traces`, `maintain`, `propose` |
| `src/engram/wikiskill/loop.py` | Algorithm 1 (`evolve`), `evaluate`, `state.json` |
| `src/engram/wikiskill/cli.py` | `add_evolve_parser(sub)` + `run_evolve(args)` |
| `src/engram/wikiskill/prompts/livemath.md` | E.1 LiveMath inference prompt (`{skill_section}` placeholder) |
| `src/engram/wikiskill/prompts/maintainer.md` | E.2 |
| `src/engram/wikiskill/prompts/proposer.md` | E.3 with `raw/iter-<k>/<id>.json` path |
| `src/engram/cli.py` | wire `evolve` subparser (modify) |
| `tests/test_wikiskill_claude.py` | argv + parsing of `run_claude` |
| `tests/test_wikiskill_bench.py` | shuffle, prompts, score, split I/O |
| `tests/test_wikiskill_workspace.py` | patch ops, proposal apply, impact entry, git restore |
| `tests/test_wikiskill_roles.py` | sampling, role argv/parse with mocked `run_claude` |
| `tests/test_wikiskill_loop.py` | gate, rollback, early stop, no_action, resume, evaluate |
| `tests/test_wikiskill_cli.py` | subparser wiring |

---

### Task 1: `run_claude` wrapper

**Files:**
- Create: `src/engram/wikiskill/__init__.py` (empty)
- Create: `src/engram/wikiskill/claude.py`
- Test: `tests/test_wikiskill_claude.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class ClaudeResult:
      text: str                 # result field
      structured: dict | None   # structured_output field
      turns: int
      cost_usd: float
      is_error: bool
      raw: dict                 # full JSON, for the trace file

  def run_claude(prompt: str, *, system: str, model: str, cwd: Path, tools: list[str],
                 max_turns: int | None = None, json_schema: dict | None = None,
                 timeout: int = 900) -> ClaudeResult
  ```
  All roles and tests monkeypatch `engram.wikiskill.claude.run_claude`'s inner `_exec(argv, cwd, timeout) -> str` (stdout) for isolation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wikiskill_claude.py
import json
from pathlib import Path

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


def test_non_json_stdout_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "_exec", lambda argv, cwd, timeout: "Not logged in")
    r = c.run_claude("hi", system="S", model="haiku", cwd=tmp_path, tools=[])
    assert r.is_error and r.text == "Not logged in" and r.structured is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_claude.py -q`
Expected: FAIL — `ModuleNotFoundError: engram.wikiskill`

- [ ] **Step 3: Implement**

```python
# src/engram/wikiskill/claude.py
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
```

`__init__.py` is empty.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_wikiskill_claude.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/engram/wikiskill/__init__.py src/engram/wikiskill/claude.py tests/test_wikiskill_claude.py
git commit -m "feat(wikiskill): run_claude — headless claude -p wrapper for the roles"
```

---

### Task 2: `Bench` protocol + LiveMath

**Files:**
- Create: `src/engram/wikiskill/bench.py`
- Create: `src/engram/wikiskill/prompts/livemath.md`
- Test: `tests/test_wikiskill_bench.py`

**Interfaces:**
- Produces:
  ```python
  class Task(TypedDict): id: str; question: str; choices: dict[str, str]; answer: str
  class Bench(Protocol):
      name: str
      tools: list[str]
      def system_prompt(self, skill_section: str) -> str: ...
      def user_prompt(self, task: Task) -> str: ...
      def score(self, task: Task, response: str) -> float: ...
  class LiveMath:  # implements Bench; name="livemath", tools=[]
      HF_REPO = "LiveMathematicianBench/LiveMathematicianBench"
      SPLIT_SIZES = {"train": 35, "val": 18, "test": 124}
      @staticmethod
      def tasks_from_records(records: list[dict], seed: int) -> list[Task]
      def download_records(self, cache_dir: Path) -> list[dict]   # all data/*/qa_*_final.json
  def make_splits(tasks: list[Task], sizes: dict[str, int], seed: int) -> dict[str, list[Task]]
  def write_split(path: Path, tasks: list[Task]) -> None      # jsonl
  def read_split(path: Path) -> list[Task]
  def get_bench(name: str) -> Bench                            # {"livemath": LiveMath}
  ```
- Dataset facts (verified 2026-09-22): each record has `mcq.question`, `mcq.correct_choice{label,text}` (label always "A"), `mcq.choices[4]` with labels B–E (distractors only), plus `no`, `month`. Hence the label shuffle.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wikiskill_bench.py
import json

import pytest

from engram.wikiskill import bench as b


def rec(no, month="202606", correct="right", distractors=("d1", "d2", "d3", "d4")):
    return {"no": no, "month": month, "mcq": {
        "question": f"Q{no}?", "correct_choice": {"label": "A", "text": correct},
        "choices": [{"label": lab, "text": t} for lab, t in zip("BCDE", distractors)]}}


def test_tasks_from_records_shuffles_labels_deterministically():
    recs = [rec(i) for i in range(20)]
    t1 = b.LiveMath.tasks_from_records(recs, seed=0)
    t2 = b.LiveMath.tasks_from_records(recs, seed=0)
    assert t1 == t2
    assert t1[0]["id"] == "202606-0" and set(t1[0]["choices"]) == set("ABCDE")
    for t in t1:  # correct text sits under the answer letter
        assert t["choices"][t["answer"]] == "right"
    assert len({t["answer"] for t in t1}) > 1  # not always "A"


def test_make_splits_disjoint_and_sized():
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(200)], seed=1)
    s = b.make_splits(tasks, {"train": 35, "val": 18, "test": 124}, seed=1)
    ids = [t["id"] for k in ("train", "val", "test") for t in s[k]]
    assert len(ids) == len(set(ids)) == 177
    assert [len(s[k]) for k in ("train", "val", "test")] == [35, 18, 124]


def test_make_splits_too_few_fails_loud():
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(10)], seed=1)
    with pytest.raises(ValueError):
        b.make_splits(tasks, {"train": 35, "val": 18, "test": 124}, seed=1)


def test_split_roundtrip(tmp_path):
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(3)], seed=0)
    b.write_split(tmp_path / "x.jsonl", tasks)
    assert b.read_split(tmp_path / "x.jsonl") == tasks
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 3


def test_prompts():
    lm = b.LiveMath()
    sp = lm.system_prompt("## Skills\nfoo")
    assert "## Skills\nfoo" in sp and "{skill_section}" not in sp
    assert "<answer>" in sp
    t = b.LiveMath.tasks_from_records([rec(1)], seed=0)[0]
    up = lm.user_prompt(t)
    assert "Q1?" in up and "A." in up and "E." in up


@pytest.mark.parametrize("resp,gold,exp", [
    ("blah <answer>C</answer>", "C", 1.0),
    ("<answer>B</answer> ... <answer>C</answer>", "C", 1.0),   # last one wins
    ("<answer> c </answer>", "C", 1.0),
    ("<answer>B</answer>", "C", 0.0),
    ("no tags", "C", 0.0),
    ("<answer>CD</answer>", "C", 0.0),
])
def test_score(resp, gold, exp):
    t = {"id": "x", "question": "q", "choices": {}, "answer": gold}
    assert b.LiveMath().score(t, resp) == exp


def test_get_bench():
    assert b.get_bench("livemath").name == "livemath"
    with pytest.raises(KeyError):
        b.get_bench("nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_bench.py -q`
Expected: FAIL — `ImportError` / `AttributeError`

- [ ] **Step 3: Write the LiveMath prompt file**

`src/engram/wikiskill/prompts/livemath.md` — E.1 verbatim:

```
You are an expert mathematical reasoning agent solving multiple-choice questions.

{skill_section}

## Task Format

You will receive one mathematics multiple-choice question and its answer choices. Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.

## Answer Format

Think step by step, then provide your final answer inside <answer>...</answer> tags. Inside the tags, output only the single choice label, such as A or C.

Example:

<answer>B</answer>
```

- [ ] **Step 4: Implement bench.py**

```python
# src/engram/wikiskill/bench.py
"""Bench protocol (load / prompt / score) + LiveMathematicianBench. The evolution loop never
sees dataset specifics; SpreadsheetBench later plugs in behind the same protocol."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Protocol, TypedDict

PROMPTS = Path(__file__).parent / "prompts"


class Task(TypedDict):
    id: str
    question: str
    choices: dict[str, str]
    answer: str


class Bench(Protocol):
    name: str
    tools: list[str]

    def system_prompt(self, skill_section: str) -> str: ...
    def user_prompt(self, task: Task) -> str: ...
    def score(self, task: Task, response: str) -> float: ...


_ANSWER = re.compile(r"<answer>\s*([A-Za-z]+)\s*</answer>")


class LiveMath:
    name = "livemath"
    tools: list[str] = []
    HF_REPO = "LiveMathematicianBench/LiveMathematicianBench"
    SPLIT_SIZES = {"train": 35, "val": 18, "test": 124}  # Table 6

    @staticmethod
    def tasks_from_records(records: list[dict], seed: int) -> list[Task]:
        """HF records always put the correct option under 'A' and list B–E distractors;
        reshuffle the five options per task (seeded) so the letter carries no signal."""
        rng = random.Random(seed)
        out: list[Task] = []
        for r in records:
            m = r["mcq"]
            texts = [m["correct_choice"]["text"]] + [c["text"] for c in m["choices"]]
            order = list(range(len(texts)))
            rng.shuffle(order)
            labels = "ABCDEFGH"[: len(texts)]
            choices = {lab: texts[i] for lab, i in zip(labels, order)}
            answer = labels[order.index(0)]
            out.append(Task(id=f"{r['month']}-{r['no']}", question=m["question"], choices=choices, answer=answer))
        return out

    def download_records(self, cache_dir: Path) -> list[dict]:
        from huggingface_hub import HfApi, hf_hub_download

        files = sorted(s.rfilename for s in HfApi().dataset_info(self.HF_REPO).siblings
                       if s.rfilename.startswith("data/") and s.rfilename.endswith(".json"))
        recs: list[dict] = []
        for f in files:
            p = hf_hub_download(self.HF_REPO, f, repo_type="dataset", cache_dir=cache_dir)
            recs.extend(json.loads(Path(p).read_text()))
        return recs

    def system_prompt(self, skill_section: str) -> str:
        return (PROMPTS / "livemath.md").read_text().replace("{skill_section}", skill_section)

    def user_prompt(self, task: Task) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(task["choices"].items()))
        return f"{task['question']}\n\n{opts}"

    def score(self, task: Task, response: str) -> float:
        hits = _ANSWER.findall(response)
        return 1.0 if hits and hits[-1].strip().upper() == task["answer"] else 0.0


def make_splits(tasks: list[Task], sizes: dict[str, int], seed: int) -> dict[str, list[Task]]:
    need = sum(sizes.values())
    if len(tasks) < need:
        raise ValueError(f"need {need} tasks for splits {sizes}, have {len(tasks)}")
    pool = list(tasks)
    random.Random(seed).shuffle(pool)
    out, i = {}, 0
    for name, n in sizes.items():
        out[name] = pool[i:i + n]
        i += n
    return out


def write_split(path: Path, tasks: list[Task]) -> None:
    path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))


def read_split(path: Path) -> list[Task]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_BENCHES = {"livemath": LiveMath}


def get_bench(name: str) -> Bench:
    return _BENCHES[name]()
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_wikiskill_bench.py -q`
Expected: all passed (10 tests)

- [ ] **Step 6: Commit**

```bash
git add src/engram/wikiskill/bench.py src/engram/wikiskill/prompts/livemath.md tests/test_wikiskill_bench.py
git commit -m "feat(wikiskill): Bench protocol + LiveMathematicianBench loader, label shuffle, exact-match score"
```

---

### Task 3: Workspace — layout, patch ops, proposal apply, impact entry, git

**Files:**
- Create: `src/engram/wikiskill/workspace.py`
- Test: `tests/test_wikiskill_workspace.py`

**Interfaces:**
- Produces:
  ```python
  INDEX_HEADER = "# Wiki Index\n\n"
  def init_workspace(ws: Path) -> None            # dirs + empty wiki files + git init + first commit
  def apply_edits(text: str, edits: list[dict]) -> tuple[str, list[str]]   # (new_text, skipped_reasons)
  def skill_section(ws: Path) -> str              # "" if no skills, else "## Skills\n\n### <name>\n<SKILL.md>\n..." per skill
  def read_wiki(ws: Path) -> str                  # index + log + every pattern page, for the Maintainer
  def apply_maintainer(ws: Path, out: dict, k: int) -> list[str]   # returns skipped reasons; writes patterns/index/log
  def apply_proposal(ws: Path, p: dict) -> tuple[str, list[str]]   # (unified_diff, skipped); writes skills/
  def append_impact(ws: Path, k: int, p: dict, diff: str, r_val: float | None, r_best: float, outcome: str) -> None
  def git(ws: Path, *args: str) -> str
  def commit_all(ws: Path, msg: str) -> None      # git add -A && commit (no-op if clean)
  def tag(ws: Path, name: str) -> None
  def restore_skills(ws: Path, ref: str) -> None  # git checkout <ref> -- skills/ ; removes skills not in ref
  ```
- Patch-op semantics (E.2/E.3): `{"op":"append","content"}`, `{"op":"replace","target","content"}`, `{"op":"insert_after","target","content"}`; `target` must be an exact substring (first occurrence); a miss is skipped, never fuzzy-matched.
- Proposal shape (E.3): `{"action":"create","name","skill_md","purpose_md"}` | `{"action":"patch","name","edits"}` | `{"action":"no_action"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wikiskill_workspace.py
import pytest

from engram.wikiskill import workspace as w


def test_init_layout(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    for rel in ("raw", "wiki/patterns", "skills", "dataset"):
        assert (ws / rel).is_dir()
    for f in ("wiki/index.md", "wiki/log.md", "wiki/skill-impact.md"):
        assert (ws / f).is_file()
    assert (ws / ".git").is_dir()
    assert w.git(ws, "log", "--oneline").count("\n") == 1


def test_apply_edits_ops_and_skip():
    text = "line1\nline2\nline3\n"
    new, skipped = w.apply_edits(text, [
        {"op": "append", "content": "tail"},
        {"op": "replace", "target": "line2", "content": "LINE2"},
        {"op": "insert_after", "target": "line1", "content": "after1"},
        {"op": "replace", "target": "nope", "content": "x"},
        {"op": "bogus", "content": "x"},
    ])
    assert new == "line1\nafter1\nLINE2\nline3\ntail\n"
    assert len(skipped) == 2 and "nope" in skipped[0] and "bogus" in skipped[1]


def test_skill_section_empty_and_filled(tmp_path):
    ws = tmp_path
    (ws / "skills").mkdir()
    assert w.skill_section(ws) == ""
    (ws / "skills/a").mkdir()
    (ws / "skills/a/SKILL.md").write_text("---\nname: a\n---\nDo A")
    (ws / "skills/b").mkdir()
    (ws / "skills/b/SKILL.md").write_text("Do B")
    s = w.skill_section(ws)
    assert s.startswith("## Skills") and s.index("Do A") < s.index("Do B")


def test_apply_maintainer(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "wiki/patterns/old.md").write_text("old body\n")
    skipped = w.apply_maintainer(ws, {
        "create_patterns": [{"name": "loop.md", "content": "# loop\nbody\n"}],
        "update_patterns": [{"name": "old.md", "edits": [{"op": "append", "content": "more"}]},
                            {"name": "missing.md", "edits": [{"op": "append", "content": "x"}]}],
        "update_index": "# idx\n- [loop](wiki/patterns/loop.md): p+rc+fix\n",
        "append_log": "iteration findings",
    }, k=1)
    assert (ws / "wiki/patterns/loop.md").read_text() == "# loop\nbody\n"
    assert (ws / "wiki/patterns/old.md").read_text() == "old body\nmore\n"
    assert (ws / "wiki/index.md").read_text().startswith("# idx")
    log = (ws / "wiki/log.md").read_text()
    assert "## iter 1" in log and "iteration findings" in log and "missing.md" in log
    assert skipped and "missing.md" in skipped[0]


def test_apply_proposal_create_then_patch(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    diff, skipped = w.apply_proposal(ws, {"action": "create", "name": "break_loop",
                                          "skill_md": "---\nname: break_loop\n---\nrule1\n",
                                          "purpose_md": "## Origin\nloop.md\n"})
    assert (ws / "skills/break_loop/SKILL.md").read_text().endswith("rule1\n")
    assert (ws / "skills/break_loop/PURPOSE.md").exists()
    assert "+rule1" in diff and not skipped
    diff, skipped = w.apply_proposal(ws, {"action": "patch", "name": "break_loop",
                                          "edits": [{"op": "append", "content": "rule2"},
                                                    {"op": "replace", "target": "zzz", "content": "q"}]})
    assert (ws / "skills/break_loop/SKILL.md").read_text().endswith("rule1\nrule2\n")
    assert "+rule2" in diff and len(skipped) == 1


def test_apply_proposal_patch_unknown_skill_raises(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    with pytest.raises(w.ProposalError):
        w.apply_proposal(ws, {"action": "patch", "name": "ghost", "edits": []})


def test_apply_proposal_rejects_bad_name(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    with pytest.raises(w.ProposalError):
        w.apply_proposal(ws, {"action": "create", "name": "../x", "skill_md": "a", "purpose_md": "b"})


def test_append_impact_format(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    w.append_impact(ws, 2, {"action": "create", "name": "s1"}, "--- a\n+++ b\n+x\n", 0.5, 0.4, "Accepted")
    w.append_impact(ws, 3, {"action": "no_action"}, "", None, 0.5, "NoAction")
    t = (ws / "wiki/skill-impact.md").read_text()
    assert "## iter 2 — create s1 — val 0.500 (best 0.400) — Accepted" in t
    assert "```diff\n--- a\n+++ b\n+x\n```" in t
    assert "## iter 3 — no_action — val n/a (best 0.500) — NoAction" in t


def test_git_commit_tag_restore(git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "skills/a").mkdir()
    (ws / "skills/a/SKILL.md").write_text("v1")
    w.commit_all(ws, "iter 1")
    w.tag(ws, "accepted-1")
    (ws / "skills/a/SKILL.md").write_text("v2")
    (ws / "skills/b").mkdir()
    (ws / "skills/b/SKILL.md").write_text("new")
    w.restore_skills(ws, "accepted-1")
    assert (ws / "skills/a/SKILL.md").read_text() == "v1"
    assert not (ws / "skills/b").exists()
    w.commit_all(ws, "noop")  # clean tree must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_workspace.py -q`
Expected: FAIL — import error

- [ ] **Step 3: Implement**

```python
# src/engram/wikiskill/workspace.py
"""The three-layer workspace (raw/ wiki/ skills/) and the harness-side mutations the paper
assigns to code, not to an LLM: patch ops, proposal apply, skill-impact.md, git audit trail."""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from pathlib import Path

INDEX_HEADER = "# Wiki Index\n\n"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ProposalError(ValueError):
    """Untrusted LLM proposal that cannot be applied (bad name, unknown skill, bad action)."""


def init_workspace(ws: Path) -> None:
    for rel in ("raw", "wiki/patterns", "skills", "dataset"):
        (ws / rel).mkdir(parents=True, exist_ok=True)
    for rel, txt in (("wiki/index.md", INDEX_HEADER), ("wiki/log.md", "# Evolution Log\n\n"),
                     ("wiki/skill-impact.md", "# Skill Impact\n\n")):
        p = ws / rel
        if not p.exists():
            p.write_text(txt)
    for rel in ("raw", "skills", "dataset", "wiki/patterns"):
        (ws / rel / ".gitkeep").touch()
    if not (ws / ".git").exists():
        git(ws, "init", "-q")
    commit_all(ws, "wikiskill: init workspace")


def apply_edits(text: str, edits: list[dict]) -> tuple[str, list[str]]:
    skipped: list[str] = []
    for e in edits:
        op, content = e.get("op"), str(e.get("content", ""))
        if op == "append":
            text = text + ("" if text.endswith("\n") or not text else "\n") + content + "\n"
        elif op in ("replace", "insert_after"):
            target = str(e.get("target", ""))
            i = text.find(target) if target else -1
            if i < 0:
                skipped.append(f"{op}: target not found: {target[:80]!r}")
                continue
            j = i + len(target)
            text = text[:i] + content + text[j:] if op == "replace" else text[:j] + "\n" + content + text[j:]
        else:
            skipped.append(f"unknown op: {op!r}")
    return text, skipped


def skills_dir_section(skills: Path) -> str:
    parts = [f"### {d.name}\n{(d / 'SKILL.md').read_text()}"
             for d in sorted(skills.iterdir()) if (d / "SKILL.md").is_file()]
    return "## Skills\n\n" + "\n\n".join(parts) + "\n" if parts else ""


def skill_section(ws: Path) -> str:
    return skills_dir_section(ws / "skills")


def read_wiki(ws: Path) -> str:
    wiki = ws / "wiki"
    parts = [f"=== wiki/index.md ===\n{(wiki / 'index.md').read_text()}",
             f"=== wiki/log.md ===\n{(wiki / 'log.md').read_text()}"]
    for p in sorted((wiki / "patterns").glob("*.md")):
        parts.append(f"=== wiki/patterns/{p.name} ===\n{p.read_text()}")
    return "\n\n".join(parts)


def _safe_md_name(name: str) -> str:
    base = name[:-3] if name.endswith(".md") else name
    if not _NAME.match(base):
        raise ProposalError(f"bad name: {name!r}")
    return base + ".md"


def apply_maintainer(ws: Path, out: dict, k: int) -> list[str]:
    patterns, skipped = ws / "wiki/patterns", []
    for c in out.get("create_patterns") or []:
        try:
            (patterns / _safe_md_name(str(c["name"]))).write_text(str(c["content"]))
        except (ProposalError, KeyError) as e:
            skipped.append(f"create_patterns: {e}")
    for u in out.get("update_patterns") or []:
        try:
            p = patterns / _safe_md_name(str(u["name"]))
        except (ProposalError, KeyError) as e:
            skipped.append(f"update_patterns: {e}")
            continue
        if not p.is_file():
            skipped.append(f"update_patterns: no such pattern {p.name}")
            continue
        text, sk = apply_edits(p.read_text(), u.get("edits") or [])
        p.write_text(text)
        skipped += [f"{p.name}: {s}" for s in sk]
    if out.get("update_index"):
        (ws / "wiki/index.md").write_text(str(out["update_index"]).rstrip() + "\n")
    entry = f"## iter {k}\n\n{str(out.get('append_log', '')).strip()}\n"
    if skipped:
        entry += "\nSkipped patches:\n" + "".join(f"- {s}\n" for s in skipped)
    with (ws / "wiki/log.md").open("a") as f:
        f.write(entry + "\n")
    return skipped


def _skill_text(d: Path) -> str:
    return (d / "SKILL.md").read_text() if (d / "SKILL.md").is_file() else ""


def apply_proposal(ws: Path, p: dict) -> tuple[str, list[str]]:
    action = p.get("action")
    if action not in ("create", "patch"):
        raise ProposalError(f"bad action: {action!r}")
    name = str(p.get("name", ""))
    if not _NAME.match(name):
        raise ProposalError(f"bad skill name: {name!r}")
    d = ws / "skills" / name
    before, skipped = _skill_text(d), []
    if action == "create":
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(str(p.get("skill_md", "")))
        (d / "PURPOSE.md").write_text(str(p.get("purpose_md", "")))
    else:
        if not (d / "SKILL.md").is_file():
            raise ProposalError(f"patch: no such skill {name!r}")
        text, skipped = apply_edits(before, p.get("edits") or [])
        (d / "SKILL.md").write_text(text)
    after = _skill_text(d)
    rel = f"skills/{name}/SKILL.md"
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return diff, skipped


def append_impact(ws: Path, k: int, p: dict, diff: str, r_val: float | None, r_best: float, outcome: str) -> None:
    action, name = p.get("action", "?"), p.get("name", "")
    val = f"{r_val:.3f}" if r_val is not None else "n/a"
    head = f"{action} {name}".strip()
    entry = (f"## iter {k} — {head} — val {val} (best {r_best:.3f}) — {outcome}\n\n"
             f"- action: {action}\n- skill: {name}\n- outcome: {outcome}\n")
    if diff:
        entry += f"\n```diff\n{diff.rstrip()}\n```\n"
    with (ws / "wiki/skill-impact.md").open("a") as f:
        f.write(entry + "\n")


def git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True, text=True).stdout


def commit_all(ws: Path, msg: str) -> None:
    git(ws, "add", "-A")
    if git(ws, "status", "--porcelain").strip():
        git(ws, "-c", "user.name=wikiskill", "-c", "user.email=wikiskill@engram", "commit", "-q", "-m", msg)


def tag(ws: Path, name: str) -> None:
    git(ws, "tag", "-f", name)


def restore_skills(ws: Path, ref: str) -> None:
    """Roll skills/ back to `ref` exactly: files from the ref, and nothing the ref lacks."""
    shutil.rmtree(ws / "skills")
    (ws / "skills").mkdir()
    git(ws, "checkout", ref, "--", "skills")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_wikiskill_workspace.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/engram/wikiskill/workspace.py tests/test_wikiskill_workspace.py
git commit -m "feat(wikiskill): workspace layout, patch ops, proposal apply, skill-impact, git audit"
```

---

### Task 4: Prompts E.2 / E.3 + roles (rollout, sampling, maintain, propose)

**Files:**
- Create: `src/engram/wikiskill/prompts/maintainer.md`, `src/engram/wikiskill/prompts/proposer.md`
- Create: `src/engram/wikiskill/roles.py`
- Test: `tests/test_wikiskill_roles.py`

**Interfaces:**
- Consumes: `run_claude` (Task 1), `Bench`/`Task` (Task 2), `skill_section`, `read_wiki` (Task 3).
- Produces:
  ```python
  class Trace(TypedDict): id: str; split: str; prompt: str; response: str; answer: str; gold: str; score: float; cost_usd: float
  def rollout(bench, ws: Path, tasks: list[Task], *, skills_text: str, model: str, parallel: int, out_dir: Path) -> list[Trace]
  def mean_score(traces: list[Trace]) -> float
  def sample_traces(traces: list[Trace], *, max_fail=5, max_pass=3, cap_chars=15000) -> list[Trace]
  def maintain(ws: Path, sample: list[Trace], *, model: str) -> dict          # Maintainer JSON (validated keys)
  def propose(ws: Path, k: int, traces: list[Trace], *, model: str, max_turns: int = 25) -> dict   # proposal JSON
  MAINTAINER_SCHEMA, PROPOSER_SCHEMA: dict
  ```
- `rollout` writes `out_dir/<task_id>.json` (the Trace + `claude` raw JSON under `"raw"`), skips tasks whose file already exists (resume), uses `ThreadPoolExecutor(parallel)`. A `run_claude` error → response "" and score 0, but the trace records `"error": text`.
- `sample_traces`: failing = score < 1.0. Stable order (by id). Cap applies to `response` text.
- Maintainer user message = `"# Execution traces\n\n" + per-trace blocks + "\n\n# Current wiki\n\n" + read_wiki(ws)`.
- Proposer user message = index + skill-impact + outcome summary lines `"<id>\t<PASS|FAIL>\tpred=<answer>\tgold=<gold>"`; system prompt is `proposer.md` with `{task_desc}` → bench description and `{iter}` → k.

- [ ] **Step 1: Write the prompt files**

`prompts/maintainer.md` — E.2 verbatim (copy the block from the paper; only fix `log.md` naming to `wiki/log.md` which the paper itself uses in E.2):

```
You are a Wiki Maintainer Agent for an LLM skill evolution system.

Your job is to maintain a structured knowledge base (wiki) that documents patterns observed during agent execution -- both successes and failures. You must perform DEEP ANALYSIS of execution logs to identify root causes, not just surface-level symptoms.

## Wiki Structure

The wiki is organized as:

- wiki/index.md -- Concise catalog of known patterns (one line per pattern)
- wiki/log.md -- Chronological evolution log (iterations, scores, accept/reject)
- wiki/skill-impact.md -- Record of which skills were tried and their outcomes
- wiki/patterns/ -- One page per pattern with detailed evidence and analysis

## Your Input

1. Execution traces from the latest iteration -- including full agent execution logs showing what actions the agent took, what commands it ran, and what environment feedback it observed
2. The current wiki context (index, log, pattern pages)

## Your Output (Incremental Edit Mode)

Return a JSON object with these keys:

- "create_patterns": list of {"name": "pattern-name.md", "content": "..."} -- new patterns (full content)
- "update_patterns": list of {"name": "existing-pattern.md", "edits": [...]} -- patch existing patterns
- "update_index": full updated content of index.md (always provide the complete index)
- "append_log": "brief summary of this iteration's findings and actions"

"update_index" and "append_log" are REQUIRED. Always provide them, even if there are no new patterns. For "update_index", always provide the complete updated index content including all existing entries plus any new ones.

### Patch Operations (for update_patterns only)

For "update_patterns", each entry uses an "edits" list of patch operations:

- {"op": "append", "content": "text to add at end"}
- {"op": "replace", "target": "exact text to find", "content": "replacement text"}
- {"op": "insert_after", "target": "exact text to find", "content": "text to insert after"}

Rules for patch operations:

1. "target" must be an EXACT substring of the existing content.
2. Use "append" to add new evidence. Use "replace" to fix or refine existing text.
3. Use "insert_after" to add entries after a specific line.
4. Keep each edit minimal -- only change what's needed.
5. For NEW patterns (create_patterns), use full "content".

## Analysis Guidelines

### Deep Trace Analysis (CRITICAL)

When execution logs are provided, you MUST:

1. Read the agent's actual actions -- what commands did it issue?
2. Compare successful vs failed tasks -- what did successful tasks do differently?
3. Identify ACTION PATTERNS and strategies, not just error messages.
4. Check whether the agent followed any active skills, and whether the skill guidance was helpful or not

### Pattern Documentation Rules

1. Each pattern page should document:
   - What the pattern is (description)
   - Root cause analysis (WHY it happens, not just WHAT happens)
   - Exact command sequences from traces (what the agent did wrong / right)
   - Known solutions or workarounds (concrete action patterns with exact syntax)
2. Capture BOTH success and failure patterns:
   - **Failure patterns**: Document what went wrong and how to avoid it
   - **Success patterns**: Document strategies that consistently lead to task completion
3. Do NOT create duplicate patterns -- update existing ones with new evidence
4. Be concise. Pattern pages should be 10-30 lines, not essays.
5. Only create patterns for meaningful, generalizable observations.

### Index Description Quality (CRITICAL)

The index.md entries are the MOST IMPORTANT part of the wiki because they determine whether inference agents will read the full pattern pages.

Each index entry MUST follow this format:

- [pattern-name](wiki/patterns/pattern-name.md): PROBLEM + ROOT CAUSE + FIX in one or two sentence.

The description must be specific enough that an agent can judge relevance without reading the full page. Include the problem, root cause, AND solution.
```

`prompts/proposer.md` — E.3 with the two Claude Code substitutions (`read_file` → the Read tool; `traces/<task_id>` → `raw/iter-{iter}/<task_id>.json`; `finish()` → the final structured answer):

```
You are a Skill Proposer Agent for an LLM agent that solves {task_desc}.

Your job is to explore the wiki knowledge base and execution traces, diagnose root causes of failures, and propose a skill change (create or patch).

## Tools Available

You have one tool: `Read` -- read a wiki file, a skill file, or an execution trace. Paths are relative to the workspace root (your current directory). When you are done, your FINAL message must be the proposal JSON object described below (it is validated against a schema).

## Workflow

1. Start by reading `wiki/index.md` to understand what patterns exist
2. Read `wiki/skill-impact.md` to see what was tried before (includes full content of rejected proposals -- DO NOT repeat rejected approaches)
3. Read specific pattern pages that seem relevant to the current failures
4. Read execution traces for failed tasks via `raw/iter-{iter}/<task_id>.json` to understand root causes
5. Read the current skills under `skills/<name>/SKILL.md` if any exist
6. Decide: create (new skill) or patch (edit existing skill), or no_action
7. Emit the proposal as your final message

## Proposal Format

For creating a new skill:

- "action": "create"
- "name": skill directory name (snake_case)
- "skill_md": full SKILL.md content with YAML frontmatter + When to Apply + When NOT to Apply + Instructions
- "purpose_md": full PURPOSE.md content with Origin + Patterns Addressed + Evolution History

For patching an existing skill:

- "action": "patch"
- "name": existing skill directory name
- "edits": list of patch operations:
  - {"op": "append", "content": "text to add at end"}
  - {"op": "replace", "target": "exact text to find", "content": "replacement"}
  - {"op": "insert_after", "target": "exact text to find", "content": "text to insert after"}
  Each "replace" target should be a short, specific section -- not the entire file. If you need to change most of the file, use "action": "create" instead.

If no action is needed: {"action": "no_action"}

## Rules

1. Read the wiki FIRST -- don't propose something that was already tried and rejected. skill-impact.md contains full content of rejected proposals.
2. Focus on action patterns and concrete strategies.
3. Keep skills concise and actionable.
4. You MUST read at least 4 execution traces before proposing a skill change. Target your exploration based on the trace summary.
5. Prefer patching existing skills over creating new ones when the existing skill is partially correct.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_wikiskill_roles.py
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
    assert (out / "t0.json").exists() and json.loads((out / "t1.json").read_text())["gold"] == "A"
    assert all("## Skills\nS" in s and tools == [] for _, s, tools in calls)
    assert r.mean_score(traces) == 0.5
    calls.clear()
    traces2 = r.rollout(LiveMath(), tmp_path, tasks, skills_text="", model="haiku", parallel=2, out_dir=out)
    assert calls == [] and [t["id"] for t in traces2] == ["t0", "t1"]  # resumed from disk, same order


def test_rollout_error_is_zero_score(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="Not logged in", structured=None, turns=0, cost_usd=0, is_error=True))
    tr = r.rollout(LiveMath(), tmp_path, [_task(0)], skills_text="", model="haiku", parallel=1, out_dir=tmp_path / "o")
    assert tr[0]["score"] == 0.0 and tr[0]["error"] == "Not logged in"


def test_sample_traces_budget_and_cap():
    traces = [_trace(i, 0.0, resp="f" * 20000) for i in range(9)] + [_trace(i, 1.0) for i in range(9, 15)]
    s = r.sample_traces(traces)
    fails = [t for t in s if t["score"] < 1.0]
    passes = [t for t in s if t["score"] == 1.0]
    assert len(fails) == 5 and len(passes) == 3
    assert all(len(t["response"]) == 15000 for t in fails)
    assert [t["id"] for t in fails] == ["t0", "t1", "t2", "t3", "t4"]


def test_maintain_builds_prompt_and_validates(monkeypatch, git_repo, tmp_path):
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


def test_maintain_missing_structured_raises(monkeypatch, git_repo, tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    monkeypatch.setattr(r, "run_claude", lambda *a, **k: ClaudeResult(
        text="prose", structured=None, turns=1, cost_usd=0, is_error=False))
    with pytest.raises(r.RoleError):
        r.maintain(ws, [], model="haiku")


def test_propose_prompt_and_result(monkeypatch, git_repo, tmp_path):
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_roles.py -q`
Expected: FAIL — import error

- [ ] **Step 4: Implement roles.py**

```python
# src/engram/wikiskill/roles.py
"""The three LLM roles of §3.2 as thin `run_claude` calls: Inference Agent (rollout), Wiki
Maintainer (one call, JSON), Skill Proposer (ReAct over Read, JSON final). Plus Appendix C sampling."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from engram.wikiskill.bench import Bench, Task
from engram.wikiskill.claude import run_claude
from engram.wikiskill.workspace import read_wiki

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


def _infer(bench: Bench, ws: Path, task: Task, system: str, model: str, out_dir: Path, split: str) -> Trace:
    f = out_dir / f"{task['id']}.json"
    if f.exists():
        return json.loads(f.read_text())
    prompt = bench.user_prompt(task)
    res = run_claude(prompt, system=system, model=model, cwd=ws, tools=list(bench.tools))
    resp = "" if res.is_error else res.text
    tr: Trace = {"id": task["id"], "split": split, "prompt": prompt, "response": resp,
                 "answer": _pred(resp), "gold": task["answer"], "score": bench.score(task, resp),
                 "cost_usd": res.cost_usd}
    if res.is_error:
        tr["error"] = res.text
    f.write_text(json.dumps({**tr, "raw": res.raw}, ensure_ascii=False, indent=1))
    return tr


def _pred(resp: str) -> str:
    import re
    hits = re.findall(r"<answer>\s*([^<]*?)\s*</answer>", resp)
    return hits[-1] if hits else ""


def rollout(bench: Bench, ws: Path, tasks: list[Task], *, skills_text: str, model: str,
            parallel: int, out_dir: Path) -> list[Trace]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name  # iter-<k> → train; val-<k> → val; eval-<split>-<label> → <split>
    split = "train" if name.startswith("iter-") else name.split("-")[1] if name.startswith("eval-") else name.split("-")[0]
    system = bench.system_prompt(skills_text)
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        return list(ex.map(lambda t: _infer(bench, ws, t, system, model, out_dir, split), tasks))


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


def maintain(ws: Path, sample: list[Trace], *, model: str) -> dict:
    prompt = ("# Execution traces\n\n" + "\n".join(_trace_block(t) for t in sample)
              + "\n\n# Current wiki\n\n" + read_wiki(ws))
    res = run_claude(prompt, system=(PROMPTS / "maintainer.md").read_text(), model=model, cwd=ws,
                     tools=[], json_schema=MAINTAINER_SCHEMA)
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
            task_desc: str = "multiple-choice mathematics questions") -> dict:
    system = (PROMPTS / "proposer.md").read_text().replace("{iter}", str(k)).replace("{task_desc}", task_desc)
    prompt = (f"# wiki/index.md\n\n{(ws / 'wiki/index.md').read_text()}\n\n"
              f"# wiki/skill-impact.md\n\n{(ws / 'wiki/skill-impact.md').read_text()}\n\n"
              f"# Training outcomes (iteration {k})\n\n{outcome_summary(traces)}\n")
    res = run_claude(prompt, system=system, model=model, cwd=ws, tools=["Read"],
                     max_turns=max_turns, json_schema=PROPOSER_SCHEMA)
    if not isinstance(res.structured, dict) or "action" not in res.structured:
        raise RoleError(f"proposer returned no structured output: {res.text[:200]!r}")
    return res.structured
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_wikiskill_roles.py -q`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/engram/wikiskill/roles.py src/engram/wikiskill/prompts/maintainer.md src/engram/wikiskill/prompts/proposer.md tests/test_wikiskill_roles.py
git commit -m "feat(wikiskill): roles — rollout, Appendix C sampling, Maintainer and Proposer with Appendix E prompts"
```

---

### Task 5: Loop — Algorithm 1, state, resume, evaluate

**Files:**
- Create: `src/engram/wikiskill/loop.py`
- Test: `tests/test_wikiskill_loop.py`

**Interfaces:**
- Consumes: Task 2 (`Bench`, `read_split`), Task 3 (workspace fns), Task 4 (`rollout`, `mean_score`, `sample_traces`, `maintain`, `propose`, `RoleError`).
- Produces:
  ```python
  def load_state(ws) -> dict      # {"iteration": int, "r_best": float|None, "history": [...], "stopped": bool}
  def save_state(ws, st) -> None
  def evolve(ws: Path, bench: Bench, *, model: str, iters: int, parallel: int, log=print) -> dict
  def evaluate(ws: Path, bench: Bench, *, split: str, model: str, parallel: int, skills_dir: Path | None = None) -> float
  ```
- `evolve` steps per iteration k (1..iters), each persisted in `state.json` so a crash resumes:
  1. `r_best is None` → val rollout with empty skills into `raw/val-0` → `r_best`; commit.
  2. `r_best == 1.0` → `stopped=True`, break.
  3. train rollout with `skill_section(ws)` into `raw/iter-k`.
  4. `maintain` → `apply_maintainer` → commit `"iter k: wiki"`.
  5. `propose` → if `no_action` (or `ProposalError`/`RoleError`): `append_impact(..., outcome="NoAction"|"Invalid")`, history entry, commit, continue.
  6. `apply_proposal` → val rollout into `raw/val-k` → `r_val`.
  7. `r_val > r_best` → `r_best = r_val`, `tag accepted-k`, `Accepted`; else `restore_skills(ws, last_accepted_ref)`, `Rejected`. `last_accepted_ref` = latest `accepted-*` tag or the init commit tag `accepted-0` (create it at baseline step).
  8. `append_impact`, history entry `{k, action, name, r_val, r_best, outcome, cost}`, commit `"iter k: <outcome>"`.
- `evaluate`: rollout `split` into `raw/eval-<split>-<skills-label>` and return mean score; `skills_dir` overrides the skill text source (transfer experiment); label = `"self"` or the other workspace's dir name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wikiskill_loop.py
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


def _ws(git_repo, tmp_path, n_train=4, n_val=2):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    mk = lambda i: {"id": f"t{i}", "question": f"q{i}", "choices": {}, "answer": "A"}
    write_split(ws / "dataset/train.jsonl", [mk(i) for i in range(n_train)])
    write_split(ws / "dataset/val.jsonl", [mk(100 + i) for i in range(n_val)])
    write_split(ws / "dataset/test.jsonl", [mk(200 + i) for i in range(3)])
    return ws


def _wire(monkeypatch, val_scores, proposals, maint=None):
    """val_scores: list of per-iteration val answers (fraction correct); proposals: list of dicts."""
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

    def fake_propose(ws, k, traces, *, model, max_turns=25, task_desc=""):
        p = proposals[it["prop"]]
        it["prop"] += 1
        return p

    monkeypatch.setattr(L, "rollout", fake_rollout)
    monkeypatch.setattr(L, "propose", fake_propose)
    monkeypatch.setattr(L, "maintain", maint or (lambda ws, sample, *, model: {
        "create_patterns": [{"name": "p.md", "content": "pat"}], "update_patterns": [],
        "update_index": "# idx\n- p", "append_log": "found p"}))
    return it


CREATE = {"action": "create", "name": "s1", "skill_md": "---\nname: s1\n---\nrule\n", "purpose_md": "why"}
PATCH = {"action": "patch", "name": "s1", "edits": [{"op": "append", "content": "rule2"}]}


def test_gate_accept_reject_and_impact(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 1.0, 0.5], proposals=[CREATE, PATCH])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, log=lambda *a: None)
    # iter1: val 1.0 > 0.5 → Accepted; iter2: 0.5 ≤ 1.0 → Rejected, but r_best hit 1.0 so loop stops before iter2
    assert st["r_best"] == 1.0 and st["stopped"] is True
    assert [h["outcome"] for h in st["history"]] == ["Accepted"]
    impact = (ws / "wiki/skill-impact.md").read_text()
    assert "— Accepted" in impact and "+rule" in impact
    assert (ws / "skills/s1/SKILL.md").read_text().endswith("rule\n")
    assert "accepted-1" in w.git(ws, "tag")
    assert (ws / "wiki/patterns/p.md").exists() and "found p" in (ws / "wiki/log.md").read_text()


def test_reject_rolls_back_skills_keeps_wiki(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 0.5, 1.0], proposals=[CREATE, CREATE])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, log=lambda *a: None)
    assert [h["outcome"] for h in st["history"]] == ["Rejected", "Accepted"]
    assert not (ws / "skills/s1").exists() or st["history"][1]["outcome"] == "Accepted"
    impact = (ws / "wiki/skill-impact.md").read_text()
    assert impact.count("## iter") == 2 and "— Rejected" in impact
    assert (ws / "wiki/patterns/p.md").exists()  # wiki never rolled back
    # after the rejected iter 1 the skills dir was restored to accepted-0 (empty)
    assert "accepted-0" in w.git(ws, "tag")


def test_strict_gate_equal_is_rejected(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    _wire(monkeypatch, val_scores=[0.5, 0.5], proposals=[CREATE])
    st = L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, log=lambda *a: None)
    assert st["history"][0]["outcome"] == "Rejected" and not (ws / "skills/s1").exists()


def test_no_action_and_invalid_proposal(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    it = _wire(monkeypatch, val_scores=[0.5], proposals=[{"action": "no_action"}, {"action": "patch", "name": "ghost", "edits": []}])
    st = L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, log=lambda *a: None)
    assert [h["outcome"] for h in st["history"]] == ["NoAction", "Invalid"]
    assert it["val"] == 1  # no validation rollouts spent
    assert "— NoAction" in (ws / "wiki/skill-impact.md").read_text()


def test_resume_from_state(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    it = _wire(monkeypatch, val_scores=[0.5, 0.75, 0.9], proposals=[CREATE, PATCH])
    L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, log=lambda *a: None)
    st = L.load_state(ws)
    assert st["iteration"] == 1 and st["r_best"] == 0.75
    L.evolve(ws, FakeBench(), model="m", iters=2, parallel=1, log=lambda *a: None)
    st = L.load_state(ws)
    assert st["iteration"] == 2 and st["r_best"] == 0.9 and it["val"] == 3
    assert (ws / "skills/s1/SKILL.md").read_text().endswith("rule\nrule2\n")


def test_role_error_is_invalid_not_crash(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
    _wire(monkeypatch, val_scores=[0.5], proposals=[])

    def boom(ws, k, traces, *, model, max_turns=25, task_desc=""):
        raise r.RoleError("no output")

    monkeypatch.setattr(L, "propose", boom)
    st = L.evolve(ws, FakeBench(), model="m", iters=1, parallel=1, log=lambda *a: None)
    assert st["history"][0]["outcome"] == "Invalid"


def test_evaluate_self_and_transfer(monkeypatch, git_repo, tmp_path):
    ws = _ws(git_repo, tmp_path)
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_loop.py -q`
Expected: FAIL — import error

- [ ] **Step 3: Implement loop.py**

```python
# src/engram/wikiskill/loop.py
"""Algorithm 1 of WikiSkill, line by line. The harness owns gating, rollback, skill-impact.md and the
git audit trail; the wiki is never rolled back. state.json makes every step resumable."""
from __future__ import annotations

import json
from pathlib import Path

from engram.wikiskill.bench import Bench, read_split
from engram.wikiskill.roles import RoleError, maintain, mean_score, propose, rollout, sample_traces
from engram.wikiskill.workspace import (ProposalError, append_impact, apply_maintainer, apply_proposal,
                                        commit_all, restore_skills, skill_section, skills_dir_section, tag)


def load_state(ws: Path) -> dict:
    p = ws / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"iteration": 0, "r_best": None, "history": [], "stopped": False}


def save_state(ws: Path, st: dict) -> None:
    (ws / "state.json").write_text(json.dumps(st, indent=1))


def _last_accepted(st: dict) -> str:
    acc = [h["k"] for h in st["history"] if h["outcome"] == "Accepted"]
    return f"accepted-{acc[-1]}" if acc else "accepted-0"


def evolve(ws: Path, bench: Bench, *, model: str, iters: int, parallel: int, log=print) -> dict:
    train, val = read_split(ws / "dataset/train.jsonl"), read_split(ws / "dataset/val.jsonl")
    st = load_state(ws)
    if st["r_best"] is None:  # Algorithm 1 line 3: baseline validation with S_0 = ∅
        st["r_best"] = mean_score(rollout(bench, ws, val, skills_text="", model=model, parallel=parallel,
                                          out_dir=ws / "raw/val-0"))
        tag(ws, "accepted-0")
        save_state(ws, st)
        commit_all(ws, f"iter 0: baseline val {st['r_best']:.3f}")
        log(f"baseline R_best={st['r_best']:.3f}")
    for k in range(st["iteration"] + 1, iters + 1):
        if st["r_best"] >= 1.0:  # line 5
            st["stopped"] = True
            save_state(ws, st)
            log("R_best = 1.0 — early stop")
            break
        traces = rollout(bench, ws, train, skills_text=skill_section(ws), model=model, parallel=parallel,
                         out_dir=ws / f"raw/iter-{k}")  # line 8
        log(f"iter {k}: train R={mean_score(traces):.3f}")
        skipped = apply_maintainer(ws, maintain(ws, sample_traces(traces), model=model), k)  # lines 9–10
        commit_all(ws, f"iter {k}: wiki" + (f" ({len(skipped)} patches skipped)" if skipped else ""))
        entry = {"k": k, "action": None, "name": None, "r_val": None, "r_best": st["r_best"], "outcome": None}
        diff, r_val = "", None
        try:
            p = propose(ws, k, traces, model=model)  # line 11
            entry["action"], entry["name"] = p.get("action"), p.get("name")
            if p["action"] == "no_action":
                entry["outcome"] = "NoAction"
            else:
                diff, _ = apply_proposal(ws, p)  # line 12
                r_val = mean_score(rollout(bench, ws, val, skills_text=skill_section(ws), model=model,
                                           parallel=parallel, out_dir=ws / f"raw/val-{k}"))  # line 13
                if r_val > st["r_best"]:  # line 14: strict
                    st["r_best"] = r_val
                    tag(ws, f"accepted-{k}")
                    entry["outcome"] = "Accepted"
                else:
                    restore_skills(ws, _last_accepted(st))  # line 17: skills only
                    entry["outcome"] = "Rejected"
        except (RoleError, ProposalError) as e:
            p = {"action": entry["action"] or "invalid", "name": entry["name"] or ""}
            entry["outcome"] = "Invalid"
            log(f"iter {k}: invalid proposal: {e}")
            restore_skills(ws, _last_accepted(st))
        entry["r_val"], entry["r_best"] = r_val, st["r_best"]
        append_impact(ws, k, p, diff, r_val, st["r_best"], entry["outcome"])  # line 19
        st["history"].append(entry)
        st["iteration"] = k
        save_state(ws, st)
        commit_all(ws, f"iter {k}: {entry['outcome']} val={r_val if r_val is None else f'{r_val:.3f}'} best={st['r_best']:.3f}")
        log(f"iter {k}: {entry['outcome']} val={r_val} best={st['r_best']:.3f}")
    return st


def evaluate(ws: Path, bench: Bench, *, split: str, model: str, parallel: int,
             skills_dir: Path | None = None) -> float:
    tasks = read_split(ws / f"dataset/{split}.jsonl")
    label = skills_dir.parent.name if skills_dir else "self"
    skills = skills_dir_section(skills_dir) if skills_dir else skill_section(ws)
    return mean_score(rollout(bench, ws, tasks, skills_text=skills, model=model, parallel=parallel,
                              out_dir=ws / f"raw/eval-{split}-{label}"))
```

Note for `test_evaluate_self_and_transfer`: `skills_dir=other.parent` is `tmp_path/other/skills`, so `skills_dir.parent.name == "other"`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_wikiskill_loop.py -q`
Expected: 7 passed. If `test_reject_rolls_back_skills_keeps_wiki` fails on the `accepted-0` restore, confirm `tag(ws, "accepted-0")` runs before the first `commit_all` of the baseline — a tag needs a commit; `init_workspace` already made one, so it's fine.

- [ ] **Step 5: Commit**

```bash
git add src/engram/wikiskill/loop.py tests/test_wikiskill_loop.py
git commit -m "feat(wikiskill): Algorithm 1 loop — strict gate, rollback, early stop, resumable state, evaluate"
```

---

### Task 6: CLI — `engram evolve init|run|eval|status`

**Files:**
- Create: `src/engram/wikiskill/cli.py`
- Modify: `src/engram/cli.py` (add subparser + dispatch)
- Test: `tests/test_wikiskill_cli.py`

**Interfaces:**
- Produces:
  ```python
  def add_evolve_parser(sub) -> None      # registers "evolve" with init/run/eval/status
  def run_evolve(args) -> int
  ```
- `init`: `--ws`, `--bench livemath`, `--seed 0` → `init_workspace`, `download_records(ws/".hf-cache")`, `tasks_from_records`, `make_splits(bench.SPLIT_SIZES)`, `write_split` ×3, write `dataset/meta.json` `{bench, seed, sizes}`, commit `"wikiskill: dataset split"`. Add `.hf-cache/` to `ws/.gitignore`.
- `run`: `--ws`, `--model haiku`, `--iters 8`, `--parallel 8` → `evolve`; prints per-iteration lines; exit 0.
- `eval`: `--ws`, `--split test`, `--model haiku`, `--parallel 8`, `--skills <other-ws-or-skills-dir>` → prints `R(<split>) = 0.xxx (skills: self|<label>)`.
- `status`: prints `state.json` history as a table.
- Bench name for `run/eval/status` comes from `dataset/meta.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wikiskill_cli.py
import argparse
import json

from engram import cli
from engram.wikiskill import cli as wcli
from engram.wikiskill import workspace as w
from engram.wikiskill.bench import write_split


def test_parser_wired():
    p = argparse.ArgumentParser()
    wcli.add_evolve_parser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["evolve", "run", "--ws", "x", "--iters", "3"])
    assert a.evolve_cmd == "run" and a.iters == 3 and a.model == "haiku" and a.parallel == 8
    a = p.parse_args(["evolve", "eval", "--ws", "x", "--skills", "y"])
    assert a.split == "test" and a.skills == "y"


def test_engram_main_dispatches(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(wcli, "run_evolve", lambda args: called.setdefault("cmd", args.evolve_cmd) or 0)
    assert cli.main(["evolve", "status", "--ws", "x"]) == 0
    assert called["cmd"] == "status"


def test_init_writes_splits(monkeypatch, git_repo, tmp_path):
    recs = [{"no": i, "month": "202606", "mcq": {"question": f"q{i}", "correct_choice": {"label": "A", "text": "r"},
             "choices": [{"label": l, "text": "d"} for l in "BCDE"]}} for i in range(200)]
    monkeypatch.setattr(wcli.LiveMath, "download_records", lambda self, cache_dir: recs)
    ws = tmp_path / "ws"
    rc = wcli.run_evolve(argparse.Namespace(evolve_cmd="init", ws=str(ws), bench="livemath", seed=0))
    assert rc == 0
    assert len((ws / "dataset/train.jsonl").read_text().splitlines()) == 35
    assert len((ws / "dataset/test.jsonl").read_text().splitlines()) == 124
    assert json.loads((ws / "dataset/meta.json").read_text())["bench"] == "livemath"
    assert ".hf-cache" in (ws / ".gitignore").read_text()
    assert w.git(ws, "status", "--porcelain").strip() == ""


def test_status_prints_history(git_repo, tmp_path, capsys):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "dataset/meta.json").write_text(json.dumps({"bench": "livemath", "seed": 0}))
    (ws / "state.json").write_text(json.dumps({"iteration": 1, "r_best": 0.5, "stopped": False,
        "history": [{"k": 1, "action": "create", "name": "s", "r_val": 0.5, "r_best": 0.5, "outcome": "Accepted"}]}))
    assert wcli.run_evolve(argparse.Namespace(evolve_cmd="status", ws=str(ws))) == 0
    out = capsys.readouterr().out
    assert "Accepted" in out and "0.500" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_wikiskill_cli.py -q`
Expected: FAIL — import error

- [ ] **Step 3: Implement wikiskill/cli.py**

```python
# src/engram/wikiskill/cli.py
"""`engram evolve init|run|eval|status` — the WikiSkill harness entrypoints."""
from __future__ import annotations

import json
from pathlib import Path

from engram.wikiskill.bench import LiveMath, get_bench, make_splits, write_split
from engram.wikiskill.loop import evaluate, evolve, load_state
from engram.wikiskill.workspace import commit_all, init_workspace


def add_evolve_parser(sub) -> None:
    pe = sub.add_parser("evolve", help="WikiSkill: co-evolve skills with a persistent wiki (arXiv:2608.27454)")
    es = pe.add_subparsers(dest="evolve_cmd", required=True)
    pi = es.add_parser("init", help="create a workspace and a frozen train/val/test split")
    pi.add_argument("--ws", required=True)
    pi.add_argument("--bench", default="livemath", choices=["livemath"])
    pi.add_argument("--seed", type=int, default=0)
    pr = es.add_parser("run", help="run Algorithm 1 for --iters iterations (resumable)")
    pr.add_argument("--ws", required=True)
    pr.add_argument("--model", default="haiku")
    pr.add_argument("--iters", type=int, default=8)
    pr.add_argument("--parallel", type=int, default=8)
    pv = es.add_parser("eval", help="score the active skills on a split (empty skills = no-skill baseline)")
    pv.add_argument("--ws", required=True)
    pv.add_argument("--split", default="test", choices=["train", "val", "test"])
    pv.add_argument("--model", default="haiku")
    pv.add_argument("--parallel", type=int, default=8)
    pv.add_argument("--skills", default=None, help="another workspace's skills/ dir (cross-model transfer)")
    ps = es.add_parser("status", help="print the iteration history")
    ps.add_argument("--ws", required=True)


def _meta(ws: Path) -> dict:
    return json.loads((ws / "dataset/meta.json").read_text())


def run_evolve(args) -> int:
    ws = Path(args.ws)
    if args.evolve_cmd == "init":
        init_workspace(ws)
        bench = LiveMath()
        (ws / ".gitignore").write_text(".hf-cache/\n")
        tasks = bench.tasks_from_records(bench.download_records(ws / ".hf-cache"), args.seed)
        splits = make_splits(tasks, bench.SPLIT_SIZES, args.seed)
        for name, ts in splits.items():
            write_split(ws / f"dataset/{name}.jsonl", ts)
        (ws / "dataset/meta.json").write_text(json.dumps({"bench": bench.name, "seed": args.seed,
                                                         "sizes": bench.SPLIT_SIZES}, indent=1))
        commit_all(ws, "wikiskill: dataset split")
        print(f"initialized {ws}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        return 0
    bench = get_bench(_meta(ws)["bench"])
    if args.evolve_cmd == "run":
        st = evolve(ws, bench, model=args.model, iters=args.iters, parallel=args.parallel)
        print(f"done: iteration={st['iteration']} R_best={st['r_best']:.3f} stopped={st['stopped']}")
        return 0
    if args.evolve_cmd == "eval":
        sk = Path(args.skills) if args.skills else None
        if sk and (sk / "skills").is_dir():
            sk = sk / "skills"
        r = evaluate(ws, bench, split=args.split, model=args.model, parallel=args.parallel, skills_dir=sk)
        print(f"R({args.split}) = {r:.3f} (skills: {sk.parent.name if sk else 'self'})")
        return 0
    st = load_state(ws)  # status
    print(f"bench={_meta(ws)['bench']} iteration={st['iteration']} R_best={st['r_best']} stopped={st['stopped']}")
    print("k\taction\tskill\tval\tbest\toutcome")
    for h in st["history"]:
        val = "n/a" if h["r_val"] is None else f"{h['r_val']:.3f}"
        print(f"{h['k']}\t{h['action']}\t{h['name']}\t{val}\t{h['r_best']:.3f}\t{h['outcome']}")
    return 0
```

- [ ] **Step 4: Wire into `src/engram/cli.py`**

After the `curate` subparser block (before `args = p.parse_args(argv)`), add:

```python
    from engram.wikiskill.cli import add_evolve_parser

    add_evolve_parser(sub)
```

And in the dispatch chain, before the final `return 0`, add:

```python
    elif args.cmd == "evolve":
        from engram.wikiskill import cli as wcli

        return wcli.run_evolve(args)
```

(Import inside the branch so the module reference `wcli.run_evolve` is looked up at call time — the test monkeypatches it.)

- [ ] **Step 5: Run all wikiskill tests and the full suite**

Run: `uv run pytest tests/test_wikiskill_cli.py -q && uv run pytest tests/test_core.py tests/test_curate.py tests/test_default_dir.py -q`
Expected: all pass; existing CLI tests unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/engram/wikiskill/cli.py src/engram/cli.py tests/test_wikiskill_cli.py
git commit -m "feat(wikiskill): engram evolve init|run|eval|status"
```

---

### Task 7: Live smoke on 3 tasks, docs, PR

**Files:**
- Modify: `CLAUDE.md` (Commands section: add `engram evolve`), `README.md` (one paragraph)
- No new source files.

- [ ] **Step 1: Init a real workspace in the scratchpad**

```bash
uv run engram evolve init --ws $SCRATCH/ws-livemath --bench livemath --seed 0
```
Expected: `initialized …: train=35, val=18, test=124`; `git -C $SCRATCH/ws-livemath log --oneline` shows 2 commits.

- [ ] **Step 2: Cut a 3/3 smoke workspace**

```bash
cp -r $SCRATCH/ws-livemath $SCRATCH/ws-smoke
head -3 $SCRATCH/ws-livemath/dataset/train.jsonl > $SCRATCH/ws-smoke/dataset/train.jsonl
head -3 $SCRATCH/ws-livemath/dataset/val.jsonl > $SCRATCH/ws-smoke/dataset/val.jsonl
git -C $SCRATCH/ws-smoke commit -qam "smoke split"
uv run engram evolve run --ws $SCRATCH/ws-smoke --model haiku --iters 1 --parallel 3
```
Expected: `baseline R_best=…`, `iter 1: train R=…`, `iter 1: <Accepted|Rejected|NoAction> …`, `done: iteration=1`. Check:
- `raw/val-0/*.json` ×3, `raw/iter-1/*.json` ×3, each with `"raw"` containing `total_cost_usd`.
- `wiki/index.md` non-empty, `wiki/log.md` has `## iter 1`, `wiki/patterns/*.md` ≥ 1.
- `wiki/skill-impact.md` has one `## iter 1` entry with a diff (unless NoAction).
- `git -C $SCRATCH/ws-smoke log --oneline` shows `iter 0`, `iter 1: wiki`, `iter 1: <outcome>`.
- Sum of `total_cost_usd` across `raw/**/*.json` + Maintainer + Proposer < 0.50 USD (report the number).

- [ ] **Step 3: Resume test on the same workspace**

```bash
uv run engram evolve run --ws $SCRATCH/ws-smoke --model haiku --iters 2 --parallel 3
uv run engram evolve status --ws $SCRATCH/ws-smoke
```
Expected: no `baseline` line (resumed), `iter 2` runs, status table has rows k=1 and k=2.

- [ ] **Step 4: Update docs**

`CLAUDE.md` → §Commands, append:

```
- `uv run engram evolve init --ws <dir> --bench livemath --seed 0` / `run --ws <dir> --model haiku --iters 8` / `eval --ws <dir> --split test [--skills <other-ws>]` / `status` — **WikiSkill** (arXiv:2608.27454) skill evolution: three-layer workspace (`raw/ wiki/ skills/`), Wiki Maintainer + ReAct Skill Proposer as headless `claude -p`, strict validation gate with skills-only rollback, `wiki/skill-impact.md` audit trail. Spec: `docs/superpowers/specs/2026-09-22-wikiskill-design.md`.
```

`README.md` → add this paragraph after the curator section:

```
### Skill evolution (WikiSkill)

`engram evolve` implements *WikiSkill* (Tang et al., arXiv:2608.27454): an inference agent
rolls out on training tasks, a Wiki Maintainer compiles the traces into a persistent
`wiki/` of patterns, a ReAct Skill Proposer reads the wiki and proposes one skill change,
and a strict validation gate keeps it only if the validation score improves — the wiki is
never rolled back. First bench: LiveMathematicianBench. Design:
`docs/superpowers/specs/2026-09-22-wikiskill-design.md`.
```

- [ ] **Step 5: Full suite + lint, commit, push, PR**

```bash
uv run pytest -q
uv run ruff check src tests   # if installed
git add CLAUDE.md README.md
git commit -m "docs: engram evolve (WikiSkill) commands"
git branch --unset-upstream 2>/dev/null; git push -u origin feature/wikiskill
gh pr create --base main --title "feat: WikiSkill skill evolution (engram evolve)" --body-file <scratch>/pr-body.md
```
PR body (write to the scratchpad first — the Bash hook blocks multi-line bodies): what it implements (Algorithm 1, three layers, Appendix E prompts, Appendix C sampling, Eq. 4 gate), the smoke numbers from Step 2, deviations from the spec §Deviations, and the trailer `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

### Task 8 (after merge, separate session): the real run

Not part of this plan's code; listed so the executor knows what "done when" needs:

```bash
uv run engram evolve eval --ws <ws> --split test --model haiku       # no-skill baseline (skills/ empty)
uv run engram evolve run  --ws <ws> --model haiku --iters 8 --parallel 8
uv run engram evolve eval --ws <ws> --split test --model haiku       # WikiSkill
```
Budget estimate at ~0.005 USD per LiveMath call: 18 + 8×(35+18) = 442 rollouts ≈ 2.5 USD + 8 Maintainer + 8 Proposer (≤25 turns) ≈ 3–5 USD; test eval 124 × 2 ≈ 1.2 USD. Record the two test numbers and `status` in `PLAN.md`.
