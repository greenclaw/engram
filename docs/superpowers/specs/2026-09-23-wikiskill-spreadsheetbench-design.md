# WikiSkill — SpreadsheetBench (second bench)

Adds SpreadsheetBench to `engram evolve` as the paper's second setup (arXiv:2608.27454,
Table 6: 80 / 40 / 280, bash tool, E.1 SpreadsheetBench prompt) and runs it on two models.
The evolution loop (Algorithm 1) does not change; everything bench-specific sits behind
the `Bench` protocol. Parent design: `2026-09-22-wikiskill-design.md`.

## Decisions (2026-09-23)

1. **Dataset** — SpreadsheetBench *Verified* 400 (`KAKA22/SpreadsheetBench`,
   `spreadsheetbench_verified_400.tar.gz`): 80 + 40 + 280 = 400, which is what the paper's
   Table 6 sizes imply. One test case per task: `1_<id>_init.xlsx` (input) and
   `1_<id>_golden.xlsx` (reference). Split by our seeded shuffle; the exact ids of the
   SkillOpt split (`split_seed=42`, their own splitter) are not published.
2. **Models** — two self-evolution runs on the **same split**: Haiku 4.5 and Sonnet. Then the
   Table 2 cross-model transfer: Haiku's skills evaluated on Sonnet and vice versa.
3. **Sandbox instead of Docker** — the paper's agent may only touch its working directory
   ("Any attempt to access files outside this directory will fail"). We enforce that with the
   Claude Code bash sandbox (verified live 2026-09-23): `denyRead ~/`, `allowRead` =
   [task workdir, agent venv, uv Python root], writes only in cwd (+ system temp), network
   off, `allowUnsandboxedCommands: false`, Bash explicitly allowed (`--allowedTools Bash`;
   `autoAllowBashIfSandboxed` alone does not cover `python -c`). SkillLens uses a 14 GB
   Docker image for the same restriction.

## Protocol change (both benches)

```python
class Bench(Protocol):
    name: str
    task_desc: str                                   # {task_desc} in the E.3 Proposer prompt
    def system_prompt(self, skill_section: str) -> str: ...
    def prepare(self, ws: Path, task: dict, workdir: Path) -> None: ...     # stage inputs
    def user_prompt(self, task: dict, workdir: Path) -> str: ...
    def claude_opts(self, ws: Path, workdir: Path) -> dict: ...            # tools, sandbox, env, turns
    def score(self, task: dict, response: str, workdir: Path) -> tuple[float, str]: ...
```

`score` returns `(score, answer)`: `answer` is what the trace records as the prediction —
the chosen letter for LiveMath, the grader's verdict (e.g. first differing cell) for
SpreadsheetBench, which is exactly the feedback the Maintainer needs. LiveMath is migrated
to this protocol with unchanged behavior (`prepare` no-op, no tools, one turn). The
Proposer gets `bench.task_desc` (critique: `loop.py` called `propose` without it, so every
bench would have been described as "multiple-choice mathematics questions").

## SpreadsheetBench specifics

- **init** downloads the tarball into `.hf-cache/`, extracts to `.data/` (gitignored), writes
  `dataset/{train,val,test}.jsonl` with `{id, instruction, instruction_type, answer_position,
  answer}` (`answer` = `answer_position`, the "gold" shown to the Maintainer), and creates
  `.venv/` (`uv venv` + `openpyxl pandas`) — the agent's Python.
- **Workdir** per rollout = `work/<rollout-dir>/<task_id>/` (gitignored — outputs are MBs),
  holding a copy of the init file as `input.xlsx`; the agent must write `output.xlsx` there.
- **Prompt** = E.1 SpreadsheetBench prompt verbatim; the user message fills the listed fields
  (`working_directory`, `instruction`, `spreadsheet_path`, `spreadsheet_content`,
  `instruction_type`, `answer_position`, `output_path`). `spreadsheet_content` = the first 5
  rows of every sheet (SpreadsheetBench's "5 rows" setting), tab-separated, capped.
- **Agent** = `claude -p` with Bash, the sandbox above, `PATH=<ws>/.venv/bin:$PATH`,
  `--max-turns 30` (SkillOpt: "up to 30 turns"), `--no-session-persistence`, and
  `--output-format stream-json --verbose`.
- **Trace** (`raw/…/<id>.json`): `response` = compact transcript of the session — every Bash
  command, its output (each capped at 2,000 chars), and the final text — so the Maintainer and
  Proposer see "what commands it ran and what the environment returned" (§3.2.2). `raw` =
  the final `result` event (same shape as `--output-format json`, so `usage_of` still works).
- **Score** = port of the official `compare_workbooks` (openpyxl `data_only=True`; numbers
  rounded to 2 decimals; datetimes to Excel serial; types must match; `""` ≡ `None`; every
  cell of every range in `answer_position`; a sheet missing from the output fails). No
  formula recalculation — a formula-only answer reads as `None`, the "library constraint"
  the paper mentions and SkillOpt's learned rule targets ("write evaluated static values").
  Missing or unreadable `output.xlsx` → 0. Hard score only (one test case).
- **Known dataset defect** — in 2 of 400 tasks (`13-1`, `60-7`) the position names no sheet and
  the official grader's default (first sheet of the golden file) differs from `answer_sheet`.
  Kept as in the official grader and SkillLens, for comparability.

## Runs

```
engram evolve init --ws <runs>/ssb-s0 --bench spreadsheetbench --seed 0
cp -R <runs>/ssb-s0 <runs>/ssb-haiku-s0 ; cp -R <runs>/ssb-s0 <runs>/ssb-sonnet-s0
per model M in {haiku, sonnet}:
  eval --ws ssb-M-s0 --split test --model M            # no skill
  run  --ws ssb-M-s0 --model M --iters 8
  eval --ws ssb-M-s0 --split test --model M            # self-evolved
cross: eval --ws ssb-haiku-s0 --model haiku --skills ssb-sonnet-s0 ; and the mirror
```

A live smoke (3 / 3 / 3 tasks, Haiku) precedes the full runs; its per-call tokens set the
consumption estimate reported before launching (calls / tokens, not USD).

## Tests

- protocol: LiveMath through the new protocol reproduces its previous prompts and scores
- spreadsheet grader: equal values, 2-decimal rounding, int vs "5" string, datetime serial,
  `""` vs `None`, multi-range + quoted sheet names, missing sheet, missing file → 0
- preview: first 5 rows per sheet, capped
- prepare: copies `init.xlsx` → `input.xlsx`; prompt carries absolute paths inside workdir
- claude_opts: sandbox JSON (denyRead ~/, allowRead set), `--allowedTools Bash`, PATH prefix
- run_claude stream mode: transcript from tool_use/tool_result events, output capping,
  result event as `raw`, `--no-session-persistence` always
- init: splits 80/40/280, `.gitignore` covers `.data/ work/ .venv/ .hf-cache/`
- loop passes `bench.task_desc` to the Proposer

## Non-goals

LibreOffice recalculation, the 912-task original set, SkillOpt's exact split ids,
per-role model overrides (both runs are self-evolution, one model per run).
