# WikiSkill in engram — design

Faithful implementation of *WikiSkill: Compiling Agent Experience into Persistent
Knowledge for Skill Evolution* (Tang et al., Google Research, arXiv:2608.27454,
Aug 2026) as an `engram` subpackage. "Faithful" means: Algorithm 1 line by line,
the three-layer workspace of §3.1, the exact role prompts of Appendix E, the
trace-sampling budget of Appendix C, and the strict validation gate of Eq. 4.
Where Claude Code forces a deviation it is listed in §Deviations.

## Decisions (2026-09-22)

1. **Dataset `D`** — a public benchmark from the paper, not personal tasks.
   First LiveMathematicianBench (single-step, no tools, exact-match scoring),
   second SpreadsheetBench via the same `Bench` interface.
2. **Placement** — subpackage `src/engram/wikiskill/` with its own workspace
   (`raw/ wiki/ skills/`). It does not touch the memory store, `core.py`, or
   `curate.py`; the paper's metric gate is not the human gate of `curate`.
3. **Models** — one model for all three roles (self-evolution, the paper's main
   setting); default `haiku` (`claude-haiku-4-5-20251001`). One `--model`
   flag; per-role overrides are a later experiment (cross-model transfer).
4. **Orchestration** — a deterministic Python harness (`engram evolve`) drives
   Algorithm 1; every LLM role is one headless `claude -p` subprocess. No API
   key, no SDK, no interactive session as orchestrator.

## Workspace layout (§3.1)

```
<ws>/
  dataset/{train,val,test}.jsonl   frozen split; {id, question, choices, answer}
  raw/iter-<k>/<task_id>.json      immutable traces: {split, prompt, response, answer, gold, score}
  raw/val-<k>/<task_id>.json       validation traces (k=0 is the empty-skill baseline)
  wiki/index.md                    one line per pattern (E.2 index format)
  wiki/log.md                      evolution log, appended by the Maintainer
  wiki/skill-impact.md             appended ONLY by the harness after gating
  wiki/patterns/<name>.md          one page per pattern
  skills/<name>/SKILL.md           full skill content (frontmatter + When to Apply + When NOT + Instructions)
  skills/<name>/PURPOSE.md         Origin + Patterns Addressed + Evolution History
  state.json                       {bench, model, iteration, r_best, history: [...]}
```

The workspace is a git repository. Every iteration ends in one commit; an
accepted iteration also gets tag `accepted-<k>`. `raw/` and `wiki/` are never
rolled back; `skills/` is restored from the last `accepted-*` tag on rejection.

## Bench interface

```python
class Task(TypedDict): id: str; question: str; choices: dict[str, str]; answer: str
class Bench(Protocol):
    name: str
    tools: list[str]                      # claude -p --tools; [] for LiveMath
    def load(self, split: str) -> list[Task]: ...
    def system_prompt(self, skill_section: str) -> str   # E.1 prompt with {skill_section}
    def user_prompt(self, task: Task) -> str
    def score(self, task: Task, response: str) -> float  # in [0, 1]
```

**LiveMath**: dataset `LiveMathematicianBench/LiveMathematicianBench` pulled via
`huggingface_hub` at `init`, shuffled with `--seed`, cut 35/18/124 (Table 6),
written to `dataset/*.jsonl`; offline afterwards. The raw records always store the
correct option under "A" with B–E as distractors, so the five options are
re-lettered per task with the same seed — otherwise "always answer A" scores 100%. Score = `1.0` iff the letter
inside the last `<answer>…</answer>` equals the gold letter, else `0.0`
(missing or malformed answer = 0).

## Roles (all `claude -p --model <m> --output-format json`)

**Inference Agent** (§3.2.1, Eq. 1). One call per task, parallel (`--parallel`,
default 8). `--system-prompt` = bench prompt with `{skill_section}` = the full
text of every `skills/*/SKILL.md` (full injection). `--tools` = bench tools
(none for LiveMath). The wiki is never in the prompt (§5.1 ablation). Output →
`raw/iter-k/<id>.json`.

**Wiki Maintainer** (§3.2.2, Eq. 2). Exactly one call per iteration. Input =
sampled traces per Appendix C (≤5 failing + ≤3 passing, each trace text capped
at 15,000 chars) + the full current wiki (index, log, every pattern page).
System prompt = E.2 verbatim. Structured output via `--json-schema`:
`create_patterns[{name, content}]`, `update_patterns[{name, edits[]}]`,
`update_index` (full text, required), `append_log` (required). The harness
applies the three patch ops `append | replace | insert_after`; a `replace` or
`insert_after` whose `target` is not an exact substring is skipped and logged
to `wiki/log.md` — the wiki is never corrupted by a bad patch. Then it revises
`index.md`, appends to `log.md`, commits.

**Skill Proposer** (§3.2.3, Eq. 3). One ReAct agent per iteration:
`claude -p` with cwd = workspace, `--tools Read`, `--max-turns 25` (paper:
10–20 ReAct turns; the flag is accepted by `claude` 2.1.278 though absent from
its `--help`). System prompt = E.3 verbatim except path aliases (see
Deviations). Initial user message = `wiki/index.md` + `wiki/skill-impact.md` +
a summary of all train outcomes (`id, pass/fail, prediction, gold`). The agent
reads pattern pages and `raw/iter-k/<id>.json` on demand. Final answer via
`--json-schema`: `{action: "create"|"patch"|"no_action", name, skill_md,
purpose_md}` or `{action: "patch", name, edits[]}`. Atomic: one proposal, one
skill.

All three prompts live as files in `src/engram/wikiskill/prompts/` so they can
be diffed against the paper.

## Gating and rollback (§3.2.4, Eq. 4, Algorithm 1)

```
R_best ← R(val rollout with S_0 = ∅)            # raw/val-0
for k in 1..K:
    if R_best == 1.0: break
    T_train ← rollout(train, S_{k-1})            # raw/iter-k
    T_sample ← sample(T_train)                   # Appendix C
    W' ← Maintainer(W, T_sample)
    P ← Proposer(W', S_{k-1}, T_train)
    if P.action == no_action: record NoAction; continue
    S' ← apply(S_{k-1}, P)
    R_val ← R(rollout(val, S'))                  # raw/val-k
    if R_val > R_best: S_k ← S'; R_best ← R_val; tag accepted-k; a=Accepted
    else:              S_k ← S_{k-1} (git restore skills/); a=Rejected
    append skill-impact.md(P, R_val, a); commit
```

`skill-impact.md` entry: `## iter k — <create|patch> <skill> — val <R_val> (best
<R_best>) — Accepted|Rejected|NoAction`, then metadata and the **full unified
diff** of the proposal (the Proposer must see rejected content in full).

`state.json` is written after every step, so `run` resumes from the last
completed iteration after a crash.

## CLI

```
engram evolve init   --ws <dir> --bench livemath --seed 0
engram evolve run    --ws <dir> --model haiku --iters 8 --parallel 8
engram evolve eval   --ws <dir> --split test [--skills <other-ws>]
engram evolve status --ws <dir>
```

`eval` reports `R(T_test)` for the active skills; `--skills` points at another
workspace's `skills/` for the Table 2 cross-model transfer experiment.
`eval` with an empty `skills/` is the no-skill baseline.

## Modules

| file | responsibility | ~LOC |
|---|---|---|
| `wikiskill/bench.py` | `Bench` protocol, `livemath` loader/score, split writer | 120 |
| `wikiskill/workspace.py` | layout, patch ops, skill apply, impact entry, git | 180 |
| `wikiskill/roles.py` | `run_claude()` + the three role wrappers, trace sampling | 160 |
| `wikiskill/loop.py` | Algorithm 1, `state.json`, resume | 150 |
| `wikiskill/cli.py` | `evolve` subparser wired into `engram/cli.py` | 80 |
| `wikiskill/prompts/*.md` | E.1 LiveMath, E.2 Maintainer, E.3 Proposer | — |

## Tests (pure logic, no LLM)

- patch ops: `append`, exact-substring `replace`/`insert_after`, skip-and-log on miss
- LiveMath score: last `<answer>` wins, case, malformed → 0
- trace sampling: ≤5 fail / ≤3 pass, 15,000-char cap, deterministic order
- gate: strict `>`, rollback restores `skills/`, early stop at 1.0, `no_action` path
- `skill-impact.md` entry format and full diff
- resume: `state.json` after iteration 3 → `run` starts at 4
- role wrappers: `run_claude` monkeypatched; each role builds the right argv
  (`--tools`, `--json-schema`, cwd) and parses its output

Live smoke (manual): `init` + `run --iters 1` on 3 train / 3 val tasks with Haiku.

## Done when

`engram evolve run` completes 8 iterations on LiveMath with Haiku unattended;
`eval --split test` yields a no-skill number and a WikiSkill number;
`skill-impact.md` holds at least one Accepted and one Rejected entry.

## Deviations from the paper (and why)

- **`traces/<id>` alias** → the Proposer reads `raw/iter-<k>/<id>.json`
  directly; the E.3 prompt is edited to name the real path. The paper's alias
  is a workspace-environment detail, not a method detail.
- **`finish(proposal)` tool** → `--json-schema` structured final output. Same
  contract, native to `claude -p`.
- **Model** — Claude Haiku 4.5 instead of Qwen/Gemma/Gemini; numbers are not
  comparable to Table 1, only the no-skill → WikiSkill delta is.
- **Runs** — one evolution run per experiment in v1 (paper: 3 runs + paired
  bootstrap). Repeating with `--seed` is the same command; bootstrap
  significance is a later addition.
- **Git per iteration** — the paper does not version the workspace; engram
  does, for the audit trail. It changes nothing in the algorithm.

## Known benchmark property (verified 2026-09-22)

LiveMathematicianBench is "substitution-resistant" by design (arXiv:2604.01754): for a fraction of
items the true theorem is replaced by the meta option *"One of the remaining options is correct, but
a stronger result can be proven."* In the released data that option appears **only** when it is the
correct answer — 81 of the 177 tasks in our split (46%). "Pick the meta option when present" is
therefore a strong shortcut, and the first live-smoke Proposer found it immediately ("default to the
meta option"). We keep the benchmark unchanged (a faithful reproduction evaluates what the paper
evaluated); when reading LiveMath gains — ours or the paper's — check how much of the delta is this
shortcut, e.g. by scoring meta and non-meta items separately.

## Non-goals (v1)

Skill retrieval/triggering (paper limitation 1), wiki pruning (limitation 3),
neutral-proposal acceptance (limitation 2), the EvoSkill/SkillOpt/Trace2Skill
baselines, per-role model overrides, SpreadsheetBench (second bench, same
interface, separate spec).
