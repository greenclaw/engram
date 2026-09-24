# PLAN — engram build

**Decision (Option i):** build a thin, **service-less** layer over the user's own `memory/*.md`
(not adopt basic-memory's server). Rationale: (a) semantic recall is commodity but building it thin
matches the user's zero-infra/transparency values and reuses their bge-m3/pgvector muscle; (b) the
**curator is the novel value** — built regardless. Validate every increment with `claude-bench`.

## Guardrails (non-negotiable)
- **Markdown files stay the source of truth.** Any index/embedding store is a *rebuildable secondary*
  (basic-memory principle). Never the authority.
- **No running service, no external DB.** Local on-disk index (sqlite-vec or a numpy/`.npy` cosine
  over a few-hundred files). Rebuild on change; no daemon.
- **Curator NEVER writes durable memory directly.** It emits a **git diff**; a human/policy gate
  approves → commit. (Provenance + poisoning defense; SSGM/Hermes.)
- **Invalidate-don't-delete** for contradictions (Zep). Git already holds history; frontmatter marks stale.
- **Don't clobber curated prose.** Evolution/UPDATE proposes edits as diffs, never silent overwrite.
- **Curation runs OFFLINE** (nightly `/retro` or Stop-hook), not inline in a task (sleep-time compute).

## Architecture
```
 SoT: memory/*.md  frontmatter{ name, description, type, importance, updated, invalidated_by, links[] }
      + MEMORY.md (L1 index, one line/fact)

 (a) RECALL  (read path, service-less)          (b) CURATOR  (write path, offline + gated)
  ┌───────────────────────────────────┐          candidate fact/lesson (from session/retro)
  │ L1  MEMORY.md one-liners (always)  │            │
  │ L2  semantic top-k topic files ────┼──recall──► │ retrieve top-k similar existing notes
  │     score = recency×importance×rel │            │ LLM adjudicate → ADD | UPDATE | DELETE | NOOP
  │ L3  read full file (only chosen id)│            │ classify pair: compatible|contradict|subsume
  └───────────────────────────────────┘            │ contradict → invalidate-don't-delete (frontmatter)
     progressive disclosure (~10× tokens)          │ emit git DIFF ──► human/gate approve ──► commit
     reinforce: bump updated/last_seen on hit       (optional A-MEM "evolution": re-touch linked notes, diffed)
```

## Increment 1 — semantic recall (a)  [START HERE]
**Build:**
- `index.py` — walk `memory/*.md`, parse frontmatter+body, embed (`description` + body head). Store
  vectors + metadata in a local `.npy`/sqlite-vec file. Rebuild on file change / `--reindex`.
- `recall.py <query|context>` — embed query → cosine over index → **score = recency × importance ×
  relevance** (Generative Agents): relevance=cosine, importance=frontmatter `importance` (default via
  `type`: gotcha/feedback high, reference low), recency=decay on `updated`. Return top-k **L2**
  (id + description), NOT full bodies. Caller reads L3 (full file) only for chosen ids.
- Wire as a recall hook / skill so surfaced facts come from semantic match, not just `description` string.
**Reuse:** basic-memory (md=SoT, index=secondary), Generative Agents (scoring), claude-mem/MemCP
(progressive disclosure), MemoryBank (reinforcement on hit).
**Embedder options (open decision):** (i) local ONNX bge-m3 int8 (already have it, `scripts/embeddings`);
(ii) a tiny local model (fastembed); (iii) the AI-gateway `/embed` if running. Prefer service-less local.
**Done when:** recall surfaces the needed note in top-k at a higher hit-rate than `description`-string
match (A/B, K≥5). ✅ **MET (2026-07-01):** semantic **100%@5 / 67%@1** vs lexical **0%** on a labelled
paraphrase set (`bench_recall.py` / `tests/test_eval.py`). Built as `embed.py`+`core.py`+`store.py`+`cli.py`;
retrieval hit-rate measured directly (needs no LLM) rather than via claude-bench's transcript harness.
Scoring uses a Generative-Agents weighted **sum** with **min-max-normalized** components — a literal
product, and un-normalized raw cosine, both let note-type/recency drown query relevance (the bench caught it).
**Validity caveats:** the set is constructed for ~zero lexical overlap (existence proof of the semantic
gap fix, NOT an effect size on realistic queries); n=12. **Quality dims measured 2026-07-07**
(`bench_quality.py`): confusables p@1 88%/hit@5 100%, negation 100%, exact-keyword 100%, RU↔EN 100%,
scale 112 notes ≡ 12-note baseline (p@1 67%, hit@5 100%, same misses — the @1 gap is the scoring-weights
knob, §6, not retrieval); off-topic abstention 2/3 (one leak inside the known cosine overlap band).
Small n per dim — smoke coverage. Still unmeasured: end-to-end effect on assistant answers (the
original claude-bench framing) — the live hook wiring now exists, so dogfooding telemetry is the path.

## Increment 2 — curator (b)  [THE NOVEL PART]
**Status (2026-07-06): increment 2 complete.** Deterministic core ✅ (`curate.py`: change-set → diff →
human gate → pathspec-only git commit; path-contained; UPDATE preserves frontmatter; INVALIDATE keeps
the note). LLM adjudication ✅ (`/engram-curate` skill, incl. the decision-5 evolution re-touch step).
`mem_curate` bench ✅. Auto-gate ✅ calibrated (see done-when below). Pending-store + Stop-hook ✅
(`pending.py`: enqueue on Stop, dedup by session_id, skill drains — the manual flow, auto-fed).
**Build:**
- `curate.py` — input = candidate facts (from a session transcript / `/retro`). For each:
  1. recall top-k similar existing notes (reuse (a)).
  2. LLM adjudicates op ∈ **ADD | UPDATE | DELETE | NOOP** + classify vs each neighbor
     (compatible|contradictory|subsumes|subsumed) — Mem0/Memory-R1 pattern.
  3. contradictory → **invalidate-don't-delete**: set neighbor `invalidated_by:` + supersede; ADD new.
  4. emit a **unified git diff** of all proposed edits (new file / edit / frontmatter change).
- **Gate:** present the diff; human or a policy gate (e.g. claude-bench-tuned confidence) approves →
  `git commit` (provenance). NOOP/rejected → nothing.
- Optional: **evolution** (A-MEM) — on ADD, re-touch `[[linked]]` notes' framing, as diffs, gated.
- Optional: **reflection/consolidation** pass (Generative Agents / sleep-time) — nightly job merges
  redundant notes, promotes 3+-recurrence learnings to MEMORY.md, evicts low-utility (the "swamping" fix).
**Reuse:** Mem0 write-engine (top-k→LLM op), AgeMem op-set, Hermes staged-diff+approve, Zep invalidate,
SSGM provenance/rollback via git, sleep-time offline cadence.
**Done when:** curator op-precision on a **labelled candidate set** → op-accuracy + contradiction-catch,
then calibrate the gate threshold like `calibrate_gate.py` (μ−Zσ over N verifier votes).
✅ **Instrument built + first numbers (2026-07-02):** `bench_curate.py` (12 labelled candidates, 4 op
classes, headless `claude -p`) → **op_accuracy 100%, contradiction_catch 100%, false_invalidate 0 (3/3
runs)**. *Caveats: clean-case set — an existence proof that skill rules + recall evidence adjudicate
unambiguous candidates correctly; scores are model-dependent (`--model` flag).*
✅ **Auto-gate calibrated (2026-07-06):** hard set (`curate_dataset_hard.json` — confusable neighbor
pairs, value-change-vs-contradiction boundaries, narrower-vs-refining boundaries, cross-lingual
duplicates) → **100/100/0 × 5 runs**; μ−2σ over per-change confidences (n=60) → **τ=0.770, coverage
95%, risk 0%**. Wired as `curate apply --auto-threshold`; body-UPDATEs (prose re-touches) always fall
back to the human gate (decision-3 carve-out). *Honest caveat: risk=0 is vacuous — five runs produced
no op errors even on the boundary set (0/60 ⇒ error rate ≲5% at 95% CI). τ is a defensible knob, but
flipping auto-commit ON by default should wait for an error-producing set or real-usage telemetry.*

## Validation (claude-bench, both increments)
- Scenario `mem_recall`: does semantic recall find the needed fact? (hit-rate; recall on vs off).
- Scenario `mem_curate`: labelled ops → curator precision/recall on ADD/UPDATE/DELETE/NOOP + contradiction.
- Reuse `~/projects/claude-bench` harness + the calibration pattern.

## WikiSkill — first full run (2026-09-23)
`engram evolve` (arXiv:2608.27454), LiveMathematicianBench, split 35/18/124 seed 0, Claude Haiku 4.5
in all three roles, 8 iterations. Workspace: `~/projects/engram-runs/livemath-haiku-s0`.

| | test (124) | meta-gold items (57) | other items (67) |
|---|---|---|---|
| no skills | 22.6% | 1.8% | 40.3% |
| WikiSkill (1 skill, 270 lines) | **58.1%** | 73.7% | 44.8% |

- Validation 0.278 → 0.722; 4 Accepted (iters 1, 3, 5, 7), 4 Rejected; every accepted step
  patched the same skill `recognize_meta_options`; a second skill (iter 4) was rejected.
- Paired test delta +35.5 pts (bootstrap 95% CI +24.2…+45.2): 52 items fixed, 8 broken.
- **Almost all of it is the benchmark shortcut.** The meta option is gold on 46% of items and never
  a distractor; the evolved skill picks it 42/57 times it is shown (vs 1/57 without skills). On the
  67 items without it the delta is +4.5 pts, CI −7.5…+16.4 — not distinguishable from zero.
- Consumption: 706 `claude -p` calls, 4.1M input / 6.7M output tokens, ~1050 API-minutes
  (≈ 3 h wall-clock at `--parallel 8`), on the claude.ai subscription.
- Caveat found later (2026-09-23): this run's Proposer could Read `raw/val-*` traces (validation
  answers) — its reads were not logged, so exposure is unknown. Fixed for later runs (explicit
  read-deny + streamed Proposer log); the test numbers are unaffected in mechanism (test traces did
  not exist during evolution) but the validation gate may have been fit.
- Takeaway: the harness works end to end and reproduces the paper's *shape* (large LiveMath gain);
  the size of the gain on this bench is not evidence of transferable reasoning skill. Next
  signal-bearing runs: a bench without a structural shortcut (SpreadsheetBench), or LiveMath scored
  on non-meta items only.

## WikiSkill — SpreadsheetBench, Haiku (2026-09-24)
Verified 400, split 80/40/280 seed 0, Claude Haiku 4.5 in all roles, 8 iterations, sandboxed bash
agent. Workspace: `~/projects/engram-runs/ssb-haiku-s0` (the Sonnet run will copy the same `ssb-s0`).

| test (280) | accuracy | vs no skill (paired, bootstrap 95% CI) |
|---|---|---|
| no skill | 31.1% | — |
| skill after iter 2 (checkpoint) | 68.9% | +37.9 pts [+31.4, +44.3] |
| **final skill (iter 4, R_best)** | **62.9%** | **+31.8 pts [+25.7, +37.9]** — 99 fixed, 10 broken |

- Validation 0.450 → 0.700; accepted iters 1, 2, 4 (all on one skill,
  `compute_cell_values_programmatically`, 544 lines at the end); iters 3, 5–8 rejected; 14 wiki patterns.
- The gain is a real environment procedure, not a benchmark shortcut: the dominant no-skill failure is
  an empty cell (formula written without a cached value — the grader reads cached values only), 174 of
  280 test verdicts; with the final skill 65. Same rule SkillOpt learned on GPT-5.5.
- **Gate overfitting**: iter 4 was accepted on +2 validation tasks (28 vs 26 of 40), yet on test it is
  6.1 pts *below* the iter-2 skill (CI [−11.4, −0.7]). The strict `>` gate on a 40-task split selects
  on noise — the paper's own Appendix B caveat; our run shows it on held-out data.
- Consumption: 1,856 `claude -p` calls, 164M input tokens (mostly cache reads), 11.2M output,
  ~1,800 API-minutes. Interruptions survived by resume: a network/DNS outage (57 calls re-rolled), a
  revoked subscription session overnight (26 calls re-rolled after `/login`), a machine sleep (runs now
  go under `caffeinate`). One extra test eval happened by accident (the iter-2 checkpoint above):
  after a restart the "no-skill eval" step evaluated the then-current skill — fixed in `ssb-run.sh`.

## Decisions (resolved 2026-07-01)
1. **Embedder** — local ONNX bge-m3 int8 (reuse `scripts/embeddings`). Service-less.
2. **Curator trigger** — **Stop-hook** (auto at session end). `/retro` stays as a manual entry point.
3. **Gate mode** — **auto-apply above a bench-calibrated confidence threshold.** ⚠️ Sequencing:
   until claude-bench (`mem_curate`) yields a μ−Zσ threshold, the curator starts in **human-review**;
   auto-commit switches on only after calibration. **Evolution re-touches of curated prose stay
   human-gated even after** (they edit hand-written text — the SOTA-hard asset).
4. **Frontmatter** — add `importance`, `updated`, `invalidated_by`. **Not** explicit `links[]` —
   parse `[[wikilinks]]` from the body (A-MEM style), no duplicate list to keep in sync.
5. **Evolution (A-MEM re-touch)** — **ON in v1**, but only as gated diffs, never silent overwrite
   (see the human-gate carve-out in decision 3).
6. **Scope** — **generic** tool over any `memory/*.md` dir (path as `--dir` arg). Project-agnostic.

## Non-goals (v1)
- No vector DB service, no knowledge-graph DB, no MCP server (keep it a CLI + hooks).
- v1 ships the **heuristic** rotation policy (score + decay + reinforcement + size-pressure eviction).
  Learned rotation (auto-tuned weights → bandit → RL) is a staged **research flow**, not a v1 build
  item — see `RESEARCH.md` §6. No RL policy trained blind without the claude-bench reward signal.
- Not trying to beat Zep on temporal — cheap frontmatter validity is sufficient for a personal store.
