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
gap fix, NOT an effect size on realistic queries); n=12; end-to-end effect on assistant answers (the
original claude-bench framing) and the live recall hook/wiring are still unvalidated/unbuilt.

## Increment 2 — curator (b)  [THE NOVEL PART]
**Status (2026-07-02):** deterministic core ✅ (`curate.py`: change-set → diff → human gate → pathspec-only
git commit; path-contained; UPDATE preserves frontmatter; INVALIDATE keeps the note). Scope cut for v1
(manual-first, a strict prefix of the auto path): pending-store + Stop-hook deferred to fast-follow.
**The LLM adjudication (skill) + `mem_curate` bench are NOT built — the novel claim is unvalidated until
they land.**
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
**Done when:** claude-bench "curator op-precision" — a **labelled candidate set** (facts that should ADD /
UPDATE-an-existing / NOOP-dup / DELETE-stale) → measure op-accuracy + contradiction-catch, calibrate the
gate threshold exactly like `calibrate_gate.py` (μ−Zσ over N verifier votes).

## Validation (claude-bench, both increments)
- Scenario `mem_recall`: does semantic recall find the needed fact? (hit-rate; recall on vs off).
- Scenario `mem_curate`: labelled ops → curator precision/recall on ADD/UPDATE/DELETE/NOOP + contradiction.
- Reuse `~/projects/claude-bench` harness + the calibration pattern.

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
