# engram

A curated, git-native, plain-text **agent memory system** for Claude Code — the missing
assembly no existing tool ships: a markdown Zettelkasten as source-of-truth + semantic
recall on top + an **LLM curator that proposes reviewable ADD/UPDATE/DELETE diffs**
(contradiction-aware, invalidate-don't-delete) + a **human/gate approval before the git
commit**. The approval gate doubles as a memory-poisoning defense.

## Status
- **Research: DONE** (2026-07-01) — see `RESEARCH.md` (3-stream survey: academic + production + open-problems/prior-art, with sources).
- **Design direction: AGREED** — build a thin, **service-less** layer over the user's own `memory/*.md` (Option **i**), reusing proven mechanisms; the novel value is the **curator (b)**, retrieval (a) is commodity we build thin for zero-infra/full control.
- **Increment 1 (semantic recall): BUILT + VALIDATED** (2026-07-01) — service-less bge-m3 (ONNX, no service/torch) + `.npy` index + `recency×importance×relevance` recall + abstention floor. Done-when met: semantic **100%@5** vs lexical **0%@5** (A/B, `bench_recall.py`). *Caveat: the labelled set is constructed for ~zero query↔note token overlap, so this is an existence proof of closing the semantic gap — not an effect size on a realistic query mix (exact-keyword regression, confusables, scale still untested).* Commands in `CLAUDE.md`.
- **Increment 2 (curator): BUILT + first validation** (2026-07-02) — `engram curate apply` (change-set → unified diff → human gate → git commit) + the `/engram-curate` skill (TDD-tested: baseline agent self-approved the gate; with the skill it stops and hands the gate back) + `bench_curate.py` (**op_accuracy 100%, contradiction_catch 100%, false_invalidate 0 — 3/3 runs** on a 12-candidate clean-case set; existence proof, not a hard-set score). First live dogfood run curated this project's own store end-to-end.
- **Live recall wiring: BUILT** (2026-07-02) — `engram hook` on UserPromptSubmit: every prompt is semantically recalled against the store, hits land in context, abstention keeps noise out; failures never block the prompt (~1s). All four thesis ingredients now run live end-to-end.
- **Next:** harder mem_curate set + μ−Zσ gate calibration → fast-follow: pending-store + Stop-hook → invalidated-note downweight in recall. See `PLAN.md`.

## Why (one paragraph)
The user already hand-runs the hard part — **curation** (high-signal, human-written topic files + a `MEMORY.md` index + `[[links]]` + `learnings/`), the exact thing automated memory engines fail at. The gaps are the two commodities they lack: **semantic recall** and a **formalized curator loop** (contradiction/staleness handling). Prior art proves each ingredient separately (basic-memory, DiffMem, Hermes, A-MEM, Mem0) but **nobody fused all four** — so this is an *integration gap, not a research gap*. Not a reinvention; a novel assembly.

## Files
- `RESEARCH.md` — the landscape, the seminal ideas to steal (with attribution), the hard/unsolved axis (forgetting, contradiction, temporal, poisoning), benchmarks, and the prior-art verdict (what's novel vs commodity). Sources inline.
- `CONTEXT.md` — the user's *current* memory setup (what exists, why) — grounding so the next session doesn't rediscover it.
- `PLAN.md` — the a+b architecture, what to reuse vs build, the build increments, and validation via `claude-bench`. Open decisions listed.

## Resume checklist (next session)
1. Read `CONTEXT.md` (current setup) → `RESEARCH.md` (verdict + ideas) → `PLAN.md` (increments).
2. Confirm the open decisions in `PLAN.md` (embedder choice, trigger, gate mode).
3. Build **increment 1** = service-less semantic recall over `memory/*.md`, with a `claude-bench` A/B scenario (recall on/off → hit-rate).
4. Then **increment 2** = the curator (propose diff + contradiction-check + approval gate).

## Related
- **claude-bench** (`~/projects/claude-bench`, github.com/greenclaw/claude-bench) — the instrument to *validate* each increment (A/B recall; calibrate the curator's op-precision like the μ−Zσ gate).
- The user's live memory: `~/.claude/projects/<proj>/memory/` (`MEMORY.md` + topic files), session-log hook, `~/.claude/learnings/`, `~/.claude/CLAUDE.md`.
