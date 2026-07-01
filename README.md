# engram

A curated, git-native, plain-text **agent memory system** for Claude Code — the missing
assembly no existing tool ships: a markdown Zettelkasten as source-of-truth + semantic
recall on top + an **LLM curator that proposes reviewable ADD/UPDATE/DELETE diffs**
(contradiction-aware, invalidate-don't-delete) + a **human/gate approval before the git
commit**. The approval gate doubles as a memory-poisoning defense.

## Status
- **Research: DONE** (2026-07-01) — see `RESEARCH.md` (3-stream survey: academic + production + open-problems/prior-art, with sources).
- **Design direction: AGREED** — build a thin, **service-less** layer over the user's own `memory/*.md` (Option **i**), reusing proven mechanisms; the novel value is the **curator (b)**, retrieval (a) is commodity we build thin for zero-infra/full control.
- **Build: NEXT SESSION.** Start with increment 1 (semantic recall). See `PLAN.md`.

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
