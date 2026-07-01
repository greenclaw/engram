# CONTEXT — the user's current memory setup (baseline we build on)

The system already exists as a **hand-run, file-based, curated** memory. curated-memory adds the two
missing commodity layers (semantic recall + a formalized curator) WITHOUT throwing this away — the
files stay the source of truth.

## What exists today
- **`~/.claude/CLAUDE.md`** (global, ~78 lines after a compression pass) — standing *rules* / how-to-work.
  Loaded IN FULL every turn → must stay small. (Not memory of facts; instructions.)
- **`~/.claude/projects/<proj>/memory/`** — the actual memory store:
  - **`MEMORY.md`** — the always-loaded **index**: one line per fact (`- [Title](file.md) — hook`),
    grouped (Key Patterns / Feedback / Gotchas / Decisions / Reference). **Already over its ~24.4KB
    budget** → this is the concrete pain (MemGPT's "memory pressure" signal). Fix = evict detail to
    topic files, keep the index terse.
  - **topic files** `*.md` — **one fact each**, YAML frontmatter (`name`, `description`, `metadata.type`
    ∈ user|feedback|project|reference), body links related facts with **`[[wikilinks]]`**. Human-written,
    high-signal. Recalled by `description` relevance + surfaced inside `<system-reminder>` blocks.
    Carries a manual staleness caveat ("reflects what was true when written — verify before asserting").
- **Session logs** — `.claude/sessions/active|archive/*.md`, appended by a Stop/session-log hook +
  `/handover`. Episodic ("what happened").
- **`~/.claude/learnings/{YYYY-MM}.md`** — monthly findings from `/retro`; rule = a pattern seen 3+
  times gets promoted to CLAUDE.md / MEMORY.md.
- **Skills:** `/retro` (end-of-session learnings), `/handover` (context save), `/improve` (propose
  rules/skills), `session-log`.

## How memory moves today (all manual/curated)
```
 during session ── I decide a fact is non-obvious ──► write/update a topic file + a MEMORY.md line
 session end ────── /retro ──► learnings/{month}.md ; 3+ recurrences ──► promote to CLAUDE.md/MEMORY.md
 next session ───── MEMORY.md injected in full + relevant topic files surfaced by description match
```

## Strengths (keep — these ARE the SOTA-hard part)
- **Curation** (high signal) — the thing automated engines fail at.
- **Transparency** — plain-text, git-versioned, grep-able, human/Cursor-readable.
- **`[[links]]`** — a lightweight knowledge graph already in place.
- **Size limit on MEMORY.md** — a forgetting *pressure* (forces eviction).
- **Manual staleness caveat** — a hand-rolled temporal-validity guard.
- **Zero infra** — no service, no DB (values to preserve; user disliked claude-mem's Bun+Chroma daemon).

## Gaps (what curated-memory adds)
1. **No semantic recall** — retrieval is `description`-string match; misses relevant facts phrased differently → **increment (a)**.
2. **No formalized curator** — curation is fully manual; no contradiction/dedup/staleness handling at write, no propose-diff→approve loop → **increment (b)**.
3. **MEMORY.md over budget** — needs the recall layer so it can shrink to a true index (detail retrieved on demand).

## Validation asset
- **claude-bench** (`~/projects/claude-bench`) — A/B each increment: (a) recall on/off → fact-hit-rate;
  (b) curator op-precision on a labelled candidate set (ADD vs UPDATE-existing vs NOOP-dup vs DELETE-stale),
  exactly like the μ−Zσ gate calibration (`calibrate_gate.py`).
