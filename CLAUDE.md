# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**engram** — a curated, git-native, plain-text **agent memory system** for Claude Code. **Increment 1 (service-less semantic recall) is built + validated** (2026-07-01); increment 2 (the curator) is next. See `PLAN.md`.

The thesis (novelty is an *integration* gap, not a research gap): fuse four ingredients that each exist separately but that **no single system combines** —
1. a git-markdown Zettelkasten as **source of truth** (`memory/*.md` + `MEMORY.md` index + `[[links]]`),
2. **semantic recall** over those plain-text files,
3. an **LLM curator** that emits reviewable **ADD / UPDATE / DELETE / NOOP** *diffs*, contradiction-aware,
4. a **human/gate approval before the git commit**.

The curator (3) is the novel contribution; recall (2) is commodity we build thin for zero-infra/full control. The gate (4) doubles as a memory-poisoning defense.

## Read in this order before doing anything

The design lives entirely in these docs — read them before proposing code, they encode decisions you'd otherwise re-litigate:
- **`CONTEXT.md`** — the user's *existing* hand-run memory setup (`~/.claude/.../memory/`, session-log, `learnings/`). engram extends this; it does **not** replace it. The files stay the source of truth.
- **`RESEARCH.md`** — the landscape, the seminal mechanisms to reuse (with attribution), the hard axis (forgetting / contradiction / temporal / poisoning), benchmarks, and the prior-art verdict.
- **`PLAN.md`** — the a+b architecture, guardrails, the two build increments with their "done when" criteria, and the **6 open decisions**.
- **`README.md`** — one-page entry point + resume checklist.

## Guardrails (non-negotiable — these define the whole approach)

Violating any of these means building the wrong system. From `PLAN.md`:
- **Markdown files are the source of truth.** Any index/embedding store is a *rebuildable secondary* — never the authority (basic-memory principle; hence `*.npy` / `index.db` are gitignored).
- **No running service, no external DB.** Local on-disk index (sqlite-vec, or numpy `.npy` cosine over a few-hundred files). Rebuild on change; no daemon. The user explicitly disliked claude-mem's Bun+Chroma daemon.
- **The curator NEVER writes durable memory directly.** It emits a **git diff**; a human/policy gate approves → commit. Provenance + poisoning defense.
- **Invalidate-don't-delete** for contradictions — mark stale via frontmatter (`invalidated_by:`); git holds history.
- **Don't clobber curated prose.** UPDATE proposes edits as diffs, never a silent overwrite. The hand-written curation is the SOTA-hard asset — preserve it.
- **Curation runs OFFLINE** (`/retro` / Stop-hook), not inline in a task (sleep-time-compute pattern).

## Decisions (resolved 2026-07-01 — do not re-ask)

Confirmed with the user; full rationale + sequencing in `PLAN.md` §Decisions:
1. **Embedder** — local ONNX bge-m3 int8 (reuse `scripts/embeddings`). Service-less.
2. **Curator trigger** — Stop-hook (auto at session end); `/retro` stays as manual entry.
3. **Gate** — auto-commit above a claude-bench-calibrated threshold. Until that threshold exists, curator runs human-review; **evolution re-touches of curated prose stay human-gated regardless**.
4. **Frontmatter** — add `importance`, `updated`, `invalidated_by`; NOT explicit `links[]` (parse `[[wikilinks]]` from body).
5. **Evolution (A-MEM re-touch)** — ON in v1, but only as gated diffs, never silent overwrite.
6. **Scope** — generic tool over any `memory/*.md` dir (`--dir` arg). Project-agnostic.

**Learned memory-rotation** (auto-tuned weights → bandit → RL) is a staged **research flow** (`RESEARCH.md` §6), not a v1 build item — v1 ships the heuristic policy.

## Commands

Python via **`uv`** (never pip). Deps are light: onnxruntime + tokenizers + numpy + pyyaml + huggingface-hub (no torch).
- `uv sync` — set up the venv.
- `uv run pytest -q` — full suite (`test_core` pure logic; `test_recall`/`test_eval` load the real bge-m3 and **skip** if it's not in the local HF cache).
- `uv run pytest tests/test_core.py -q` — fast pure-logic tests only (no model).
- `uv run engram index --dir <memory/>` then `uv run engram recall "<query>" --dir <memory/> -k 5` — the CLI.
- `uv run python bench_recall.py` — the increment-1 done-when A/B (semantic vs lexical hit-rate over `tests/fixtures/recall_dataset.json`).

Embedder: local **ONNX bge-m3** resolved from the HF cache with `local_files_only` (never downloads). Override the repo with `ENGRAM_EMBED_REPO`.

## Modules (increment 1 — built)

- `embed.py` — service-less bge-m3 via onnxruntime; ONNX graph already bakes in CLS pooling → tokenize → `sentence_embedding` → L2-normalize.
- `core.py` — note parsing (frontmatter+body, handles `type:` and nested `metadata.type`) + scoring. **Score is a Generative-Agents weighted sum, not a literal product** (a product zeroes an old-but-critical fact); weights = the L1 auto-tune knobs (`RESEARCH.md` §6).
- `store.py` — build `.engram/index.npy` + `meta.json` (rebuildable secondary), recall = one matmul → **min-max-normalized** relevance × importance × recency. Normalization matters: raw cosine is compressed (~0.4–0.7), so without it importance/recency drown query match (the bench caught this). **Discovery is recursive** (rglob, skips MEMORY.md at any level + `.engram/`); **recall auto-rebuilds** when the source drifts (note added/edited/deleted, via count + mtime).
- `cli.py`, `eval.py` (the A/B instrument).

**Robustness policy:** bad notes **fail loud** — `core.MemoryNoteError` names the offending file (invalid YAML, or non-numeric `importance`); one corrupt note aborts the index rather than being silently skipped (curated data must surface).

**Abstention:** recall drops hits below a raw-cosine floor (`RELEVANCE_FLOOR`, default 0.35, env `ENGRAM_RELEVANCE_FLOOR`) → an unrelated query returns `[]` instead of a confident wrong hit. Deliberately conservative: bge-m3's relevant/irrelevant cosine bands **overlap** (~0.37–0.45; measured real-hit min 0.425 vs junk up to 0.44), so no clean τ exists — the default sits safely below real hits (never false-abstains) and only catches blatant off-topic. It's a bench-calibratable §6 knob.

**Still untested quality dims (next):** confusables/precision, negation, cross-lingual matrix, exact-keyword regression, scale (>100 notes).

## Build plan — next

**Increment 2 — curator (the novel part):** `curate.py` (candidate facts → recall top-k neighbors → LLM adjudicates op + classifies pair compatible|contradictory|subsumes|subsumed → emit unified git diff → gate → commit). Validate with `claude-bench` `mem_curate`: op-precision on a *labelled* candidate set + contradiction-catch; calibrate the gate threshold like `calibrate_gate.py` (μ−Zσ over N verifier votes).

Increment-1 validation (done): `mem_recall` A/B measured directly (retrieval hit-rate needs no LLM) — semantic **100%@5** vs lexical **0%@5** on the labelled paraphrase set. claude-bench's transcript harness (`~/projects/claude-bench`) is the right tool for the curator/gate work above.

## Non-goals (v1)

No vector-DB service, no knowledge-graph DB, no MCP server (CLI + hooks only). No blind RL policy in v1 (learned rotation is a staged research flow — `RESEARCH.md` §6). Not trying to beat Zep on temporal — cheap frontmatter validity is enough for a personal store.
