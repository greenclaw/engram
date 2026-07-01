# RESEARCH — agent long-term memory (survey, 2026-07-01)

Three parallel research streams (academic architectures / production systems / open problems +
note-native prior art). Mechanism-first, not marketing. "Steal" = what to reuse for a curated
markdown + `[[links]]` + git memory. **All vendor benchmark numbers are self-reported and, for
LoCoMo/LongMemEval, actively disputed — trust the qualitative findings, not the deltas.**

## 0. Two philosophies (the field splits cleanly)
1. **Automated engines** — Mem0, Zep/Graphiti, Letta, Cognee, Supermemory. LLM extraction
   pipelines, contradiction resolved at *write* time, running services + vector/graph DBs.
2. **File/graph stores** — basic-memory, official MCP memory, DiffMem. Human-legible artifacts,
   forgetting punted to humans/git.
Our system is a **hybrid**: file-store SoT (philosophy 2) + an engine's *curator loop* (philosophy 1),
gated by human review.

## 1. Seminal academic ideas → what to steal
| Idea (mechanism) | Paper | Steal |
|---|---|---|
| retrieval score = `α·recency + α·importance + α·relevance`; **reflection** writes higher-level memories when importance sum crosses a threshold | Generative Agents (Park 2023, arXiv 2304.03442) | rank by recency×importance×relevance, not cosine alone; tag `importance` at write time; reflection = periodic synthesis pass |
| OS paging: small **core** (in-context) vs unbounded **archive** (searched); self-edit fns; **memory-pressure** trigger | MemGPT/Letta (Packer 2023, 2310.08560) | `MEMORY.md` index = core; topic files = archive; the 24.4KB budget overflow **is** the paging signal → evict detail |
| Ebbinghaus decay `S=e^(−Δt/S)`; recall resets Δt, ++strength; un-recalled prunes | MemoryBank (Zhong 2023, 2305.10250) | `last_accessed` + recall counter; reading a note reinforces it |
| **atomic notes {content, kw, tags, ctx-desc, embedding, links}; LLM-CONFIRMED links (shortlist→LLM decides); memory EVOLUTION (new note re-touches linked neighbors' framing)** | **A-MEM (Xu 2025, 2502.12110)** — *the blueprint* | links via shortlist→LLM (not threshold); store LLM ctx-desc+tags per note; gated evolution (diff, don't clobber curated text) |
| hippocampal index: KG of triples + **Personalized PageRank** from query-entity seeds for multi-hop | HippoRAG (2405.14831) | the `[[links]]` graph is the KG; run propagation, not just leaf-embedding |
| taxonomy: **working / episodic / semantic / procedural** memory + decision loop | CoALA (Sumers 2023, 2309.02427) | type notes; keep how-tos (procedural) out of the facts (semantic) index |
| store the **distilled lesson**, not the raw event, feed-forward | Reflexion (Shinn 2023, 2303.11366) | gotchas stay imperative + failure-derived |
| **sleep-time compute**: consolidate memory OFFLINE while idle (~5× less test-time compute, +13–18% acc) | Letta 2025 (2504.13171) | run curation as nightly `/retro`, not inline |
| consolidation as first-class ops: **consolidate / update / filter / enhance**; "swamping" makes curation mandatory | surveys 2603.07670, 2604.01707 | the curator must merge/evict, not only append |
| self-editing memory = corruption surface → provenance, diffs, rollback | SSGM governance (2603.11768) | git = provenance+rollback; gate the curator |

## 2. The hard, unsolved axis
- **Forgetting.** Pure RAG-over-vectors is append-only → "cannot selectively discard, only avoid
  retrieving" (survey 2603.07670 §9.4). Uncurated accumulation poisons precision (~10% degradation
  over long runs). **Selective forgetting is an open problem 2026** — MemoryAgentBench (2507.05257):
  *no system masters it*.
- **Contradiction.** Defensible default = **recency-wins + invalidate-don't-delete** (preserve
  history). Classify candidate-vs-stored as **compatible / contradictory / subsumes / subsumed**;
  only contradictory triggers suppression. Single choke-point writer emitting **ADD/UPDATE/DELETE/
  NOOP** (Memory-R1; Mem0's engine: top-10 similar → LLM adjudicates).
- **Temporal validity.** Attach validity to the fact. Zep/Graphiti (2501.13956): **bi-temporal** —
  valid-time (event) vs transaction-time (recorded), `valid_at/invalid_at/created_at/expired_at`;
  graph never self-contradicts; supports "what did we believe at T". Cheap markdown version:
  `updated:` / `invalidated_by:` frontmatter, prefer newest non-invalidated.
- **Security.** Memory is an attack surface: **MINJA** (2503.03704) plants malicious records via
  *query-only* interaction (>95% success), fires on a later victim query. → an **approval/provenance
  gate is also a poisoning defense** (under-marketed but real).
- **Action-space taxonomies.** AgeMem (2601.01885): ADD/UPDATE/DELETE + RETRIEVE/SUMMARY/FILTER
  (the only systematic formalization; RL-trained policy is the hard, early part). Memory-R1: +NOOP.

## 3. Benchmarks (what to test against)
- **LongMemEval** — 500 Qs over multi-session chat; uniquely tests **knowledge updates** and
  **abstention** (`_abs`: decline vs fabricate). ~30% drop for long-context assistants.
- **MemoryAgentBench** (2507.05257) — the important one: scores retrieval, test-time learning,
  long-range understanding, **and selective forgetting** (only bench that does). *Most systems fail
  conspicuously on forgetting.*
- **LoCoMo** — 1540 QA (single/multi-hop/temporal); recall+temporal, **no** updates/abstention;
  ~81 QA pairs → easy to overfit. Do NOT rely on it alone.
- Robust qualitative finding: **structured graph recall > flat chunk recall on multi-hop/temporal**;
  full-context still wins raw accuracy at ~10–50× latency/tokens.

## 4. Production systems (condensed)
- **Mem0** — vector (Qdrant) + SQLite history (immutable ADD/UPDATE/DELETE audit) + optional Neo4j
  graph. Two-phase: async rolling-summary extraction → **top-10 similar → LLM ADD/UPDATE/DELETE/NOOP**
  (dedup=contradiction, resolved at write, no decay). File or hosted. (Live repo v3 = single-call,
  different algo — don't conflate with the paper 2504.19413.)
- **Zep/Graphiti** — bi-temporal KG (episode→entity→community subgraphs), invalidate-don't-delete,
  hybrid cosine+BM25+BFS retrieval, **no LLM in the default rerank loop**. Self-host lib (BYO graph
  DB) / hosted. Best temporal.
- **Letta** — memory **blocks** {label,desc,value,limit}; core (pinned) vs archival (pgvector) vs
  recall; heartbeat self-edit; memory-pressure eviction; **sleep-time agent** shares blocks, consolidates
  offline. Running server (Postgres/SQLite).
- **Cognee** — Extract→Cognify→Load into a **typed pydantic-graph** (ontology-validated), pluggable
  graph+vector+relational; batch capture; temporal opt-in. Library/self-host.
- **Supermemory** — Cloudflare (D1+Vectorize+R2); transparent OpenAI-proxy "Infinite Chat" injects
  retrieved context past ~20k tokens. Forgetting *claimed, not documented* (marketing).
- **basic-memory (MCP)** — **markdown files ARE the SoT**; graph parsed FROM the markdown (entities=
  notes, observations=`- [category] text`, relations=`- rel [[WikiLink]]`); SQLite index (FTS +
  FastEmbed semantic) is a **rebuildable secondary**; bidirectional file-watcher; `memory://` +
  `build_context` re-hydrate a prior graph. Deliberate capture (LLM `write_note` or human edits).
  **No forgetting (by design); LLM writes files directly (no propose-diff→approve).** File-based +
  one local MCP process, no Docker. **= the mature productization of the user's manual pattern.**
- **Official MCP memory** (`@modelcontextprotocol/server-memory`) — single JSONL flat file, entity/
  relation/observation, **substring search only (no embeddings)**, manual delete, no dedup/temporal.
- **Claude-Code tools:** **claude-mem** (SQLite+FTS5 + optional Chroma + Bun worker daemon; captures
  on the **`Stop` hook**, NOT PreCompact); **MemCP** (intercepts `/compact`, blocks until saved;
  "~20× tokens" = lazy-loading/retrieval-scoping, not compression); **Memory Keeper** (SQLite WAL,
  manual `context_save`/checkpoints, session merge conflict policies, 38 tools).

## 5. Prior-art verdict — NOT a reinvention (integration gap)
Every ingredient exists; **no single system fuses all four**:
```
  git-markdown Zettelkasten (SoT)          basic-memory, DiffMem        ✅ exists
+ semantic recall over plain-text          basic-memory, sqlite-memory  ✅ exists
+ LLM curator: reviewable ADD/UPDATE/DELETE Mem0 engine, A-MEM, AgeMem   ✅ exists (not over git-md)
  DIFFS, contradiction-aware
+ human/gate approval BEFORE commit         Hermes                       ✅ exists (over skills)
──────────────────────────────────────────────────────────────────────
= the fused assembly + contradiction-aware curator + gate                ❌ NOBODY → our contribution
```
- **DiffMem** — closest shape (git+md SoT + LLM writer + diff history) but **no semantic recall**
  (grep/git only) and **no approval gate** (auto-commit).
- **basic-memory** — md-SoT + wiki-links + hybrid semantic, but LLM writes files directly (no
  propose-diff→approve, no contradiction reconciliation step).
- **Hermes** — staged unified-diff + approve/reject, but over skills, not a git-Zettelkasten w/ semantic.
- **A-MEM** — active curation + note-evolution + LLM-confirmed links, but auto (no gate, no git).

**Reuse (don't reinvent):** markdown-SoT+git, Zettelkasten atomic-notes+links, ADD/UPDATE/DELETE/NOOP
op-set, staged-diff approval, invalidate-don't-delete, semantic recall, recency×importance×relevance
scoring, decay, offline consolidation.
**Novel = the assembly + a contradiction-aware curator that emits reviewable diffs over a git-committed
markdown Zettelkasten, gated before commit.**

## Sources
Academic: [Generative Agents](https://arxiv.org/abs/2304.03442) · [MemGPT](https://arxiv.org/pdf/2310.08560) · [MemoryBank](https://arxiv.org/pdf/2305.10250) · [A-MEM](https://arxiv.org/abs/2502.12110) · [HippoRAG](https://arxiv.org/abs/2405.14831) · [CoALA](https://arxiv.org/abs/2309.02427) · [Reflexion](https://arxiv.org/abs/2303.11366) · [Sleep-time compute](https://arxiv.org/html/2504.13171v1) · [survey 2603.07670](https://arxiv.org/html/2603.07670v1) · [Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)
Open problems: [MemoryAgentBench](https://arxiv.org/pdf/2507.05257) · [AgeMem](https://arxiv.org/html/2601.01885v1) · [MINJA](https://arxiv.org/html/2503.03704) · [SSGM governance](https://arxiv.org/html/2603.11768v1) · [Consolidation problem](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation)
Production: [Mem0](https://arxiv.org/abs/2504.19413) · [Zep/Graphiti](https://arxiv.org/abs/2501.13956) · [Letta blocks](https://docs.letta.com/guides/agents/memory-blocks/) · [Cognee](https://github.com/topoteretes/cognee) · [Supermemory engine](https://supermemory.ai/blog/memory-engine)
Note-native prior art: [basic-memory](https://github.com/basicmachines-co/basic-memory) · [DiffMem](https://github.com/Growth-Kinetics/DiffMem) · [Hermes gate](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md) · [Karpathy llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) · [official MCP memory](https://github.com/modelcontextprotocol/servers/blob/main/src/memory/README.md)
