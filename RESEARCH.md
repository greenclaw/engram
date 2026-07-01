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
- **Hermes-agent (Nous Research)** — **the closest prior-art to our curator (b)**; cloned at
  `~/projects/hermes-agent` (Python, public). Findings below are **verified from code**, not marketing.
  *"Self-learning"* = a **background self-improvement review after each turn** that replays the
  conversation and distills *repeated corrections* + *hard-won workflow lessons* (concrete triggers:
  save after a 5+-tool-call task, or after a dead-end→working-path) into either a **semantic memory
  entry** or a **procedural skill** (facts↔procedures split = CoALA); review can run on a cheap aux
  model. Store (`tools/memory_tool.py`) = markdown `MEMORY.md`(env) + `USER.md`(user), `§`-delimited,
  **hard CHAR limits (2200/1375, explicitly "model-independent") as the forgetting pressure**; loaded
  as a **frozen system-prompt snapshot at session start** (mid-session writes apply next session);
  extra: a **drift-guard** (errors if the file was edited outside the §-format) + **consolidation
  when over the char limit**. Gate (`tools/write_approval.py`) = per-subsystem `write_approval`,
  **default OFF ("writes flow freely")**, ON → stage to `pending/{memory,skills}/<id>.json` as a
  **replayable payload**; skills reviewed via unified `diff`, **foreground memory writes always stage**
  ("too big to eyeball mid-loop"). Past-conversation search = SQLite **FTS5 (keyword, ~20ms), NOT
  semantic**. **Two verified gaps (= our integration edge): NO contradiction/reconciliation module**
  (compatible/contradictory/subsumes absent) and **NO semantic recall over the store** (the whole
  capped file is injected; FTS5 only over transcripts).
  → **Steal for increment 2:** candidate-triggers (repeated correction / 5+-tool workflow / dead-end→fix),
  cheap-aux-model review, replayable-staged-payload gate, drift-guard. → **Our edge:** engram adds
  *semantic recall over the store* + *contradiction-aware reconciliation* + *gate-on-by-default*
  (Hermes ships the gate OFF → the exact MINJA poisoning surface RESEARCH §2 warns about).

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
- **Hermes** — staged unified-diff + approve/reject + a real per-turn self-improvement loop (closest to
  our curator), but **gate defaults OFF**, memory is a char-capped frozen snapshot with **no semantic
  recall**, and there is **no contradiction-reconciliation step** (all verified from code, §4). Not over
  a git-Zettelkasten.
- **A-MEM** — active curation + note-evolution + LLM-confirmed links, but auto (no gate, no git).

**Reuse (don't reinvent):** markdown-SoT+git, Zettelkasten atomic-notes+links, ADD/UPDATE/DELETE/NOOP
op-set, staged-diff approval, invalidate-don't-delete, semantic recall, recency×importance×relevance
scoring, decay, offline consolidation.
**Novel = the assembly + a contradiction-aware curator that emits reviewable diffs over a git-committed
markdown Zettelkasten, gated before commit.**

## 6. Adaptive rotation policy — research flow (heuristic → auto-tuned → learned)
The "what to evict / decay / promote / consolidate / invalidate, and with what weights" decision is a
*policy*. v1 hard-codes it; this flow is how it becomes dynamic **without becoming a research project**.
**Reward signal for every learned tier is the same instrument: claude-bench (`mem_recall` hit-rate,
`mem_curate` op-precision, index size). No bench → no honest learning.**

**The binding constraint is label signal, not model capacity** — a personal store is data-starved, so a
method's viability = *does a signal feed it*, not how sophisticated it is. Three signals gate everything:
- **S1 — claude-bench:** constructed "fact F is needed" scenarios, clean ground-truth, offline, dozens–
  hundreds. Feeds weight-tuning + gate-threshold calibration + op-classifier *eval*.
- **S2 — recall telemetry:** `last_accessed` + hit-counter per note, accumulates online, noisy. Feeds
  utility regression + decay-τ estimation + eviction.
- **S3 — git curation history:** approved/rejected diffs = labelled ops, slow accrual (tens/week). Feeds
  a curator op-classifier once large enough.
- (S4 — "was the surfaced fact actually used": near-uninstrumentable, noisy — don't rely on it.)

**Learnable decision surface → technique fit** (viability set by signal, not sophistication):

| decision | technique | fit |
|---|---|---|
| predict note utility → `importance` + eviction | **logistic/linear regression** on {type, age, recall-count, `[[links]]`-centrality, size} | ✅ **best fit** — tiny-data-safe, interpretable (coeffs = "what makes a memory live"); needs S2 |
| recall ranking | learning-to-rank | pointwise-LTR = the regression above ✅; pairwise/listwise ❌ early (needs many labelled query–doc) |
| curator op ADD/UPDATE/DELETE/NOOP | classifier | ⚠️ later — LLM already does it; pays only once S3 is large. Keep LLM+gate for now |
| score weights + decay τ | auto-ML = HPO (Bayesian / CMA-ES) | ✅ = L1; "full auto-ML" model-search ❌ (a ~6-feature regression has nothing to search) |
| online keep/evict/promote | contextual bandit | ✅ v2 (S2 = reward); honest RL-lite for single-step decisions |
| (whole policy) | full RL / neuroevolution | ❌ decision is single-step not sequential → temporal credit-assignment wasted; data-starved |

**Machine-assisted, not machine-decided:** learned models rank/propose to the human gate; they don't
auto-decide until bench-validated (engram is human-in-the-loop by thesis). Climb only to the tier that pays:
- **L0 — heuristic (v1, build).** score = recency×importance×relevance (Generative Agents) + Ebbinghaus
  decay (MemoryBank) + reinforcement-on-hit + `MEMORY.md` size-pressure eviction (MemGPT) + LLM
  op-adjudication for contradictions. A *fixed-weight dynamic* policy — adaptive per-note, hand-tuned.
- **L1 — auto-tuned weights (v1.5, the justified "auto-ML").** Don't hand-pick the ~5–7 knobs
  (α_recency, α_importance, α_relevance, decay τ, evict/promote thresholds); optimize them against the
  bench objective. Non-differentiable + noisy → derivative-free: **Bayesian optimization** (optuna/skopt)
  or **CMA-ES / (μ,λ)-ES** (the "evolutionary" option — legit here: black-box search over a few knobs,
  *not* GP over a whole policy). Generalizes the μ−Zσ gate calibration. Output = a transparent tuned
  YAML, still service-less. **The tier that actually pays off — few-hundred LOC.**
- **L2 — contextual bandit (v2, online + data-efficient).** Per-note keep/evict/promote/consolidate as
  arms; context = note features (type, age, recall-count, `[[links]]`-graph centrality, size); reward =
  later recall hit / precision held. Thompson sampling / LinUCB. Learns online from real outcomes,
  degrades gracefully, far more sample-efficient than RL. Where "dynamic / logistics-based" genuinely lives.
- **L3 — full RL / neuroevolution (research-only, deferred).** RL policy over the AgeMem action space
  (ADD/UPDATE/DELETE/RETRIEVE/SUMMARY/FILTER) or evolutionary policy search. Honest blocker: needs a
  reward-labeled env/simulator + many episodes; a personal few-hundred-note store is **data-starved**;
  AgeMem itself flags the RL-trained policy as the hard, early part. Prototype-able, unlikely to beat
  L1+L2 on a personal store. Document, don't build.

**Verdict:** the highest-leverage *machine-assisted* piece is **not RL** — it's a **utility-prediction
regression** (logreg on note features, fed by S2) driving `importance`+eviction, plus **auto-tuned score
weights** (Bayesian/CMA-ES, fed by S1). Both are tiny-data-safe, interpretable, service-less, and gated on
telemetry+bench existing → they land **after increment 1** starts logging hits. RL/neuroevolution is the
seductive-wrong tool here (single-step, data-starved, human-in-loop) — a 6-feature logreg matches it and
you can read the weights. Prototype it as research, don't ship it in v1.

## Sources
Academic: [Generative Agents](https://arxiv.org/abs/2304.03442) · [MemGPT](https://arxiv.org/pdf/2310.08560) · [MemoryBank](https://arxiv.org/pdf/2305.10250) · [A-MEM](https://arxiv.org/abs/2502.12110) · [HippoRAG](https://arxiv.org/abs/2405.14831) · [CoALA](https://arxiv.org/abs/2309.02427) · [Reflexion](https://arxiv.org/abs/2303.11366) · [Sleep-time compute](https://arxiv.org/html/2504.13171v1) · [survey 2603.07670](https://arxiv.org/html/2603.07670v1) · [Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)
Open problems: [MemoryAgentBench](https://arxiv.org/pdf/2507.05257) · [AgeMem](https://arxiv.org/html/2601.01885v1) · [MINJA](https://arxiv.org/html/2503.03704) · [SSGM governance](https://arxiv.org/html/2603.11768v1) · [Consolidation problem](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation)
Production: [Mem0](https://arxiv.org/abs/2504.19413) · [Zep/Graphiti](https://arxiv.org/abs/2501.13956) · [Letta blocks](https://docs.letta.com/guides/agents/memory-blocks/) · [Cognee](https://github.com/topoteretes/cognee) · [Supermemory engine](https://supermemory.ai/blog/memory-engine)
Note-native prior art: [basic-memory](https://github.com/basicmachines-co/basic-memory) · [DiffMem](https://github.com/Growth-Kinetics/DiffMem) · [Hermes-agent repo](https://github.com/NousResearch/hermes-agent) · [Hermes memory docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) · [Hermes skills docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) · [Karpathy llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) · [official MCP memory](https://github.com/modelcontextprotocol/servers/blob/main/src/memory/README.md)
