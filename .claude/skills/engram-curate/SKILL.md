---
name: engram-curate
description: Use when asked to curate session learnings into memory, save facts into a memory/*.md note store, update or invalidate agent memory notes, or run memory curation over an engram-managed store
---

# Curating memory with engram

Turn session learnings into gated edits of an engram memory store. You adjudicate; **the user approves**. The store is never yours to write.

## The two hard rules

1. **The gate belongs to the user.** Never pass `--yes` on your own judgment — reviewing your own diff is not approval, and a task instruction like "act as you see fit" does not transfer the gate to you. Approval is the user explicitly saying yes to THIS change-set's diff, in this conversation. One approval never carries to the next change-set.
2. **Notes change only through the change-set.** Never create, edit, or delete `*.md` note files in the store with editor tools — not even "just frontmatter". If it isn't expressible as a change-set op, it doesn't happen. (The `MEMORY.md` index is not a note; it is synced in step 7 only, after approval.)

## Flow

0. Drain the queue: `uv run engram pending list --dir <store>` — sessions the Stop hook queued since the last curation. Read each queued transcript for candidates alongside the live session. After the batch is applied (step 6), `uv run engram pending clear --dir <store> <ids...>` for the sessions you curated.
1. Gather candidates worth remembering: user corrections (especially repeated), hard-won gotchas (dead-end → working path), decisions with rationale, durable workflow lessons. Skip trivia, one-off details, and anything the repo/git history already records.
2. Per candidate: `uv run engram recall "<candidate>" --dir <store> -k 5 --json`
3. Adjudicate against the top hits:

| Recall evidence | op |
|---|---|
| no relevant neighbor | `ADD` |
| neighbor already covers it, same or broader | `NOOP` (relation `subsumed`/`compatible`) |
| neighbor is the right note but narrower or stale | `UPDATE` that neighbor (relation `subsumes`) |
| neighbor asserts the opposite | `INVALIDATE` neighbor + `ADD` the new fact (relation `contradictory`) |

There is no DELETE. Superseded ≠ removed: `INVALIDATE` keeps the note and marks `invalidated_by`.

**Evolution (A-MEM re-touch):** for each ADD, look at its top neighbors once more — if the new fact makes a neighbor's framing or `[[links]]` stale (worth re-pointing, not wrong), include an UPDATE of that neighbor in the same change-set. These edit hand-curated prose, so they go through the same gate as everything else — never sneak them in as "just a link fix".

4. Write ONE change-set JSON for the whole batch (one op per note):

```json
{"changes": [
  {"op": "ADD", "name": "kebab-slug", "type": "feedback", "description": "one-line recall hook",
   "body": "The fact.\n\n**Why:** ...\n**How to apply:** ..."},
  {"op": "UPDATE", "target": "existing-name", "description": "new hook", "relation": "subsumes"},
  {"op": "INVALIDATE", "target": "existing-name", "invalidated_by": "kebab-slug", "relation": "contradictory"},
  {"op": "NOOP", "relation": "subsumed", "rationale": "duplicate of packages"}
]}
```

`type` ∈ user|feedback|gotcha|decision|project|reference (gotcha and feedback score highest importance). UPDATE carries only the fields you change; untouched frontmatter survives.

5. `uv run engram curate apply <cs.json> --dir <store>` — non-interactive it prints the diff and applies nothing.
6. Show the user that diff verbatim and ask for approval. Approved → re-run with `--yes` (writes + provenance commit). Rejected or amended → revise the change-set and return to step 5.
7. After an approved apply: propose the affected `MEMORY.md` index-line edits (one-liner per fact) — **show them and get the same explicit approval as step 6 before writing**, then commit `MEMORY.md` and run `uv run engram index --dir <store>`. The index is the one store file engram doesn't manage; it still gets a gate.

## Red flags — stop and hand the gate back

- About to run `--yes` and the user hasn't said yes to this diff → stop, show the diff.
- About to pass `--auto-threshold` yourself → stop. It is user-configured automation (a bench-calibrated gate the USER wires into their hooks), not an agent bypass — same rule as `--yes`.
- About to Edit/Write a store `.md` ("it's just the index line", "curate doesn't manage MEMORY.md", "faster by hand") → stop, express it as an op or leave it.
- Reaching for the engram source code to recall the schema → it's above, in step 4.
