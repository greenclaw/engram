"""mem_curate bench (increment-2 done-when): curator op-precision on a labelled candidate set.

Measures the LLM adjudication decision in isolation — the mechanical steps (recall, apply) are already
unit-tested. Each run: build the fixture store, recall evidence per candidate, ask Claude (headless
`claude -p`, Claude-Code-driven — no API key) for ONE change-set, score ops against labels.

    uv run python bench_curate.py [--runs N] [--model MODEL] [--dry]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = Path(__file__).parent / "tests" / "fixtures" / "curate_dataset.json"

RULES = """You are the engram memory curator. For EACH candidate below, decide one op against the store:
- ADD: no relevant neighbor covers the fact. Include "name" (kebab-slug).
- NOOP: a neighbor already covers it, same or broader.
- UPDATE: a neighbor is the right note but narrower or stale — refine it. Include "target".
- INVALIDATE: a neighbor asserts the opposite. Include "target" + "invalidated_by" (the new note's slug),
  AND a second change: ADD the superseding fact (same "candidate" index).
There is no DELETE. Every change object must carry "candidate": <index of the candidate it resolves>.
Reply with ONLY a JSON object: {"changes": [...]} — no prose, no code fences."""


def score_changeset(labels: list[dict], changes: list[dict]) -> dict:
    """Score a predicted change-set against per-candidate labels. Pure function (unit-tested)."""
    # primary predicted op per candidate: INVALIDATE wins over its paired ADD
    by_cand: dict[int, dict] = {}
    for ch in changes:
        c = ch.get("candidate")
        if c is None:
            continue
        if c not in by_cand or ch.get("op") == "INVALIDATE":
            by_cand[c] = ch

    correct = 0
    contra_total = contra_caught = false_invalidate = 0
    for i, label in enumerate(labels):
        pred = by_cand.get(i, {})
        op_ok = pred.get("op") == label["op"]
        if label["op"] in ("UPDATE", "INVALIDATE"):
            op_ok = op_ok and pred.get("target") == label["target"]
        correct += op_ok
        if label["op"] == "INVALIDATE":
            contra_total += 1
            contra_caught += op_ok
        elif pred.get("op") == "INVALIDATE":
            false_invalidate += 1
    return {
        "op_accuracy": correct / len(labels),
        "contradiction_catch": (contra_caught / contra_total) if contra_total else 1.0,
        "false_invalidate": false_invalidate,
    }


def build_prompt(dataset: dict, mem_dir: Path) -> str:
    from engram.store import recall

    lines = [RULES, "", "STORE (name: description):"]
    for n in dataset["notes"]:
        lines.append(f"- {n['name']}: {n['description']}")
    lines.append("\nCANDIDATES with recall evidence (top hits, cosine):")
    for i, cand in enumerate(dataset["candidates"]):
        lines.append(f"\n[{i}] {cand['text']}")
        hits = recall(mem_dir, cand["text"], k=3)
        for h in hits:
            lines.append(f"    hit: {h.name} (cos={h.relevance:.2f}) — {h.description}")
        if not hits:
            lines.append("    hit: (none above relevance floor)")
    return "\n".join(lines)


def ask_claude(prompt: str, model: str | None) -> list[dict]:
    cmd = ["claude", "-p", prompt]
    if model:
        cmd += ["--model", model]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    m = re.search(r"\{.*\}", out, re.DOTALL)  # tolerate stray prose/fences around the JSON
    if not m:
        raise ValueError(f"no JSON in model reply: {out[:200]!r}")
    return json.loads(m.group(0))["changes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default=None, help="claude -p --model override")
    ap.add_argument("--dry", action="store_true", help="print the prompt, don't call the model")
    args = ap.parse_args()

    from engram.eval import write_memory
    from engram.store import build_index

    dataset = json.loads(DATASET.read_text())
    labels = [{k: v for k, v in c.items() if k != "text"} for c in dataset["candidates"]]

    with tempfile.TemporaryDirectory() as d:
        mem = Path(d) / "mem"
        write_memory(dataset["notes"], mem)
        build_index(mem)
        prompt = build_prompt(dataset, mem)

    if args.dry:
        print(prompt)
        return

    print(f"mem_curate: {len(labels)} labelled candidates, {args.runs} run(s)\n")
    for r in range(args.runs):
        try:
            changes = ask_claude(prompt, args.model)
        except (ValueError, subprocess.TimeoutExpired) as e:
            print(f"  run {r + 1}: FAILED — {e}")
            continue
        s = score_changeset(labels, changes)
        print(f"  run {r + 1}: op_accuracy={s['op_accuracy']:.0%}  "
              f"contradiction_catch={s['contradiction_catch']:.0%}  "
              f"false_invalidate={s['false_invalidate']}")


if __name__ == "__main__":
    sys.exit(main())
