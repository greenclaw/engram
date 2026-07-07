"""Recall quality dims the increment-1 A/B left untested (it used a zero-lexical-overlap set):
confusables (precision@1 among near-neighbors), negation polarity, exact-keyword regression,
cross-lingual matrix, and scale (>100 notes) with abstention checks.

    uv run python bench_quality.py [--scale N]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from engram.eval import write_memory
from engram.store import build_index, recall

FIXTURES = Path(__file__).parent / "tests" / "fixtures"

# ~100 deterministic distractors for the scale dim (20 subjects × 5 fact templates) — the realistic
# "big personal store" filler the labelled queries must still cut through.
_SUBJECTS = ["billing", "search", "onboarding", "notifications", "analytics", "exports", "imports",
             "webhooks", "profiles", "dashboards", "invoices", "reports", "sessions", "uploads",
             "queues", "emails", "themes", "locales", "audits", "backfills"]
_FACTS = [
    ("{s}-owner", "The {s} module is owned by the platform team."),
    ("{s}-slo", "The {s} service targets a 99.9 percent availability SLO."),
    ("{s}-flag", "The {s} rollout is behind a feature flag."),
    ("{s}-alerts", "Alerts for {s} page the on-call engineer."),
    ("{s}-docs", "Architecture notes for {s} live in the team wiki."),
]


def distractors(n: int) -> list[dict]:
    out = [{"name": name.format(s=s), "type": "reference", "description": desc.format(s=s)}
           for s in _SUBJECTS for name, desc in _FACTS]
    return out[:n]


def _eval(notes: list[dict], queries: list[dict], mem_dir: Path) -> dict:
    """hit@1 / hit@5 / abstained (query returned nothing) over one store."""
    write_memory(notes, mem_dir)
    build_index(mem_dir)
    top1 = top5 = abstained = 0
    misses = []
    for q in queries:
        hits = recall(mem_dir, q["query"], k=5)
        names = [h.name for h in hits]
        top1 += bool(names) and names[0] == q["expect"]
        top5 += q["expect"] in names
        abstained += not names
        if not names or names[0] != q["expect"]:
            misses.append(f"{q['query']!r} → {names[:3]} (want {q['expect']})")
    n = len(queries)
    return {"n": n, "p@1": top1 / n, "hit@5": top5 / n, "abstained": abstained, "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=100, help="distractor notes for the scale dim")
    ap.add_argument("-v", action="store_true", help="print misses")
    args = ap.parse_args()

    quality = json.loads((FIXTURES / "recall_quality.json").read_text())
    base = json.loads((FIXTURES / "recall_dataset.json").read_text())

    with tempfile.TemporaryDirectory() as d:
        rows = []
        for dim in ("confusables", "negation", "keyword", "cross-lingual"):
            sec = quality[dim]
            rows.append((dim, _eval(sec["notes"], sec["queries"], Path(d) / dim)))

        # scale: the increment-1 labelled set must still hit with ~100 distractors in the store
        scale_notes = base["notes"] + distractors(args.scale)
        scale_dir = Path(d) / "scale"
        rows.append((f"scale ({len(scale_notes)} notes)", _eval(scale_notes, base["queries"], scale_dir)))

        # abstention true-negatives measured on the big store (the realistic case)
        offtopic = quality["abstention_offtopic"]
        silent = sum(not recall(scale_dir, q, k=5) for q in offtopic)

    print(f"recall quality dims (p@1 = top hit is THE labelled note)\n")
    for dim, r in rows:
        print(f"  {dim:22s} n={r['n']:2d}  p@1={r['p@1']:4.0%}  hit@5={r['hit@5']:4.0%}"
              + (f"  false-abstain={r['abstained']}" if r["abstained"] else ""))
        if args.v:
            for m in r["misses"]:
                print(f"      miss: {m}")
    print(f"  {'abstention (off-topic)':22s} n={len(offtopic):2d}  silent={silent}/{len(offtopic)}")


if __name__ == "__main__":
    main()
