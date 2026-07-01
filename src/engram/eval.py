"""Increment-1 done-when instrument: semantic recall vs a lexical (description-string) baseline.

Measures top-k hit-rate for both on a labelled query→note set. This is the `mem_recall` metric —
retrieval hit-rate needs no LLM, so we measure it directly rather than through claude-bench's
transcript harness (which is the right tool later for curator op-precision / gate calibration).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from engram.store import build_index, recall

# tiny stoplist so the lexical baseline isn't dominated by function words
_STOP = set(
    "a an the of to and or not is are be do we i you my our your it this that how what where which "
    "should me with via for on in at from into out over about as по на в и не как что где".split()
)


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", s.lower(), flags=re.UNICODE) if len(w) > 1 and w not in _STOP}


def lexical_topk(query: str, notes: list[dict], k: int) -> list[str]:
    """Baseline: rank notes by query↔(name+description) token overlap. A string baseline that shares
    zero tokens with a note does NOT retrieve it — so overlap-0 notes are never counted as hits."""
    q = _tokens(query)
    scored = [(len(q & _tokens(f"{n['name']} {n['description']}")), n["name"]) for n in notes]
    scored = sorted((s for s in scored if s[0] > 0), key=lambda s: s[0], reverse=True)
    return [name for _, name in scored[:k]]


def write_memory(notes: list[dict], mem_dir) -> None:
    mem_dir = Path(mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    for n in notes:
        front = yaml.safe_dump(
            {"name": n["name"], "description": n["description"], "type": n.get("type", "reference")},
            allow_unicode=True,
            sort_keys=False,
        )
        (mem_dir / f"{n['name']}.md").write_text(
            f"---\n{front}---\n{n.get('body', '')}\n", encoding="utf-8"
        )


def hit_rate_ab(dataset: dict, mem_dir, k: int = 5) -> dict:
    notes, queries = dataset["notes"], dataset["queries"]
    write_memory(notes, mem_dir)
    build_index(mem_dir)

    sem = lex = 0
    for item in queries:
        expect = item["expect"]
        sem += expect in [h.name for h in recall(mem_dir, item["query"], k=k)]
        lex += expect in lexical_topk(item["query"], notes, k)
    n = len(queries)
    return {"n": n, "k": k, "semantic": sem / n, "lexical": lex / n}
