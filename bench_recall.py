"""Increment-1 A/B report: semantic recall vs lexical baseline hit-rate.

    uv run python bench_recall.py [dataset.json]
"""
import json
import sys
import tempfile
from pathlib import Path

from engram.eval import hit_rate_ab


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/recall_dataset.json")
    dataset = json.loads(path.read_text())
    print(f"mem_recall A/B over {path}  (n={len(dataset['queries'])} labelled queries)\n")
    with tempfile.TemporaryDirectory() as d:
        for k in (1, 5):
            r = hit_rate_ab(dataset, Path(d) / f"m{k}", k=k)
            print(f"  k={k}:  semantic {r['semantic']:6.0%}   lexical {r['lexical']:6.0%}")


if __name__ == "__main__":
    main()
