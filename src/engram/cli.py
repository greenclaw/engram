"""engram CLI: `engram index --dir X` and `engram recall "query" --dir X`."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="engram", description="Curated agent memory: service-less semantic recall.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="(re)build the recall index over a memory dir")
    pi.add_argument("--dir", required=True, help="path to a memory/ dir of *.md notes")

    pr = sub.add_parser("recall", help="semantic recall over a memory dir")
    pr.add_argument("query")
    pr.add_argument("--dir", required=True)
    pr.add_argument("-k", type=int, default=5)
    pr.add_argument("--json", action="store_true", help="emit hits as JSON")

    args = p.parse_args(argv)

    from engram.store import build_index, recall  # lazy import → `engram --help` stays instant

    if args.cmd == "index":
        n = build_index(args.dir)
        print(f"indexed {n} notes → {Path(args.dir) / '.engram'}")
    elif args.cmd == "recall":
        hits = recall(args.dir, args.query, k=args.k)
        if args.json:
            import json

            print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))
        elif not hits:
            print("(no sufficiently relevant memory)")
        else:
            for h in hits:
                print(f"{h.score:.3f}  [{h.type}] {h.name} — {h.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
