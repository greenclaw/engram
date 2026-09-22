"""`engram evolve init|run|eval|status` — the WikiSkill harness entrypoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from engram.wikiskill.bench import LiveMath, get_bench, make_splits, write_split
from engram.wikiskill.loop import EvolveError, evaluate, evolve, load_state
from engram.wikiskill.workspace import commit_all, init_workspace


def add_evolve_parser(sub) -> None:
    pe = sub.add_parser("evolve", help="WikiSkill: co-evolve skills with a persistent wiki (arXiv:2608.27454)")
    es = pe.add_subparsers(dest="evolve_cmd", required=True)
    pi = es.add_parser("init", help="create a workspace and a frozen train/val/test split")
    pi.add_argument("--ws", required=True)
    pi.add_argument("--bench", default="livemath", choices=["livemath"])
    pi.add_argument("--seed", type=int, default=0)
    pr = es.add_parser("run", help="run Algorithm 1 for --iters iterations (resumable)")
    pr.add_argument("--ws", required=True)
    pr.add_argument("--model", default="haiku")
    pr.add_argument("--iters", type=int, default=8)
    pr.add_argument("--parallel", type=int, default=8)
    pv = es.add_parser("eval", help="score the active skills on a split (empty skills = no-skill baseline)")
    pv.add_argument("--ws", required=True)
    pv.add_argument("--split", default="test", choices=["train", "val", "test"])
    pv.add_argument("--model", default="haiku")
    pv.add_argument("--parallel", type=int, default=8)
    pv.add_argument("--skills", default=None, help="another workspace (or its skills/ dir) — cross-model transfer")
    ps = es.add_parser("status", help="print the iteration history")
    ps.add_argument("--ws", required=True)


def _meta(ws: Path) -> dict:
    return json.loads((ws / "dataset/meta.json").read_text())


def run_evolve(args) -> int:
    ws = Path(args.ws)
    if args.evolve_cmd == "init":
        if (ws / "dataset/meta.json").exists():  # a new split under an existing state.json would silently mix runs
            print(f"engram evolve: {ws} is already initialized; use a new --ws for a new split", file=sys.stderr)
            return 2
        init_workspace(ws)
        bench = LiveMath()
        (ws / ".gitignore").write_text(".hf-cache/\n")
        tasks = bench.tasks_from_records(bench.download_records(ws / ".hf-cache"), args.seed)
        splits = make_splits(tasks, bench.SPLIT_SIZES, args.seed)
        for name, ts in splits.items():
            write_split(ws / f"dataset/{name}.jsonl", ts)
        (ws / "dataset/meta.json").write_text(json.dumps({"bench": bench.name, "seed": args.seed,
                                                         "sizes": bench.SPLIT_SIZES}, indent=1))
        commit_all(ws, "wikiskill: dataset split")
        print(f"initialized {ws}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        return 0
    bench = get_bench(_meta(ws)["bench"])
    if args.evolve_cmd == "run":
        try:
            st = evolve(ws, bench, model=args.model, iters=args.iters, parallel=args.parallel)
        except EvolveError as e:
            print(f"engram evolve: {e}", file=sys.stderr)
            return 2
        print(f"done: iteration={st['iteration']} R_best={st['r_best']:.3f} stopped={st['stopped']}")
        return 0
    if args.evolve_cmd == "eval":
        sk = Path(args.skills) if args.skills else None
        if sk and (sk / "skills").is_dir():
            sk = sk / "skills"
        r = evaluate(ws, bench, split=args.split, model=args.model, parallel=args.parallel, skills_dir=sk)
        print(f"R({args.split}) = {r:.3f} (skills: {sk.parent.name if sk else 'self'})")
        return 0
    st = load_state(ws)  # status
    print(f"bench={_meta(ws)['bench']} iteration={st['iteration']} R_best={st['r_best']} stopped={st['stopped']}")
    print("k\taction\tskill\tval\tbest\toutcome\tcost_usd")
    for h in st["history"]:
        val = "n/a" if h["r_val"] is None else f"{h['r_val']:.3f}"
        print(f"{h['k']}\t{h['action']}\t{h['name']}\t{val}\t{h['r_best']:.3f}\t{h['outcome']}\t{h.get('cost_usd', 0.0):.3f}")
    return 0
