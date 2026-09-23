"""`engram evolve init|run|eval|status` — the WikiSkill harness entrypoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from engram.wikiskill.bench import BENCH_NAMES, get_bench, write_split
from engram.wikiskill.loop import (
    EvolveError,
    evaluate,
    evolve,
    iteration_usage,
    load_state,
    usage_of,
)
from engram.wikiskill.workspace import commit_all, init_workspace


def add_evolve_parser(sub) -> None:
    pe = sub.add_parser("evolve", help="WikiSkill: co-evolve skills with a persistent wiki (arXiv:2608.27454)")
    es = pe.add_subparsers(dest="evolve_cmd", required=True)
    pi = es.add_parser("init", help="create a workspace and a frozen train/val/test split")
    pi.add_argument("--ws", required=True)
    pi.add_argument("--bench", default="livemath", choices=BENCH_NAMES)
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
        bench = get_bench(args.bench)
        (ws / ".gitignore").write_text(".hf-cache/\n.data/\nwork/\n.venv/\n")  # derived or bulky, never committed
        splits = bench.init(ws, args.seed)
        for name, ts in splits.items():
            write_split(ws / f"dataset/{name}.jsonl", ts)
        (ws / "dataset/meta.json").write_text(json.dumps({"bench": bench.name, "seed": args.seed,
                                                         "sizes": {k: len(v) for k, v in splits.items()}}, indent=1))
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
    print("k\taction\tskill\tval\tbest\toutcome\tcalls\tin_tok\tout_tok\tapi_min")
    for h in st["history"]:
        val = "n/a" if h["r_val"] is None else f"{h['r_val']:.3f}"
        u = iteration_usage(ws, h["k"])  # from raw/, so histories written before usage tracking still render
        print(f"{h['k']}\t{h['action']}\t{h['name']}\t{val}\t{h['r_best']:.3f}\t{h['outcome']}\t{_usage_cols(u)}")
    t = usage_of(sorted((ws / "raw").rglob("*.json")))
    print(f"total (all raw/ incl. baseline and evals)\t\t\t\t\t\t{_usage_cols(t)}")
    return 0


def _usage_cols(u: dict) -> str:
    return f"{u['calls']}\t{u['input_tokens']}\t{u['output_tokens']}\t{u['api_seconds'] / 60:.1f}"
