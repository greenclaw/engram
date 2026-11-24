"""engram CLI: `engram index --dir X` and `engram recall "query" --dir X`."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def default_memory_dir(cwd=None) -> Path | None:
    """The Claude Code auto-memory store for the project at cwd, or None if it doesn't exist.
    Slug = path with non-alphanumerics replaced by '-'; worktree sessions map to the MAIN
    project's store (cut at /.worktrees/)."""
    base = str(Path(cwd) if cwd else Path.cwd()).split("/.worktrees/")[0]
    slug = re.sub(r"[^A-Za-z0-9]", "-", base)
    d = Path.home() / ".claude" / "projects" / slug / "memory"
    return d if d.is_dir() else None


_STRAIGHT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _in_store(path: Path) -> bool:
    """A note of an engram store: under a `memory` dir that holds a MEMORY.md index, or under a Claude
    auto-memory store (~/.claude/projects/<slug>/memory) — guarded before its index exists."""
    for d in path.parents:
        if d.name == "memory":
            return (d / "MEMORY.md").exists() or d.parent.parent == Path.home() / ".claude" / "projects"
    return False


def check_note() -> int:
    """PreToolUse hook on Write/Edit: refuse a memory note engram can't read (exit 2, reason on stderr →
    Claude rewrites it). Never blocks on its own failure. Registered with an `if` path filter, so it
    only spawns for memory/ files."""
    import json
    import os
    import sys

    if os.environ.get("ENGRAM_DISABLE", "") not in ("", "0"):
        return 0
    try:
        data = json.load(sys.stdin)
        inp = data.get("tool_input") or {}
        path = Path(inp.get("file_path") or "")
        if path.suffix != ".md" or path.name == "MEMORY.md" or not _in_store(path):
            return 0
        new = inp.get("new_string") or ""
        if data.get("tool_name") == "Write":
            text = inp.get("content") or ""
        elif data.get("tool_name") == "Edit":
            old = inp.get("old_string") or ""
            if not old:  # Edit with an empty old_string creates the file
                if path.exists():
                    return 0
                text = new
            else:
                text = path.read_text(encoding="utf-8")
                if old not in text:  # Claude Code's Edit also matches curly quotes as straight ones
                    text = text.translate(_STRAIGHT)
                    if old not in text:
                        return 0  # the Edit itself will fail; nothing to judge
                text = text.replace(old, new, -1 if inp.get("replace_all") else 1)
        else:
            return 0
        text = text.replace("\r\n", "\n")  # what read_text gives the indexer on disk
        from engram.core import frontmatter_error

        err = frontmatter_error(text, path)
    except Exception:  # noqa: BLE001 — a guard bug must never block a write
        return 0
    if err:
        print(f"engram: memory note frontmatter is not valid — {err}", file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["check-note"]:  # ponytail: dispatched before the parser — skips the evolve imports (hot path)
        return check_note()
    p = argparse.ArgumentParser(prog="engram", description="Curated agent memory: service-less semantic recall.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-note", help="PreToolUse hook: refuse a Write/Edit that leaves a memory note's "
                                       "frontmatter unreadable (stdin hook JSON)")

    pi = sub.add_parser("index", help="(re)build the recall index over a memory dir")
    pi.add_argument("--dir", default=None,
                    help="path to a memory/ dir of *.md notes (default: this project's Claude auto-memory store)")

    pr = sub.add_parser("recall", help="semantic recall over a memory dir")
    pr.add_argument("query")
    pr.add_argument("--dir", default=None)
    pr.add_argument("-k", type=int, default=5)
    pr.add_argument("--json", action="store_true", help="emit hits as JSON")

    ph = sub.add_parser("hook", help="UserPromptSubmit hook: print semantic recall for the prompt (stdin JSON)")
    ph.add_argument("--dir", default=None)
    ph.add_argument("-k", type=int, default=3)

    pp = sub.add_parser("pending", help="pending-store: sessions queued by the Stop hook for curation")
    pps = pp.add_subparsers(dest="pending_cmd", required=True)
    ppa = pps.add_parser("add", help="Stop hook: enqueue the finished session (stdin hook JSON); never blocks")
    ppa.add_argument("--dir", default=None)
    ppl = pps.add_parser("list", help="list queued sessions as JSON")
    ppl.add_argument("--dir", default=None)
    ppc = pps.add_parser("clear", help="drop queued sessions (all, or the given session ids)")
    ppc.add_argument("--dir", default=None)
    ppc.add_argument("ids", nargs="*", help="session ids to drop (default: all)")

    pc = sub.add_parser("curate", help="apply a Claude-produced change-set (gated git commit)")
    pcs = pc.add_subparsers(dest="curate_cmd", required=True)
    pca = pcs.add_parser("apply", help="validate a change-set, show its diff, gate, commit")
    pca.add_argument("changeset", help="path to change-set JSON, or - for stdin")
    pca.add_argument("--dir", default=None, help="the memory/ dir to apply into (default: auto-memory store)")
    pca.add_argument("--yes", action="store_true",
                     help="apply without the interactive prompt — bypasses the human gate (for scripting)")
    pca.add_argument("--auto-threshold", type=float, default=None, metavar="TAU",
                     help="auto-approve iff every change's confidence ≥ TAU and no body-UPDATE "
                          "(bench-calibrated, μ−Zσ); otherwise fall back to the interactive gate")

    from engram.wikiskill.cli import add_evolve_parser

    add_evolve_parser(sub)

    args = p.parse_args(argv)

    if args.cmd == "evolve":  # its own workspace (--ws), never the memory store
        from engram.wikiskill import cli as wcli

        return wcli.run_evolve(args)

    # ENGRAM_DISABLE kill switch: silences the ambient hook entrypoints only (recall hook +
    # Stop-hook enqueue) — instant rollback without editing settings or restarting sessions.
    # Explicit commands (index/recall/curate/pending list|clear) stay live. Env read at use.
    ambient = args.cmd == "hook" or (args.cmd == "pending" and args.pending_cmd == "add")
    if ambient:
        import os

        if os.environ.get("ENGRAM_DISABLE", "") not in ("", "0"):
            return 0

    if args.dir is None:  # global-hook mode: one registration serves every project with a store
        args.dir = default_memory_dir()
        if args.dir is None:
            if ambient:
                return 0  # no store for this project — the hook stays silent, never blocks
            import sys

            print("engram: no --dir given and no auto-memory store exists for this project", file=sys.stderr)
            return 2

    if args.cmd == "index":
        import sys

        from engram.core import MemoryNoteError
        from engram.store import build_index

        try:
            n = build_index(args.dir)
        except (FileNotFoundError, ValueError, MemoryNoteError) as e:  # missing dir / bad env knob → clean error, not traceback
            print(f"engram index: {e}", file=sys.stderr)
            return 1
        print(f"indexed {n} notes → {Path(args.dir) / '.engram'}")
    elif args.cmd == "recall":
        import sys

        from engram.core import MemoryNoteError
        from engram.store import recall

        try:
            hits = recall(args.dir, args.query, k=args.k)
        except (FileNotFoundError, ValueError, MemoryNoteError) as e:
            print(f"engram recall: {e}", file=sys.stderr)
            return 1
        if args.json:
            import json

            print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))
        elif not hits:
            print("(no sufficiently relevant memory)")
        else:
            for h in hits:
                print(f"{h.score:.3f}  [{h.type}] {h.name} — {h.description}")
    elif args.cmd == "hook":
        # A recall failure must NEVER block the user's prompt: any problem → silent exit 0
        # (warning on stderr). ponytail: availability over strictness is deliberate for a hook.
        import json
        import sys

        try:
            prompt = (json.load(sys.stdin).get("prompt") or "").strip()
            if len(prompt) < 15 or prompt.startswith("/"):
                return 0
            from engram.core import QUOTE_HINT, MemoryNoteError
            from engram.store import recall

            try:
                hits = recall(args.dir, prompt, k=args.k)
            except MemoryNoteError as e:  # a bad note slipped past the write guard (Bash/python write)
                why = " ".join(str(e).split())[:300]  # note-controlled text: one bounded line
                print(f"engram: semantic recall is OFF for this project until this memory note is fixed — {why}. "
                      f"Fix its frontmatter ({QUOTE_HINT}).")
                return 0
            if hits:
                print("engram semantic recall (background context; verify before asserting — read the full note at its path if it matters):")
                for h in hits:
                    # collapse whitespace/newlines: a note's text is untrusted, and a raw newline would
                    # inject a fake extra hit line into the agent's context (prompt injection via memory).
                    name = " ".join(str(h.name).split())
                    desc = " ".join(str(h.description).split())
                    print(f"- [{h.type}] {name} — {desc} ({h.path})")
        except Exception as e:  # noqa: BLE001
            print(f"engram hook: {e}", file=sys.stderr)
        return 0
    elif args.cmd == "pending":
        import json
        import sys

        from engram import pending

        if args.pending_cmd == "add":
            # Stop-hook semantics (like `hook`): any failure → warn on stderr, exit 0, never block
            try:
                data = json.load(sys.stdin)
                sid, tp = data.get("session_id"), data.get("transcript_path")
                if sid and tp:
                    pending.add(args.dir, sid, tp)
            except Exception as e:  # noqa: BLE001
                print(f"engram pending: {e}", file=sys.stderr)
            return 0
        if args.pending_cmd == "list":
            print(json.dumps(pending.load(args.dir), ensure_ascii=False, indent=2))
        else:  # clear
            print(f"cleared {pending.clear(args.dir, args.ids or None)}")
    elif args.cmd == "curate":
        import sys

        from engram.curate import CurateError, apply, auto_approvable, load_changeset

        def confirm(diff):
            print(diff or "(no textual diff)")
            if args.yes:
                return True
            if args.auto_threshold is not None and auto_approvable(cs, args.auto_threshold):
                print(f"auto-approved: all confidences ≥ {args.auto_threshold}, no prose re-touch")
                return True
            prompt = "apply these changes? [y/N] "
            try:  # stdin may be a piped change-set ('-') — read the gate answer from the terminal
                with open("/dev/tty") as tty:
                    print(prompt, end="", flush=True)
                    return tty.readline().strip().lower() in ("y", "yes")
            except OSError:
                pass
            try:
                return input(prompt).strip().lower() in ("y", "yes")
            except (EOFError, OSError):  # non-interactive (agent-run): show diff, apply only via --yes
                print("non-interactive: nothing applied — review the diff above, re-run with --yes to apply")
                return False

        try:  # a bad change-set (untrusted LLM output) is a clean error, not a traceback
            cs = load_changeset(args.changeset)
            print("applied" if apply(args.dir, cs, confirm=confirm) else "no changes applied")
        except CurateError as e:
            print(f"engram curate: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
