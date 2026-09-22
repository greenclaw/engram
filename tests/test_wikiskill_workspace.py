import pytest

from engram.wikiskill import workspace as w


def test_init_layout(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    for rel in ("raw", "wiki/patterns", "skills", "dataset"):
        assert (ws / rel).is_dir()
    for f in ("wiki/index.md", "wiki/log.md", "wiki/skill-impact.md"):
        assert (ws / f).is_file()
    assert (ws / ".git").is_dir()
    assert w.git(ws, "log", "--oneline").count("\n") == 1


def test_apply_edits_ops_and_skip():
    text = "line1\nline2\nline3\n"
    new, skipped = w.apply_edits(text, [
        {"op": "append", "content": "tail"},
        {"op": "replace", "target": "line2", "content": "LINE2"},
        {"op": "insert_after", "target": "line1", "content": "after1"},
        {"op": "replace", "target": "nope", "content": "x"},
        {"op": "bogus", "content": "x"},
    ])
    assert new == "line1\nafter1\nLINE2\nline3\ntail\n"
    assert len(skipped) == 2 and "nope" in skipped[0] and "bogus" in skipped[1]


def test_skill_section_empty_and_filled(tmp_path):
    ws = tmp_path
    (ws / "skills").mkdir()
    assert w.skill_section(ws) == ""
    (ws / "skills/a").mkdir()
    (ws / "skills/a/SKILL.md").write_text("---\nname: a\n---\nDo A")
    (ws / "skills/b").mkdir()
    (ws / "skills/b/SKILL.md").write_text("Do B")
    s = w.skill_section(ws)
    assert s.startswith("## Skills") and s.index("Do A") < s.index("Do B")


def test_apply_maintainer(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "wiki/patterns/old.md").write_text("old body\n")
    skipped = w.apply_maintainer(ws, {
        "create_patterns": [{"name": "loop.md", "content": "# loop\nbody\n"}],
        "update_patterns": [{"name": "old.md", "edits": [{"op": "append", "content": "more"}]},
                            {"name": "missing.md", "edits": [{"op": "append", "content": "x"}]}],
        "update_index": "# idx\n- [loop](wiki/patterns/loop.md): p+rc+fix\n",
        "append_log": "iteration findings",
    }, k=1)
    assert (ws / "wiki/patterns/loop.md").read_text() == "# loop\nbody\n"
    assert (ws / "wiki/patterns/old.md").read_text() == "old body\nmore\n"
    assert (ws / "wiki/index.md").read_text().startswith("# idx")
    log = (ws / "wiki/log.md").read_text()
    assert "## iter 1" in log and "iteration findings" in log and "missing.md" in log
    assert skipped and "missing.md" in skipped[0]


def test_apply_maintainer_undoes_double_escaped_strings(tmp_path):
    """Live smoke: Haiku sometimes JSON-escapes a string field twice, so index.md arrived as one line
    with literal \\n and \\". A single-line value with escape sequences is decoded once; real text is untouched."""
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "wiki/patterns/p.md").write_text("x\n")
    w.apply_maintainer(ws, {
        "create_patterns": [{"name": "q.md", "content": "# Q\\n\\nbody \\\"quoted\\\""}],
        "update_patterns": [{"name": "p.md", "edits": [{"op": "append", "content": "line1\\nline2"}]}],
        "update_index": "## Idx\\n\\n- [a](wiki/patterns/a.md): \\\"fix\\\"",
        "append_log": "found it\\nsecond line"}, k=1)
    assert (ws / "wiki/index.md").read_text() == '## Idx\n\n- [a](wiki/patterns/a.md): "fix"\n'
    assert "found it\nsecond line" in (ws / "wiki/log.md").read_text()
    assert (ws / "wiki/patterns/q.md").read_text() == '# Q\n\nbody "quoted"'
    assert (ws / "wiki/patterns/p.md").read_text() == "x\nline1\nline2\n"
    # multi-line values are real text: a literal backslash-n inside them (e.g. LaTeX) is kept
    w.apply_maintainer(ws, {"update_index": "## Idx\n- uses \\n in C", "append_log": "l"}, k=2)
    assert (ws / "wiki/index.md").read_text() == "## Idx\n- uses \\n in C\n"


def test_apply_maintainer_rejects_bad_pattern_name(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    skipped = w.apply_maintainer(ws, {"create_patterns": [{"name": "../evil.md", "content": "x"}],
                                      "update_index": "# i", "append_log": "l"}, k=1)
    assert skipped and not (tmp_path / "evil.md").exists() and not (ws / "wiki/evil.md").exists()


def test_apply_proposal_create_then_patch(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    diff, skipped = w.apply_proposal(ws, {"action": "create", "name": "break_loop",
                                          "skill_md": "---\nname: break_loop\n---\nrule1\n",
                                          "purpose_md": "## Origin\nloop.md\n"})
    assert (ws / "skills/break_loop/SKILL.md").read_text().endswith("rule1\n")
    assert (ws / "skills/break_loop/PURPOSE.md").exists()
    assert "+rule1" in diff and not skipped
    diff, skipped = w.apply_proposal(ws, {"action": "patch", "name": "break_loop",
                                          "edits": [{"op": "append", "content": "rule2"},
                                                    {"op": "replace", "target": "zzz", "content": "q"}]})
    assert (ws / "skills/break_loop/SKILL.md").read_text().endswith("rule1\nrule2\n")
    assert "+rule2" in diff and len(skipped) == 1


def test_apply_proposal_patch_unknown_skill_raises(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    with pytest.raises(w.ProposalError):
        w.apply_proposal(ws, {"action": "patch", "name": "ghost", "edits": []})


def test_apply_proposal_rejects_bad_name(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    with pytest.raises(w.ProposalError):
        w.apply_proposal(ws, {"action": "create", "name": "../x", "skill_md": "a", "purpose_md": "b"})
    with pytest.raises(w.ProposalError):
        w.apply_proposal(ws, {"action": "no_action"})


def test_append_impact_format(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    w.append_impact(ws, 2, {"action": "create", "name": "s1"}, "--- a\n+++ b\n+x\n", 0.5, 0.4, "Accepted")
    w.append_impact(ws, 3, {"action": "no_action"}, "", None, 0.5, "NoAction")
    t = (ws / "wiki/skill-impact.md").read_text()
    assert "## iter 2 — create s1 — val 0.500 (best 0.400) — Accepted" in t
    assert "```diff\n--- a\n+++ b\n+x\n```" in t
    assert "## iter 3 — no_action — val n/a (best 0.500) — NoAction" in t


def test_git_commit_tag_restore(tmp_path):
    ws = tmp_path / "ws"
    w.init_workspace(ws)
    (ws / "skills/a").mkdir()
    (ws / "skills/a/SKILL.md").write_text("v1")
    w.commit_all(ws, "iter 1")
    w.tag(ws, "accepted-1")
    (ws / "skills/a/SKILL.md").write_text("v2")
    (ws / "skills/b").mkdir()
    (ws / "skills/b/SKILL.md").write_text("new")
    w.restore_skills(ws, "accepted-1")
    assert (ws / "skills/a/SKILL.md").read_text() == "v1"
    assert not (ws / "skills/b").exists()
    w.commit_all(ws, "noop")  # clean tree must not raise
