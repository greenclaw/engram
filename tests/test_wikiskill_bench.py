import pytest

from engram.wikiskill import bench as b


def rec(no, month="202606", correct="right", distractors=("d1", "d2", "d3", "d4")):
    return {"no": no, "month": month, "mcq": {
        "question": f"Q{no}?", "correct_choice": {"label": "A", "text": correct},
        "choices": [{"label": lab, "text": t} for lab, t in zip("BCDE", distractors)]}}


def test_tasks_from_records_shuffles_labels_deterministically():
    recs = [rec(i) for i in range(20)]
    t1 = b.LiveMath.tasks_from_records(recs, seed=0)
    t2 = b.LiveMath.tasks_from_records(recs, seed=0)
    assert t1 == t2
    assert t1[0]["id"] == "202606-0" and set(t1[0]["choices"]) == set("ABCDE")
    for t in t1:  # correct text sits under the answer letter
        assert t["choices"][t["answer"]] == "right"
    assert len({t["answer"] for t in t1}) > 1  # not always "A"


def test_make_splits_disjoint_and_sized():
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(200)], seed=1)
    s = b.make_splits(tasks, {"train": 35, "val": 18, "test": 124}, seed=1)
    ids = [t["id"] for k in ("train", "val", "test") for t in s[k]]
    assert len(ids) == len(set(ids)) == 177
    assert [len(s[k]) for k in ("train", "val", "test")] == [35, 18, 124]


def test_make_splits_too_few_fails_loud():
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(10)], seed=1)
    with pytest.raises(ValueError):
        b.make_splits(tasks, {"train": 35, "val": 18, "test": 124}, seed=1)


def test_split_roundtrip(tmp_path):
    tasks = b.LiveMath.tasks_from_records([rec(i) for i in range(3)], seed=0)
    b.write_split(tmp_path / "x.jsonl", tasks)
    assert b.read_split(tmp_path / "x.jsonl") == tasks
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 3


def test_prompts():
    lm = b.LiveMath()
    sp = lm.system_prompt("## Skills\nfoo")
    assert "## Skills\nfoo" in sp and "{skill_section}" not in sp
    assert "<answer>" in sp
    t = b.LiveMath.tasks_from_records([rec(1)], seed=0)[0]
    up = lm.user_prompt(t)
    assert "Q1?" in up and "A." in up and "E." in up


@pytest.mark.parametrize("resp,gold,exp", [
    ("blah <answer>C</answer>", "C", 1.0),
    ("<answer>B</answer> ... <answer>C</answer>", "C", 1.0),   # last one wins
    ("<answer> c </answer>", "C", 1.0),
    ("<answer>B</answer>", "C", 0.0),
    ("no tags", "C", 0.0),
    ("<answer>CD</answer>", "C", 0.0),
])
def test_score(resp, gold, exp):
    t = {"id": "x", "question": "q", "choices": {}, "answer": gold}
    assert b.LiveMath().score(t, resp) == exp


def test_get_bench():
    assert b.get_bench("livemath").name == "livemath"
    with pytest.raises(KeyError):
        b.get_bench("nope")
