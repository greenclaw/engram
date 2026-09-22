"""Bench protocol (load / prompt / score) + LiveMathematicianBench. The evolution loop never
sees dataset specifics; SpreadsheetBench later plugs in behind the same protocol."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Protocol, TypedDict

PROMPTS = Path(__file__).parent / "prompts"


class Task(TypedDict):
    id: str
    question: str
    choices: dict[str, str]
    answer: str


class Bench(Protocol):
    name: str
    tools: list[str]

    def system_prompt(self, skill_section: str) -> str: ...
    def user_prompt(self, task: Task) -> str: ...
    def score(self, task: Task, response: str) -> float: ...


_ANSWER = re.compile(r"<answer>\s*([A-Za-z]+)\s*</answer>")


class LiveMath:
    name = "livemath"
    tools: list[str] = []
    HF_REPO = "LiveMathematicianBench/LiveMathematicianBench"
    SPLIT_SIZES = {"train": 35, "val": 18, "test": 124}  # Table 6

    @staticmethod
    def tasks_from_records(records: list[dict], seed: int) -> list[Task]:
        """HF records always put the correct option under 'A' and list B–E distractors;
        reshuffle the five options per task (seeded) so the letter carries no signal."""
        rng = random.Random(seed)
        out: list[Task] = []
        for r in records:
            m = r["mcq"]
            texts = [m["correct_choice"]["text"]] + [c["text"] for c in m["choices"]]
            order = list(range(len(texts)))
            rng.shuffle(order)
            labels = "ABCDEFGH"[: len(texts)]
            choices = {lab: texts[i] for lab, i in zip(labels, order)}
            answer = labels[order.index(0)]
            out.append(Task(id=f"{r['month']}-{r['no']}", question=m["question"], choices=choices, answer=answer))
        return out

    def download_records(self, cache_dir: Path) -> list[dict]:
        from huggingface_hub import HfApi, hf_hub_download

        files = sorted(s.rfilename for s in HfApi().dataset_info(self.HF_REPO).siblings
                       if s.rfilename.startswith("data/") and s.rfilename.endswith(".json"))
        recs: list[dict] = []
        for f in files:
            p = hf_hub_download(self.HF_REPO, f, repo_type="dataset", cache_dir=cache_dir)
            recs.extend(json.loads(Path(p).read_text()))
        return recs

    def system_prompt(self, skill_section: str) -> str:
        return (PROMPTS / "livemath.md").read_text().replace("{skill_section}", skill_section)

    def user_prompt(self, task: Task) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(task["choices"].items()))
        return f"{task['question']}\n\n{opts}"

    def score(self, task: Task, response: str) -> float:
        hits = _ANSWER.findall(response)
        return 1.0 if hits and hits[-1].strip().upper() == task["answer"] else 0.0


def make_splits(tasks: list[Task], sizes: dict[str, int], seed: int) -> dict[str, list[Task]]:
    need = sum(sizes.values())
    if len(tasks) < need:
        raise ValueError(f"need {need} tasks for splits {sizes}, have {len(tasks)}")
    pool = list(tasks)
    random.Random(seed).shuffle(pool)
    out, i = {}, 0
    for name, n in sizes.items():
        out[name] = pool[i:i + n]
        i += n
    return out


def write_split(path: Path, tasks: list[Task]) -> None:
    path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))


def read_split(path: Path) -> list[Task]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_BENCHES = {"livemath": LiveMath}


def get_bench(name: str) -> Bench:
    return _BENCHES[name]()
