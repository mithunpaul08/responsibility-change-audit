"""Dataset download + normalisation into one JSONL schema.

Item schema (data/processed/<name>.jsonl, one JSON object per line):
    id            str   unique, stable across runs
    dataset       str   "babi" | "hotpot"
    task          str   bAbI task number as "qa1".."qa20"; HotpotQA type "bridge"/"comparison"
    context       list[str]   sentences, in original order
    question      str
    answers       list[str]   acceptable gold answers (bAbI: one; HotpotQA: one)
    supporting    list[int]   0-based indices into `context` that are gold supporting facts

Paraphrase pair schema (data/processed/pairs_<name>.jsonl):
    id, dataset ("paws"|"mrpc"), split, s1, s2, label (1 = paraphrase)

Raw files are downloaded with plain HTTP into data/raw/ so the pipeline does not
depend on HuggingFace dataset *scripts* (which newer `datasets` refuse to run).
"""
from __future__ import annotations

import io
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Keras mirror of the original Facebook tarball (the thespermwhale.com URL is dead).
BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
BABI_URL_FALLBACK = "https://huggingface.co/datasets/facebook/babi_qa/resolve/main/tasks_1-20_v1-2.tar.gz"
# HuggingFace's parquet conversion of HotpotQA distractor/validation (7 405 items).
HOTPOT_DEV_URL = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet"

# bAbI tasks used in the design (§2.3). Keys are the file-name stems.
BABI_TASKS = {
    "qa1": "qa1_single-supporting-fact",
    "qa2": "qa2_two-supporting-facts",
    "qa3": "qa3_three-supporting-facts",
    "qa5": "qa5_three-arg-relations",
    "qa6": "qa6_yes-no-questions",
    "qa11": "qa11_basic-coreference",
    "qa13": "qa13_compound-coreference",
    "qa14": "qa14_time-reasoning",
    "qa16": "qa16_basic-induction",
    "qa20": "qa20_agents-motivations",
}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[download] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "sapc/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


# --------------------------------------------------------------------------- bAbI
def _parse_babi_file(text: str, task: str, split: str) -> list[dict]:
    """Parse the bAbI line format into items.

    Lines look like:
        1 Mary moved to the bathroom.
        2 John went to the hallway.
        3 Where is Mary? \tbathroom\t1
    A line numbered 1 starts a new story. Supporting-fact numbers refer to the
    original line numbers; we map them to indices into the *context* list,
    which excludes question lines.
    """
    items = []
    story_lines: dict[int, int] = {}   # original line no -> context index
    context: list[str] = []
    story_no = -1
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        num_str, rest = raw.split(" ", 1)
        num = int(num_str)
        if num == 1:
            story_no += 1
            story_lines, context = {}, []
        if "\t" in rest:
            q, a, sup = rest.split("\t")
            sup_idx = [story_lines[int(s)] for s in sup.split() if int(s) in story_lines]
            items.append({
                "id": f"babi-{split}-{task}-s{story_no}-q{num}",
                "dataset": "babi", "task": task,
                "context": list(context), "question": q.strip(),
                "answers": [a.strip()], "supporting": sup_idx,
            })
        else:
            story_lines[num] = len(context)
            context.append(rest.strip())
    return items


def load_babi(tasks: list[str] | None = None, per_task: int | None = 300,
              split: str = "test", seed: int = 0) -> list[dict]:
    tasks = tasks or list(BABI_TASKS)
    tar_path = RAW / "babi" / "tasks_1-20_v1-2.tar.gz"
    try:
        _download(BABI_URL, tar_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[babi] primary URL failed ({exc}); trying fallback")
        _download(BABI_URL_FALLBACK, tar_path)
    out = []
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        for task in tasks:
            stem = BABI_TASKS[task]
            member = next(n for n in names if n.endswith(f"en-10k/{stem}_{split}.txt"))
            text = tf.extractfile(member).read().decode("utf-8")
            items = _parse_babi_file(text, task, split)
            if per_task is not None and len(items) > per_task:
                rng = random.Random(seed)
                items = rng.sample(items, per_task)
                items.sort(key=lambda x: x["id"])
            out.extend(items)
    return out


# ----------------------------------------------------------------------- HotpotQA
def load_hotpot(n: int | None = 500, seed: int = 0) -> list[dict]:
    import pandas as pd
    path = _download(HOTPOT_DEV_URL, RAW / "hotpot" / "distractor_validation.parquet")
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(seed)
    if n is not None and len(rows) > n:
        rows = rng.sample(rows, n)
    out = []
    for ex in rows:
        sf = ex["supporting_facts"]
        gold = {(t, int(i)) for t, i in zip(sf["title"], sf["sent_id"])}
        context, supporting = [], []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            for i, s in enumerate(sents):
                s = str(s).strip()
                if not s:
                    continue
                if (title, i) in gold:
                    supporting.append(len(context))
                context.append(s)
        out.append({
            "id": f"hotpot-dev-{ex['id']}", "dataset": "hotpot", "task": ex["type"],
            "context": context, "question": ex["question"].strip(),
            "answers": [ex["answer"].strip()], "supporting": supporting,
            "level": ex.get("level"),
        })
    return out


# -------------------------------------------------------------- paraphrase pairs
def load_pairs(name: str) -> list[dict]:
    """PAWS-Wiki labeled_final and GLUE/MRPC via HuggingFace parquet (no scripts)."""
    from datasets import load_dataset
    out = []
    if name == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final")
        for split in ("validation", "test"):
            for i, ex in enumerate(ds[split]):
                out.append({"id": f"paws-{split}-{ex['id']}", "dataset": "paws",
                            "split": "dev" if split == "validation" else "test",
                            "s1": ex["sentence1"], "s2": ex["sentence2"], "label": int(ex["label"])})
    elif name == "mrpc":
        ds = load_dataset("nyu-mll/glue", "mrpc")
        for split in ("validation", "test"):
            for ex in ds[split]:
                out.append({"id": f"mrpc-{split}-{ex['idx']}", "dataset": "mrpc",
                            "split": "dev" if split == "validation" else "test",
                            "s1": ex["sentence1"], "s2": ex["sentence2"], "label": int(ex["label"])})
    else:
        raise ValueError(name)
    return out


# ------------------------------------------------------------------------ I/O
def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
