"""Backend registry + per-item JSONL cache so structure extraction is resumable."""
from __future__ import annotations

import json
import time
from pathlib import Path

from sapc.structure.text_circuit import TextCircuit

BACKENDS = ("discocirc", "coref", "surface")


def build_circuit(backend: str, sentences: list[str], question: str, **kw) -> TextCircuit:
    if backend == "discocirc":
        from sapc.structure import discocirc_backend as b
    elif backend == "coref":
        from sapc.structure import coref_backend as b
    elif backend == "surface":
        from sapc.structure import surface_backend as b
    else:
        raise ValueError(backend)
    return b.build(sentences, question, **kw)


class StructureStore:
    """One JSONL per (dataset, backend). Lines: {"id":..., "circuit":..., "error":..., "secs":...}"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.done[row["id"]] = row

    def get(self, item_id: str) -> TextCircuit | None:
        row = self.done.get(item_id)
        if row is None or row.get("circuit") is None:
            return None
        return TextCircuit.from_dict(row["circuit"])

    def has(self, item_id: str) -> bool:
        return item_id in self.done

    def error(self, item_id: str) -> str | None:
        row = self.done.get(item_id)
        return None if row is None else row.get("error")

    def add(self, item_id: str, circuit: TextCircuit | None, error: str | None, secs: float):
        row = {"id": item_id, "circuit": circuit.to_dict() if circuit else None,
               "error": error, "secs": secs}
        self.done[item_id] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_and_store(store: StructureStore, backend: str, item: dict, timeout_s: float = 120.0,
                    **kw) -> TextCircuit | None:
    if store.has(item["id"]):
        return store.get(item["id"])
    t0 = time.time()
    try:
        c = build_circuit(backend, item["context"], item["question"], **kw)
        err = None
        if time.time() - t0 > timeout_s:
            err = f"timeout>{timeout_s}s"
    except Exception as exc:  # noqa: BLE001
        c, err = None, f"{type(exc).__name__}: {exc}"[:500]
    store.add(item["id"], c if err is None else None, err, time.time() - t0)
    return c if err is None else None
