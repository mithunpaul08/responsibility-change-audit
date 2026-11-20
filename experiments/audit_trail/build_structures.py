"""Angle 4, step 2: build a TextCircuit per document with each backend (resumable).

    python experiments/audit_trail/build_structures.py --backends surface coref discocirc

Writes data/processed/structures/audit_<backend>.jsonl via the Angle 2 store.
The question slot is a fixed dummy ("Which party is bound?"); its wires are unused.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sapc.data.loaders import PROCESSED, read_jsonl  # noqa: E402
from sapc.structure.build import StructureStore, build_and_store  # noqa: E402

DOCS = PROCESSED / "audit" / "docs.jsonl"
STRUCT_DIR = PROCESSED / "structures"
QUESTION = "Which party is bound?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["surface", "coref", "discocirc"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="use docs_<tag>.jsonl and audit_<backend>_<tag>.jsonl")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    docs = read_jsonl(DOCS.with_name(f"docs{sfx}.jsonl"))[: args.limit]
    for backend in args.backends:
        store = StructureStore(STRUCT_DIR / f"audit_{backend}{sfx}.jsonl")
        kw = {} if backend == "surface" else {"device": args.device}
        t0, n_new = time.time(), 0
        for k, d in enumerate(docs):
            if store.has(d["id"]):
                continue
            build_and_store(store, backend, {"id": d["id"], "context": d["sentences"], "question": QUESTION}, **kw)
            n_new += 1
            if n_new % 20 == 0:
                print(f"[{backend}] {k+1}/{len(docs)}  {(time.time()-t0)/n_new:.2f}s/doc", flush=True)
        ok = sum(1 for d in docs if store.get(d["id"]) is not None)
        err = [(d["id"], store.error(d["id"])) for d in docs if store.error(d["id"])]
        pf = sum(1 for d in docs if (c := store.get(d["id"])) and c.meta.get("parse_failed"))
        secs = [store.done[d["id"]]["secs"] for d in docs if store.has(d["id"])]
        print(f"=== audit/{backend}: ok={ok} errors={len(err)} docs_with_parse_fallback={pf} "
              f"mean_secs={sum(secs)/max(1,len(secs)):.2f}", flush=True)
        for e in err[:5]:
            print("   ", e)


if __name__ == "__main__":
    main()
