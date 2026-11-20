"""Angle 4, step 3: predict affects(s) per sentence with every method.

    python experiments/audit_trail/run_methods.py                 # structural methods
    python experiments/audit_trail/run_methods.py --llm            # + llm_zs (14B reader, cached)
    python experiments/audit_trail/run_methods.py --llm --limit 5  # smoke

Writes results/a4_audit/predictions.jsonl, one row per (doc, method):
    {id, method, pred: [{affects: [party ids], change_type}], secs, cached}
Re-running overwrites the rows of the methods it ran and keeps the others.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sapc.data.loaders import PROCESSED, read_jsonl  # noqa: E402
from sapc.structure.build import StructureStore  # noqa: E402
from sapc.structure.text_circuit import TextCircuit  # noqa: E402

DOCS = PROCESSED / "audit" / "docs.jsonl"
STRUCT_DIR = PROCESSED / "structures"
OUT_DIR = ROOT / "results" / "a4_audit"
PRED = OUT_DIR / "predictions.jsonl"
CHANGE_TABLE = yaml.safe_load(open(ROOT / "configs" / "audit_change_types.yaml"))
LEMMA2TYPE = {lem: typ for typ, lems in CHANGE_TABLE.items() for lem in lems}

STRUCTURAL = ["surface_any", "coref_any", "coref_subj", "dc_any", "dc_subj",
              "coref_any_strict", "dc_any_strict"]
BACKEND_OF = {"surface": "surface", "coref": "coref", "dc": "discocirc"}


def paths(tag: str | None) -> dict:
    """Input/output locations for a document set. tag=None is the seed-0 set."""
    sfx = f"_{tag}" if tag else ""
    out_dir = ROOT / "results" / f"a4_audit{sfx}"
    return {"docs": PROCESSED / "audit" / f"docs{sfx}.jsonl", "out_dir": out_dir,
            "pred": out_dir / "predictions.jsonl",
            "struct": (lambda backend: STRUCT_DIR / f"audit_{backend}{sfx}.jsonl")}


# ------------------------------------------------------------ wires -> parties
def wire_owner(circuit: TextCircuit, parties: list[dict], strict: bool) -> dict[int, set[int]]:
    """Lenient: a party owns every wire with a mention equal to one of its surface
    tokens. Strict: only wires whose *first* mention is such a token (so a wire
    created by an unlinked alias or pronoun belongs to nobody)."""
    owner: dict[int, set[int]] = {}
    for e in circuit.entities:
        texts = [e.mentions[0][1].lower()] if strict and e.mentions else [m[1].lower() for m in e.mentions]
        texts = set(texts) | ({e.name.lower()} if not strict else set())
        for p in parties:
            if texts & set(p["surface_tokens"]):
                owner.setdefault(e.id, set()).add(p["id"])
    return owner


def change_type(pred_lemma: str | None, affects: set[int]) -> str:
    if not affects:
        return "NONE"
    return LEMMA2TYPE.get((pred_lemma or "").lower(), "CREATE")


def structural_predict(method: str, circuit: TextCircuit, parties: list[dict]) -> list[dict]:
    strict = method.endswith("_strict")
    rule = "subj" if "_subj" in method else "any"
    owner = wire_owner(circuit, parties, strict)
    out = []
    for g in circuit.gates:
        wires = [g.subject] if rule == "subj" else g.wires
        affects = set()
        for w in wires:
            if w is not None:
                affects |= owner.get(w, set())
        out.append({"affects": sorted(affects), "change_type": change_type(g.predicate, affects)})
    return out


# ------------------------------------------------------------------- LLM arm
AUDIT_SYSTEM = ("You are auditing a document for a reviewer. You answer only in the exact "
                "line format requested, with no explanation.")

LLM_TEMPLATE = """Parties:
{parties}

Document (one numbered sentence per line):
{sentences}

Task: for every sentence that creates, modifies, ends or transfers a duty or responsibility of a party, output one line in the form
<sentence number>: <party ids separated by commas> | <CREATE or MODIFY or END or TRANSFER>
A party's duties change only if the sentence imposes, changes, ends or moves an obligation or responsibility of that party. Merely mentioning a party, granting it a right or permission, recording what it observed or said, or describing events does not count. In a sentence like "A shall pay B", only A's duty changes. In a transfer or assignment, both the party giving up and the party taking on the duty change. Resolve pronouns and defined terms to the party they refer to. Output nothing for sentences with no such change, and nothing else."""

# E2 (pre-registered in DESIGN_AUDIT_TRAIL.md): two guidance sentences shared by llm_zs_v2 and hyb_sent_v2.
GUIDE_EXTRA = ("A sentence that says a party finished, completed or closed out a task, or was relieved of or "
               "released from it, ends that party's responsibility (END). A sentence that says a party handed, "
               "passed or reassigned a task to another party transfers it (TRANSFER). When one party must do "
               "something to, for or with another party (pay, deliver, notify, brief, escort, report to), only "
               "the party that must act is affected; the party that receives, is notified or benefits is not, "
               "even though it is named.")
LLM_TEMPLATE_V2 = LLM_TEMPLATE.replace(" Resolve pronouns", f" {GUIDE_EXTRA} Resolve pronouns")
assert LLM_TEMPLATE_V2 != LLM_TEMPLATE

LINE_RE = re.compile(r"^\s*\[?\s*(\d+)\s*\]?\s*[:\-\.]\s*([^|]*?)\s*(?:\|\s*([A-Za-z]+))?\s*$")


def llm_prompt(doc: dict, v2: bool = False) -> str:
    parties = "\n".join(f"P{p['id']}: {p['name']}" + (f" (referred to as {p['alias']})" if p["alias"] != p["name"] else "")
                        for p in doc["parties"])
    sents = "\n".join(f"[{i}] {s}" for i, s in enumerate(doc["sentences"]))
    return (LLM_TEMPLATE_V2 if v2 else LLM_TEMPLATE).format(parties=parties, sentences=sents)


def parse_llm(text: str, n_sent: int, n_parties: int) -> list[dict]:
    pred = [{"affects": [], "change_type": "NONE"} for _ in range(n_sent)]
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if not 0 <= idx < n_sent:
            continue
        ids = {int(x) for x in re.findall(r"P?(\d+)", m.group(2)) if 0 <= int(x) < n_parties}
        typ = (m.group(3) or "CREATE").upper()
        if typ not in {"CREATE", "MODIFY", "END", "TRANSFER"}:
            typ = "CREATE"
        pred[idx] = {"affects": sorted(ids), "change_type": typ if ids else "NONE"}
    return pred


class AuditReader:
    def __init__(self, model_id: str, max_tokens: int = 512):
        from sapc.readers.mlx_reader import MLXReader
        self.r = MLXReader(model_id=model_id, max_tokens=max_tokens)
        self.r._chat = self._chat  # type: ignore[method-assign]

    def _chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": user}]
        return self.r.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def run(self, doc: dict, v2: bool = False) -> tuple[str, int, float, bool]:
        prompt = llm_prompt(doc, v2)
        key = self.r.cache.key(self.r.model_id, self._chat(prompt), self.r.max_tokens)
        cached = self.r.cache.get(key) is not None
        t0 = time.time()
        out, ptoks = self.r.generate(prompt)
        return out, ptoks, time.time() - t0, cached


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=STRUCTURAL)
    ap.add_argument("--llm", action="store_true", help="also run llm_zs")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-14B-Instruct-4bit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print", action="store_true", help="print raw LLM output for the first doc")
    ap.add_argument("--llm-v2", action="store_true", help="also run llm_zs_v2 (E2 guidance)")
    ap.add_argument("--tag", default=None, help="document set tag (e.g. seed1); default: the seed-0 set")
    args = ap.parse_args()
    P = paths(args.tag)
    P["out_dir"].mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(P["docs"])[: args.limit]
    llm_methods = (["llm_zs"] if args.llm else []) + (["llm_zs_v2"] if args.llm_v2 else [])
    methods = list(args.methods) + llm_methods

    rows = {}
    if P["pred"].exists():
        for r in read_jsonl(P["pred"]):
            if r["method"] not in methods:
                rows[(r["id"], r["method"])] = r

    stores = {b: StructureStore(P["struct"](b)) for b in BACKEND_OF.values()}
    for m in [m for m in methods if not m.startswith("llm")]:
        backend = BACKEND_OF[m.split("_")[0]]
        store = stores[backend]
        for d in docs:
            c = store.get(d["id"])
            if c is None:
                rows[(d["id"], m)] = {"id": d["id"], "method": m, "pred": None,
                                      "error": store.error(d["id"]) or "missing", "secs": None}
                continue
            rows[(d["id"], m)] = {"id": d["id"], "method": m, "pred": structural_predict(m, c, d["parties"]),
                                  "secs": store.done[d["id"]]["secs"], "parse_failed": c.meta.get("parse_failed", [])}
        print(f"[{m}] done", flush=True)

    if llm_methods:
        reader = AuditReader(args.model)
        for m in llm_methods:
            v2 = m == "llm_zs_v2"
            t0 = time.time()
            for k, d in enumerate(docs):
                out, ptoks, secs, cached = reader.run(d, v2)
                if args.print and k == 0:
                    print(out)
                rows[(d["id"], m)] = {"id": d["id"], "method": m,
                                      "pred": parse_llm(out, len(d["sentences"]), len(d["parties"])),
                                      "secs": secs, "cached": cached, "prompt_tokens": ptoks,
                                      "raw": out, "model": args.model}
                if (k + 1) % 10 == 0:
                    print(f"[{m}] {k+1}/{len(docs)}  {(time.time()-t0)/(k+1):.1f}s/doc", flush=True)
            print(f"[{m}] done", flush=True)

    with open(P["pred"], "w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {P['pred']}")


if __name__ == "__main__":
    main()
