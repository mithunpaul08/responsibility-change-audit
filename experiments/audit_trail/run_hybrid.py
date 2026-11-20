"""Angle 4, extension E1 (post hoc): two-stage hybrid audit.

Stage 1: coref_any candidates (sentence, party) with mention evidence.
Stage 2: the 14B reader judges only the candidates.

    python experiments/audit_trail/run_hybrid.py --variants hyb_doc hyb_sent [--limit N] [--print]
    python experiments/audit_trail/run_hybrid.py --tag seed1 --variants hyb_sent hyb_sent_v2   # E2

Adds rows with method hyb_doc / hyb_sent to results/a4_audit/predictions.jsonl
(other methods' rows are kept). Requires run_methods.py to have run (coref_any rows).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sapc.data.loaders import PROCESSED, read_jsonl  # noqa: E402
from sapc.structure.build import StructureStore  # noqa: E402
from experiments.audit_trail.run_methods import (AuditReader, GUIDE_EXTRA,  # noqa: E402
                                                 parse_llm, paths, wire_owner)

STAGE1 = "coref_any"
GUIDE = ("A party's duty or responsibility changes only if the sentence imposes, changes, ends or moves an "
         "obligation or responsibility of that party. Merely mentioning a party, granting it a right or "
         "permission, recording what it observed or said, or describing events does not count. In a sentence "
         "like \"A shall pay B\", only A's duty changes. In a transfer or assignment, both the party giving up "
         "and the party taking on the duty change.")

DOC_TEMPLATE = """Parties:
{parties}

Candidate sentences from the document (one numbered sentence per line; each line first lists the parties the sentence refers to and the words used):
{sentences}

Task: for every candidate sentence that creates, modifies, ends or transfers a duty or responsibility of one of the parties it refers to, output one line in the form
<sentence number>: <party ids separated by commas> | <CREATE or MODIFY or END or TRANSFER>
{guide} Output nothing for sentences with no such change, and nothing else."""

SENT_TEMPLATE = """Previous sentence (context only): {prev}
Sentence: {sent}
Parties this sentence refers to:
{cands}

Question: whose duty or responsibility does this sentence create, modify, end or transfer? {guide}
Answer with one line per affected party in the form
P<id> | <CREATE or MODIFY or END or TRANSFER>
or the single word NONE. Nothing else."""

SENT_RE = re.compile(r"P?(\d+)\s*\|\s*([A-Za-z]+)")

# E2 (pre-registered): v2 guidance shared with llm_zs_v2, plus one line after the candidate list.
GUIDE_V2 = GUIDE + " " + GUIDE_EXTRA
SENT_TEMPLATE_V2 = SENT_TEMPLATE.replace(
    "{cands}\n", "{cands}\nOften only one of the listed parties is affected, and sometimes none.\n")
assert SENT_TEMPLATE_V2 != SENT_TEMPLATE
SENT_VARIANTS = {"hyb_sent": (SENT_TEMPLATE, GUIDE), "hyb_sent_v2": (SENT_TEMPLATE_V2, GUIDE_V2)}


def party_label(p: dict) -> str:
    return f"P{p['id']}: {p['name']}" + (f" (referred to as {p['alias']})" if p["alias"] != p["name"] else "")


def evidence(circuit, doc: dict) -> list[dict[int, list[str]]]:
    """Per sentence: party id -> mention texts of that party's wires in the sentence."""
    owner = wire_owner(circuit, doc["parties"], strict=False)
    out = [dict() for _ in doc["sentences"]]
    for e in circuit.entities:
        for pid in owner.get(e.id, ()):
            for si, text in e.mentions:
                if si < len(out):
                    out[si].setdefault(pid, [])
                    if text not in out[si][pid]:
                        out[si][pid].append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["hyb_doc", "hyb_sent"])
    ap.add_argument("--model", default="mlx-community/Qwen2.5-14B-Instruct-4bit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--tag", default=None, help="document set tag (e.g. seed1); default: the seed-0 set")
    args = ap.parse_args()
    P = paths(args.tag)
    docs = read_jsonl(P["docs"])[: args.limit]
    rows = {(r["id"], r["method"]): r for r in read_jsonl(P["pred"]) if r["method"] not in args.variants}
    cands = {r["id"]: r["pred"] for r in rows.values() if r["method"] == STAGE1}
    store = StructureStore(P["struct"]("coref"))
    readers = {}
    if "hyb_doc" in args.variants:
        readers["hyb_doc"] = AuditReader(args.model, max_tokens=512)
    sent_variants = [v for v in SENT_VARIANTS if v in args.variants]
    if sent_variants:
        sent_reader = AuditReader(args.model, max_tokens=48)  # shared by the sentence variants
    t0, n_calls = time.time(), 0
    for k, d in enumerate(docs):
        n_sent, n_p = len(d["sentences"]), len(d["parties"])
        c1 = cands[d["id"]]
        ev = evidence(store.get(d["id"]), d)
        cand_sents = [i for i in range(n_sent) if c1[i]["affects"]]
        if "hyb_doc" in readers:
            lines = []
            for i in cand_sents:
                refs = "; ".join(f"P{pid} as \"{'/'.join(ev[i].get(pid, ['?']))}\"" for pid in c1[i]["affects"])
                lines.append(f"[{i}] (refers to {refs}) {d['sentences'][i]}")
            prompt = DOC_TEMPLATE.format(parties="\n".join(party_label(p) for p in d["parties"]),
                                         sentences="\n".join(lines), guide=GUIDE)
            out, ptoks, secs, cached = _run(readers["hyb_doc"], prompt)
            pred = parse_llm(out, n_sent, n_p)
            for i in range(n_sent):
                keep = sorted(set(pred[i]["affects"]) & set(c1[i]["affects"]))
                pred[i] = {"affects": keep, "change_type": pred[i]["change_type"] if keep else "NONE"}
            if args.print and k == 0:
                print(prompt, "\n---\n", out)
            rows[(d["id"], "hyb_doc")] = {"id": d["id"], "method": "hyb_doc", "pred": pred, "secs": secs,
                                          "cached": cached, "prompt_tokens": ptoks, "raw": out,
                                          "n_candidates": len(cand_sents), "model": args.model}
            n_calls += 1
        for variant in sent_variants:
            template, guide = SENT_VARIANTS[variant]
            pred = [{"affects": [], "change_type": "NONE"} for _ in range(n_sent)]
            secs_total, ptoks_total, raws = 0.0, 0, {}
            for i in cand_sents:
                cl = "\n".join(f"{party_label(d['parties'][pid])}, mentioned here as \"{'/'.join(ev[i].get(pid, ['?']))}\""
                               for pid in c1[i]["affects"])
                prompt = template.format(prev=d["sentences"][i - 1] if i else "(none)", sent=d["sentences"][i],
                                              cands=cl, guide=guide)
                out, ptoks, secs, _ = _run(sent_reader, prompt)
                secs_total += secs; ptoks_total += ptoks; raws[i] = out; n_calls += 1
                if args.print and k == 0 and i == cand_sents[0]:
                    print(prompt, "\n---\n", out)
                affects, typ = set(), "CREATE"
                if "NONE" not in out.upper().split()[:1]:
                    for m in SENT_RE.finditer(out):
                        pid = int(m.group(1))
                        if pid in c1[i]["affects"]:
                            affects.add(pid)
                            t = m.group(2).upper()
                            typ = t if t in {"CREATE", "MODIFY", "END", "TRANSFER"} else "CREATE"
                pred[i] = {"affects": sorted(affects), "change_type": typ if affects else "NONE"}
            rows[(d["id"], variant)] = {"id": d["id"], "method": variant, "pred": pred, "secs": secs_total,
                                           "prompt_tokens": ptoks_total, "raw": raws,
                                           "n_candidates": len(cand_sents), "model": args.model}
        if (k + 1) % 10 == 0:
            print(f"[hybrid] {k+1}/{len(docs)} docs, {n_calls} calls, {(time.time()-t0)/(k+1):.1f}s/doc", flush=True)
    with open(P["pred"], "w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {P['pred']}")


def _run(reader: AuditReader, prompt: str):
    key = reader.r.cache.key(reader.r.model_id, reader._chat(prompt), reader.r.max_tokens)
    cached = reader.r.cache.get(key) is not None
    t0 = time.time()
    out, ptoks = reader.r.generate(prompt)
    return out, ptoks, time.time() - t0, cached


if __name__ == "__main__":
    main()
