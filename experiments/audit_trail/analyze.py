"""Angle 4, step 4: score predictions against gold and write the report.

    python experiments/audit_trail/analyze.py

Reads results/a4_audit/predictions.jsonl; writes summary.csv, comparisons.csv,
breakdowns.csv and report.md next to it. Documents on which the discocirc
backend failed are excluded from every method (DESIGN_AUDIT_TRAIL.md section 3).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sapc.data.loaders import PROCESSED, read_jsonl  # noqa: E402
from sapc.eval.metrics import paired_bootstrap  # noqa: E402

DOCS = PROCESSED / "audit" / "docs.jsonl"
OUT_DIR = ROOT / "results" / "a4_audit"
ORDER = ["surface_any", "coref_any", "coref_subj", "dc_any", "dc_subj",
         "coref_any_strict", "dc_any_strict", "llm_zs", "hyb_doc", "hyb_sent", "llm_zs_v2", "hyb_sent_v2"]
PAIRS = [("dc_any", "surface_any"), ("dc_any", "coref_any"), ("dc_subj", "coref_subj"),
         ("dc_any", "dc_subj"), ("dc_any", "llm_zs"), ("dc_subj", "llm_zs"), ("coref_any", "llm_zs"),
         ("dc_any", "dc_any_strict"), ("hyb_sent", "llm_zs"), ("hyb_doc", "llm_zs"),
         ("hyb_sent", "hyb_doc"), ("hyb_sent", "coref_any"),
         # E2 confirmation
         ("hyb_sent_v2", "llm_zs_v2"), ("hyb_sent_v2", "hyb_sent"), ("llm_zs_v2", "llm_zs"), ("hyb_sent_v2", "llm_zs")]
TAG: str | None = None


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_doc(doc: dict, pred: list[dict]) -> dict:
    tp = fp = fn = 0
    exact = 0
    ct_ok = ct_n = 0
    by_ref = defaultdict(lambda: [0, 0])          # ref_type -> [recalled, gold]
    by_change = defaultdict(lambda: [0, 0])
    hard_tp = hard_fp = rest_tp = rest_fp = 0
    for g, p in zip(doc["gold"], pred):
        G, P = set(g["affects"]), set(p["affects"])
        tp += len(G & P); fp += len(P - G); fn += len(G - P)
        exact += int(G == P)
        if G and P:
            ct_n += 1
            ct_ok += int(g["change_type"] == p["change_type"])
        if G:
            by_ref[g["ref_type"]][0] += len(G & P); by_ref[g["ref_type"]][1] += len(G)
            by_change[g["change_type"]][0] += len(G & P); by_change[g["change_type"]][1] += len(G)
        if set(g["mentions"]) != G:
            hard_tp += len(G & P); hard_fp += len(P - G)
        else:
            rest_tp += len(G & P); rest_fp += len(P - G)
    traces_exact = traces_rec = traces_prec = 0
    for party in doc["parties"]:
        pid = party["id"]
        gt = {i for i, g in enumerate(doc["gold"]) if pid in g["affects"]}
        pt = {i for i, p in enumerate(pred) if pid in p["affects"]}
        traces_exact += int(gt == pt)
        traces_rec += len(gt & pt) / len(gt) if gt else 1.0
        traces_prec += len(gt & pt) / len(pt) if pt else (1.0 if not gt else 0.0)
    n_p = len(doc["parties"])
    return {"tp": tp, "fp": fp, "fn": fn, "f1": prf(tp, fp, fn)[2], "exact": exact, "n_sent": len(pred),
            "ct_ok": ct_ok, "ct_n": ct_n, "by_ref": dict(by_ref), "by_change": dict(by_change),
            "hard": (hard_tp, hard_fp), "rest": (rest_tp, rest_fp),
            "trace_exact": traces_exact / n_p, "trace_rec": traces_rec / n_p, "trace_prec": traces_prec / n_p}


def agreement(pa: list[dict], pb: list[dict], n_parties: int) -> tuple[int, int]:
    agree = total = 0
    for a, b in zip(pa, pb):
        A, B = set(a["affects"]), set(b["affects"])
        for pid in range(n_parties):
            total += 1
            agree += int((pid in A) == (pid in B))
    return agree, total


def main() -> None:
    global DOCS, OUT_DIR, TAG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="document set tag (e.g. seed1); default: the seed-0 set")
    TAG = ap.parse_args().tag
    if TAG:
        DOCS = DOCS.with_name(f"docs_{TAG}.jsonl")
        OUT_DIR = OUT_DIR.with_name(f"a4_audit_{TAG}")
    docs = {d["id"]: d for d in read_jsonl(DOCS)}
    rows = read_jsonl(OUT_DIR / "predictions.jsonl")
    preds: dict[str, dict[str, list[dict] | None]] = defaultdict(dict)
    meta: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        preds[r["method"]][r["id"]] = r["pred"]
        meta[r["method"]][r["id"]] = r
    methods = [m for m in ORDER if m in preds]
    excluded = {i for m in methods for i, p in preds[m].items() if p is None}
    ids = sorted(i for i in docs if all(i in preds[m] for m in methods) and i not in excluded)
    genres = sorted({docs[i]["genre"] for i in ids})

    scores = {m: {i: score_doc(docs[i], preds[m][i]) for i in ids} for m in methods}
    summary, breakdown = [], []
    for m in methods:
        for genre in ["all"] + genres:
            sel = [i for i in ids if genre == "all" or docs[i]["genre"] == genre]
            S = [scores[m][i] for i in sel]
            tp, fp, fn = (sum(s[k] for s in S) for k in ("tp", "fp", "fn"))
            p, r, f = prf(tp, fp, fn)
            ct = sum(s["ct_ok"] for s in S) / max(1, sum(s["ct_n"] for s in S))
            secs = [meta[m][i].get("secs") for i in sel if meta[m][i].get("secs") is not None]
            summary.append({"method": m, "genre": genre, "n_docs": len(sel), "precision": round(p, 4),
                            "recall": round(r, 4), "f1": round(f, 4),
                            "sent_exact": round(sum(s["exact"] for s in S) / sum(s["n_sent"] for s in S), 4),
                            "trace_exact": round(float(np.mean([s["trace_exact"] for s in S])), 4),
                            "trace_recall": round(float(np.mean([s["trace_rec"] for s in S])), 4),
                            "trace_precision": round(float(np.mean([s["trace_prec"] for s in S])), 4),
                            "change_type_acc": round(ct, 4),
                            "secs_per_doc_median": round(float(np.median(secs)), 2) if secs else None,
                            "prompt_tokens_mean": round(float(np.mean([meta[m][i]["prompt_tokens"] for i in sel])), 0)
                            if m.startswith(("llm", "hyb")) else None})
            if genre == "all":
                ref, chg = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
                for s in S:
                    for k, v in s["by_ref"].items():
                        ref[k][0] += v[0]; ref[k][1] += v[1]
                    for k, v in s["by_change"].items():
                        chg[k][0] += v[0]; chg[k][1] += v[1]
                for k, v in sorted(ref.items()):
                    breakdown.append({"method": m, "axis": "ref_type", "value": k, "metric": "recall",
                                      "score": round(v[0] / v[1], 4), "n_gold_pairs": v[1]})
                for k, v in sorted(chg.items()):
                    breakdown.append({"method": m, "axis": "change_type", "value": k, "metric": "recall",
                                      "score": round(v[0] / v[1], 4), "n_gold_pairs": v[1]})
                htp, hfp = (sum(s["hard"][0] for s in S), sum(s["hard"][1] for s in S))
                rtp, rfp = (sum(s["rest"][0] for s in S), sum(s["rest"][1] for s in S))
                breakdown.append({"method": m, "axis": "mentions!=affects", "value": "hard", "metric": "precision",
                                  "score": round(prf(htp, hfp, 0)[0], 4), "n_gold_pairs": htp + hfp})
                breakdown.append({"method": m, "axis": "mentions!=affects", "value": "rest", "metric": "precision",
                                  "score": round(prf(rtp, rfp, 0)[0], 4), "n_gold_pairs": rtp + rfp})

    comparisons = []
    for a, b in PAIRS:
        if a in methods and b in methods:
            for genre in ["all"] + genres:
                sel = [i for i in ids if genre == "all" or docs[i]["genre"] == genre]
                bs = paired_bootstrap(np.array([scores[a][i]["f1"] for i in sel]),
                                      np.array([scores[b][i]["f1"] for i in sel]))
                comparisons.append({"a": a, "b": b, "genre": genre, "metric": "doc_f1", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in bs.items()}})

    # H4-null: decision-level agreement discocirc vs coref
    null = {}
    for a, b in (("dc_any", "coref_any"), ("dc_subj", "coref_subj")):
        if a in methods and b in methods:
            ag = tot = 0
            for i in ids:
                x, y = agreement(preds[a][i], preds[b][i], len(docs[i]["parties"]))
                ag += x; tot += y
            null[f"{a} vs {b}"] = ag / tot

    for name, data in (("summary.csv", summary), ("comparisons.csv", comparisons), ("breakdowns.csv", breakdown)):
        with open(OUT_DIR / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    # ------------------------------------------------------------- report
    L = ["# Angle 4: audit trails from text circuits" + (f" (document set: {TAG})" if TAG else ""), "",
         f"Documents: {len(docs)} generated, {len(ids)} scored, {len(excluded)} excluded "
         f"(structure error on some backend). Methods: {', '.join(methods)}.", "",
         "## Primary metrics, all documents", "",
         "| method | P | R | F1 | sentence exact | trace exact | trace R | trace P | change-type acc | s/doc (median) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if s["genre"] == "all":
            L.append(f"| {s['method']} | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** | {s['sent_exact']:.3f} | "
                     f"{s['trace_exact']:.3f} | {s['trace_recall']:.3f} | {s['trace_precision']:.3f} | {s['change_type_acc']:.3f} | {s['secs_per_doc_median']} |")
    for genre in genres:
        L += ["", f"## {genre}", "", "| method | P | R | F1 | sentence exact | trace exact |", "|---|---|---|---|---|---|"]
        for s in summary:
            if s["genre"] == genre:
                L.append(f"| {s['method']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['sent_exact']:.3f} | {s['trace_exact']:.3f} |")

    L += ["", "## Recall by how the affected party is referred to (gold pairs)", ""]
    refs = sorted({b["value"] for b in breakdown if b["axis"] == "ref_type"})
    L += ["| method | " + " | ".join(refs) + " |", "|---|" + "---|" * len(refs)]
    for m in methods:
        cells = {b["value"]: b for b in breakdown if b["axis"] == "ref_type" and b["method"] == m}
        L.append(f"| {m} | " + " | ".join(f"{cells[r]['score']:.3f} (n={cells[r]['n_gold_pairs']})" if r in cells else "-" for r in refs) + " |")
    L += ["", "## Recall by change type", ""]
    chs = [c for c in ("CREATE", "MODIFY", "END", "TRANSFER") if any(b["value"] == c for b in breakdown)]
    L += ["| method | " + " | ".join(chs) + " |", "|---|" + "---|" * len(chs)]
    for m in methods:
        cells = {b["value"]: b for b in breakdown if b["axis"] == "change_type" and b["method"] == m}
        L.append(f"| {m} | " + " | ".join(f"{cells[c]['score']:.3f}" if c in cells else "-" for c in chs) + " |")
    L += ["", "## Precision on sentences where mentions != affects (bilateral, rights, third parties, witness) vs the rest", "",
          "| method | hard | rest |", "|---|---|---|"]
    for m in methods:
        cells = {b["value"]: b for b in breakdown if b["axis"] == "mentions!=affects" and b["method"] == m}
        L.append(f"| {m} | {cells['hard']['score']:.3f} | {cells['rest']['score']:.3f} |")

    L += ["", "## Paired bootstrap on per-document F1 (a - b), 10,000 resamples", "",
          "| a | b | genre | mean diff | 95% CI | significant |", "|---|---|---|---|---|---|"]
    for c in comparisons:
        L.append(f"| {c['a']} | {c['b']} | {c['genre']} | {c['mean_diff']:+.3f} | [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | {'yes' if c['significant'] else 'no'} |")

    L += ["", "## Pre-registered verdicts (DESIGN_AUDIT_TRAIL.md section 5)", ""]
    allsum = {s["method"]: s for s in summary if s["genre"] == "all"}
    ref_of = lambda m, r: next((b["score"] for b in breakdown if b["axis"] == "ref_type" and b["method"] == m and b["value"] == r), None)  # noqa: E731
    if "dc_any" in methods and "surface_any" in methods:
        for r in ("pronoun", "possessive"):
            L.append(f"* H4a ({r}): recall dc_any {ref_of('dc_any', r)} vs surface_any {ref_of('surface_any', r)}")
    if "llm_zs" in methods:
        best = max((m for m in methods if not m.startswith(("llm", "hyb")) and not m.endswith("_strict")), key=lambda m: allsum[m]["f1"])
        gap = allsum["llm_zs"]["f1"] - allsum[best]["f1"]
        verdict = ("competitive (gap <= 5 points)" if gap <= 0.05 else
                   "structure alone is not the product (gap > 10 points)" if gap > 0.10 else "inconclusive (5 to 10 points)")
        L.append(f"* H4b: llm_zs F1 {allsum['llm_zs']['f1']:.3f} vs best structural {best} {allsum[best]['f1']:.3f}; gap {gap*100:+.1f} points -> {verdict}")
    if "dc_subj" in methods:
        L.append(f"* H4c (dc_subj recall): name {ref_of('dc_subj', 'name')}, passive {ref_of('dc_subj', 'passive')}, "
                 f"possessive {ref_of('dc_subj', 'possessive')}, object {ref_of('dc_subj', 'object')}")
    if "dc_any" in methods:
        cells = {b["value"]: b["score"] for b in breakdown if b["axis"] == "mentions!=affects" and b["method"] == "dc_any"}
        L.append(f"* H4c (dc_any precision): hard {cells['hard']:.3f} vs rest {cells['rest']:.3f}")
    for h in ("hyb_doc", "hyb_sent"):
        if h in methods and "llm_zs" in methods:
            L.append(f"* E1 (post hoc) {h}: P {allsum[h]['precision']:.3f} R {allsum[h]['recall']:.3f} F1 {allsum[h]['f1']:.3f} vs llm_zs F1 {allsum['llm_zs']['f1']:.3f}")
    for k, v in null.items():
        L.append(f"* H4-null: decision agreement {k} = {v:.4f} ({'>= 0.95: signal is coreference' if v >= 0.95 else '< 0.95: the formalism changes decisions'})")

    # ------------------------------------------ E2 confirmation (DESIGN_AUDIT_TRAIL.md, extension E2)
    if "hyb_sent_v2" in methods and "llm_zs_v2" in methods:
        L += ["", "## E2 confirmation verdicts (DESIGN_AUDIT_TRAIL.md, extension E2)", ""]
        comp_of = lambda a, b: next((c for c in comparisons if c["a"] == a and c["b"] == b and c["genre"] == "all"), None)  # noqa: E731
        for tag, a, b in (("E2a", "hyb_sent", "llm_zs"), ("E2b", "hyb_sent_v2", "llm_zs_v2")):
            c = comp_of(a, b)
            if c:
                ok = c["significant"] and c["mean_diff"] > 0
                L.append(f"* {tag}: {a} - {b} per-doc F1 {c['mean_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] -> {'pass' if ok else 'FAIL'}")

        def end_recall(m: str) -> float:
            S = [scores[m][i] for i in ids if docs[i]["genre"] == "incident"]
            rec = sum(s["by_change"].get("END", [0, 0])[0] for s in S)
            n = sum(s["by_change"].get("END", [0, 0])[1] for s in S)
            return rec / n if n else float("nan")

        def hard(m: str) -> float:
            return next(b["score"] for b in breakdown if b["axis"] == "mentions!=affects" and b["method"] == m and b["value"] == "hard")

        if "hyb_sent" in methods:
            d_end = end_recall("hyb_sent_v2") - end_recall("hyb_sent")
            L.append(f"* E2c (incident END recall): hyb_sent {end_recall('hyb_sent'):.3f} -> hyb_sent_v2 {end_recall('hyb_sent_v2'):.3f} "
                     f"({d_end:+.3f}; rule >= +0.15 -> {'pass' if d_end >= 0.15 else 'FAIL'})")
        for a, b in (("hyb_sent_v2", "hyb_sent"), ("llm_zs_v2", "llm_zs")):
            if a in methods and b in methods:
                L.append(f"* E2c (precision on mentions != affects): {b} {hard(b):.3f} -> {a} {hard(a):.3f} ({'rises' if hard(a) > hard(b) else 'does not rise'})")
        if "hyb_sent" in methods and "llm_zs" in methods:
            L.append(f"* E2c (predictions): hyb_sent_v2 F1 {allsum['hyb_sent_v2']['f1']:.3f} (predicted >= 0.80); "
                     f"v2 - v1 gain: hybrid {allsum['hyb_sent_v2']['f1'] - allsum['hyb_sent']['f1']:+.3f}, "
                     f"llm {allsum['llm_zs_v2']['f1'] - allsum['llm_zs']['f1']:+.3f}")
        c = comp_of("llm_zs", "coref_any")
        if c:
            L.append(f"* Replication: llm_zs - coref_any {c['mean_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] (section 8: +0.052)")
    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
