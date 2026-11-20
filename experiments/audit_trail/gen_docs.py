"""Angle 4, step 1: generate synthetic contracts and incident reports with gold
duty-change labels known by construction (DESIGN_AUDIT_TRAIL.md section 2).

    python experiments/audit_trail/gen_docs.py            # 100 docs per genre, seed 0
    python experiments/audit_trail/gen_docs.py --n 5 --print

Output: data/processed/audit/docs.jsonl, one document per line:
    id, genre, sentences: [str], parties: [{id, name, alias, surface_tokens, pronoun}],
    gold: [{affects: [party ids], mentions: [party ids], change_type, ref_type}]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sapc.data.loaders import PROCESSED, write_jsonl  # noqa: E402

OUT = PROCESSED / "audit" / "docs.jsonl"

CHANGE_MIX = [("CREATE", 0.45), ("MODIFY", 0.20), ("END", 0.20), ("TRANSFER", 0.15)]
DUTY_SHARE = 0.55

# --------------------------------------------------------------------- vocab
ORG_A = ["Acme", "Bramble", "Castellan", "Dunmore", "Everly", "Fenwick", "Greystone", "Halvard",
         "Ironbridge", "Juniper", "Kestrel", "Lindqvist", "Marlow", "Northwind", "Oakhurst", "Pellerin",
         "Quarrie", "Rosedale", "Sterling", "Tamsin", "Umber", "Valcourt", "Westbrook", "Yarrow"]
ORG_B = ["Logistics", "Retail", "Holdings", "Systems", "Freight", "Energy", "Textiles", "Foods",
         "Marine", "Capital", "Materials", "Pharma"]
ORG_SUFFIX = ["Ltd", "plc", "GmbH", "Inc", "LLC", "AG", "SA"]
ROLES = ["Supplier", "Buyer", "Distributor", "Carrier", "Licensor", "Licensee", "Guarantor",
         "Contractor", "Lender", "Borrower"]
FIRMS = ["Meridian LLP", "Harrow and Vale LLP", "Oldcastle Bank", "Trent Insurance Brokers",
         "Ashby Surveyors", "Corvid Analytics"]
GOODS = ["goods", "equipment", "components", "software", "materials", "consignment"]
DATA = ["pricing data", "customer lists", "technical drawings", "source code"]
PLACES = ["England and Wales", "Scotland", "the State of Delaware", "Ontario", "New South Wales"]
DATES = ["1 March 2027", "30 June 2027", "15 September 2027", "31 December 2027", "1 February 2028"]
AMOUNTS = ["40,000 euros", "120,000 pounds", "75,000 dollars", "a monthly fee of 9,500 euros"]

FIRST = {"he": ["Jamal", "Tomasz", "Daniel", "Kwame", "Rafael", "Hiroshi", "Mateo", "Sven"],
         "she": ["Dana", "Priya", "Aisha", "Ingrid", "Lucia", "Mei", "Noor", "Yara"],
         "they": ["Sam", "Robin", "Alex", "Jordan", "Casey", "Morgan"]}
LAST = ["Reyes", "Kim", "Patel", "Okafor", "Novak", "Lindgren", "Haddad", "Moreau", "Tanaka",
        "Brennan", "Castillo", "Varga", "Oyelaran", "Fischer"]
PROLES = ["operator", "supervisor", "technician", "electrician", "inspector", "driver", "fitter"]
TASKS = ["inspect the loading bay", "isolate the main conveyor", "cordon off dock 3",
         "complete the spill report", "test the emergency stop", "escort the contractor",
         "check the pallet racking", "log the equipment faults", "secure the chemical store",
         "brief the night shift"]
TASK_NOUNS = {"inspect the loading bay": "the inspection of the loading bay",
              "isolate the main conveyor": "the isolation of the main conveyor",
              "cordon off dock 3": "the cordon at dock 3",
              "complete the spill report": "the spill report",
              "test the emergency stop": "the emergency stop test",
              "escort the contractor": "the contractor escort",
              "check the pallet racking": "the racking check",
              "log the equipment faults": "the fault log",
              "secure the chemical store": "the chemical store lock-down",
              "brief the night shift": "the night shift briefing"}
TIMES = ["09:15", "10:40", "11:05", "13:20", "14:10", "15:45", "16:30"]
SITES = ["the Greenford depot", "the Lowfield warehouse", "the Ravenscar plant", "the Dockside yard"]
EQUIPMENT = ["forklift", "conveyor belt", "pallet wrapper", "dock leveller", "overhead crane"]


class Doc:
    def __init__(self, rng: random.Random, genre: str, doc_id: str):
        self.rng, self.genre, self.id = rng, genre, doc_id
        self.parties: list[dict] = []
        self.sentences: list[str] = []
        self.gold: list[dict] = []
        self.clause = 0

    def add(self, text: str, affects: list[int], mentions: list[int], change: str, ref: str):
        assert set(affects) <= set(mentions), (text, affects, mentions)
        self.clause += 1
        self.sentences.append(text)
        self.gold.append({"affects": sorted(set(affects)), "mentions": sorted(set(mentions)),
                          "change_type": change, "ref_type": ref})

    def prev_mentions(self) -> list[int]:
        return self.gold[-1]["mentions"] if self.gold else []

    def pick_party(self, ids: list[int]) -> dict:
        """Half the time stay with the party the previous sentence was solely
        about, so pronoun references ("It shall...", "She completed...") occur
        often enough to measure."""
        prev = self.prev_mentions()
        if len(prev) == 1 and prev[0] in ids and self.rng.random() < 0.5:
            return self.parties[prev[0]]
        return self.parties[self.rng.choice(ids)]

    def pick_change(self) -> str:
        r, acc = self.rng.random(), 0.0
        for c, p in CHANGE_MIX:
            acc += p
            if r < acc:
                return c
        return "CREATE"


# ------------------------------------------------------------------ contract
def new_org(rng: random.Random, used: set[str]) -> dict:
    while True:
        a, b, s = rng.choice(ORG_A), rng.choice(ORG_B), rng.choice(ORG_SUFFIX)
        if a not in used and b not in used and s not in used:
            used.update({a, b, s})
            return {"name": f"{a} {b} {s}", "tokens": [a, b, s]}


def contract(rng: random.Random, doc_id: str, n_sent: int) -> Doc:
    d = Doc(rng, "contract", doc_id)
    used: set[str] = set()
    roles = rng.sample(ROLES, 4)
    n_intro = rng.choice([2, 2, 3, 4])
    for k in range(n_intro):
        org = new_org(rng, used)
        d.parties.append({"id": k, "name": org["name"], "alias": f"the {roles[k]}",
                          "surface_tokens": [t.lower() for t in org["tokens"]] + [roles[k].lower()],
                          "pronoun": "it"})
    intro_ids = list(range(n_intro))
    goods, data = rng.choice(GOODS), rng.choice(DATA)

    parts = " and ".join(f"{p['name']}, referred to as {p['alias']}," for p in d.parties)
    d.add(f"This agreement is made between {parts} on {rng.choice(DATES)}.", [], intro_ids, "NONE", "none")
    d.add(f"In this agreement, the {goods} means the items listed in Schedule 1.", [], [], "NONE", "none")

    def ref(p: dict, kind: str) -> str:
        if kind == "name":
            return p["name"]
        if kind == "alias":
            return p["alias"]
        if kind == "pronoun":
            return "it"
        raise ValueError(kind)

    def cap(s: str) -> str:
        return s[0].upper() + s[1:]

    def choose_ref(p: dict) -> str:
        kinds = ["name", "alias", "alias", "pronoun", "pronoun"]
        k = rng.choice(kinds)
        if k == "pronoun" and d.prev_mentions() != [p["id"]]:
            k = rng.choice(["name", "alias"])
        return k

    def other(p: dict) -> dict:
        return rng.choice([q for q in d.parties if q["id"] != p["id"] and q["id"] in intro_ids] or
                          [q for q in d.parties if q["id"] != p["id"]])

    while len(d.sentences) < n_sent:
        p = d.pick_party(intro_ids)
        o = other(p)
        if rng.random() < DUTY_SHARE:
            change = d.pick_change()
            if change == "TRANSFER" and len(d.parties) >= n_intro + 2:
                change = "CREATE"      # at most two assignees per contract (name pool)
            form = rng.random()
            if change == "CREATE":
                if form < 0.15:   # passive, agent is the affected party
                    k = rng.choice(["name", "alias"])
                    d.add(f"The {goods} shall be insured against loss by {ref(p, k)}.", [p["id"]], [p["id"]], change, "passive")
                elif form < 0.30:
                    k = rng.choice(["name", "alias"])
                    d.add(f"{rng.choice(AMOUNTS).capitalize()} shall be paid to {o['alias']} by {ref(p, k)} within {rng.choice([14, 30, 45])} days.",
                          [p["id"]], [p["id"], o["id"]], change, "passive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    t = rng.choice([
                        (f"{r} shall deliver the {goods} to {o['alias']} within {rng.choice([30, 45, 60])} days.", [p["id"], o["id"]]),
                        (f"{r} shall pay {o['alias']} {rng.choice(AMOUNTS)} within {rng.choice([14, 30])} days of delivery.", [p["id"], o["id"]]),
                        (f"{r} shall notify {o['alias']} in writing of any delay.", [p["id"], o["id"]]),
                        (f"{r} shall maintain records of each shipment for {rng.choice([3, 5, 7])} years.", [p["id"]]),
                        (f"{r} shall keep the {data} confidential.", [p["id"]]),
                        (f"{r} shall provide monthly reports to {o['alias']}.", [p["id"], o["id"]]),
                        (f"{r} shall insure the {goods} against loss.", [p["id"]]),
                    ])
                    d.add(t[0], [p["id"]], t[1], change, k)
            elif change == "MODIFY":
                if form < 0.35:
                    k = rng.choice(["name", "alias"])
                    t = rng.choice([
                        f"{cap(ref(p, k))}'s delivery period is extended to {rng.choice([75, 90, 120])} days.",
                        f"{cap(ref(p, k))}'s payment term is shortened to {rng.choice([7, 10])} days.",
                        f"{cap(ref(p, k))}'s reporting obligation is reduced to quarterly reports.",
                    ])
                    d.add(t, [p["id"]], [p["id"]], change, "possessive")
                else:
                    k = choose_ref(p)
                    r = ref(p, k)
                    t = rng.choice([
                        (f"Notwithstanding clause {max(1, d.clause - rng.randint(1, 4))}, {r} shall deliver the {goods} within {rng.choice([75, 90])} days.", [p["id"]]),
                        (f"The parties agree that {r} shall instead pay {rng.choice(AMOUNTS)}.", [p["id"]]),
                        (f"From {rng.choice(DATES)}, {r} shall provide weekly rather than monthly reports to {o['alias']}.", [p["id"], o["id"]]),
                        (f"{cap(r)} shall extend the warranty to {rng.choice([18, 24, 36])} months.", [p["id"]]),
                    ])
                    d.add(t[0], [p["id"]], t[1], change, k)
            elif change == "END":
                if form < 0.4:
                    k = rng.choice(["name", "alias"])
                    t = rng.choice([
                        f"{cap(ref(p, k))}'s obligation to insure the {goods} ceases on {rng.choice(DATES)}.",
                        f"{cap(ref(p, k))}'s duty to provide reports ends on {rng.choice(DATES)}.",
                    ])
                    d.add(t, [p["id"]], [p["id"]], change, "possessive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    t = rng.choice([
                        f"{r} is released from the obligation to keep the {data} confidential.",
                        f"{r} is no longer required to insure the {goods} after {rng.choice(DATES)}.",
                        f"On {rng.choice(DATES)} the obligations of {ref(p, k)} under clause {max(1, d.clause - rng.randint(1, 4))} terminate.",
                    ])
                    d.add(t, [p["id"]], [p["id"]], change, k)
            else:  # TRANSFER: assignment to a new party, introduced here
                k = choose_ref(p)
                r = cap(ref(p, k))
                org = new_org(rng, used)
                new = {"id": len(d.parties), "name": org["name"], "alias": org["name"],
                       "surface_tokens": [t.lower() for t in org["tokens"]], "pronoun": "it"}
                d.parties.append(new)
                t = rng.choice([
                    f"{r} assigns its obligation to deliver the {goods} to {new['name']}.",
                    f"{r} transfers its payment obligations under clause {max(1, d.clause - rng.randint(1, 4))} to {new['name']}.",
                    f"{new['name']} assumes the obligations of {ref(p, k)} under clause {max(1, d.clause - rng.randint(1, 4))}.",
                ])
                d.add(t, [p["id"], new["id"]], [p["id"], new["id"]], change, k)
        else:
            form = rng.random()
            if form < 0.25:   # right, not duty
                k = choose_ref(p)
                r = cap(ref(p, k))
                t = rng.choice([
                    (f"{r} may inspect the premises of {o['alias']} on reasonable notice.", [p["id"], o["id"]]),
                    (f"{r} may terminate this agreement on {rng.choice([30, 60, 90])} days' notice.", [p["id"]]),
                    (f"{r} may appoint a sub-contractor with the consent of {o['alias']}.", [p["id"], o["id"]]),
                ])
                d.add(t[0], [], t[1], "NONE", k)
            elif form < 0.40:  # third party
                k = rng.choice(["name", "alias"])
                d.add(f"{cap(ref(p, k))}'s auditor, {rng.choice(FIRMS)}, prepared the valuation report.", [], [p["id"]], "NONE", k)
            elif form < 0.55:  # recital
                d.add(f"{cap(o['alias'])} wishes to purchase the {goods} from {p['alias']}.", [], [p["id"], o["id"]], "NONE", "alias")
            else:
                d.add(rng.choice([
                    f"This agreement is governed by the laws of {rng.choice(PLACES)}.",
                    "Notices under this agreement shall be sent to the addresses in Schedule 2.",
                    f"The {goods} were manufactured in {rng.choice(PLACES)}.",
                    f"Schedule {rng.choice([1, 2, 3])} forms part of this agreement.",
                    "Headings are for convenience only and do not affect interpretation.",
                ]), [], [], "NONE", "none")
    return d


# ------------------------------------------------------------------ incident
def incident(rng: random.Random, doc_id: str, n_sent: int) -> Doc:
    d = Doc(rng, "incident", doc_id)
    n_people = rng.choice([2, 3, 3, 4])
    lasts = rng.sample(LAST, n_people)
    roles = rng.sample(PROLES, n_people)
    firsts_used: set[str] = set()
    for k in range(n_people):
        pron = rng.choice(["he", "she", "they"])
        first = rng.choice([f for f in FIRST[pron] if f not in firsts_used])
        firsts_used.add(first)
        d.parties.append({"id": k, "name": f"{first} {lasts[k]}", "alias": f"the {roles[k]}",
                          "surface_tokens": [first.lower(), lasts[k].lower(), roles[k].lower()],
                          "pronoun": pron, "_last": lasts[k], "_role": roles[k]})
    ids = list(range(n_people))
    site = rng.choice(SITES)
    names = ", ".join(p["name"] for p in d.parties[:-1]) + f" and {d.parties[-1]['name']}"
    d.add(f"On {rng.choice(DATES)}, {names} were on shift at {site}.", [], ids, "NONE", "name")

    def ref(p: dict, kind: str) -> str:
        if kind == "name":
            return p["name"]
        if kind == "alias":
            return rng.choice([p["alias"], f"{p['_role'].capitalize()} {p['_last']}"])
        if kind == "pronoun":
            return p["pronoun"]
        raise ValueError(kind)

    def cap(s: str) -> str:
        return s[0].upper() + s[1:]

    def choose_ref(p: dict) -> str:
        k = rng.choice(["name", "alias", "alias", "pronoun", "pronoun"])
        if k == "pronoun" and d.prev_mentions() != [p["id"]]:
            k = rng.choice(["name", "alias"])
        return k

    def other(p: dict) -> dict:
        return rng.choice([q for q in d.parties if q["id"] != p["id"]])

    while len(d.sentences) < n_sent:
        p = d.pick_party(ids)
        o = other(p)
        task = rng.choice(TASKS)
        tn = TASK_NOUNS[task]
        time = rng.choice(TIMES)
        if rng.random() < DUTY_SHARE:
            change = d.pick_change()
            form = rng.random()
            if change == "CREATE":
                if form < 0.20:   # affected party as object of another party's instruction
                    k = rng.choice(["name", "alias"])
                    d.add(f"{cap(ref(o, rng.choice(['name', 'alias'])))} instructed {ref(p, k)} to {task} at {time}.",
                          [p["id"]], [p["id"], o["id"]], change, "object")
                elif form < 0.30:
                    k = rng.choice(["name", "alias"])
                    d.add(f"{cap(tn)} was assigned to {ref(p, k)} at {time}.", [p["id"]], [p["id"]], change, "passive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    t = rng.choice([
                        (f"{r} was assigned to {task} at {time}.", [p["id"]]),
                        (f"{r} took responsibility for {tn} at {time}.", [p["id"]]),
                        (f"{r} agreed to {task} before the end of the shift.", [p["id"]]),
                        (f"{r} agreed with {ref(o, 'alias')} to {task}.", [p["id"], o["id"]]),
                    ])
                    d.add(t[0], [p["id"]], t[1], change, k)
            elif change == "MODIFY":
                if form < 0.4:
                    k = rng.choice(["name", "alias"])
                    d.add(f"{cap(ref(p, k))}'s task was changed to {task} at {time}.", [p["id"]], [p["id"]], change, "possessive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    d.add(rng.choice([
                        f"{r} was additionally asked to {task}.",
                        f"At {time} {ref(p, k)} extended the check to include the {rng.choice(EQUIPMENT)}.",
                    ]), [p["id"]], [p["id"]], change, k)
            elif change == "END":
                if form < 0.25:
                    k = rng.choice(["name", "alias"])
                    d.add(f"{cap(tn)} was completed by {ref(p, k)} at {time}.", [p["id"]], [p["id"]], change, "passive")
                elif form < 0.45:
                    k = rng.choice(["name", "alias"])
                    d.add(f"{cap(ref(p, k))}'s responsibility for {tn} ended at {time}.", [p["id"]], [p["id"]], change, "possessive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    d.add(rng.choice([
                        f"{r} completed {tn} at {time}.",
                        f"{r} was relieved of {tn} at {time}.",
                        f"{r} finished {tn} before the break.",
                    ]), [p["id"]], [p["id"]], change, k)
            else:  # TRANSFER between two existing people
                if form < 0.35:
                    k = rng.choice(["name", "alias"])
                    d.add(f"Responsibility for {tn} was transferred from {ref(p, k)} to {ref(o, rng.choice(['name', 'alias']))} at {time}.",
                          [p["id"], o["id"]], [p["id"], o["id"]], change, "passive")
                else:
                    k = choose_ref(p)
                    r = cap(ref(p, k))
                    d.add(rng.choice([
                        f"{r} handed over {tn} to {ref(o, rng.choice(['name', 'alias']))} at {time}.",
                        f"{r} passed {tn} to {ref(o, rng.choice(['name', 'alias']))} before leaving.",
                    ]), [p["id"], o["id"]], [p["id"], o["id"]], change, k)
        else:
            form = rng.random()
            if form < 0.30:   # observation by a party, no responsibility change
                k = choose_ref(p)
                r = cap(ref(p, k))
                d.add(rng.choice([
                    f"{r} noted that the floor was wet near dock {rng.choice([1, 2, 3, 4])}.",
                    f"{r} observed that the {rng.choice(EQUIPMENT)} alarm was sounding.",
                    f"{r} reported that visibility in the yard was poor.",
                ]), [], [p["id"]], "NONE", k)
            elif form < 0.45:  # witness statement mentioning two parties
                d.add(f"{cap(ref(p, rng.choice(['name', 'alias'])))} stated that {ref(o, rng.choice(['name', 'alias']))} had been near the {rng.choice(EQUIPMENT)}.",
                      [], [p["id"], o["id"]], "NONE", "name")
            else:
                d.add(rng.choice([
                    f"At {time} the temperature in the warehouse was {rng.randint(4, 28)} degrees.",
                    f"A {rng.choice(EQUIPMENT)} was parked beside the dock.",
                    f"The {rng.choice(EQUIPMENT)} was last serviced in {rng.choice(['March', 'June', 'August'])}.",
                    "Rain had been falling since the early morning.",
                    f"The site alarm was tested at {time} as scheduled.",
                ]), [], [], "NONE", "none")
    for p in d.parties:
        p.pop("_last", None)
        p.pop("_role", None)
    return d


# --------------------------------------------------------------------- main
def validate(d: Doc) -> None:
    text = " ".join(d.sentences).lower()
    for p in d.parties:
        assert any(t in text for t in p["surface_tokens"]), (d.id, p)
    toks = [t for p in d.parties for t in p["surface_tokens"]]
    assert len(toks) == len(set(toks)), (d.id, "surface tokens collide across parties", toks)
    assert len(d.sentences) == len(d.gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="documents per genre")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-sents", type=int, default=12)
    ap.add_argument("--max-sents", type=int, default=30)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--tag", default=None, help="write docs_<tag>.jsonl with ids audit-s<seed>-...; default: docs.jsonl")
    args = ap.parse_args()
    out = OUT.with_name(f"docs_{args.tag}.jsonl") if args.tag else OUT
    prefix = f"audit-s{args.seed}-" if args.tag else "audit-"
    rng = random.Random(args.seed)
    docs = []
    for genre, fn in (("contract", contract), ("incident", incident)):
        for i in range(args.n):
            n_sent = rng.randint(args.min_sents, args.max_sents)
            d = fn(rng, f"{prefix}{genre}-{i:03d}", n_sent)
            validate(d)
            docs.append({"id": d.id, "genre": genre, "sentences": d.sentences,
                         "parties": d.parties, "gold": d.gold})
    write_jsonl(docs, out)
    n_s = sum(len(x["sentences"]) for x in docs)
    n_duty = sum(1 for x in docs for g in x["gold"] if g["affects"])
    from collections import Counter
    ref = Counter(g["ref_type"] for x in docs for g in x["gold"] if g["affects"])
    ch = Counter(g["change_type"] for x in docs for g in x["gold"])
    hard = sum(1 for x in docs for g in x["gold"] if set(g["affects"]) != set(g["mentions"]))
    print(f"wrote {len(docs)} docs, {n_s} sentences, {n_duty} duty-changing, "
          f"{hard} with mentions != affects -> {out}")
    print("ref_type (duty-changing):", dict(ref))
    print("change_type:", dict(ch))
    if args.print:
        for x in docs[:1] + docs[args.n:args.n + 1]:
            print("\n==", x["id"], [p["name"] + " / " + p["alias"] for p in x["parties"]])
            for s, g in zip(x["sentences"], x["gold"]):
                print(f"  {g['affects']!s:8} {g['change_type']:8} {g['ref_type']:10} {s}")


if __name__ == "__main__":
    main()
