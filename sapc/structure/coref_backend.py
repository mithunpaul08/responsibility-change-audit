"""Backend `coref`: the ablation. Same Maverick coreference, same lemma merge,
but entity wires come from spaCy noun tokens and the gate's subject/predicate
from the spaCy dependency parse. No lambeq, no CCG, no pregroup tree.

If `lightcone` makes the same choices from this backend as from `discocirc`,
the structural signal is coreference, not the categorical formalism (H2-null).
"""
from __future__ import annotations

from functools import lru_cache

from sapc.structure.coref import PresplitMaverick, first_mention_map
from sapc.structure.text_circuit import Entity, Gate, TextCircuit

NOUN_POS = {"NOUN", "PROPN", "PRON"}


@lru_cache(maxsize=1)
def _coref(device: str = "cpu"):
    return PresplitMaverick(device=device)


def build(sentences: list[str], question: str, device: str = "cpu",
          merge_same_noun: bool = True) -> TextCircuit:
    coref = _coref(device)
    all_sents = list(sentences) + [question]
    tokens, docs, corefs = coref.coref(all_sents)
    canon = first_mention_map(corefs)
    lemma_wire: dict[str, tuple[int, int]] = {}

    def canonical(nid):
        c = canon.get(nid, nid)
        if merge_same_noun:
            tok = docs[c[0]][c[1]]
            if tok.pos_ == "NOUN":
                c = lemma_wire.setdefault(tok.lemma_.lower(), c)
        return c

    noun2wire: dict[tuple[int, int], int] = {}
    entities: list[Entity] = []
    gates: list[Gate] = []
    q_wires: list[int] = []
    for i, doc in enumerate(docs):
        wires, tok2wire = [], {}
        for t in doc:
            if t.pos_ not in NOUN_POS:
                continue
            key = canonical((i, t.i))
            if key not in noun2wire:
                noun2wire[key] = len(entities)
                entities.append(Entity(id=len(entities), name=tokens[key[0]][key[1]]))
            w = noun2wire[key]
            entities[w].mentions.append((i, t.text))
            tok2wire[t.i] = w
            if w not in wires:
                wires.append(w)
        if i == len(sentences):
            q_wires = wires
            break
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        subj = None
        if root is not None:
            for t in root.children:
                if t.dep_ in {"nsubj", "nsubjpass"}:
                    subj = tok2wire.get(t.i) or next((tok2wire[c.i] for c in t.subtree if c.i in tok2wire), None)
                    break
        if subj is None and wires:
            subj = wires[0]
        pred = root.lemma_.lower() if root is not None else None
        gates.append(Gate(idx=i, text=sentences[i], wires=wires, subject=subj, predicate=pred,
                          objects=[w for w in wires if w != subj]))
    return TextCircuit(sentences=list(sentences), entities=entities, gates=gates,
                       question=question, question_wires=q_wires, backend="coref",
                       meta={"n_wires": len(entities), "merge_same_noun": merge_same_noun})
