"""Backend `surface`: no coreference at all. An entity is a distinct noun
*string* (lower-cased lemma); a sentence touches it iff the string occurs in
the sentence. Pronouns are their own 'entities' and never link to anything.

This is the `lexical_entity` baseline: what light-cone pruning degenerates to
when you throw the coreference wires away.
"""
from __future__ import annotations

from sapc.structure.coref import spacy_nlp
from sapc.structure.text_circuit import Entity, Gate, TextCircuit

NOUN_POS = {"NOUN", "PROPN", "PRON"}


def build(sentences: list[str], question: str, **_) -> TextCircuit:
    nlp = spacy_nlp()
    docs = list(nlp.pipe(list(sentences) + [question]))
    key2wire: dict[str, int] = {}
    entities: list[Entity] = []
    gates: list[Gate] = []
    q_wires: list[int] = []
    for i, doc in enumerate(docs):
        wires = []
        for t in doc:
            if t.pos_ not in NOUN_POS:
                continue
            key = t.lemma_.lower() if t.pos_ != "PRON" else f"<pron:{i}:{t.i}>"
            if key not in key2wire:
                key2wire[key] = len(entities)
                entities.append(Entity(id=len(entities), name=t.text))
            w = key2wire[key]
            entities[w].mentions.append((i, t.text))
            if w not in wires:
                wires.append(w)
        if i == len(sentences):
            q_wires = wires
            break
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        gates.append(Gate(idx=i, text=sentences[i], wires=wires,
                          subject=wires[0] if wires else None,
                          predicate=root.lemma_.lower() if root is not None else None,
                          objects=wires[1:]))
    return TextCircuit(sentences=list(sentences), entities=entities, gates=gates,
                       question=question, question_wires=q_wires, backend="surface",
                       meta={"n_wires": len(entities)})
