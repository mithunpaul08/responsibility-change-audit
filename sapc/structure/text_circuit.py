"""The shared intermediate representation for Angle 2.

Every structure backend (DisCoCirc via lambeq, coreference-only, surface
string match) produces a ``TextCircuit``. Selectors never see the backend;
they only see this object. That is what makes the backends comparable.

Vocabulary (mirrors DisCoCirc):
  * an *entity* is a noun wire that runs through the whole text
  * a *gate* is one sentence acting on the wires it mentions
  * the *question wires* are the entity wires the question mentions
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Entity:
    id: int
    name: str                      # canonical surface form (first mention)
    mentions: list[tuple[int, str]] = field(default_factory=list)
    # (sentence index, mention text) for every mention, including pronouns


@dataclass
class Gate:
    idx: int                       # sentence index in the original context
    text: str
    wires: list[int]               # entity ids touched by the sentence
    subject: int | None = None     # entity id of the grammatical subject
    predicate: str | None = None   # root verb lemma (spaCy), used by DCE
    objects: list[int] = field(default_factory=list)
    frame: str | None = None       # DisCoCirc frame label (pregroup-tree root word), informational


@dataclass
class TextCircuit:
    sentences: list[str]
    entities: list[Entity]
    gates: list[Gate]
    question: str
    question_wires: list[int]
    backend: str
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def wires_of(self, idx: int) -> set[int]:
        return set(self.gates[idx].wires)

    def entity_name(self, eid: int) -> str:
        return self.entities[eid].name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextCircuit":
        ents = [Entity(id=e["id"], name=e["name"],
                       mentions=[tuple(m) for m in e["mentions"]]) for e in d["entities"]]
        gates = [Gate(**g) for g in d["gates"]]
        return cls(sentences=d["sentences"], entities=ents, gates=gates,
                   question=d["question"], question_wires=list(d["question_wires"]),
                   backend=d["backend"], meta=d.get("meta", {}))
