"""Backend `discocirc`: entity wires from lambeq's DisCoCirc text-circuit construction.

We do not build the full composed diagram for pruning (its boxes carry no
sentence index). Instead we run the exact per-sentence steps of
`lambeq.experimental.discocirc.DisCoCircReader.text2circuit`:

    Bobcat CCG parse -> pregroup tree -> TreeRewriter(determiner, auxiliary)
    -> _tree2circuital (drags noun leaves out as wires) -> noun ids
    -> coreference canonicalisation -> wire ids

and record, for every sentence, the wires its gate acts on, the gate's label
(the tree root word = the frame name in the circuit) and the first wire in
the gate's domain (the subject in DisCoCirc's wire ordering).

The question is appended as the final sentence so its nouns are resolved
against the same wires; it becomes `question_wires` and is not a gate.

Faithfulness check: `build_full_diagram()` builds the real lambeq diagram for
the same text so a test can confirm the wire count and noun boxes agree with
what this backend reports (tests/test_discocirc_backend.py).
"""
from __future__ import annotations

import os
from functools import lru_cache

from sapc.structure.coref import PresplitMaverick, first_mention_map
from sapc.structure.text_circuit import Entity, Gate, TextCircuit

REWRITE_RULES = ("determiner", "auxiliary")


@lru_cache(maxsize=1)
def _parser(device: str = "cpu"):
    from lambeq import BobcatParser
    kw = {"device": device, "verbose": "suppress"}
    # lambeq's model host (qnlp.cambridgequantum.com) no longer resolves, so the
    # parser must be pointed at a local copy of the Bobcat weights. Default to
    # lambeq's own cache directory; override with BOBCAT_MODEL_PATH.
    path = os.environ.get("BOBCAT_MODEL_PATH") or os.path.expanduser("~/.cache/lambeq/bobcat")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Bobcat weights not found at {path}. Set BOBCAT_MODEL_PATH to a directory "
            "containing config.json, pytorch_model.bin, grammar.json (see RUNBOOK §1.4).")
    kw["model_name_or_path"] = path
    return BobcatParser(**kw)


@lru_cache(maxsize=1)
def _reader(device: str = "cpu"):
    """A DisCoCircReader whose coref resolver is never used (we supply our own)."""
    from lambeq.experimental.discocirc import DisCoCircReader, CoreferenceResolver

    class _NullCoref(CoreferenceResolver):
        def tokenise_and_coref(self, text):  # pragma: no cover
            raise RuntimeError("not used")
    return DisCoCircReader(parser=_parser(device), coref_resolver=_NullCoref())


@lru_cache(maxsize=1)
def _coref(device: str = "cpu"):
    return PresplitMaverick(device=device)


class _SentenceParseCache:
    """Bobcat is context-free per sentence; bAbI repeats sentences constantly."""
    def __init__(self):
        self.store: dict[tuple[str, ...], tuple[list[int], str, list[int]] | None] = {}

    def get(self, toks):
        return self.store.get(tuple(toks), "MISS")

    def put(self, toks, value):
        self.store[tuple(toks)] = value


_PARSE_CACHE = _SentenceParseCache()


def _sentence_nouns(reader, tokens: list[str], rewrite_rules=REWRITE_RULES):
    """Replicates one iteration of text2circuit's loop for a single sentence.

    Returns (noun token indices in gate-domain order, root word, all words of tree)
    or None if Bobcat cannot parse the sentence.
    """
    from lambeq.experimental.discocirc import TreeRewriter
    hit = _PARSE_CACHE.get(tokens)
    if hit != "MISS":
        return hit
    try:
        tree = reader._sentence2tree(tokens, True)
        tree_toks = tree.get_words()
        tree_idx = tree.get_word_indices()
        reidx = reader._calculate_reindices(tokens, tree_toks, tree_idx)
        reidx[None] = None
        tree = TreeRewriter(rewrite_rules)(tree)
        tree = reader._reindex_nodes(tree, reidx)
        root_word = tree.word
        _sdiag, _nouns, nids = reader._tree2circuital(tree, False, (), True)
        value = (list(nids), root_word, tree_toks)
    except Exception:  # noqa: BLE001  (parser failure -> caller falls back)
        value = None
    _PARSE_CACHE.put(tokens, value)
    return value


def build(sentences: list[str], question: str, device: str = "cpu",
          merge_same_noun: bool = True) -> TextCircuit:
    reader = _reader(device)
    coref = _coref(device)
    all_sents = list(sentences) + [question]
    tokens, docs, corefs = coref.coref(all_sents)
    canon = first_mention_map(corefs)

    # --- optional lemma merge: "the football" in s1 and "the football" in s6
    # become one wire if Maverick did not already cluster them. Proper nouns
    # and pronouns are left to Maverick alone.
    lemma_wire: dict[str, tuple[int, int]] = {}

    def canonical(nid):
        c = canon.get(nid, nid)
        if merge_same_noun:
            tok = docs[c[0]][c[1]]
            if tok.pos_ == "NOUN":
                key = tok.lemma_.lower()
                c = lemma_wire.setdefault(key, c)
        return c

    noun2wire: dict[tuple[int, int], int] = {}
    entities: list[Entity] = []
    gates: list[Gate] = []
    parse_failed = []
    q_wires: list[int] = []

    for i, toks in enumerate(tokens):
        parsed = _sentence_nouns(reader, toks)
        if parsed is None:
            parse_failed.append(i)
            nids = [t.i for t in docs[i] if t.pos_ in {"NOUN", "PROPN", "PRON"}]
            root_word = next((t.lemma_ for t in docs[i] if t.dep_ == "ROOT"), None)
        else:
            nids, root_word, _ = parsed
        wires = []
        for nid in nids:
            key = canonical((i, nid))
            if key not in noun2wire:
                noun2wire[key] = len(entities)
                entities.append(Entity(id=len(entities), name=tokens[key[0]][key[1]]))
            w = noun2wire[key]
            entities[w].mentions.append((i, toks[nid]))
            if w not in wires:
                wires.append(w)
        if i == len(sentences):          # the question
            q_wires = wires
            break
        subj = wires[0] if wires else None
        # The pregroup-tree root is often a preposition/particle frame ("to",
        # "up"); DCE needs the verb, so the predicate is the spaCy root lemma
        # in every backend, and the frame label is kept for inspection.
        root_tok = next((t for t in docs[i] if t.dep_ == "ROOT"), None)
        pred = root_tok.lemma_.lower() if root_tok is not None else None
        gates.append(Gate(idx=i, text=sentences[i], wires=wires, subject=subj,
                          predicate=pred, objects=wires[1:], frame=root_word))

    return TextCircuit(sentences=list(sentences), entities=entities, gates=gates,
                       question=question, question_wires=q_wires, backend="discocirc",
                       meta={"parse_failed": parse_failed,
                             "n_wires": len(entities), "merge_same_noun": merge_same_noun})


def build_full_diagram(sentences: list[str], question: str | None = None, device: str = "cpu"):
    """The genuine lambeq circuit for the same text (for tests / drawing only)."""
    from lambeq import BobcatParser
    from lambeq.experimental.discocirc import DisCoCircReader, MaverickCoreferenceResolver
    r = DisCoCircReader(parser=_parser(device),
                        coref_resolver=MaverickCoreferenceResolver(device=device))
    text = " ".join(sentences + ([question] if question else []))
    return r.text2circuit(text, rewrite_rules=list(REWRITE_RULES))
