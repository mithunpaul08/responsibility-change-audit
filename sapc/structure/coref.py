"""Coreference over *pre-split* sentences, shared by the discocirc and coref backends.

lambeq's MaverickCoreferenceResolver re-splits the text with spaCy, which can
disagree with the dataset's sentence boundaries (abbreviations, quotes). We
keep the dataset's boundaries: each sentence is tokenised separately, the
token lists are handed to Maverick as one document, and clusters come back
as (sentence index, token index) mentions.

The returned structure is identical to lambeq's `CorefDataT`
(list over clusters, each a list over sentences of token offsets) so the
discocirc backend can feed it straight into the reader's wire bookkeeping.
"""
from __future__ import annotations

from functools import lru_cache

SPACY_NOUN_POS = {"NOUN", "PROPN", "PRON"}


@lru_cache(maxsize=1)
def spacy_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _allow_omegaconf_in_torch_load() -> None:
    """The Maverick checkpoint (sapienzanlp/maverick-mes-ontonotes) pickles its
    hydra/omegaconf config. torch>=2.6 `weights_only` loading rejects those
    classes unless they are allow-listed explicitly. We allow-list exactly the
    omegaconf container/metadata classes and nothing else."""
    import torch
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode, StringNode, IntegerNode, FloatNode, BooleanNode
    import typing
    import collections
    torch.serialization.add_safe_globals([
        DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, StringNode,
        IntegerNode, FloatNode, BooleanNode, typing.Any, collections.defaultdict, dict, list,
    ])


class PresplitMaverick:
    def __init__(self, device: str = "cpu",
                 hf_name_or_path: str = "sapienzanlp/maverick-mes-ontonotes"):
        from maverick import Maverick
        _allow_omegaconf_in_torch_load()
        self.model = Maverick(hf_name_or_path=hf_name_or_path, device=device)
        # transformers>=5 loads the encoder in the checkpoint's fp16 while the
        # classifier heads are fp32; unify to fp32 (CPU has no fp16 matmul).
        self.model.model = self.model.model.float()
        self.nlp = spacy_nlp()

    def tokenise(self, sentences: list[str]):
        """Returns per-sentence spaCy docs (tokens with POS, lemma, dep)."""
        return list(self.nlp.pipe(sentences))

    @staticmethod
    def _mention_head(doc, a: int, b: int, pos_flat, fa: int, fb: int) -> int:
        """Reduce a mention span (token offsets a..b within `doc`) to one token.

        lambeq's rule takes the *last* proper noun in the span, which turns
        "Mat Hoffman (born January 9, 1972)" into "January" and then links
        every later "He" to a wire that is not Hoffman's. We instead take the
        span's syntactic head (the token whose dependency head lies outside the
        span) when it is a noun/pronoun, and fall back to lambeq's rule otherwise.
        """
        head = None
        for t in doc[a:b + 1]:
            if t.head.i < a or t.head.i > b or t.head.i == t.i:
                head = t
                break
        if head is not None and head.pos_ in SPACY_NOUN_POS:
            return head.i
        is_propn, start = False, a
        for k in range(fa, fb + 1):
            pos = pos_flat[k]
            if not is_propn:
                is_propn = pos == "PROPN"
            if pos in SPACY_NOUN_POS and ((is_propn and pos == "PROPN")
                                          or (not is_propn and pos != "PROPN")):
                start = k - fa + a
        return start

    def coref(self, sentences: list[str]):
        """Returns (tokens, docs, corefs).

        tokens : list[list[str]]
        docs   : list[spacy.tokens.Doc]  (one per sentence, dependency-parsed)
        corefs : CorefDataT – one entry per cluster; entry[i] lists token
                 offsets in sentence i that start a mention of that cluster.
                 Multi-token mentions are reduced to their syntactic head noun
                 (see `_mention_head`; lambeq's last-PROPN rule is the fallback).
                 Trivial singleton clusters are appended for every noun token so
                 that every noun gets a wire, again mirroring lambeq.
        """
        docs = self.tokenise(sentences)
        tokens = [[t.text for t in d] for d in docs]
        n = len(sentences)
        sent_of, pos_flat, offset = [], [], [0]
        for i, d in enumerate(docs):
            sent_of.extend([i] * len(d))
            pos_flat.extend([t.pos_ for t in d])
            offset.append(offset[-1] + len(d))

        clusters = []
        if sum(len(t) for t in tokens) > 0:
            out = self.model.predict(tokens)
            for cluster in out["clusters_token_offsets"]:
                per_sent = [[] for _ in range(n)]
                for (a, b) in cluster:
                    if sent_of[a] != sent_of[b]:
                        continue
                    si = sent_of[a]
                    start = self._mention_head(docs[si], a - offset[si], b - offset[si], pos_flat, a, b)
                    per_sent[si].append(start)
                clusters.append(per_sent)
        # trivial clusters for every noun token (lambeq does the same)
        for i, d in enumerate(docs):
            for t in d:
                if t.pos_ in SPACY_NOUN_POS:
                    per_sent = [[] for _ in range(n)]
                    per_sent[i] = [t.i]
                    clusters.append(per_sent)
        return tokens, docs, clusters


def first_mention_map(corefs) -> dict[tuple[int, int], tuple[int, int]]:
    """(sent, tok) -> (sent, tok) of the cluster's first mention. Same as
    lambeq's `dict_from_corefs`: earlier (real) clusters win over the trivial ones."""
    m: dict[tuple[int, int], tuple[int, int]] = {}
    for cluster in corefs:
        mentions = [(i, t) for i, toks in enumerate(cluster) for t in toks]
        if not mentions:
            continue
        for men in mentions:
            m.setdefault(men, mentions[0])
    return m
