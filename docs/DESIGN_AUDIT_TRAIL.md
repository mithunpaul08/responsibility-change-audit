# Angle 4: audit trails from text circuits (design, pre-registered 2026-11-02)

Status: design written before any document was generated or scored. Written
with Claude (Fable 5.1) for Mithun Paul, NAU. Uses the Angle 2 structure
backends unchanged (`sapc/structure/`).

## 0. Question

A reviewer of a contract or incident report wants a map: for every sentence,
which party's duties did it change, and how. Law, finance and healthcare pay
for this traceability. A DisCoCirc text circuit gives exactly this shape: one
wire per entity, one gate per sentence, a gate acts on the wires it touches.
The question is whether that map, read off the circuit, is good enough to be
useful, and whether it beats (a) plain string matching, (b) a coreference
resolver without the formalism, and (c) a local LLM asked for the same map.

Time is not a constraint here: a document is parsed once, offline, and the
reviewer already spends hours on it. So this is the application where the
parser's cost is easiest to justify. The result must still show the parse buys
something an LLM plus coreference does not: a checkable per-sentence trace with
the exact mention as evidence, at comparable accuracy.

## 1. Task

Input: a document as a list of sentences, and its party list (names and
defined-term aliases, as a reviewer would have them).

Output: for each sentence `s`, the set `affects(s)` of parties whose duty or
responsibility the sentence creates, modifies, ends or transfers. Sentences
that only mention a party (definitions, rights without duties, third parties,
recitals) have `affects(s) = {}`.

Derived output: for each party `p`, its trace `trace(p) = {s : p in affects(s)}`
in document order.

## 2. Data: synthetic documents with gold by construction

Real contracts have no sentence-level gold for "whose duty changed", and
annotating them is out of scope. Documents are generated from templates with a
fixed seed, so the gold is exact and the hard cases are controlled. Two genres:

| Genre | Parties | Duty-changing sentences | Distractors |
|---|---|---|---|
| contract | 2 to 4 organisations, each introduced as `Name ("the Role")`, later referred to by name, role alias, or `it`/`its` | shall-clauses (CREATE), amendments (MODIFY), cessation (END), assignment to a new party (TRANSFER) | definitions, recitals, governing law, notices, rights (`may`), third-party mentions (auditors, banks) |
| incident | 2 to 4 named people with he/she/they pronouns | assignment of responsibility (CREATE), re-assignment (TRANSFER), completion/discharge (END), change of scope (MODIFY) | timestamps, equipment, weather, witness statements, observations |

Each generated sentence carries: `affects` (party ids), `mentions` (party ids
mentioned at all, a superset of `affects`), `change_type` in
{CREATE, MODIFY, END, TRANSFER, NONE}, `ref_type` describing how the affected
party is referred to in {name, alias, pronoun, possessive, passive, none}.

Controlled hard cases, each at a fixed share of duty-changing sentences:
* bilateral clause: "The Buyer shall pay the Supplier": `affects = {Buyer}`,
  `mentions = {Buyer, Supplier}`
* right, not duty: "The Supplier may inspect the premises": `affects = {}`
* pronoun / alias reference: "Notwithstanding clause 3, it shall deliver within 45 days"
* possessive: "The Supplier's obligation to insure the goods ceases on 1 March"
* passive: "The goods shall be insured by the Supplier"
* third-party mention: "The Buyer's auditor, Meridian LLP, prepared the report": `affects = {}`
* new party mid-document (TRANSFER): "The Buyer assigns its payment obligation to Northwind plc"

Size: 100 documents per genre, 12 to 30 sentences each, seed 0. The generator
asserts `affects ⊆ mentions` and that every party in `affects` is reachable by
string from the party list (name or alias) somewhere in the document.

## 3. Methods

All structural methods build a `TextCircuit` with an Angle 2 backend, map
entity wires to parties, and read `affects(s)` off the gate for sentence `s`.

Wire-to-party mapping: a party owns every wire that has at least one mention
whose text equals one of the party's surface tokens (name head word or alias
noun), as given in the party list. This is the lenient mapping and is the
primary one: the party list is an input to every method, including the LLM.
The strict mapping (a party owns only the wire of its first mention) is
reported as a diagnostic of how much the coreference link itself contributes.

| Name | Backend | `affects(s)` |
|---|---|---|
| `surface_any` | surface (no coreference) | parties whose name or alias string occurs in `s` |
| `coref_any` | coref (Maverick + spaCy, no lambeq) | parties owning any wire in `gate(s).wires` |
| `coref_subj` | coref | party owning `gate(s).subject`, else {} |
| `dc_any` | discocirc (lambeq Bobcat + Maverick) | parties owning any wire in `gate(s).wires` |
| `dc_subj` | discocirc | party owning `gate(s).subject` (first wire in the gate's domain) |
| `llm_zs` | none | local `mlx-community/Qwen2.5-14B-Instruct-4bit`, greedy, one prompt per document: party list plus numbered sentences, asked to list `index: parties` for every sentence that creates, changes, ends or transfers a party's duty or responsibility |
| `gold` | bound | the generator's labels |

Change type for structural methods: the gate predicate lemma mapped through a
small table (`configs/audit_change_types.yaml`: e.g. `cease, end, expire,
terminate -> END`; `assign, transfer, delegate, novate -> TRANSFER`;
`amend, extend, reduce, replace, vary -> MODIFY`; anything else with a non-empty
`affects` -> CREATE). This table is the only task-informed component and its
effect is reported separately. The LLM is asked for the change type in the same
output line. Change type is a secondary metric.

Parse failures: if the discocirc backend raises on a document, the document
is excluded from every method's aggregate and the rate is reported.

## 4. Metrics

Primary, over (sentence, party) pairs within the non-excluded documents:
* micro precision, recall, F1 of `affects`
* sentence exact-set accuracy (predicted set equals gold set, including {})
* per-party trace recall and precision; share of parties with an exact trace

Breakdowns: genre; `ref_type`; `change_type`; `mentions != affects`
(bilateral and third-party sentences) versus the rest.

Secondary: change-type accuracy on sentences where both gold and prediction
have a non-empty `affects`.

Cost: structure build seconds per document; LLM seconds and prompt tokens per
document. The machine is shared with the Angle 3 grid during this run, so
times are indicative only and are labelled as such.

Statistics: paired bootstrap over documents (10,000 resamples, seed 0) on
per-document F1 for every method pair reported; two methods differ only if the
95% CI excludes 0.

## 5. Hypotheses

* **H4a (coreference).** On sentences with `ref_type` in {pronoun, possessive}
  the recall of `dc_any` exceeds that of `surface_any`, CI above 0. This only
  confirms the coreference stage works on this text; it is the precondition.
* **H4b (against the LLM).** Prediction, stated before running: `llm_zs` has
  the highest micro F1. The decision rule is about the gap: if the best
  structural method is within 5 F1 points of `llm_zs`, the circuit map is
  competitive and its advantages (deterministic, every attribution carries a
  mention span as evidence, no generation) make it worth pursuing. If the gap
  exceeds 10 points, structure alone is not the product; at most it is a
  candidate generator for an LLM. Between 5 and 10: inconclusive, report.
* **H4c (predicted failure modes).** `dc_subj` loses recall on `ref_type` in
  {passive, possessive} relative to its recall on `name`; `dc_any` loses
  precision on `mentions != affects` sentences relative to the rest. These
  are predictions of where reading the map off the raw circuit is not enough.
* **H4-null (formalism).** `dc_any` and `coref_any` agree on at least 95% of
  (sentence, party) decisions. If so, the structural signal is coreference and
  the finding must be worded that way, as in Angle 2.

## 6. Procedure

1. `experiments/audit_trail/gen_docs.py` -> `data/processed/audit/docs.jsonl`
2. `experiments/audit_trail/build_structures.py` -> `data/processed/structures/audit_{surface,coref,discocirc}.jsonl` (resumable, Angle 2 store)
3. `experiments/audit_trail/run_methods.py` -> `results/a4_audit/predictions.jsonl` (structural methods first, then `llm_zs`, reader answers cached)
4. `experiments/audit_trail/analyze.py` -> `results/a4_audit/{summary.csv,comparisons.csv,report.md}`

## 7. What a result would and would not show

A positive result shows that a per-sentence, per-party change map read off a
text circuit is accurate enough on controlled documents to serve as an audit
trail, and says whether the categorical construction adds anything over
coreference. It does not show this holds on real contracts (the parser's
coverage on legal prose is unknown and is the first thing to measure next),
and it says nothing about quantum hardware.

## 8. Results (2026-11-02, all 200 documents, none excluded)

Full tables: `results/a4_audit/report.md`. Headline micro F1 over (sentence,
party) pairs: `llm_zs` 0.706, `coref_any` 0.653, `dc_any` 0.649, `surface_any`
0.642, `dc_subj` 0.599, `coref_subj` 0.576.

* **H4a: pass.** Pronoun recall `dc_any` 0.83 vs `surface_any` 0.14.
  Defined-term aliases are linked by Maverick (strict and lenient wire mapping
  give the same numbers), so coreference does its job on this text.
* **H4b: inconclusive by the pre-registered rule.** The LLM leads the best
  structural method by 5.3 F1 points (per-document CI excludes 0 in both
  genres). The error profiles are opposite: structural methods recall 0.98 to
  0.99 of duty changes at precision 0.49; the LLM recalls 0.76 at precision
  0.66. For an audit, a missed change is the costlier error, so neither is
  usable alone.
* **H4c: confirmed, with one surprise.** `dc_subj` recall is 0.00 on passives
  and 0.01 when the affected party is the object ("X instructed Y to ...").
  `dc_any` precision is 0.17 on sentences where mentions differ from affects
  and 0.97 elsewhere: every structural false positive is a bilateral clause,
  a right, a recital or a witness statement. The surprise: `dc_subj` recalls
  1.00 of possessive references ("The Supplier's obligation ... ceases") where
  `coref_subj` recalls 0.14, because the DisCoCirc wire order puts the possessor
  first while spaCy's nsubj is the head noun. It also has the best change-type
  accuracy (0.88).
* **H4-null: the signal is coreference.** `dc_any` and `coref_any` agree on
  99.8% of decisions; `dc_subj` and `coref_subj` on 95.7%. Coref alone is
  nine times faster (0.57 s/doc vs 3.9 s/doc).

Reading: the circuit answers "which sentences touch this party" almost
perfectly and with a mention span as evidence, but it cannot answer "whose
duty changed", which depends on modality (shall vs may vs wishes), voice and
argument role. That is a semantic filter the formalism does not provide. The
natural next step, not run here and post hoc if run, is a two-stage audit:
circuit (or coref) candidates for recall and evidence, then a per-candidate
filter (an LLM asked about one sentence and one party, or a modality and
role rule) for precision. The LLM's own recall gap (it missed 24% of duty
changes, including plain "shall" clauses) is the argument for keeping the
structural stage.

## Extension E1: two-stage hybrid (added 2026-11-02 after reading section 8; post hoc, not pre-registered)

Stage 1: `coref_any` candidates, i.e. every (sentence, party) pair where the
sentence touches a wire the party owns, with the mention text in that sentence
as evidence. `coref_any` stands in for `dc_any` (99.8% agreement, 9x faster).
Stage 2: the same 14B reader, greedy, asked only about the candidates:

| Name | Stage 2 prompt |
|---|---|
| `hyb_doc` | one prompt per document listing only the candidate sentences, each annotated with the parties it refers to and how ("P0 as 'it'"); same output format as `llm_zs` |
| `hyb_sent` | one prompt per candidate sentence with the previous sentence as context and the candidate parties with their mention text; output `P<id> \| <type>` lines or `NONE` |

Predictions are intersected with the stage-1 candidates, so recall is bounded
by `coref_any` recall (0.989). Expectation written before running: precision
rises above `llm_zs` (0.658) because pronouns and aliases arrive resolved and
the model only has to judge modality and role; recall stays above `llm_zs`
(0.761) if the model accepts most true candidates. Any result needs
confirmation on new documents before it is claimed. Cost: `hyb_doc` 200
calls; `hyb_sent` one call per sentence with at least one candidate.

### E1 results (2026-11-02, all 200 documents; post hoc)

| Method | P | R | F1 | sent. exact | change-type acc | calls/doc | prompt tok/doc |
|---|---|---|---|---|---|---|---|
| `coref_any` (stage 1 alone) | 0.488 | 0.989 | 0.653 | 0.594 | 0.83 | 0 | 0 |
| `llm_zs` | 0.658 | 0.761 | 0.706 | 0.703 | 0.60 | 1 | 695 |
| `hyb_doc` | 0.606 | 0.843 | 0.705 | 0.690 | 0.54 | 1 | 921 |
| `hyb_sent` | 0.714 | 0.834 | 0.769 | 0.760 | 0.72 | 16.7 | 4,354 |

Paired bootstrap on per-document F1 (10,000 resamples, seed 0):
`hyb_sent` − `llm_zs` +0.068 [+0.050, +0.088] (contract +0.088, incident
+0.049, both CIs above 0); `hyb_doc` − `llm_zs` +0.005 [−0.015, +0.025], not
different; `hyb_sent` − `coref_any` +0.120 [+0.105, +0.135].

* **The expectation held for `hyb_sent`, not for `hyb_doc`.** Asking the
  model one sentence at a time with resolved candidates raised precision from
  0.658 to 0.714 and recall from 0.761 to 0.834 at once. The whole-document
  variant with annotated candidates is indistinguishable from the plain LLM:
  recall rose but precision fell, and it labels most changes as TRANSFER
  (change-type accuracy 0.54).
* **Recall is far below the stage-1 bound (0.834 vs 0.989).** The misses are
  almost all in incident reports: completion sentences the generator labels
  END ("The electrician finished the cordon", "The inspection ... was
  completed by the inspector"; 62 of 77 alias-END and 48 of 56 passive-END
  gold pairs missed) and hand-overs labelled TRANSFER ("passed the racking
  check to the inspector"; 49 of 116 missed). The model does not read task
  completion as a duty ending. On contracts `hyb_sent` recall is 0.959. This
  is a label-convention gap between the generator and the prompt, not a
  coreference failure; it would be fixed by a sentence in the prompt or by a
  different gold convention, and either change must be made before a new
  document set, not on this one.
* **Precision is still the bilateral clause.** `hyb_sent` precision on
  sentences where mentions differ from affects is 0.357 (0.982 elsewhere); the
  plain LLM is 0.339 there. The model still tends to list both parties in "A
  shall pay B".
* **Cost.** `hyb_sent` makes one call per candidate sentence (16.7 of 21.8
  sentences per document on average) and reads 6x the prompt tokens of
  `llm_zs`; median 30 s/doc against 8.5 s/doc for `llm_zs` on the same GPU
  (uncached documents only; the GPU was free for this run, unlike the main
  run). `hyb_doc` costs the same as `llm_zs` and buys nothing.

Reading: the product shape from section 8 survives: structure for recall and
evidence, a narrow per-candidate question for the judgment. The gain over the
plain LLM is 6.8 F1 points at 3.5x the wall-clock, and every accepted pair
carries the mention span from stage 1. Because the hybrid was designed after
reading section 8, this needs a fresh document set (new seed, and ideally a
genre not used here) before it is claimed; the END-completion and bilateral
gaps are the two prompt changes to pre-register for that run.

## Extension E2: confirmation run (pre-registered 2026-11-02, written before any seed-1 document was generated)

Purpose: E1 was designed after reading section 8, and the two prompt fixes
below were chosen after reading E1's error profile on the seed-0 documents.
E2 tests both on documents nobody has looked at. Nothing is tuned after
launch: one run, no prompt edits, every arm reported.

**Data.** The generator code is unchanged from the seed-0 run; only the seed
differs: seed 1, 100 documents per genre, written to
`data/processed/audit/docs_seed1.jsonl` with ids `audit-s1-<genre>-<nnn>`.
Same two genres. A third genre would be the stronger test of transfer but
needs new templates and new gold conventions, which is a design of its own
and is deferred. Structures go to `audit_{surface,coref,discocirc}_seed1.jsonl`;
predictions to `results/a4_audit_seed1/`.

**Arms.** `surface_any`, `coref_any`, `coref_subj`, `dc_any`, `dc_subj`
(replication of section 8, secondary); `llm_zs` with the seed-0 prompt,
byte-identical; `hyb_sent` with the seed-0 prompts, byte-identical;
`llm_zs_v2` and `hyb_sent_v2` with the v2 guidance. `hyb_doc` is dropped
(E1: no gain over `llm_zs`). Same reader, greedy, same cache.

**The v2 guidance.** Two sentences are added to the guidance text of both
`llm_zs_v2` and `hyb_sent_v2`, so the LLM arm gets the same label conventions
as the hybrid and the comparison stays fair:

1. (label convention for completions and hand-overs) "A sentence that says a
   party finished, completed or closed out a task, or was relieved of or
   released from it, ends that party's responsibility (END). A sentence that
   says a party handed, passed or reassigned a task to another party
   transfers it (TRANSFER)."
2. (bilateral clauses) "When one party must do something to, for or with
   another party (pay, deliver, notify, brief, escort, report to), only the
   party that must act is affected; the party that receives, is notified or
   benefits is not, even though it is named."

`hyb_sent_v2` additionally gets, after its candidate list, "Often only one of
the listed parties is affected, and sometimes none." This has no counterpart
in the document prompt, which already says to output nothing for unaffected
sentences; it is reported as part of the v2 hybrid.

**Hypotheses and decision rules** (paired bootstrap on per-document F1,
10,000 resamples, seed 0; a difference counts only if the 95% CI excludes 0):

* **E2a (E1 replicates).** `hyb_sent` − `llm_zs` > 0 on the seed-1
  documents, both genres. Predicted size about +0.07. If the CI includes 0,
  E1 was a seed-0 artefact and is withdrawn.
* **E2b (the gain survives fair prompts).** `hyb_sent_v2` − `llm_zs_v2` > 0.
  The claim "structure for candidates and evidence, a per-candidate LLM
  judgment for precision, beats one LLM pass over the document" is made only
  if E2a and E2b both hold.
* **E2c (the fixes do what they were written for).** On incident documents,
  `hyb_sent_v2` recall on gold END pairs exceeds `hyb_sent` recall on the
  same pairs by at least 0.15 (point estimate; it was 0.64 overall and 0.2
  on incident END in E1). Precision on sentences where mentions differ from
  affects rises for both v2 arms over their v1 counterpart. Prediction:
  `hyb_sent_v2` micro F1 at least 0.80; `llm_zs_v2` gains less than
  `hyb_sent_v2` because the plain LLM's misses are spread across change
  types, not concentrated on completions.
* **Replication of section 8 (secondary).** `llm_zs` − `coref_any` between
  +0.03 and +0.08; `dc_any` vs `coref_any` agreement at least 0.95.

**Cost.** Calls, prompt tokens and seconds per document per arm, labelled
indicative: the machine is shared with an Angle 3 training job and a
circuit-bench reader at launch.

**Procedure.** `gen_docs.py --seed 1 --tag seed1`;
`build_structures.py --tag seed1`; `run_methods.py --tag seed1 --llm --llm-v2`;
`run_hybrid.py --tag seed1 --variants hyb_sent hyb_sent_v2`;
`analyze.py --tag seed1`, chained in one background script, log
`results/logs/audit_confirm.txt`.

### E2 results (2026-11-02, seed-1 set, all 200 documents, none excluded, one run)

Full tables: `results/a4_audit_seed1/report.md`.

| Method | P | R | F1 | sent. exact | change-type acc | prompt tok/doc | median s/doc |
|---|---|---|---|---|---|---|---|
| `coref_any` | 0.492 | 0.986 | 0.656 | 0.601 | 0.83 | 0 | 0.9 |
| `llm_zs` | 0.656 | 0.770 | 0.708 | 0.695 | 0.61 | 699 | 11.0 |
| `llm_zs_v2` | 0.680 | 0.792 | 0.732 | 0.719 | 0.59 | 807 | 15.0 |
| `hyb_sent` | 0.714 | 0.833 | 0.769 | 0.756 | 0.72 | 4,405 | 38.0 |
| `hyb_sent_v2` | 0.710 | 0.969 | **0.820** | 0.796 | 0.66 | 6,462 | 50.4 |

Paired bootstrap on per-document F1 (10,000 resamples, seed 0):

* **E2a: pass.** `hyb_sent` − `llm_zs` +0.060 [+0.040, +0.080]; contract
  +0.060, incident +0.061, both CIs above 0. E1 replicates on documents
  nobody had looked at, at almost the same numbers (seed 0: hybrid 0.769,
  LLM 0.706; seed 1: 0.769, 0.708).
* **E2b: pass.** `hyb_sent_v2` − `llm_zs_v2` +0.094 [+0.074, +0.114];
  contract +0.030 [+0.002, +0.061], incident +0.157. The gain survives giving
  the plain LLM the same label conventions. With E2a and E2b both holding,
  the claim is made: on these documents, coreference candidates with a
  per-candidate LLM judgment beat one LLM pass over the document.
* **E2c: one fix worked, one did not.** Incident END recall for the hybrid
  went from 0.376 to 0.963 (rule: at least +0.15), and `hyb_sent_v2` recall
  overall is 0.969 against the stage-1 bound of 0.986: the completion and
  hand-over conventions are now read correctly. The bilateral fix did
  nothing for the hybrid: precision on sentences where mentions differ from
  affects is 0.366 before and 0.357 after; for the plain LLM it rose from
  0.354 to 0.383. The prediction that `hyb_sent_v2` reaches F1 0.80 held
  (0.820), and the LLM gained less from v2 than the hybrid (+0.023 vs
  +0.050), as predicted. On contracts, which have no completion sentences,
  v2 changed nothing for the hybrid (−0.007 [−0.017, +0.002]); on incidents
  it added +0.121.
* **Section 8 replicates.** `llm_zs` − `coref_any` +0.055 (seed 0: +0.052);
  H4b gap 5.2 points, again inconclusive by the rule; `dc_any` vs `coref_any`
  agreement 0.997; `dc_subj` passive recall 0.003, possessive 1.00;
  `dc_subj` change-type accuracy 0.89, still the best of any method.

What remains: the hybrid's errors are now almost entirely false positives on
bilateral clauses, 0.36 precision there against 0.99 elsewhere, and
instructing the model in prose did not move it. The 14B model lists both
parties of "A shall pay B" whichever way it is asked. That is a role
question (who must act), which is the one thing the gate's argument
structure does encode: `dc_subj` has 0.73 recall on name references with
near-perfect precision on bilateral clauses but misses passives and objects.
A rule that takes the subject wire for active clauses and the agent wire
for passives, used as a prior or a tie-breaker for the LLM, is the next
experiment and would need its own pre-registration. Cost: `hyb_sent_v2` is
4.6x the wall-clock and 9x the prompt tokens of `llm_zs` on the same GPU,
all indicative (shared machine).
