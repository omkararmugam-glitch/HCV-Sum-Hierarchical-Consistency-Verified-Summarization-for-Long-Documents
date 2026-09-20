# Labelled pair set for evaluating Stage 3 on real documents

100 cross-section sentence pairs from four real public documents, labelled contradiction /
not-contradiction. Built to answer a question the four synthetic samples could not: on the kinds of
document this pipeline is meant for, what do the NLI rule and the candidate pre-filters (similarity
floor, comparison-wording filter, shared-entity gate) actually do?

## Frozen before evaluation

Every label below was written **before any NLI model scored any of these pairs**.
`scripts/build_pair_candidates.py` samples pairs using embeddings only, and the NLI model is never
loaded while the set is built. The files were hashed at **2026-09-18T17:31:00Z**, before the first run
of `scripts/eval_pairs.py`, which verifies the hashes and refuses to run if a label file has changed.

| File | SHA-256 |
|---|---|
| `candidates.jsonl` | `7baea4243e822a292dce07a47191ef90a5e000fc6cdae9f357037497754adc13` |
| `natural_labels.json` | `108eb267970c5d613cec02cb035002aeaa74a5b942f1e95e408d14d3bc8ffcc0` |
| `constructed_pairs.jsonl` | `2d64a9e043de6191f01a5a2beca3ae5a2f4dec661f9eae12fea640e19b6153da` |

If a label turns out to be wrong after evaluation, it must be corrected in a new, separately hashed
file with the reason stated -- never silently edited.

## Sources (one per document type in the project brief)

| Type | Document | Status |
|---|---|---|
| research | Chen et al. 2025, *On Reference (In-)Determinacy in NLI* (ar5iv) | CC BY-SA 4.0 |
| financial | FOMC minutes, September 17-18, 2024 | US government, public domain |
| contract | ImageWare Systems maintenance agreement (CUAD v1) | CC BY 4.0 |
| transcript | FOMC press conference, September 18, 2024 | US government, public domain |

**Substitutions:**
- The financial document is a central-bank report, not a corporate 10-K. SEC EDGAR requires a
  personal contact e-mail in every request, and none was sent.
- The transcript is a central-bank press conference, not an earnings call. It has the same structure:
  a prepared statement followed by Q&A.

**Circularity worth noting:** the research paper is the RefNLI paper itself. Its sentences were used
only as ordinary research prose, not for what they say about NLI.

Source files are rebuilt with `scripts/build_eval_sources.py` into `data/external/` (git-ignored).
Only the short quoted sentences in these files are kept here.

## Two kinds of pair

**64 natural pairs** (`candidates.jsonl` + `natural_labels.json`): 16 per document. Both sentences
are real, come from different sections, and pass the pipeline's own gates (both count as claims, and
cosine >= 0.20). Per document, the 8 highest-similarity pairs (the look-alikes most likely to be
flagged) plus 8 sampled at random from the rest (seed 13).

**All 64 are labelled not-contradiction.** None of the sampled pairs was a genuine contradiction, as
expected of real, edited documents. They measure false positives on realistic input. Eleven are hard
negatives, sentences that look contradictory but are not:
- 50 basis points vs 1/2 percentage point
- 2.2 percent now vs 2 percent by 2026
- MORPHO indemnifies XIMAGE vs XIMAGE's liability to MORPHO
- a reporter quoting the Chair and adding "even though..."

**36 constructed contradictions** (`constructed_pairs.jsonl`): 9 per document. Sentence **A is a
verbatim real sentence** (checked against the source text), and sentence **B is a counter-claim
written by hand** to contradict it in the document's own register.

| Category | Count | What it needs |
|---|---|---|
| explicit_opposition | 14 | direct negation or antonym |
| derived_numeric | 7 | arithmetic or unit conversion (72 hours vs five days; 7 - 2.2 is not "roughly 3") |
| numeric_direct | 5 | two different values for the same quantity |
| implicit | 5 | an inference step (e.g. what an agreement statistic presupposes) |
| quantifier | 3 | "almost all" vs "most", "every" vs "several" |
| causal_denial | 2 | "did not depend on", "has nothing to do with" |

Balanced by what the claim is about: **18 "named"** (the subject is a proper name, acronym or figure,
e.g. XIMAGE, the Committee, RefNLI) and **18 "common"** (an ordinary noun phrase, e.g. "the labor
market", "mortgage rates"). That balance is deliberate: it is the variable the entity-gate question
turns on (FINDINGS.md 9.5).

## Limitations -- read before quoting any number from this set

1. **One labeller, who also built the pipeline.** Every label has a written rationale so it can be
   audited, but the labels have not been independently checked.
2. **The contradictions are constructed, not found.** Counter-claims written by hand may be cleaner
   and more explicit than contradictions that occur naturally, so recall measured here is likely an
   optimistic upper bound.
3. **Small.** 36 contradictions and 64 non-contradictions. A difference of two or three pairs is noise.
4. **One document per type**, so a type's numbers describe that document, not the type.
5. The natural pairs come from the pipeline's own candidate sampler. They are representative of what
   Stage 3 compares, not of sentence pairs in general.
